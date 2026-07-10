"""
scripts/infra/smoke_training.py — Step 5: Short Real Training Run

Runs 50–100 actual training steps (not a full epoch) on GCP for both
encoder configs (scratch and dnabert2), measuring:
  - Loss trajectory (confirm it decreases)
  - Wall-clock time per step (for cost estimation)
  - Peak GPU memory usage

This is NOT a full training run. It uses the data/augmented/ splits produced
by augment.py. If augmented splits aren't available, it falls back to a random
sample of master_dataset_v2.csv with fake protein embeddings (and notes this).

Cost estimate math (from timing results):
  T seconds/step * N_steps_per_epoch * N_epochs = total_training_time
  GCP T4 cost: ~$0.35/hr; A100: ~$3.67/hr
  If scratch path = X sec/step, full run ≈ X * 15904 * 100 / 3600 hours

Usage:
    python scripts/infra/smoke_training.py
    python scripts/infra/smoke_training.py --n-steps 100 --encoder scratch
    python scripts/infra/smoke_training.py --n-steps 50 --encoder dnabert2
    python scripts/infra/smoke_training.py --n-steps 50 --batch-size 8
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def get_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data_for_smoke(n_rows: int = 256) -> tuple[pd.DataFrame, dict, dict]:
    """Load dataset rows + ViennaRNA cache."""
    from config import DATA_PROCESSED, DATA_AUGMENTED, VIENNA_CACHE

    # Prefer augmented training splits (have full pipeline applied)
    aug_train = os.path.join(DATA_AUGMENTED, "tier1_train.csv")
    v2_path   = os.path.join(DATA_PROCESSED, "master_dataset_v2.csv")

    if os.path.exists(aug_train):
        df = pd.read_csv(aug_train)
        source = "data/augmented/tier1_train.csv"
    elif os.path.exists(v2_path):
        df = pd.read_csv(v2_path)
        source = "data/processed/master_dataset_v2.csv"
        print(f"{WARN} Augmented splits not found; using master_dataset_v2.csv")
        print(f"     Run `python scripts/data/augment.py` first for the real pipeline.")
    else:
        print(f"{FAIL} No dataset found — cannot run training smoke test.")
        sys.exit(1)

    # Filter to rows with valid sequences
    df = df[df["aptamer_sequence"].notna() & df["protein_sequence"].notna()]
    df = df.sample(min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"{INFO} Loaded {len(df)} rows from {source}")

    # ViennaRNA cache
    vienna_cache: dict = {}
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vienna_cache = pickle.load(f)
        print(f"{INFO} Vienna cache: {len(vienna_cache)} entries")

    # Protein embeddings: use pre-computed if available, else generate fakes
    import hashlib
    from config import ESM_EMBED_DIM, PROT_MAX_TOKENS

    emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    seq_to_path: dict = {}
    n_fake = 0

    for seq in df["protein_sequence"].dropna().unique():
        key  = hashlib.md5(f"{PROT_MAX_TOKENS}:{seq}".encode()).hexdigest()
        path = os.path.join(emb_dir, f"{key}.npy")
        seq_to_path[seq] = path
        if not os.path.exists(path):
            fake = np.random.randn(min(len(seq), PROT_MAX_TOKENS), ESM_EMBED_DIM).astype(np.float32)
            np.save(path, fake)
            n_fake += 1

    if n_fake:
        print(f"{WARN} {n_fake} FAKE protein embeddings generated (ESM-2 not run).")
        print(f"     Loss/metrics from this run are NOT meaningful for model quality.")
        print(f"     Purpose: verify pipeline mechanics and timing only.")
    else:
        print(f"{PASS} Using real pre-computed protein embeddings.")

    return df, seq_to_path, vienna_cache


def build_loader(df: pd.DataFrame, seq_to_path: dict, vienna_cache: dict,
                 batch_size: int, encoder_type: str) -> DataLoader:
    """Build a DataLoader for the smoke run."""
    import config as _cfg
    orig_type = _cfg.DNA_ENCODER_TYPE
    _cfg.DNA_ENCODER_TYPE = encoder_type

    from scripts.model.tokenizer import DNATokenizer
    from scripts.training.train import AptamerDataset, collate_fn

    tokenizer = DNATokenizer()
    ds = AptamerDataset(df, tokenizer, vienna_cache, seq_to_path)

    _num_workers = 0 if _cfg.DEVICE in ("mps", "cpu") else 2
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=_num_workers, pin_memory=(_num_workers > 0),
        persistent_workers=_num_workers > 0,
    )

    _cfg.DNA_ENCODER_TYPE = orig_type
    return loader


def run_smoke_training(
    encoder_type: str,
    n_steps: int,
    batch_size: int,
    device: torch.device,
    df: pd.DataFrame,
    seq_to_path: dict,
    vienna_cache: dict,
    use_amp: bool = False,
) -> dict:
    """
    Run n_steps training steps for the given encoder type.
    Returns timing and loss statistics.
    """
    import config as _cfg
    orig_type = _cfg.DNA_ENCODER_TYPE
    _cfg.DNA_ENCODER_TYPE = encoder_type

    try:
        from models.condaptnet import CondAptNet
        from scripts.training.losses import CondAptNetLoss
        from config import LEARNING_RATE_BASE, LEARNING_RATE_LORA, WEIGHT_DECAY, GRAD_CLIP

        header(f"Training smoke: DNA_ENCODER_TYPE={encoder_type}  device={device}")

        # Build model
        print(f"{INFO} Building CondAptNet (encoder={encoder_type})...")
        t0 = time.perf_counter()
        model = CondAptNet(predict_kd=True)
        model.set_stage1()
        model = model.to(device)
        model.protein_encoder = model.protein_encoder.to("cpu")
        build_time = time.perf_counter() - t0

        n_params   = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{INFO} Total params    : {n_params:,}")
        print(f"{INFO} Trainable params: {n_trainable:,} ({100*n_trainable/n_params:.2f}%)")
        print(f"{INFO} Build time      : {build_time:.2f}s")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Build data loader
        loader = build_loader(df, seq_to_path, vienna_cache, batch_size, encoder_type)
        loader_iter = iter(loader)

        # Optimizer — same param groups as train.py
        lora_params = [p for n, p in model.named_parameters()
                       if "lora_" in n and p.requires_grad]
        base_params  = [p for n, p in model.named_parameters()
                        if "lora_" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": LEARNING_RATE_BASE},
            {"params": lora_params, "lr": LEARNING_RATE_LORA},
        ], weight_decay=WEIGHT_DECAY)

        criterion = CondAptNetLoss()

        use_amp_here = use_amp and device.type == "cuda"
        amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp_here
                   else torch.amp.autocast("cpu", enabled=False))

        losses: list[float] = []
        step_times: list[float] = []

        print(f"\n{INFO} Running {n_steps} steps (batch_size={batch_size}, AMP={use_amp_here})...")
        print(f"{'Step':>6} {'Loss':>10} {'Step time':>12} {'GPU mem':>12}")
        print("-" * 45)

        model.train()
        for step in range(n_steps):
            try:
                apt, v, prot_tok, cond, labels, kds, prot_emb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                apt, v, prot_tok, cond, labels, kds, prot_emb = next(loader_iter)

            apt      = apt.to(device)
            v        = v.to(device)
            prot_emb = prot_emb.to(device)
            cond     = cond.to(device)
            labels   = labels.to(device)
            kds      = kds.to(device)

            t_step = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                out = model(apt, v, prot_tok, cond, protein_emb=prot_emb)

            loss, bce, kd_l = criterion(
                out.binding_prob.float(), labels,
                out.kd_pred.float() if out.kd_pred is not None else None, kds,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - t_step

            losses.append(loss.item())
            step_times.append(step_time)

            gpu_mem = ""
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / 1e6
                gpu_mem = f"{alloc:>8.1f} MB"

            if step % 10 == 0 or step == n_steps - 1:
                print(f"{step+1:>6} {loss.item():>10.4f} {step_time:>10.3f}s  {gpu_mem}")

        # Summary statistics
        peak_mem_mb = 0.0
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        mean_step_s  = sum(step_times) / len(step_times)
        med_step_s   = sorted(step_times)[len(step_times) // 2]

        # Check loss trend (last 20% should be lower than first 20%)
        n_check = max(1, n_steps // 5)
        first_n_loss = sum(losses[:n_check]) / n_check
        last_n_loss  = sum(losses[-n_check:]) / n_check
        loss_decreased = last_n_loss < first_n_loss

        results = {
            "encoder_type":   encoder_type,
            "n_steps":        n_steps,
            "batch_size":     batch_size,
            "device":         str(device),
            "amp":            use_amp_here,
            "losses":         losses,
            "first_loss":     losses[0],
            "last_loss":      losses[-1],
            "first_n_mean":   first_n_loss,
            "last_n_mean":    last_n_loss,
            "loss_decreased": loss_decreased,
            "mean_step_s":    mean_step_s,
            "median_step_s":  med_step_s,
            "peak_mem_mb":    peak_mem_mb,
        }

        print(f"\n{'─'*60}")
        print(f"  Results for encoder={encoder_type}")
        print(f"{'─'*60}")
        print(f"  First {n_check} steps mean loss  : {first_n_loss:.4f}")
        print(f"  Last  {n_check} steps mean loss  : {last_n_loss:.4f}")
        if loss_decreased:
            print(f"  {PASS} Loss decreased ({first_n_loss:.4f} → {last_n_loss:.4f})")
        else:
            print(f"  {WARN} Loss did NOT decrease ({first_n_loss:.4f} → {last_n_loss:.4f})")
            print(f"       This may be OK with random/fake embeddings or very few steps.")
        print(f"  Mean step time   : {mean_step_s:.3f}s")
        print(f"  Median step time : {med_step_s:.3f}s")
        if peak_mem_mb:
            print(f"  Peak GPU memory  : {peak_mem_mb:.1f} MB")

        # Cost estimation
        steps_per_epoch = 15904 // batch_size   # ~15904 tier1_train rows
        epoch_time_min  = mean_step_s * steps_per_epoch / 60
        full_run_h      = epoch_time_min * 100 / 60   # 100 epochs max
        t4_cost_usd     = full_run_h * 0.35           # $0.35/hr T4
        a100_cost_usd   = full_run_h * 3.67           # $3.67/hr A100

        print(f"\n  Cost estimate (encoder={encoder_type}, batch={batch_size}, 100 epochs):")
        print(f"    Steps/epoch       : ~{steps_per_epoch}")
        print(f"    Time/epoch        : ~{epoch_time_min:.1f} min")
        print(f"    Full run (100ep)  : ~{full_run_h:.1f} hours")
        print(f"    GCP T4  cost      : ~${t4_cost_usd:.2f}")
        print(f"    GCP A100 cost     : ~${a100_cost_usd:.2f}")

        return results

    except torch.cuda.OutOfMemoryError:
        print(f"{FAIL} GPU OOM with encoder={encoder_type}, batch_size={batch_size}")
        print(f"     Try: --batch-size 8 --n-steps 50")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        _cfg.DNA_ENCODER_TYPE = orig_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Short training run smoke test")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of training steps per encoder (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16; use 8 if OOM)")
    parser.add_argument("--encoder", type=str, default="both",
                        choices=["scratch", "dnabert2", "both"],
                        help="Which encoder to test (default: both)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable BF16 AMP (A100 only; halves activation memory)")
    parser.add_argument("--n-rows", type=int, default=256,
                        help="Dataset rows to sample for smoke run (default: 256)")
    args = parser.parse_args()

    header("Step 5: Short Real Training Run Smoke Test")

    device = get_device(args.device)
    print(f"{INFO} Device: {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"{INFO} GPU: {props.name}  VRAM: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free")

    df, seq_to_path, vienna_cache = load_data_for_smoke(args.n_rows)

    encoders_to_test = (
        ["scratch", "dnabert2"] if args.encoder == "both" else [args.encoder]
    )

    all_results: list[dict] = []
    for enc in encoders_to_test:
        res = run_smoke_training(
            encoder_type=enc,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=device,
            df=df,
            seq_to_path=seq_to_path,
            vienna_cache=vienna_cache,
            use_amp=args.use_amp,
        )
        all_results.append(res)

    # Final comparison table
    header("Summary: Training Smoke Test Results")
    print(f"\n{'Encoder':<12} {'Steps':>6} {'First loss':>12} {'Last loss':>12} "
          f"{'sec/step':>10} {'Peak MB':>10} {'Loss OK':>8}")
    print("─" * 75)
    for r in all_results:
        loss_ok = PASS if r["loss_decreased"] else WARN
        print(f"{r['encoder_type']:<12} {r['n_steps']:>6} {r['first_loss']:>12.4f} "
              f"{r['last_loss']:>12.4f} {r['mean_step_s']:>10.3f} "
              f"{r['peak_mem_mb']:>10.1f} {loss_ok:>8}")

    if len(all_results) == 2:
        ratio = all_results[1]["mean_step_s"] / max(all_results[0]["mean_step_s"], 1e-9)
        print(f"\n{INFO} dnabert2/scratch step-time ratio: {ratio:.2f}×")
        print(f"     (DNABERT-2 is {ratio:.1f}× {'slower' if ratio > 1 else 'faster'} per step)")

    header("Training smoke test complete")
    all_passed = all(r["loss_decreased"] for r in all_results)
    if all_passed:
        print(f"\n{PASS} Training smoke test PASSED — loss decreased for all encoder configs")
    else:
        print(f"\n{WARN} Loss did not decrease for all configs. "
              f"Check if fake embeddings were used (expected with random data).")

    print(f"\nNEXT STEP:")
    print(f"  Run full Stage 1 training:")
    print(f"  python scripts/training/train.py --use-amp --batch-size 32 --grad-accum 1")


if __name__ == "__main__":
    main()
