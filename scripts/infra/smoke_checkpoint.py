"""
scripts/infra/smoke_checkpoint.py — Step 4: Checkpoint Save/Load Smoke Test

Verifies that the checkpoint save/load cycle works correctly on GCP:
  1. Build CondAptNet and run a few training steps to produce real gradients
  2. Save a checkpoint in the exact format that train.py uses
  3. Load it back with weights_only=True (the security fix from PR #6)
  4. Verify model, optimizer, and scheduler state_dicts restore correctly
  5. Confirm the resumed model produces the same output as the saved one

The weights_only=True requirement is already in train.py (line 617) and
finetune.py (lines 332, 415) from PR #6 (security/pin-dnabert2-revision-
and-weights-only). This test confirms that the checkpoint format this
codebase writes is actually compatible with weights_only=True loading —
i.e., no non-tensor objects are embedded in the checkpoint that would
cause the load to fail.

Usage:
    python scripts/infra/smoke_checkpoint.py
    python scripts/infra/smoke_checkpoint.py --device cuda
    python scripts/infra/smoke_checkpoint.py --n-steps 10
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_and_optimizer(device: torch.device):
    """Build CondAptNet with the same Stage 1 setup as train.py."""
    from models.condaptnet import CondAptNet
    from config import LEARNING_RATE_BASE, LEARNING_RATE_LORA, WEIGHT_DECAY

    model = CondAptNet(predict_kd=True)
    model.set_stage1()
    model = model.to(device)
    # Pre-compute path puts protein encoder on CPU; mirror that
    model.protein_encoder = model.protein_encoder.to("cpu")

    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    base_params  = [p for n, p in model.named_parameters()
                    if "lora_" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": LEARNING_RATE_BASE},
        {"params": lora_params, "lr": LEARNING_RATE_LORA},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    return model, optimizer, scheduler


def make_dummy_batch(device: torch.device, batch_size: int = 4):
    """Minimal batch that matches AptamerDataset output shapes."""
    import config as cfg
    import torch.nn.functional as F

    B       = batch_size
    apt_len = cfg.DNA_MAX_LEN
    prot_len = 64   # small for speed
    esm_dim  = cfg.ESM_EMBED_DIM

    apt      = torch.randint(0, 66, (B, apt_len)).to(device)
    v        = torch.zeros(B, 6).to(device)
    prot_tok = torch.zeros(B, 1, dtype=torch.long).to(device)
    cond     = torch.tensor([[7.4, 150.0, 37.0, 0.0, 2.0]] * B).to(device)
    labels   = torch.randint(0, 2, (B, 1)).float().to(device)
    kds      = torch.full((B, 1), float("nan")).to(device)
    prot_emb = torch.randn(B, prot_len, esm_dim).to(device)
    return apt, v, prot_tok, cond, labels, kds, prot_emb


def run_steps(model, optimizer, criterion, device, n_steps: int) -> list[float]:
    """Run n_steps forward+backward passes; return list of loss values."""
    from config import GRAD_CLIP

    losses = []
    model.train()
    for i in range(n_steps):
        apt, v, prot_tok, cond, labels, kds, prot_emb = make_dummy_batch(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(apt, v, prot_tok, cond, protein_emb=prot_emb)
        loss, _, _ = criterion(out.binding_prob.float(), labels,
                               out.kd_pred.float() if out.kd_pred is not None else None, kds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(loss.item())
    return losses


def state_dicts_equal(sd1: dict, sd2: dict) -> tuple[bool, list[str]]:
    """Compare two state_dicts; return (equal, list of mismatched keys)."""
    mismatched = []
    if set(sd1.keys()) != set(sd2.keys()):
        only_1 = set(sd1.keys()) - set(sd2.keys())
        only_2 = set(sd2.keys()) - set(sd1.keys())
        if only_1:
            mismatched.append(f"Keys only in original: {list(only_1)[:5]}")
        if only_2:
            mismatched.append(f"Keys only in loaded: {list(only_2)[:5]}")
        return False, mismatched

    for k in sd1:
        v1, v2 = sd1[k], sd2[k]
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if not torch.allclose(v1.float(), v2.float(), atol=1e-6):
                mismatched.append(f"{k}: values differ")
        elif v1 != v2:
            mismatched.append(f"{k}: {v1} != {v2}")

    return len(mismatched) == 0, mismatched


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint save/load smoke test")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/mps/cpu). Default: auto-detect.")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Training steps before saving checkpoint (default: 5)")
    args = parser.parse_args()

    header("Step 4: Checkpoint Save/Load Smoke Test")

    device = torch.device(args.device) if args.device else get_device()
    print(f"{INFO} Device: {device}")

    from scripts.training.losses import CondAptNetLoss
    criterion = CondAptNetLoss()

    # 4a: Build model and run a few steps
    header("4a: Build model + run training steps")
    print(f"{INFO} Building CondAptNet (Stage 1)...")
    model, optimizer, scheduler = build_model_and_optimizer(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{INFO} Total params    : {n_params:,}")
    print(f"{INFO} Trainable params: {n_trainable:,} ({100*n_trainable/n_params:.2f}%)")

    print(f"{INFO} Running {args.n_steps} training steps...")
    t0 = time.perf_counter()
    losses = run_steps(model, optimizer, criterion, device, args.n_steps)
    elapsed = time.perf_counter() - t0
    print(f"{PASS} {args.n_steps} steps completed in {elapsed:.2f}s")
    print(f"{INFO} Loss trajectory: {[f'{l:.4f}' for l in losses]}")
    scheduler.step()

    # 4b: Save checkpoint in train.py's exact format
    header("4b: Save checkpoint (weights_only-compatible format)")
    ckpt = {
        "epoch":          1,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "val_mcc":        0.0,
        "best_val_mcc":   0.0,
        "patience_count": 0,
        "train_loss":     losses[-1],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "smoke_epoch_001.pt")
        t0 = time.perf_counter()
        torch.save(ckpt, ckpt_path)
        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"{PASS} Checkpoint saved: {ckpt_path}")
        print(f"{INFO}   Size    : {size_mb:.1f} MB")
        print(f"{INFO}   Saved in: {elapsed:.2f}s")

        # 4c: Load with weights_only=True
        header("4c: Load checkpoint with weights_only=True")
        print(f"{INFO} Loading from {ckpt_path} with weights_only=True...")
        t0 = time.perf_counter()
        try:
            loaded_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            elapsed = time.perf_counter() - t0
            print(f"{PASS} torch.load(..., weights_only=True) succeeded in {elapsed:.2f}s")
        except Exception as e:
            print(f"{FAIL} torch.load(..., weights_only=True) raised: {e}")
            print(f"     This means the checkpoint contains non-tensor objects that are")
            print(f"     blocked by weights_only=True. Check what's stored in the checkpoint.")
            print(f"     Note: plain Python dicts/ints/floats ARE allowed by weights_only=True.")
            import traceback; traceback.print_exc()
            sys.exit(1)

        # Verify checkpoint keys
        required_keys = {"epoch", "model", "optimizer", "scheduler", "train_loss"}
        missing_keys = required_keys - set(loaded_ckpt.keys())
        if missing_keys:
            print(f"{FAIL} Checkpoint missing required keys: {missing_keys}")
            sys.exit(1)
        print(f"{PASS} Checkpoint has all required keys: {sorted(loaded_ckpt.keys())}")

        # 4d: Restore model state
        header("4d: Restore model state")
        model2, optimizer2, scheduler2 = build_model_and_optimizer(device)
        missing, unexpected = model2.load_state_dict(loaded_ckpt["model"], strict=False)
        benign_missing = [k for k in missing if "_film_heads" in k]
        real_missing   = [k for k in missing if k not in benign_missing]

        if real_missing:
            print(f"{FAIL} Critical model weights missing after restore: {real_missing[:5]}")
            sys.exit(1)
        if unexpected:
            print(f"{WARN} Unexpected keys in checkpoint (ignored by strict=False): {unexpected[:5]}")
        if benign_missing:
            print(f"{INFO} Lazily-init keys absent (OK): {benign_missing[:5]}")
        print(f"{PASS} Model state_dict restored (strict=False, {len(benign_missing)} lazy-init keys OK)")

        # 4e: Restore optimizer state
        optimizer2.load_state_dict(loaded_ckpt["optimizer"])
        print(f"{PASS} Optimizer state_dict restored")

        # 4f: Restore scheduler state
        scheduler2.load_state_dict(loaded_ckpt["scheduler"])
        print(f"{PASS} Scheduler state_dict restored")

        # 4g: Verify model state matches (compare parameter values)
        header("4e: Verify saved == restored (parameter-level check)")
        ok, diffs = state_dicts_equal(
            {k: v for k, v in model.state_dict().items() if "film_heads" not in k},
            {k: v for k, v in model2.state_dict().items() if "film_heads" not in k},
        )
        if ok:
            print(f"{PASS} All model parameters match between original and restored model")
        else:
            print(f"{FAIL} Parameter mismatch after restore:")
            for d in diffs[:10]:
                print(f"     {d}")
            sys.exit(1)

        # 4h: Forward pass with restored model produces identical output
        header("4f: Reproducibility — same output from restored model")
        model.eval()
        model2.eval()
        apt, v, prot_tok, cond, labels, kds, prot_emb = make_dummy_batch(device)
        with torch.no_grad():
            out1 = model(apt, v, prot_tok, cond, protein_emb=prot_emb)
            out2 = model2(apt, v, prot_tok, cond, protein_emb=prot_emb)

        if torch.allclose(out1.binding_prob, out2.binding_prob, atol=1e-5):
            print(f"{PASS} Original and restored model produce identical binding_prob output")
        else:
            max_diff = (out1.binding_prob - out2.binding_prob).abs().max().item()
            print(f"{FAIL} binding_prob mismatch (max diff: {max_diff:.6f})")
            sys.exit(1)

    header("Checkpoint smoke test complete")
    print(f"\n{PASS} All checkpoint save/load checks passed on device={device}")
    print(f"{INFO} weights_only=True is compatible with this codebase's checkpoint format.")
    print(f"{INFO} (weights_only=True was added in PR #6 — security/pin-dnabert2-revision-and-weights-only)")


if __name__ == "__main__":
    main()
