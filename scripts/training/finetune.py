"""
CondAptNet Stage 2/3 — Fine-tuning.

Loads a Stage 1 pretrained checkpoint, switches the model to stage 2 mode
(ESM-2 LoRA unfrozen), and fine-tunes on either:

  Stage 2 — validation targets  (VALIDATION_TARGETS from config.py)
             insulin, myoglobin, NT-proBNP, troponin I/T, albumin
             Purpose: benchmark against published Kd values.
             Checkpoint: models/checkpoints/validation/

  Stage 3 — deployment targets  (DEPLOYMENT_TARGETS from config.py)
             TBD — update DEPLOYMENT_TARGETS in config.py when Continuity
             confirms the real device biomarker set, then run this script.
             Checkpoint: models/checkpoints/deployment/

Key differences vs train.py (Stage 1):
  - model.set_stage2() unfreezes LoRA (frozen in Stage 1)
  - Data filtered to tier-specific target proteins
  - Default LRs are 10× lower (fine-tuning, not pretraining)
  - Optional --extra-data flag for scraped_dataset.csv rows (new data from
    the mining pipeline — appended to the filtered master rows)
  - --pretrain-checkpoint required to load Stage 1 weights

Usage:
    # Stage 2 (validation)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py \\
        --stage validation

    # Stage 3 (deployment) — run after updating DEPLOYMENT_TARGETS in config.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py \\
        --stage deployment

    # Include new rows from the mining pipeline
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py \\
        --stage validation \\
        --extra-data data/raw/scraped_dataset.csv

    # Resume interrupted fine-tune
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py \\
        --stage validation --resume
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import glob
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, RANDOM_SEED,
    DATA_PROCESSED, CHECKPOINTS_DIR, VIENNA_CACHE,
    BATCH_SIZE, LEARNING_RATE_BASE, LEARNING_RATE_LORA, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, GRAD_CLIP,
    DNA_MAX_LEN, PROT_MAX_TOKENS,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
    VALIDATION_TARGETS, DEPLOYMENT_TARGETS,
)
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer
from scripts.training.losses import CondAptNetLoss
from scripts.training.train import (
    AptamerDataset, collate_fn,
    split_by_protein_family,
    train_epoch, eval_epoch,
    precompute_protein_embeddings,
)
from scripts.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Fine-tuning uses 10× lower LRs than pretraining
_FT_LR_BASE = LEARNING_RATE_BASE / 10   # 1e-5
_FT_LR_LORA = LEARNING_RATE_LORA / 10  # 1e-6


# ── Target filtering ──────────────────────────────────────────────────────────

def filter_by_tier_targets(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    Keep only rows whose target_protein fuzzy-matches any name in targets.

    Matching rules (all case-insensitive):
      - Normalise both sides: lowercase, replace '_' and '-' with space
      - Keep row if the target keyword appears anywhere in target_protein,
        OR if target_protein appears anywhere in the target keyword
        (handles "human serum albumin" ↔ "albumin")

    Returns empty DataFrame (with same columns) when targets is empty.
    """
    if not targets:
        log.warning("Target list is empty — returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    def _norm(s: str) -> str:
        return str(s).lower().replace("_", " ").replace("-", " ")

    lower_targets = [_norm(t) for t in targets]

    def matches(target_protein) -> bool:
        if not isinstance(target_protein, str):
            return False
        tp = _norm(target_protein)
        return any(t in tp or tp in t for t in lower_targets)

    mask = df["target_protein"].apply(matches)
    result = df[mask].copy()
    log.info(
        "filter_by_tier_targets: %d / %d rows match targets %s",
        len(result), len(df), targets,
    )
    return result


def _merge_extra_data(
    base_df:    pd.DataFrame,
    extra_path: str,
    targets:    list[str],
) -> pd.DataFrame:
    """
    Optionally load extra rows from scraped_dataset.csv (20-column scraper schema),
    map them to the master_dataset.csv schema, filter by targets, and append.

    Only columns present in both schemas are carried over; missing ones default
    to the physiological values from config.py.
    """
    if not extra_path or not os.path.exists(extra_path):
        log.debug("--extra-data path not found or not set; skipping.")
        return base_df

    try:
        extra = pd.read_csv(extra_path, dtype=str).fillna("")
    except Exception as exc:
        log.warning("Could not load extra data from %s: %s", extra_path, exc)
        return base_df

    # Rename scraper → master schema column names
    col_map = {
        "aptamer_sequence":    "sequence",
        "target_name":         "target_protein",
        "kd_value":            "Kd_nM",
        "ph":                  "pH",
        "na_concentration_mM": "salt_mM",
        "temperature_C":       "temp_C",
        "mg_concentration_mM": "mg_mM",
        "source_doi":          "source_pmid",
    }
    extra = extra.rename(columns={k: v for k, v in col_map.items() if k in extra.columns})

    # Add columns required by AptamerDataset but absent from scraped schema
    for col, default in [
        ("protein_sequence",         None),
        ("uniprot_id",               None),
        ("label",                    1),     # scraped rows assumed positive (binders)
        ("training_tier",            2),
        ("augmented",                False),
        ("needs_sequence_enrichment", False),
        ("source",                   "scraper"),
        ("buffer_type",              DEFAULT_BUFFER),
    ]:
        if col not in extra.columns:
            extra[col] = default

    # Filter by tier targets
    extra_filtered = filter_by_tier_targets(extra, targets)
    if extra_filtered.empty:
        log.info("No extra-data rows matched the tier targets.")
        return base_df

    # Only keep sequences that are pure ATGC (same QC as master)
    atgc_mask = extra_filtered["sequence"].str.match(r"^[ATGC]+$", na=False)
    extra_filtered = extra_filtered[atgc_mask]

    combined = pd.concat([base_df, extra_filtered], ignore_index=True)
    log.info(
        "Appended %d extra rows from %s (total: %d)",
        len(extra_filtered), extra_path, len(combined),
    )
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CondAptNet Stage 2/3 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["validation", "deployment"],
        required=True,
        help="Stage 2 (validation targets) or Stage 3 (deployment targets).",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default=os.path.join(CHECKPOINTS_DIR, "pretrain", "best.pt"),
        help="Path to Stage 1 best.pt checkpoint (default: models/checkpoints/pretrain/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(DATA_PROCESSED, "master_dataset.csv"),
    )
    parser.add_argument(
        "--extra-data",
        type=str,
        default="",
        help="Optional path to scraped_dataset.csv for additional fine-tuning rows.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="",
        help="Override default checkpoint dir (default: models/checkpoints/{stage})")
    parser.add_argument("--max-epochs",    type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr-base",       type=float, default=_FT_LR_BASE)
    parser.add_argument("--lr-lora",       type=float, default=_FT_LR_LORA)
    parser.add_argument("--prot-max-tokens", type=int, default=PROT_MAX_TOKENS)
    parser.add_argument("--max-prot-len",    type=int, default=PROT_MAX_TOKENS)
    parser.add_argument("--max-batches",   type=int,   default=None,
        help="Limit batches per epoch (smoke-test mode).")
    parser.add_argument("--resume",        action="store_true",
        help="Resume from latest epoch_*.pt in --checkpoint-dir.")
    args = parser.parse_args()

    # ── Resolve stage-specific settings ──────────────────────────────────────
    if args.stage == "validation":
        tier_targets = VALIDATION_TARGETS
        tier_label   = "Stage 2 (validation)"
    else:
        tier_targets = DEPLOYMENT_TARGETS
        tier_label   = "Stage 3 (deployment)"

    if not tier_targets:
        log.error(
            "%s targets list is empty. "
            "Update %s_TARGETS in config.py before running fine-tuning.",
            tier_label,
            args.stage.upper(),
        )
        sys.exit(1)

    checkpoint_dir = (
        args.checkpoint_dir
        or os.path.join(CHECKPOINTS_DIR, args.stage)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Runtime device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)
    torch.manual_seed(RANDOM_SEED)

    # ── Load and filter data ──────────────────────────────────────────────────
    log.info("Loading master dataset from %s", args.data)
    master = pd.read_csv(args.data)

    ready = (
        master["sequence"].notna() &
        (master["needs_sequence_enrichment"] == False) &
        master["protein_sequence"].notna()
    )
    df = master[ready].copy()
    log.info("Training-ready rows (all tiers): %d / %d total", len(df), len(master))

    df = filter_by_tier_targets(df, tier_targets)

    if args.extra_data:
        df = _merge_extra_data(df, args.extra_data, tier_targets)

    if len(df) == 0:
        log.error(
            "No rows found for targets: %s\n"
            "Check that master_dataset.csv has rows with matching target_protein "
            "values and that protein_sequence enrichment has been run.",
            tier_targets,
        )
        sys.exit(1)

    train_df, val_df, test_df = split_by_protein_family(df)
    log.info(
        "%s data: %d train / %d val / %d test",
        tier_label, len(train_df), len(val_df), len(test_df),
    )

    if len(train_df) == 0:
        log.error("Train split is empty — not enough protein families for a split.")
        sys.exit(1)

    # ── Auxiliary tools ───────────────────────────────────────────────────────
    tokenizer = DNATokenizer()

    vienna_cache: dict = {}
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vienna_cache = pickle.load(f)
        log.info("Vienna cache: %d entries", len(vienna_cache))
    else:
        log.warning("Vienna cache not found — structure features computed on-the-fly")

    # ── Build model ───────────────────────────────────────────────────────────
    log.info("Building CondAptNet...")
    model = CondAptNet(predict_kd=True)

    # Load Stage 1 pretrained weights
    if os.path.exists(args.pretrain_checkpoint):
        log.info("Loading Stage 1 checkpoint: %s", args.pretrain_checkpoint)
        ckpt_pretrain = torch.load(args.pretrain_checkpoint, map_location="cpu")
        state = ckpt_pretrain.get("model", ckpt_pretrain)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Missing keys when loading pretrain ckpt: %s", missing[:5])
        if unexpected:
            log.warning("Unexpected keys in pretrain ckpt: %s", unexpected[:5])
        log.info("Stage 1 weights loaded successfully.")
    else:
        log.warning(
            "Pretrain checkpoint not found at %s — fine-tuning from random init. "
            "Run Stage 1 training first for best results.",
            args.pretrain_checkpoint,
        )

    # Stage 2/3: unfreeze LoRA adapters in ESM-2
    model.set_stage2()
    model = model.to(device)
    model.protein_encoder = model.protein_encoder.to("cpu")

    log.info("Total params:     %d", model.total_params())
    log.info("Trainable params: %d (%.2f%%)",
             model.trainable_params(),
             100 * model.trainable_params() / model.total_params())

    # ── Pre-compute protein embeddings ────────────────────────────────────────
    emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
    all_prot_seqs = df["protein_sequence"].dropna().unique().tolist()
    seq_to_emb = precompute_protein_embeddings(
        all_prot_seqs,
        model.protein_encoder,
        emb_dir,
        device=torch.device("cpu"),
        prot_max_tokens=args.prot_max_tokens,
    )
    model.protein_encoder = model.protein_encoder.to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = AptamerDataset(train_df, tokenizer, vienna_cache, seq_to_emb,
                               max_prot_len=args.max_prot_len)
    val_ds = (
        AptamerDataset(val_df, tokenizer, vienna_cache, seq_to_emb,
                       max_prot_len=args.max_prot_len)
        if len(val_df) > 0 else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader   = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   collate_fn=collate_fn, num_workers=0, pin_memory=False)
        if val_ds else None
    )

    # ── Optimizer: lower LRs for fine-tuning ─────────────────────────────────
    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    base_params = [p for n, p in model.named_parameters()
                   if "lora_" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr_base},
        {"params": lora_params, "lr": args.lr_lora},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-7,
    )
    criterion  = CondAptNetLoss()

    # ── Training state ────────────────────────────────────────────────────────
    best_val_mcc   = -1.0
    patience_count = 0
    start_epoch    = 1
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    if args.resume:
        ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
        if ckpts:
            resume_path = ckpts[-1]
            log.info("Resuming from %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            benign_missing = [k for k in missing if "_film_heads" in k]
            real_missing   = [k for k in missing if k not in benign_missing]
            if real_missing:
                raise RuntimeError(f"Critical weights missing from checkpoint: {real_missing}")
            if unexpected:
                log.warning("Checkpoint has unexpected keys (ignored): %s", unexpected)
            if benign_missing:
                log.info("Lazily-init keys not in checkpoint (random init, OK): %s",
                         benign_missing)
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            best_val_mcc   = ckpt.get("best_val_mcc", ckpt.get("val_mcc", -1.0))
            patience_count = ckpt.get("patience_count", 0)
            start_epoch    = ckpt.get("epoch", 0) + 1
            log.info("Resumed at epoch %d  best_val_mcc=%.4f  patience=%d",
                     start_epoch - 1, best_val_mcc, patience_count)
        else:
            log.warning("--resume set but no epoch_*.pt found in %s — starting fresh",
                        checkpoint_dir)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log.info("GPU memory: %.1fGB free / %.1fGB total", free / 1e9, total / 1e9)

    log.info("=" * 65)
    log.info("%s fine-tuning on %s", tier_label, device)
    log.info("Targets: %s", tier_targets)
    log.info("lr_base=%.1e  lr_lora=%.1e  epochs=%d", args.lr_base, args.lr_lora, args.max_epochs)
    log.info("=" * 65)

    # ── Training loop (identical structure to train.py) ───────────────────────
    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()

        (tr_loss, tr_bce, tr_kd,
         tr_labels, tr_probs,
         tr_kd_t,  tr_kd_p) = train_epoch(model, train_loader, optimizer,
                                           criterion, device,
                                           max_batches=args.max_batches)
        tr_m = compute_metrics(tr_labels, tr_probs,
                               kd_true=tr_kd_t, kd_pred=tr_kd_p)
        elapsed = time.time() - t0

        if val_loader is not None:
            (va_loss, va_bce, va_kd,
             va_labels, va_probs,
             va_kd_t,  va_kd_p) = eval_epoch(model, val_loader, criterion, device,
                                              max_batches=args.max_batches)
            va_m    = compute_metrics(va_labels, va_probs,
                                      kd_true=va_kd_t, kd_pred=va_kd_p)
            val_mcc = va_m["mcc"]
            log.info(
                "Epoch %3d/%d | loss=%.4f (bce=%.4f kd=%.4f) | "
                "train MCC=%.3f AUC=%.3f | "
                "val loss=%.4f MCC=%.3f AUC=%.3f | %.0fs",
                epoch, args.max_epochs,
                tr_loss, tr_bce, tr_kd,
                tr_m["mcc"], tr_m["auroc"],
                va_loss, va_m["mcc"], va_m["auroc"],
                elapsed,
            )
        else:
            val_mcc = tr_m["mcc"]
            log.info(
                "Epoch %3d/%d | loss=%.4f (bce=%.4f kd=%.4f) | "
                "train MCC=%.3f AUC=%.3f | (no val split) | %.0fs",
                epoch, args.max_epochs,
                tr_loss, tr_bce, tr_kd,
                tr_m["mcc"], tr_m["auroc"],
                elapsed,
            )

        # Save every epoch
        ckpt = {
            "epoch":          epoch,
            "stage":          args.stage,
            "tier_targets":   tier_targets,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "val_mcc":        val_mcc,
            "best_val_mcc":   best_val_mcc,
            "patience_count": patience_count,
            "train_loss":     tr_loss,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if val_mcc > best_val_mcc:
            best_val_mcc   = val_mcc
            patience_count = 0
            torch.save(ckpt, best_ckpt_path)
            log.info("  → New best MCC=%.3f  saved to %s", best_val_mcc, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOPPING_PATIENCE:
                log.info("Early stopping at epoch %d (patience=%d)",
                         epoch, EARLY_STOPPING_PATIENCE)
                break

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Fine-tuning complete. Best val MCC=%.3f", best_val_mcc)
    log.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
