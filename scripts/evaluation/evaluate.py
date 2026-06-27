"""
CondAptNet evaluation script.

Loads a checkpoint and evaluates on a held-out split (protein-family split,
zero sequence leakage). Reports MCC (primary), AUC-ROC, AUC-PR, sensitivity,
specificity, and Kd Pearson r.

Reads the same augmented splits as Stage 1 training (data/augmented/) so the
test set here is exactly the one held out during training:
    tier1_train.csv / val.csv / test.csv

Protein embeddings are pre-computed once with ESM-2 and cached to disk, then the
model consumes them (identical path to train.py — ESM-2 is never run per-batch).

Usage:
    python scripts/evaluation/evaluate.py \
        --checkpoint models/checkpoints/pretrain/best.pt

    python scripts/evaluation/evaluate.py \
        --checkpoint models/checkpoints/pretrain/best.pt \
        --split test          # test | val | train
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, DATA_PROCESSED, DATA_AUGMENTED, VIENNA_CACHE,
    BATCH_SIZE, PROT_MAX_TOKENS,
)
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer
from scripts.training.train import (
    AptamerDataset, collate_fn, precompute_protein_embeddings,
)
from scripts.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Maps --split to the augmented CSV that holds it
_SPLIT_FILES = {
    "train": "tier1_train.csv",
    "val":   "val.csv",
    "test":  "test.csv",
}


@torch.no_grad()
def run_eval(model, loader, device) -> tuple:
    model.eval()
    all_labels: list = []
    all_probs:  list = []
    all_kd_true: list = []
    all_kd_pred: list = []

    for apt, v, prot_tok, cond, labels, kds, prot_emb in loader:
        apt      = apt.to(device)
        v        = v.to(device)
        prot_emb = prot_emb.to(device)
        cond     = cond.to(device)

        out = model(apt, v, prot_tok, cond, protein_emb=prot_emb)

        all_labels.extend(labels.squeeze(-1).tolist())
        all_probs.extend(out.binding_prob.cpu().squeeze(-1).tolist())
        all_kd_true.extend(kds.squeeze(-1).tolist())
        all_kd_pred.extend(
            out.kd_pred.cpu().squeeze(-1).tolist()
            if out.kd_pred is not None else [float("nan")] * labels.shape[0]
        )

    return all_labels, all_probs, all_kd_true, all_kd_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="CondAptNet evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt checkpoint file")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--augmented-dir", type=str, default=DATA_AUGMENTED,
                        help="Directory holding tier1_train.csv / val.csv / test.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prot-max-tokens", type=int, default=PROT_MAX_TOKENS)
    parser.add_argument("--max-prot-len", type=int, default=PROT_MAX_TOKENS)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # ── Load the requested split (same files train.py uses) ───────────────────
    split_path = os.path.join(args.augmented_dir, _SPLIT_FILES[args.split])
    if not os.path.exists(split_path):
        log.error("Split file not found: %s\nRun `python scripts/data/augment.py` first.",
                  split_path)
        sys.exit(1)

    eval_df = pd.read_csv(split_path)
    ready = eval_df["aptamer_sequence"].notna() & eval_df["protein_sequence"].notna()
    eval_df = eval_df[ready].reset_index(drop=True)
    log.info("Evaluating on '%s' split: %d rows (%s)", args.split, len(eval_df), split_path)

    if len(eval_df) == 0:
        log.error("'%s' split is empty.", args.split)
        sys.exit(1)

    # ── Load tools ────────────────────────────────────────────────────────────
    tokenizer = DNATokenizer()

    vienna_cache: dict = {}
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vienna_cache = pickle.load(f)
        log.info("Vienna cache: %d entries", len(vienna_cache))
    else:
        log.warning("Vienna cache not found — features will be computed on-the-fly")

    # ── Build model and load checkpoint ──────────────────────────────────────
    model = CondAptNet(predict_kd=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # strict=False tolerates lazily-init keys (e.g. condition_encoder._film_heads)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    real_missing = [k for k in missing if "_film_heads" not in k]
    if real_missing:
        raise RuntimeError(f"Critical weights missing from checkpoint: {real_missing}")
    if unexpected:
        log.warning("Checkpoint has unexpected keys (ignored): %s", unexpected)
    model = model.to(device)
    log.info("Loaded checkpoint from epoch %d (val_mcc=%.3f)",
             ckpt.get("epoch", -1),
             ckpt.get("best_val_mcc", ckpt.get("val_mcc", float("nan"))))

    # ── Pre-compute protein embeddings (ESM-2, run ONCE per unique protein) ───
    emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
    model.protein_encoder = model.protein_encoder.to("cpu")
    seq_to_emb = precompute_protein_embeddings(
        eval_df["protein_sequence"].dropna().unique().tolist(),
        model.protein_encoder,
        emb_dir,
        device=torch.device("cpu"),
        prot_max_tokens=args.prot_max_tokens,
    )
    model.protein_encoder = model.protein_encoder.to(device)

    dataset = AptamerDataset(eval_df, tokenizer, vienna_cache, seq_to_emb,
                             max_prot_len=args.max_prot_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    labels, probs, kd_true, kd_pred = run_eval(model, loader, device)
    m = compute_metrics(labels, probs, kd_true=kd_true, kd_pred=kd_pred)

    log.info("=" * 65)
    log.info("Evaluation results — split: %s  rows: %d", args.split, len(labels))
    log.info("=" * 65)
    print_metrics(m, prefix=args.split.upper())
    log.info("")
    log.info("  MCC          : %.4f  (primary metric)", m["mcc"])
    log.info("  AUC-ROC      : %.4f", m["auroc"])
    log.info("  AUC-PR       : %.4f", m["auprc"])
    log.info("  Sensitivity  : %.4f", m["sensitivity"])
    log.info("  Specificity  : %.4f", m["specificity"])
    if m["n_kd_pairs"] >= 3:
        log.info("  Pearson r    : %.4f  (n=%d)", m["pearson_r_kd"], m["n_kd_pairs"])
    else:
        log.info("  Pearson r    : N/A  (fewer than 3 Kd pairs)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
