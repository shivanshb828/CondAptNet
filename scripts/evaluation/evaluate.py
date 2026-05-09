"""
CondAptNet evaluation script.

Loads a checkpoint and evaluates on the test split (protein-family held-out).
Reports MCC, AUC-ROC, AUC-PR, sensitivity, specificity, and Kd Pearson r.

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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, RANDOM_SEED, DATA_PROCESSED, CHECKPOINTS_DIR, VIENNA_CACHE,
    BATCH_SIZE, DNA_MAX_LEN, PROT_MAX_TOKENS,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
)
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer
from scripts.training.train import (
    AptamerDataset, collate_fn, split_by_protein_family,
)
from scripts.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def run_eval(model, loader, device) -> tuple:
    model.eval()
    all_labels: list = []
    all_probs:  list = []
    all_kd_true: list = []
    all_kd_pred: list = []

    for apt, v, prot, cond, labels, kds in loader:
        apt    = apt.to(device)
        v      = v.to(device)
        prot   = prot.to(device)
        cond   = cond.to(device)

        out = model(apt, v, prot, cond)

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
    parser.add_argument("--data", default=os.path.join(DATA_PROCESSED, "master_dataset.csv"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # ── Load data + same split as training ────────────────────────────────────
    master = pd.read_csv(args.data)
    ready = (
        master["sequence"].notna() &
        (master["needs_sequence_enrichment"] == False) &
        master["protein_sequence"].notna()
    )
    df = master[ready].copy()
    log.info("Training-ready rows: %d", len(df))

    train_df, val_df, test_df = split_by_protein_family(df)
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    log.info("Evaluating on '%s' split: %d rows", args.split, len(eval_df))

    if len(eval_df) == 0:
        log.error("'%s' split is empty.", args.split)
        sys.exit(1)

    # ── Load tools ────────────────────────────────────────────────────────────
    tokenizer = DNATokenizer()

    vienna_cache: dict = {}
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vienna_cache = pickle.load(f)

    # ── Build model and load checkpoint ──────────────────────────────────────
    model = CondAptNet(predict_kd=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    log.info("Loaded checkpoint from epoch %d (val_mcc=%.3f)",
             ckpt.get("epoch", -1), ckpt.get("val_mcc", float("nan")))

    esm_alphabet = model.protein_encoder.alphabet
    dataset = AptamerDataset(eval_df, tokenizer, vienna_cache, esm_alphabet)
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
