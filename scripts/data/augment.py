"""
Data augmentation for CondAptNet Stage 1 training.

Reads master_dataset.csv, splits by protein family, then augments the
training split only (val/test are never touched).

Augmentations applied to label=1 (positive) training rows:
  1. reverse_complement  → new label=1 rows
  2. truncations         → remove 2 or 3 nt from left OR right (4 variants)
  3. cross_target_neg    → assign aptamer to a different protein → label=0
  4. scrambled           → shuffle nucleotides → label=0

Outputs:
  data/augmented/tier1_train.csv   training rows (all tiers), augmented
  data/augmented/val.csv           val rows, no augmentation
  data/augmented/test.csv          test rows, no augmentation

Usage:
    python scripts/data/augment.py
    python scripts/data/augment.py --no-cross-neg   # skip cross-target negatives
    python scripts/data/augment.py --dry-run        # report counts, no write
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED, DATA_AUGMENTED, RANDOM_SEED, DNA_MAX_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_RC = str.maketrans("ACGTacgt", "TGCAtgca")
MIN_LEN = 20


# ── Augmentation primitives ───────────────────────────────────────────────────

def reverse_complement(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def is_valid(seq: str) -> bool:
    """Minimal validation: length, alphabet, GC content."""
    s = seq.upper()
    if not (MIN_LEN <= len(s) <= DNA_MAX_LEN):
        return False
    if set(s) - {"A", "T", "G", "C"}:
        return False
    gc = (s.count("G") + s.count("C")) / len(s)
    return 0.20 <= gc <= 0.80


def scramble(seq: str, rng: np.random.Generator) -> str:
    chars = list(seq.upper())
    rng.shuffle(chars)
    return "".join(chars)


# ── Split (mirrors train.py exactly — same seed, same fractions) ──────────────

def split_by_protein_family(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = sorted(df["target_protein"].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    n       = len(families)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))

    train_f = set(families[:n_train])
    val_f   = set(families[n_train: n_train + n_val])
    test_f  = set(families[n_train + n_val:])

    tr = df[df["target_protein"].isin(train_f)].reset_index(drop=True)
    va = df[df["target_protein"].isin(val_f)].reset_index(drop=True)
    te = df[df["target_protein"].isin(test_f)].reset_index(drop=True)

    log.info("Split: %d train / %d val / %d test rows (%d / %d / %d families)",
             len(tr), len(va), len(te), len(train_f), len(val_f), len(test_f))
    return tr, va, te


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_train(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    cross_neg: bool = True,
) -> pd.DataFrame:
    """
    Apply all augmentations to the training split.
    Returns the original rows PLUS all augmented rows concatenated.
    """
    positives = train_df[train_df["label"] == 1].copy()
    log.info("Positive rows to augment: %d", len(positives))

    aug_rows: list[pd.DataFrame] = []

    # ── 1. Reverse complement ──────────────────────────────────────────────────
    rc_rows = positives.copy()
    rc_rows["sequence"]   = rc_rows["sequence"].apply(reverse_complement)
    rc_rows["augmented"]  = True
    rc_rows["aug_method"] = "rev_comp"
    valid_rc = rc_rows["sequence"].apply(is_valid)
    rc_rows = rc_rows[valid_rc]
    aug_rows.append(rc_rows)
    log.info("  rev_comp       : +%d rows", len(rc_rows))

    # ── 2. Truncations (2 and 3 nt from each end) ─────────────────────────────
    trunc_all: list[pd.DataFrame] = []
    for n in (2, 3):
        for side in ("left", "right"):
            t = positives.copy()
            if side == "left":
                t["sequence"] = t["sequence"].apply(lambda s: s[n:])
            else:
                t["sequence"] = t["sequence"].apply(lambda s: s[:-n])
            t["augmented"]  = True
            t["aug_method"] = f"trunc_{side[0].upper()}{n}"
            valid_t = t["sequence"].apply(is_valid)
            t = t[valid_t]
            trunc_all.append(t)
    trunc_df = pd.concat(trunc_all, ignore_index=True)
    aug_rows.append(trunc_df)
    log.info("  truncations    : +%d rows", len(trunc_df))

    # ── 3. Cross-target negatives ─────────────────────────────────────────────
    if cross_neg:
        protein_list = train_df["target_protein"].unique().tolist()
        cross_rows = []
        for _, row in positives.iterrows():
            other = [p for p in protein_list if p != row["target_protein"]]
            if not other:
                continue
            other_protein = rng.choice(other)
            # Find one row with that protein to copy its protein_sequence + condition
            ref = train_df[train_df["target_protein"] == other_protein].iloc[0]
            new = row.copy()
            new["target_protein"]  = other_protein
            new["protein_sequence"] = ref["protein_sequence"]
            new["uniprot_id"]       = ref.get("uniprot_id", None)
            new["label"]            = 0
            new["Kd_nM"]            = float("nan")
            new["augmented"]        = True
            new["aug_method"]       = "cross_neg"
            cross_rows.append(new)
        if cross_rows:
            cross_df = pd.DataFrame(cross_rows).reset_index(drop=True)
            aug_rows.append(cross_df)
            log.info("  cross_target_neg: +%d rows", len(cross_df))

    # ── 4. Scrambled sequences ────────────────────────────────────────────────
    scr_rows = []
    for _, row in positives.iterrows():
        seq = scramble(row["sequence"], rng)
        if not is_valid(seq):
            continue
        new = row.copy()
        new["sequence"]   = seq
        new["label"]      = 0
        new["Kd_nM"]      = float("nan")
        new["augmented"]  = True
        new["aug_method"] = "scrambled"
        scr_rows.append(new)
    if scr_rows:
        scr_df = pd.DataFrame(scr_rows).reset_index(drop=True)
        aug_rows.append(scr_df)
        log.info("  scrambled      : +%d rows", len(scr_df))

    # ── Combine + deduplicate ─────────────────────────────────────────────────
    combined = pd.concat([train_df] + aug_rows, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["sequence", "target_protein"], keep="first"
    ).reset_index(drop=True)
    log.info("Dedup: %d → %d rows (-%d)", before, len(combined), before - len(combined))

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CondAptNet data augmentation")
    parser.add_argument("--data",          default=os.path.join(DATA_PROCESSED, "master_dataset.csv"))
    parser.add_argument("--output-dir",    default=DATA_AUGMENTED)
    parser.add_argument("--no-cross-neg",  action="store_true", help="Skip cross-target negatives")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load and filter ───────────────────────────────────────────────────────
    master = pd.read_csv(args.data)
    ready = (
        master["sequence"].notna() &
        (master["needs_sequence_enrichment"] == False) &
        master["protein_sequence"].notna()
    )
    df = master[ready].copy()
    log.info("Training-ready rows: %d / %d total", len(df), len(master))

    # ── Split by protein family ───────────────────────────────────────────────
    train_df, val_df, test_df = split_by_protein_family(df, seed=RANDOM_SEED)

    # ── Augment training split only ───────────────────────────────────────────
    aug_train = augment_train(train_df, rng, cross_neg=not args.no_cross_neg)

    # ── Tier split for training output ────────────────────────────────────────
    tier1 = aug_train  # all rows go to tier1; tier2 fine-tuning handled separately

    log.info("=" * 55)
    log.info("tier1_train rows : %d", len(tier1))
    log.info("val rows         : %d", len(val_df))
    log.info("test rows        : %d", len(test_df))
    log.info("=" * 55)

    if args.dry_run:
        log.info("Dry-run — not writing")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tier1.to_csv(os.path.join(args.output_dir, "tier1_train.csv"),  index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"),          index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"),        index=False)
    log.info("Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
