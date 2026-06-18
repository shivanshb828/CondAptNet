"""
Data augmentation for CondAptNet Stage 1 training.

Reads master_dataset_cleaned.csv (new 23-column schema), respects the pre-assigned
`split` column from the 7-phase cleaning pipeline (train/val/test/unassigned),
then augments the training split only. Val/test are never touched.

Unassigned rows (no protein-family split could be determined during cleaning) are
folded into train.

Augmentations applied to label=1 (positive) training rows:
  1. reverse_complement  → new label=0 rows (hard negatives: different 3D fold)
  2. truncations         → remove 2 or 3 nt from left OR right (4 variants, keep label)
  3. cross_target_neg    → assign aptamer to a different protein → label=0
  4. scrambled           → shuffle nucleotides → label=0

Outputs (same schema as input + `augmented` + `aug_method` columns):
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

# Physiological defaults (Continuity device context)
DEFAULT_PH           = 7.4
DEFAULT_NA_MM        = 150.0
DEFAULT_MG_MM        = 2.0
DEFAULT_TEMP_C       = 37.0
DEFAULT_BUFFER       = "PBS"


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


# ── Fill physiological defaults ───────────────────────────────────────────────

def fill_condition_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ph"]                = df["ph"].fillna(DEFAULT_PH)
    df["na_concentration_mM"] = df["na_concentration_mM"].fillna(DEFAULT_NA_MM)
    df["mg_concentration_mM"] = df["mg_concentration_mM"].fillna(DEFAULT_MG_MM)
    df["temperature_C"]     = df["temperature_C"].fillna(DEFAULT_TEMP_C)
    df["binding_buffer"]    = df["binding_buffer"].fillna(DEFAULT_BUFFER)
    return df


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
    # RC is labeled 0 (non-binder): the RC folds into a different 3D structure
    # and is NOT a confirmed binder. It serves as a hard negative — same base
    # composition, completely different folding → teaches the model that
    # sequence alone doesn't determine binding.
    rc_rows = positives.copy()
    rc_rows["aptamer_sequence"] = rc_rows["aptamer_sequence"].apply(reverse_complement)
    rc_rows["label"]      = 0
    rc_rows["kd_value"]   = float("nan")
    rc_rows["kd_unit"]    = pd.NA
    rc_rows["augmented"]  = True
    rc_rows["aug_method"] = "rev_comp"
    valid_rc = rc_rows["aptamer_sequence"].apply(is_valid)
    rc_rows = rc_rows[valid_rc]
    aug_rows.append(rc_rows)
    log.info("  rev_comp       : +%d rows (label=0, hard negatives)", len(rc_rows))

    # ── 2. Truncations (2 and 3 nt from each end) ─────────────────────────────
    # Truncated aptamers may still bind — keep original label and Kd.
    trunc_all: list[pd.DataFrame] = []
    for n in (2, 3):
        for side in ("left", "right"):
            t = positives.copy()
            if side == "left":
                t["aptamer_sequence"] = t["aptamer_sequence"].apply(lambda s: s[n:])
            else:
                t["aptamer_sequence"] = t["aptamer_sequence"].apply(lambda s: s[:-n])
            t["augmented"]  = True
            t["aug_method"] = f"trunc_{side[0].upper()}{n}"
            valid_t = t["aptamer_sequence"].apply(is_valid)
            t = t[valid_t]
            trunc_all.append(t)
    trunc_df = pd.concat(trunc_all, ignore_index=True)
    aug_rows.append(trunc_df)
    log.info("  truncations    : +%d rows", len(trunc_df))

    # ── 3. Cross-target negatives ─────────────────────────────────────────────
    if cross_neg:
        protein_list = train_df["target_name"].unique().tolist()
        cross_rows = []
        for _, row in positives.iterrows():
            other = [p for p in protein_list if p != row["target_name"]]
            if not other:
                continue
            other_protein = rng.choice(other)
            ref = train_df[train_df["target_name"] == other_protein].iloc[0]
            new = row.copy()
            new["target_name"]      = other_protein
            new["target_id"]        = ref.get("target_id", pd.NA)
            new["target_id_source"] = ref.get("target_id_source", pd.NA)
            new["protein_sequence"] = ref["protein_sequence"]
            new["label"]            = 0
            new["kd_value"]         = float("nan")
            new["kd_unit"]          = pd.NA
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
        seq = scramble(row["aptamer_sequence"], rng)
        if not is_valid(seq):
            continue
        new = row.copy()
        new["aptamer_sequence"] = seq
        new["label"]            = 0
        new["kd_value"]         = float("nan")
        new["kd_unit"]          = pd.NA
        new["augmented"]        = True
        new["aug_method"]       = "scrambled"
        scr_rows.append(new)
    if scr_rows:
        scr_df = pd.DataFrame(scr_rows).reset_index(drop=True)
        aug_rows.append(scr_df)
        log.info("  scrambled      : +%d rows", len(scr_df))

    # ── Combine + deduplicate ─────────────────────────────────────────────────
    combined = pd.concat([train_df] + aug_rows, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["aptamer_sequence", "target_name"], keep="first"
    ).reset_index(drop=True)
    log.info("Dedup: %d → %d rows (-%d)", before, len(combined), before - len(combined))

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CondAptNet data augmentation")
    parser.add_argument("--data",         default=os.path.join(DATA_PROCESSED, "master_dataset_cleaned.csv"))
    parser.add_argument("--output-dir",   default=DATA_AUGMENTED)
    parser.add_argument("--no-cross-neg", action="store_true", help="Skip cross-target negatives")
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load ──────────────────────────────────────────────────────────────────
    master = pd.read_csv(args.data)
    log.info("Loaded %d rows from %s", len(master), args.data)

    # ── Filter: protein targets with sequences only ───────────────────────────
    ready = (
        master["aptamer_sequence"].notna() &
        (master["target_type"] == "protein") &
        master["protein_sequence"].notna()
    )
    df = master[ready].copy()
    log.info("Training-ready rows (protein + sequence): %d / %d total", len(df), len(master))

    # ── Add augmented/aug_method columns to originals ─────────────────────────
    df["augmented"]  = False
    df["aug_method"] = pd.NA

    # ── Fill physiological defaults for missing condition fields ──────────────
    df = fill_condition_defaults(df)

    # ── Use pre-assigned split column; unassigned → train ────────────────────
    # The 7-phase cleaning pipeline assigned splits by protein family to prevent
    # leakage. Unassigned rows (no family context) are folded into train.
    train_df = df[df["split"].isin(["train", "unassigned"])].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    train_families = train_df["target_name"].nunique()
    val_families   = val_df["target_name"].nunique()
    test_families  = test_df["target_name"].nunique()
    log.info(
        "Split (from cleaned): %d train / %d val / %d test rows  (%d / %d / %d families)",
        len(train_df), len(val_df), len(test_df),
        train_families, val_families, test_families,
    )

    # ── Augment training split only ───────────────────────────────────────────
    aug_train = augment_train(train_df, rng, cross_neg=not args.no_cross_neg)

    log.info("=" * 55)
    log.info("tier1_train rows : %d", len(aug_train))
    log.info("val rows         : %d", len(val_df))
    log.info("test rows        : %d", len(test_df))
    log.info("=" * 55)

    if args.dry_run:
        log.info("Dry-run — not writing")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    aug_train.to_csv(os.path.join(args.output_dir, "tier1_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"),             index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"),           index=False)
    log.info("Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
