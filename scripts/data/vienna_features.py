"""
Precompute and cache ViennaRNA secondary-structure features for DNA aptamers.

ViennaRNA is slow (~10-50 ms/sequence). This script runs it once and stores
results in data/processed/vienna_cache.pkl so the model loader can do a fast
dict lookup at training time.

Features extracted per sequence:
    mfe          : float  — minimum free energy (kcal/mol)
    structure    : str    — dot-bracket notation
    stem_count   : int    — number of helical stem regions
    loop_count   : int    — number of loop regions
    bp_prob_mean : float  — mean base-pair probability (from partition function)
    bp_prob_max  : float  — max  base-pair probability

Usage:
    python scripts/data/vienna_features.py --input data/processed/master_dataset.csv
"""

import os
import sys
import pickle
import logging
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import VIENNA_CACHE, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import RNA  # ViennaRNA Python bindings
    _VIENNA_AVAILABLE = True
except ImportError:
    _VIENNA_AVAILABLE = False
    log.warning("ViennaRNA (RNA module) not importable. Features will use fallback zeros.")


def _count_stems_loops(structure: str) -> tuple[int, int]:
    """Count stem and loop regions from a dot-bracket string."""
    stems = 0
    loops = 0
    i = 0
    n = len(structure)
    while i < n:
        # A stem starts when '(' appears after a non-'(' or at the start
        if structure[i] == "(" and (i == 0 or structure[i - 1] != "("):
            stems += 1
        # A hairpin loop: '(' followed eventually by ')' with only '.' in between
        if structure[i] == "(":
            j = i + 1
            while j < n and structure[j] == ".":
                j += 1
            if j < n and structure[j] == ")":
                loops += 1
        i += 1
    return stems, loops


def compute_vienna_features(seq: str) -> dict:
    """
    Compute ViennaRNA features for a single DNA aptamer sequence.

    ViennaRNA works with RNA by default; we pass the DNA sequence directly —
    it will interpret T as U internally, but the structure topology is the
    same since base-pairing rules are conserved for our purposes.
    """
    if not _VIENNA_AVAILABLE:
        return {"mfe": 0.0, "structure": "." * len(seq),
                "stem_count": 0, "loop_count": 0,
                "bp_prob_mean": 0.0, "bp_prob_max": 0.0}

    md = RNA.md()  # model details, default params
    fc = RNA.fold_compound(seq, md)

    structure, mfe = fc.mfe()
    stem_count, loop_count = _count_stems_loops(structure)

    # Partition function for base-pair probabilities
    fc.pf()
    bpp = fc.bpp()  # upper-triangular matrix; bpp[i][j] = P(i paired with j)

    n = len(seq)
    probs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p > 0:
                probs.append(p)

    bp_prob_mean = sum(probs) / len(probs) if probs else 0.0
    bp_prob_max  = max(probs) if probs else 0.0

    return {
        "mfe":          float(mfe),
        "structure":    structure,
        "stem_count":   stem_count,
        "loop_count":   loop_count,
        "bp_prob_mean": float(bp_prob_mean),
        "bp_prob_max":  float(bp_prob_max),
    }


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        log.info("Loaded %d cached sequences from %s", len(cache), cache_path)
        return cache
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    log.info("Cache saved (%d entries) → %s", len(cache), cache_path)


def build_cache(sequences: list[str], cache_path: str) -> dict:
    cache = load_cache(cache_path)
    new_count = 0

    for i, seq in enumerate(sequences):
        seq_upper = seq.strip().upper()
        if seq_upper in cache:
            continue
        if (i + 1) % 500 == 0:
            log.info("  %d/%d processed (%d new)", i + 1, len(sequences), new_count)
            save_cache(cache, cache_path)

        try:
            feats = compute_vienna_features(seq_upper)
            cache[seq_upper] = feats
            new_count += 1
        except Exception as exc:
            log.warning("ViennaRNA failed on %s: %s", seq_upper[:30], exc)
            cache[seq_upper] = {"mfe": 0.0, "structure": "." * len(seq_upper),
                                "stem_count": 0, "loop_count": 0,
                                "bp_prob_mean": 0.0, "bp_prob_max": 0.0}

    save_cache(cache, cache_path)
    log.info("Done. %d new sequences added; cache total: %d", new_count, len(cache))
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ViennaRNA features")
    parser.add_argument("--input", required=True,
                        help="CSV with a 'sequence' column")
    parser.add_argument("--cache", default=VIENNA_CACHE,
                        help="Output pickle path (default: data/processed/vienna_cache.pkl)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    seq_col = "aptamer_sequence" if "aptamer_sequence" in df.columns else "sequence"
    if seq_col not in df.columns:
        sys.exit("Input CSV must have an 'aptamer_sequence' or 'sequence' column")

    sequences = df[seq_col].dropna().tolist()
    log.info("Processing %d sequences", len(sequences))
    build_cache(sequences, args.cache)


if __name__ == "__main__":
    # Self-test with a short example
    test_seq = "ATGCATGCATGCATGCATGCATGC"
    print(f"Testing ViennaRNA on: {test_seq}")
    feats = compute_vienna_features(test_seq)
    for k, v in feats.items():
        print(f"  {k}: {v}")

    if _VIENNA_AVAILABLE:
        print("ViennaRNA self-test passed.")
    else:
        print("ViennaRNA not available — returned zero features (install ViennaRNA to fix).")

    if len(sys.argv) > 1:
        main()
