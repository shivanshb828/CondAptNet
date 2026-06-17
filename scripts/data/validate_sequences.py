"""
Sequence quality control for aptamer records.

Applies the validation rules from CLAUDE.md Section 5 before any sequence
enters the dataset. Logs warnings for borderline cases; never silently drops.

Input:  CSV with at least a `sequence` column
Output: Same CSV with added `valid` (bool) and `fail_reason` columns

Usage:
    python scripts/data/validate_sequences.py --input data/raw/pubmed_results.csv \
                                               --output data/processed/validated.csv
"""

import re
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SEQ_MIN_LEN, SEQ_MAX_LEN, GC_MIN, GC_MAX, MAX_HOMOPOLYMER, VALID_BASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Pre-compiled pattern for homopolymer detection
_HOMOPOLYMER_RE = re.compile(r"(A{n}|T{n}|G{n}|C{n})".replace("{n}", "{" + str(MAX_HOMOPOLYMER + 1) + ",}"))


def validate_sequence(seq: str, existing_keys: set[str] | None = None, key: str | None = None) -> tuple[bool, str]:
    """
    Validate a single DNA aptamer sequence.

    Returns (is_valid, fail_reason). fail_reason is '' when valid.
    existing_keys: set of already-seen (sequence, target_protein) composite keys.
    key: the composite key for this row (sequence + target_protein).
    """
    if not isinstance(seq, str):
        return False, "not_a_string"

    seq = seq.strip().upper()

    if len(seq) < SEQ_MIN_LEN:
        return False, f"too_short ({len(seq)} < {SEQ_MIN_LEN})"
    if len(seq) > SEQ_MAX_LEN:
        return False, f"too_long ({len(seq)} > {SEQ_MAX_LEN})"

    illegal = set(seq) - VALID_BASES
    if illegal:
        return False, f"invalid_bases ({','.join(sorted(illegal))})"

    gc = (seq.count("G") + seq.count("C")) / len(seq)
    if gc < GC_MIN:
        return False, f"gc_too_low ({gc:.2f} < {GC_MIN})"
    if gc > GC_MAX:
        return False, f"gc_too_high ({gc:.2f} > {GC_MAX})"

    if _HOMOPOLYMER_RE.search(seq):
        return False, f"homopolymer_run (>{MAX_HOMOPOLYMER} identical bases)"

    # Duplicate = same (sequence, target_protein) pair, not just same sequence.
    # The same aptamer against two different proteins is valid (cross-target context).
    if existing_keys is not None and key is not None and key in existing_keys:
        return False, "duplicate"

    return True, ""


def validate_dataframe(df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
    """
    Add `valid` and `fail_reason` columns to df.
    Tracks seen (sequence, target_protein) pairs to flag true duplicates.
    """
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found in dataframe")

    protein_col = "target_protein" if "target_protein" in df.columns else None
    seen: set[str] = set()
    valid_flags = []
    fail_reasons = []

    for i, seq in enumerate(df[seq_col]):
        protein = str(df[protein_col].iloc[i]).strip().lower() if protein_col else ""
        key = f"{str(seq).strip().upper()}|||{protein}"
        is_valid, reason = validate_sequence(seq, seen, key)
        if is_valid:
            seen.add(key)
        valid_flags.append(is_valid)
        fail_reasons.append(reason)

    df = df.copy()
    df["valid"] = valid_flags
    df["fail_reason"] = fail_reasons
    return df


def report(df: pd.DataFrame) -> None:
    total = len(df)
    passed = df["valid"].sum()
    failed = total - passed
    log.info("Validation summary: %d total, %d passed (%.1f%%), %d failed",
             total, passed, 100 * passed / total if total else 0, failed)
    if failed:
        reasons = df[~df["valid"]]["fail_reason"].value_counts()
        for reason, count in reasons.items():
            log.info("  %-40s %d", reason, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate aptamer sequences")
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seq-col", default="sequence", help="Name of sequence column")
    args = parser.parse_args()

    log.info("Reading %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows", len(df))

    df = validate_dataframe(df, seq_col=args.seq_col)
    report(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Written to %s", args.output)


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        ("ATGCATGCATGCATGCATGCATGC", True),    # valid 24-mer
        ("ATGC",                                False),   # too short
        ("AAAAAAAAATGC" * 12,                  False),   # too long
        ("ATGCATGCATGCRZATGCATGCAT",            False),   # invalid bases
        ("AAAAAAAAAATGCATGCATGCATG",            False),   # homopolymer run
        ("ATATATATATATATATATATAT",              False),   # low GC
        ("GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC", False),  # high GC
    ]
    print("Running self-tests...")
    for seq, expected_valid in test_cases:
        is_valid, reason = validate_sequence(seq)
        status = "OK" if is_valid == expected_valid else "FAIL"
        print(f"  [{status}] seq={seq[:30]!r}... valid={is_valid} reason={reason!r}")

    main() if len(sys.argv) > 1 else print("\nSelf-test complete. Pass --input and --output to run on real data.")
