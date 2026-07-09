"""
Leakage detection for master_dataset_v2.csv.

Finds two classes of problematic pairs:
  1. Exact-duplicate sequences: same aptamer_sequence + same target_name but
     contradictory labels (1 vs 0). These are genuine data conflicts.
  2. Near-duplicate sequences: Levenshtein distance <= 2 between any two
     sequences. Uses a 6-mer inverted index to prune the candidate set before
     running full Levenshtein, reducing comparisons from O(n²) to tractable.

Why 6-mers as a pruning filter:
  For sequences of length L with Levenshtein distance <= 2, the pigeonhole
  principle guarantees they share at least one 6-mer (for L >= 20). Two
  edits can destroy at most 2*6=12 consecutive positions, leaving at least
  one intact 6-mer in any sequence of length >= 20. This filter eliminates
  >99% of pairs from consideration without missing any true near-dupes.

Outputs:
  data/processed/leakage_exact_conflicts.csv  — exact-dupe label conflicts
  data/processed/leakage_near_dupes.csv       — near-dupe pairs (Lev <= 2)
  Prints a summary to stdout.

The near-dupe results are consumed by assign_splits.py to ensure all members
of a near-dupe cluster land in the same split.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

KMER_LEN = 6
MAX_DIST = 2


def levenshtein(s1: str, s2: str, max_dist: int = MAX_DIST) -> int:
    """
    DP Levenshtein with early-exit when the running minimum exceeds max_dist.
    Returns min(actual_distance, max_dist + 1).
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist:
        return max_dist + 1

    # Keep shorter string in inner loop
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        row_min = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_dist:
            return max_dist + 1
        prev, curr = curr, prev
    return prev[len2]


def kmers(seq: str, k: int = KMER_LEN) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def find_exact_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows where (aptamer_sequence, target_name) pairs have contradictory
    labels (label=1 in one row, label=0 in another).
    """
    key = ["aptamer_sequence", "target_name"]
    grouped = df.groupby(key)["label"].nunique()
    conflict_keys = grouped[grouped > 1].index
    if len(conflict_keys) == 0:
        return pd.DataFrame()
    mask = df.set_index(key).index.isin(conflict_keys)
    conflicts = df[mask].copy()
    conflicts["conflict_type"] = "exact_sequence_label_conflict"
    return conflicts.reset_index(drop=True)


def find_near_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return pairs with Levenshtein distance <= MAX_DIST using a 6-mer index
    to prune candidates.
    """
    seqs = df["aptamer_sequence"].tolist()
    n = len(seqs)

    # Build inverted index: 6-mer → set of row indices
    index: dict[str, list[int]] = defaultdict(list)
    for i, seq in enumerate(seqs):
        if len(seq) < KMER_LEN:
            continue
        for km in kmers(seq):
            index[km].append(i)

    # Collect candidate pairs sharing at least one 6-mer
    candidates: set[tuple[int, int]] = set()
    for km, idxs in index.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i != j:
                    candidates.add((min(i, j), max(i, j)))

    print(f"  6-mer index pruned to {len(candidates):,} candidate pairs "
          f"(from {n*(n-1)//2:,} total possible)")

    rows = []
    for a, b in candidates:
        if seqs[a] == seqs[b]:
            continue  # exact matches handled separately
        dist = levenshtein(seqs[a], seqs[b])
        if dist <= MAX_DIST:
            ra = df.iloc[a]
            rb = df.iloc[b]
            split_a = ra["split"] if pd.notna(ra["split"]) else "unassigned"
            split_b = rb["split"] if pd.notna(rb["split"]) else "unassigned"
            # Cross-split: both are in a concrete split (not unassigned) and differ
            holdout = {"val", "test"}
            cross = (
                split_a in holdout or split_b in holdout
            ) and split_a != split_b

            rows.append(
                {
                    "idx_a": a,
                    "idx_b": b,
                    "seq_a": ra["aptamer_sequence"],
                    "seq_b": rb["aptamer_sequence"],
                    "target_a": ra["target_name"],
                    "target_b": rb["target_name"],
                    "label_a": int(ra["label"]),
                    "label_b": int(rb["label"]),
                    "split_a": split_a,
                    "split_b": split_b,
                    "levenshtein_dist": dist,
                    "cross_split": cross,
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    csv_path = DATA / "master_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"Loaded {n_total} rows from {csv_path.name}")

    # ── 1. Exact conflicts ────────────────────────────────────────────────────
    print("\n[1/2] Checking for exact-sequence label conflicts …")
    conflicts = find_exact_conflicts(df)
    n_conflict_rows = len(conflicts)
    n_conflict_pairs = n_conflict_rows // 2
    if conflicts.empty:
        print("  None found.")
    else:
        out = DATA / "leakage_exact_conflicts.csv"
        conflicts.to_csv(out, index=False)
        print(f"  {n_conflict_rows} conflict rows ({n_conflict_pairs} pairs) → {out.name}")

    # ── 2. Near-duplicates ────────────────────────────────────────────────────
    print("\n[2/2] Checking for near-duplicate sequences (Levenshtein ≤ 2) …")
    near = find_near_dupes(df)

    n_near = 0
    n_cross = 0
    if not near.empty:
        n_near = len(near)
        n_cross = int(near["cross_split"].sum())
        out_near = DATA / "leakage_near_dupes.csv"
        near.to_csv(out_near, index=False)
        print(f"  {n_near} near-dupe pairs → {out_near.name}")
        if n_cross > 0:
            cross_df = near[near["cross_split"]]
            print(f"  ⚠  {n_cross} pairs already cross split boundaries:")
            for _, row in cross_df.iterrows():
                print(
                    f"     [{row['split_a']} vs {row['split_b']}] dist={row['levenshtein_dist']}"
                    f"  {row['seq_a'][:30]}… / {row['seq_b'][:30]}…"
                )
    else:
        print("  None found.")
        # Write empty file so downstream scripts can always read it
        pd.DataFrame(
            columns=[
                "idx_a","idx_b","seq_a","seq_b","target_a","target_b",
                "label_a","label_b","split_a","split_b","levenshtein_dist","cross_split"
            ]
        ).to_csv(DATA / "leakage_near_dupes.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── LEAKAGE SUMMARY ──────────────────────────────────────────────────")
    print(f"  Total rows analysed: {n_total}")
    print(f"  Exact-sequence label conflicts: {n_conflict_rows} rows ({n_conflict_pairs} pairs)")
    print(f"  Near-dupe pairs (Levenshtein ≤ {MAX_DIST}): {n_near}")
    print(f"    of which cross existing split boundaries: {n_cross}")
    if not conflicts.empty:
        print(
            "\n  ⚠  Exact conflicts are NOT auto-resolved — logged to "
            "leakage_exact_conflicts.csv for manual review."
        )
    print("─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
