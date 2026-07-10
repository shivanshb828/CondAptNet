"""
Audit and normalize target_name case/whitespace collisions in master_dataset_v2.csv.

Outputs (written BEFORE modifying the CSV):
  data/processed/target_name_normalization_log.csv
    columns: old_value, new_value, rows_affected

Then applies the renaming to master_dataset_v2.csv in place.

Canonical selection rule:
  - Prefer the form used by the majority of rows.
  - For exact ties, prefer the more complete/formally-cased form
    (title case for multi-word protein/compound names; chemical
    conventions for stereo-descriptors like (S), L-).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def deep_norm(s: str) -> str:
    """Lowercase + strip + collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def pick_canonical(variants: list[str], counts: dict[str, int]) -> str:
    """
    Choose the canonical form for a collision group.
    Rule: highest count wins; ties resolved by formal/complete name preference.
    """
    max_count = max(counts[v] for v in variants)
    top = [v for v in variants if counts[v] == max_count]
    if len(top) == 1:
        return top[0]
    # Tie-break: prefer title case (first char uppercase, rest natural)
    titled = [v for v in top if v[0].isupper()]
    if len(titled) == 1:
        return titled[0]
    # Further tie-break: prefer the one whose first non-paren char is uppercase
    def first_alpha_upper(s: str) -> bool:
        for c in s:
            if c.isalpha():
                return c.isupper()
        return False
    upper_first = [v for v in top if first_alpha_upper(v)]
    if upper_first:
        return sorted(upper_first)[0]  # alphabetical as last resort
    return sorted(top)[0]


def main() -> None:
    csv_path = DATA / "master_dataset_v2.csv"
    log_path = DATA / "target_name_normalization_log.csv"

    df = pd.read_csv(csv_path)
    n_before = len(df)
    print(f"Loaded {n_before} rows from {csv_path.name}")
    print(f"Unique target_name (raw):      {df['target_name'].nunique()}")
    print(f"Unique target_name (deep-norm): {df['target_name'].apply(deep_norm).nunique()}")

    # ── Find collision groups ─────────────────────────────────────────────────
    groups: dict[str, list[str]] = defaultdict(list)
    for name in df["target_name"].unique():
        groups[deep_norm(name)].append(name)

    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\nCollision groups found: {len(collisions)}")

    # ── Determine canonical for each group ───────────────────────────────────
    counts = df["target_name"].value_counts().to_dict()
    rename_map: dict[str, str] = {}  # old → new (only non-canonical variants)

    for key, variants in sorted(collisions.items()):
        canon = pick_canonical(variants, counts)
        non_canonical = [v for v in variants if v != canon]
        for old in non_canonical:
            rename_map[old] = canon

        print(f"\n  [{key}]  →  canonical: '{canon}'")
        for v in sorted(variants, key=lambda x: -counts.get(x, 0)):
            flag = "✓ keep" if v == canon else f"→ '{canon}'"
            print(f"    {counts.get(v, 0):4d}  '{v}'  {flag}")

    # ── Write normalization log BEFORE touching the CSV ───────────────────────
    log_rows = [
        {"old_value": old, "new_value": new, "rows_affected": counts.get(old, 0)}
        for old, new in sorted(rename_map.items())
    ]
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False)
    print(f"\nNormalization log written → {log_path.name} ({len(log_df)} renames)")

    # ── Apply renames ─────────────────────────────────────────────────────────
    total_changed = 0
    for old, new in rename_map.items():
        mask = df["target_name"] == old
        n = mask.sum()
        df.loc[mask, "target_name"] = new
        total_changed += n
        print(f"  Renamed {n:3d} rows: '{old}' → '{new}'")

    assert len(df) == n_before, "Row count changed — bug!"
    print(f"\nTotal rows renamed: {total_changed}")
    print(f"Unique target_name after: {df['target_name'].nunique()}")

    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
