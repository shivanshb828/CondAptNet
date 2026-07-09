"""
Split assignment for master_dataset_v2.csv.

Assigns train/val/test splits to the 858 'unassigned' and 2 NaN-split rows.
Targeting an 80/10/10 overall split ratio.

Rules (in priority order):
  1. Keep all existing train/val/test assignments unchanged.
  2. Fix 5 cross-split near-dupe pairs (train member → test, to match held-out cluster).
  3. Assign Tier-2 (validation-target) unassigned/NaN rows to val so benchmarks
     have coverage across splits.
  4. Group remaining unassigned rows by target_name (protein-family proxy).
     Assign each target group entirely to one split to prevent leakage.
  5. Near-dupe clusters (from leakage_near_dupes.csv) must land in the same split.
     (Constraint satisfied for unassigned rows after step 2: no unassigned row is
     a near-dupe of an existing val/test row.)
  6. Assign target groups to splits in a greedy round-robin by group size,
     targeting 80/10/10 overall (accounting for already-assigned rows).

Output: overwrites data/processed/master_dataset_v2.csv with split column filled.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def _union_find(pairs: list[tuple[int, int]], n: int) -> dict[int, list[int]]:
    """Union-Find returning {root: [member_indices]} components."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return comps


def main() -> None:
    csv_path = DATA / "master_dataset_v2.csv"
    near_path = DATA / "leakage_near_dupes.csv"

    df = pd.read_csv(csv_path)
    n_before = len(df)
    near = pd.read_csv(near_path)
    print(f"Loaded {n_before} rows | {len(near)} near-dupe pairs")

    # Normalise: NaN split → 'unassigned'
    df["split"] = df["split"].fillna("unassigned")
    needs_mask = df["split"] == "unassigned"
    print(f"Rows needing assignment: {needs_mask.sum()}")

    # ── Step 2: Fix cross-split near-dupe pairs ───────────────────────────────
    cross = near[near["cross_split"] == True]
    print(f"\n[2] Fixing {len(cross)} cross-split near-dupe pairs …")
    moved = []
    for _, row in cross.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        sa, sb = df.at[ia, "split"], df.at[ib, "split"]
        if sa == "test" and sb == "train":
            df.at[ib, "split"] = "test"
            moved.append(ib)
        elif sa == "train" and sb == "test":
            df.at[ia, "split"] = "test"
            moved.append(ia)
    print(f"  Moved {len(moved)} train rows → test")

    # ── Step 3: Tier-2 unassigned → val ──────────────────────────────────────
    t2_ua = df[(df["split"] == "unassigned") & (df["training_tier"] == 2)]
    print(f"\n[3] Assigning {len(t2_ua)} Tier-2 unassigned rows → val")
    df.loc[t2_ua.index, "split"] = "val"

    # ── Steps 4-6: Assign remaining unassigned by target group ────────────────
    rem_mask = df["split"] == "unassigned"
    print(f"\n[4-6] Assigning {rem_mask.sum()} remaining unassigned rows …")

    train_n = (df["split"] == "train").sum()
    val_n   = (df["split"] == "val").sum()
    test_n  = (df["split"] == "test").sum()

    target_train = math.ceil(n_before * 0.80)
    target_val   = math.ceil(n_before * 0.10)
    target_test  = n_before - target_train - target_val

    budget = {
        "train": max(0, target_train - train_n),
        "val":   max(0, target_val   - val_n),
        "test":  max(0, target_test  - test_n),
    }
    print(f"  Targets: train={target_train}, val={target_val}, test={target_test}")
    print(f"  Budget:  train+{budget['train']}, val+{budget['val']}, test+{budget['test']}")

    ua_idx_list = sorted(df[rem_mask].index.tolist())
    real_to_c = {r: c for c, r in enumerate(ua_idx_list)}

    # Union by near-dupe AND by same target_name
    pairs_c: list[tuple[int, int]] = []
    for _, row in near.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        if ia in real_to_c and ib in real_to_c:
            pairs_c.append((real_to_c[ia], real_to_c[ib]))

    target_to_cidxs: dict[str, list[int]] = defaultdict(list)
    for r in ua_idx_list:
        target_to_cidxs[df.at[r, "target_name"]].append(real_to_c[r])
    for cidxs in target_to_cidxs.values():
        for i in range(1, len(cidxs)):
            pairs_c.append((cidxs[0], cidxs[i]))

    comps = _union_find(pairs_c, len(ua_idx_list))
    groups = sorted(comps.values(), key=len, reverse=True)
    print(f"  {len(groups)} target/cluster groups (largest: {max(len(g) for g in groups)} rows)")

    used = {"train": 0, "val": 0, "test": 0}

    def pick_split(sz: int) -> str:
        for s in ["val", "test", "train"]:
            if budget[s] - used[s] >= sz:
                return s
        return max(budget, key=lambda s: budget[s] - used[s])

    for group in groups:
        split = pick_split(len(group))
        used[split] += len(group)
        for ci in group:
            df.at[ua_idx_list[ci], "split"] = split

    # ── Verify ────────────────────────────────────────────────────────────────
    assert (df["split"] == "unassigned").sum() == 0
    assert df["split"].isna().sum() == 0
    assert len(df) == n_before

    counts = df["split"].value_counts().to_dict()
    print(f"\n── FINAL SPLITS ──────────────────────────────────")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {counts[s]} ({counts[s]/n_before*100:.1f}%)")
    print(f"  Total: {sum(counts.values())}")

    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
