"""
Split assignment for master_dataset_v2.csv.

Assigns train/val/test splits to the 858 'unassigned' and 2 NaN-split rows.
Targeting an 80/10/10 overall split ratio.

Rules (in priority order):
  1. Keep all existing train/val/test assignments unchanged.
  2. Fix cross-split near-dupe pairs where test is the anchor: train member → test.
  3. Build a global Union-Find over ALL rows using near-dupe pairs AND same
     target_name. Any component containing a val or test member forces all
     unassigned members in that component to the same held-out split.
  4. Remaining unassigned rows (no val/test neighbor) are grouped by target_name
     and assigned greedily to reach the 80/10/10 budget.
  5. Tier-2 unassigned rows with no forced split go to val.

Output: overwrites data/processed/master_dataset_v2.csv with split column filled.
Postcondition: zero near-dupe pairs cross a train/val or train/test boundary.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    csv_path = DATA / "master_dataset_v2.csv"
    near_path = DATA / "leakage_near_dupes.csv"

    df = pd.read_csv(csv_path)
    n = len(df)
    near = pd.read_csv(near_path)
    print(f"Loaded {n} rows | {len(near)} near-dupe pairs")

    # Normalise NaN splits → 'unassigned'
    df["split"] = df["split"].fillna("unassigned")
    print(f"Rows needing assignment: {(df['split']=='unassigned').sum()}")

    # ── Step 2: Fix original cross-split near-dupe pairs (test is anchor) ────
    cross = near[near["cross_split"] == True]
    print(f"\n[2] Fixing {len(cross)} original cross-split pairs …")
    moved = []
    for _, row in cross.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        sa, sb = df.at[ia, "split"], df.at[ib, "split"]
        if sa == "test" and sb == "train":
            df.at[ib, "split"] = "test"; moved.append(ib)
        elif sa == "train" and sb == "test":
            df.at[ia, "split"] = "test"; moved.append(ia)
    print(f"  Moved {len(moved)} train rows → test")

    # ── Step 3: Global Union-Find over ALL rows ───────────────────────────────
    # Union by (a) near-dupe pairs and (b) same target_name
    print(f"\n[3] Building global Union-Find …")
    uf = _UF(n)

    for _, row in near.iterrows():
        uf.union(int(row["idx_a"]), int(row["idx_b"]))

    # Also union by target_name so whole protein families stay together
    target_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, tname in enumerate(df["target_name"]):
        target_to_rows[tname].append(i)
    for rows in target_to_rows.values():
        for j in range(1, len(rows)):
            uf.union(rows[0], rows[j])

    # Build component map
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)

    # For each component, determine the forced split (val/test takes priority)
    # and collect unassigned members that need assignment
    print(f"[3] Determining forced splits from {len(comp)} components …")
    ua_rows = set(df[df["split"] == "unassigned"].index)
    holdout = {"val", "test"}

    forced: dict[int, str] = {}   # row_index → forced split
    unforced: list[int] = []      # row indices that have no val/test neighbor

    for root, members in comp.items():
        ua_in_comp = [m for m in members if m in ua_rows]
        if not ua_in_comp:
            continue
        # Determine if any non-unassigned member is val or test
        anchors: dict[str, int] = {}  # split → count
        for m in members:
            s = df.at[m, "split"]
            if s in holdout:
                anchors[s] = anchors.get(s, 0) + 1

        if anchors:
            # If both val and test are in the same component (rare), prefer test
            # (test is more valuable to keep clean)
            forced_split = "test" if "test" in anchors else "val"
            for ua in ua_in_comp:
                forced[ua] = forced_split
        else:
            unforced.extend(ua_in_comp)

    print(f"  {len(forced)} unassigned rows forced by val/test neighbor")
    print(f"  {len(unforced)} unassigned rows free to assign by budget")

    # Apply forced assignments
    for idx, split in forced.items():
        df.at[idx, "split"] = split

    # ── Step 4: Budget-based assignment for remaining unassigned rows ─────────
    print(f"\n[4] Greedy budget assignment for {len(unforced)} free rows …")
    train_n = (df["split"] == "train").sum()
    val_n   = (df["split"] == "val").sum()
    test_n  = (df["split"] == "test").sum()

    target_train = math.ceil(n * 0.80)
    target_val   = math.ceil(n * 0.10)
    target_test  = n - target_train - target_val

    budget = {
        "train": max(0, target_train - train_n),
        "val":   max(0, target_val   - val_n),
        "test":  max(0, target_test  - test_n),
    }
    print(f"  Budget after forced assignments: train+{budget['train']}, "
          f"val+{budget['val']}, test+{budget['test']}")

    # Group free unassigned rows by their component root (already unionised by target)
    # and assign whole groups at once
    free_ua = set(unforced)
    free_comps: dict[int, list[int]] = defaultdict(list)
    for i in free_ua:
        # Only group with other free unassigned rows in same component
        free_comps[uf.find(i)].append(i)

    groups = sorted(free_comps.values(), key=len, reverse=True)

    # Tier-2 free rows go to val first (benchmark coverage)
    used = {"train": 0, "val": 0, "test": 0}

    def pick(sz: int) -> str:
        for s in ["val", "test", "train"]:
            if budget[s] - used[s] >= sz:
                return s
        return max(budget, key=lambda s: budget[s] - used[s])

    for group in groups:
        # If any row in group is tier-2 and val has budget, go to val
        has_tier2 = any(df.at[i, "training_tier"] == 2 for i in group)
        if has_tier2 and budget["val"] - used["val"] >= len(group):
            sp = "val"
        else:
            sp = pick(len(group))
        used[sp] += len(group)
        for i in group:
            df.at[i, "split"] = sp

    # ── Verify ────────────────────────────────────────────────────────────────
    assert (df["split"] == "unassigned").sum() == 0, "Some rows still unassigned!"
    assert df["split"].isna().sum() == 0, "NaN splits remain!"
    assert len(df) == n, "Row count changed!"

    # Verify no near-dupe pairs cross train/holdout boundaries
    cross_left = 0
    for _, row in near.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        sa, sb = df.at[ia, "split"], df.at[ib, "split"]
        if sa != sb and (sa in holdout or sb in holdout):
            cross_left += 1
    assert cross_left == 0, f"{cross_left} cross-split near-dupe pairs remain after assignment!"

    counts = df["split"].value_counts().to_dict()
    print(f"\n── FINAL SPLITS ──────────────────────────────────────────")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {counts[s]} ({counts[s]/n*100:.1f}%)")
    print(f"  Total: {sum(counts.values())}")
    print(f"  Cross-split near-dupe pairs remaining: 0 ✓")

    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
