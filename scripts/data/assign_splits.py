"""
Split assignment for master_dataset_v2.csv — corrected algorithm.

Key change from the previous version: Union-Find clusters are built
ONLY from genuine near-dupe sequence pairs (Levenshtein ≤ 2 edges from
leakage_near_dupes.csv). The old version also unioned all rows with the
same target_name, which forced entire protein families into one cluster
regardless of whether their sequences were actually similar — that's what
caused myoglobin, NT-proBNP, and troponin to end up 100% in train.

Algorithm:
  1. Keep all existing train/val/test assignments from the pre-split state
     (train=3344, val=159, test=136).
  2. Fix the 5 original cross-split near-dupe pairs by moving the train-side
     member to match its held-out neighbor.
  3. Assign the 4 Tier-2 unassigned rows → val (benchmark coverage).
  4. For all remaining unassigned rows, build Union-Find using ONLY
     near-dupe sequence edges. Any cluster that contains a val or test row
     forces all its unassigned members into that same held-out split.
  5. Remaining unassigned rows (no held-out neighbor) are assigned by
     80/10/10 budget: each cluster of near-dupe sequences is assigned
     together, but no cross-cluster target_name merging.
  6. Run convergence pass: iteratively move train rows into val/test if
     they are near-dupes of val/test rows, until no cross-split pairs remain.

Postcondition: zero near-dupe pairs cross a train<->val or train<->test boundary.
"""

from __future__ import annotations

import io
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

HOLDOUT = {"val", "test"}

# Git SHA of the last commit *before* split assignment ran in PR #9.
# This gives us the original unassigned/NaN state to reset to.
PRE_SPLIT_SHA = "92865cf"


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


def _convergence_pass(df: pd.DataFrame, near: pd.DataFrame) -> int:
    """Move train near-dupes of held-out rows → held-out. Iterate until stable."""
    total = 0
    for _ in range(20):
        moved = 0
        for _, row in near.iterrows():
            ia, ib = int(row["idx_a"]), int(row["idx_b"])
            sa, sb = df.at[ia, "split"], df.at[ib, "split"]
            if sa == sb:
                continue
            if sa in HOLDOUT and sb == "train":
                df.at[ib, "split"] = sa
                moved += 1
            elif sb in HOLDOUT and sa == "train":
                df.at[ia, "split"] = sb
                moved += 1
        total += moved
        if moved == 0:
            break
    return total


def main() -> None:
    csv_path = DATA / "master_dataset_v2.csv"
    near_path = DATA / "leakage_near_dupes.csv"

    cur_df = pd.read_csv(csv_path)
    near = pd.read_csv(near_path)
    n = len(cur_df)
    print(f"Loaded {n} rows | {len(near)} near-dupe pairs")

    # ── Restore pre-split state ────────────────────────────────────────────────
    raw = subprocess.run(
        ["git", "show", f"{PRE_SPLIT_SHA}:data/processed/master_dataset_v2.csv"],
        capture_output=True, text=True,
    ).stdout
    pre_df = pd.read_csv(io.StringIO(raw))
    assert len(pre_df) == n
    assert (pre_df["aptamer_sequence"] == cur_df["aptamer_sequence"]).all(), \
        "Row order differs between pre-split snapshot and current CSV!"

    pre_df["split"] = pre_df["split"].fillna("unassigned")

    # Use current (normalised) target names but original split column
    df = cur_df.copy()
    df["split"] = pre_df["split"]

    orig_assigned = df["split"].isin(["train", "val", "test"])
    print(f"Pre-existing assignments: {orig_assigned.sum()} rows  "
          f"(train={(df['split']=='train').sum()}, "
          f"val={(df['split']=='val').sum()}, "
          f"test={(df['split']=='test').sum()})")
    print(f"Originally unassigned/NaN: {(~orig_assigned).sum()} rows")

    # ── Step 2: Fix 5 original cross-split near-dupe pairs ────────────────────
    cross = near[near["cross_split"] == True]
    print(f"\n[2] Fixing {len(cross)} original cross-split pairs …")
    moved2 = 0
    for _, row in cross.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        sa, sb = df.at[ia, "split"], df.at[ib, "split"]
        if sa == "test" and sb == "train":
            df.at[ib, "split"] = "test"; moved2 += 1
        elif sa == "train" and sb == "test":
            df.at[ia, "split"] = "test"; moved2 += 1
    print(f"  Moved {moved2} train rows → test")

    # ── Step 3: Tier-2 unassigned rows → val ──────────────────────────────────
    t2_ua = df[(df["split"] == "unassigned") & (df["training_tier"] == 2)]
    print(f"\n[3] Assigning {len(t2_ua)} Tier-2 unassigned rows → val")
    for idx, row in t2_ua.iterrows():
        df.at[idx, "split"] = "val"
        print(f"  Row {idx}: {row['target_name']} → val")

    # ── Step 4: Union-Find on near-dupe edges ONLY (no target_name grouping) ──
    rem_mask = df["split"] == "unassigned"
    ua_idx = sorted(df[rem_mask].index.tolist())
    ua_set = set(ua_idx)
    real_to_c = {r: c for c, r in enumerate(ua_idx)}

    print(f"\n[4] Assigning {len(ua_idx)} remaining unassigned rows …")
    print("    (Union-Find on sequence near-dupe edges only — no target_name grouping)")

    uf = _UF(len(ua_idx))

    # Union ONLY on near-dupe sequence pairs
    near_ua_edges = 0
    for _, row in near.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        if ia in ua_set and ib in ua_set:
            uf.union(real_to_c[ia], real_to_c[ib])
            near_ua_edges += 1

    print(f"  Near-dupe edges within unassigned set: {near_ua_edges}")

    # Determine forced splits from assigned held-out neighbors
    forced: dict[int, str] = {}   # row_index → forced split
    for _, row in near.iterrows():
        ia, ib = int(row["idx_a"]), int(row["idx_b"])
        sa = df.at[ia, "split"]
        sb = df.at[ib, "split"]
        if ia in ua_set and sb in HOLDOUT:
            prev = forced.get(ia)
            if prev is None or (prev == "val" and sb == "test"):
                forced[ia] = sb
        if ib in ua_set and sa in HOLDOUT:
            prev = forced.get(ib)
            if prev is None or (prev == "val" and sa == "test"):
                forced[ib] = sa

    # Propagate forced split through UF clusters (test > val)
    cluster_forced: dict[int, str] = {}
    for row_idx, fsplit in forced.items():
        root = uf.find(real_to_c[row_idx])
        prev = cluster_forced.get(root)
        if prev is None or (prev == "val" and fsplit == "test"):
            cluster_forced[root] = fsplit

    forced_count = 0
    for ua_row in ua_idx:
        root = uf.find(real_to_c[ua_row])
        if root in cluster_forced:
            df.at[ua_row, "split"] = cluster_forced[root]
            forced_count += 1

    print(f"  Forced by held-out neighbor: {forced_count} rows")

    # ── Step 5: Budget assignment for remaining free unassigned rows ──────────
    still_ua = [i for i in ua_idx if df.at[i, "split"] == "unassigned"]
    print(f"\n[5] Budget assignment for {len(still_ua)} free unassigned rows …")

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
    print(f"  Targets: train={target_train}, val={target_val}, test={target_test}")
    print(f"  Budget:  +train={budget['train']}, +val={budget['val']}, +test={budget['test']}")

    # Group free unassigned rows by near-dupe cluster only
    free_clusters: dict[int, list[int]] = defaultdict(list)
    for row_idx in still_ua:
        root = uf.find(real_to_c[row_idx])
        free_clusters[root].append(row_idx)

    groups = sorted(free_clusters.values(), key=len, reverse=True)
    print(f"  {len(groups)} free clusters (largest: {max(len(g) for g in groups)} rows)")

    used = {"train": 0, "val": 0, "test": 0}

    def pick(sz: int) -> str:
        for s in ["val", "test", "train"]:
            if budget[s] - used[s] >= sz:
                return s
        return max(budget, key=lambda s: budget[s] - used[s])

    for grp in groups:
        sp = pick(len(grp))
        used[sp] += len(grp)
        for row_idx in grp:
            df.at[row_idx, "split"] = sp

    assert (df["split"] == "unassigned").sum() == 0
    assert df["split"].isna().sum() == 0

    # ── Step 6: Convergence pass ──────────────────────────────────────────────
    print(f"\n[6] Convergence pass …")
    conv_moved = _convergence_pass(df, near)
    print(f"  Moved {conv_moved} additional rows to holdout in convergence pass")

    # ── Final verification ────────────────────────────────────────────────────
    cross_left = sum(
        1 for _, row in near.iterrows()
        if (sa := df.at[int(row["idx_a"]), "split"]) != (sb := df.at[int(row["idx_b"]), "split"])
        and (sa in HOLDOUT or sb in HOLDOUT)
    )
    assert cross_left == 0, f"{cross_left} cross-split near-dupe pairs remain!"

    counts = df["split"].value_counts().to_dict()
    print(f"\n── FINAL SPLITS ─────────────────────────────────────────────────")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {counts[s]} ({counts[s]/n*100:.1f}%)")
    print(f"  Total: {sum(counts.values())}  |  cross-split near-dupe pairs: 0 ✓")
    print(f"  Row count unchanged: {sum(counts.values()) == n} ✓")

    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path.name}")


if __name__ == "__main__":
    main()
