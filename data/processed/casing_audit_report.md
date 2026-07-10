# Target Name Casing Audit Report
**Branch:** data/fix-target-casing-and-resplit  
**Date:** 2026-07-10  
**Dataset:** data/processed/master_dataset_v2.csv (4,499 rows)

---

## 1. Full Audit — All Naming Inconsistencies Found

Deep normalization (strip + lowercase + collapse whitespace) found exactly **8 collision groups** spanning 61 rows. No hidden punctuation or internal-whitespace variants were found beyond the 8 already known.

| Canonical (post-fix) | Variants before fix | Counts | Tie-break rule |
|---|---|---|---|
| `(S)-ibuprofen` | `(S)-ibuprofen`, `(s)-ibuprofen` | 2, 2 | Tie → chemical convention `(S)` cap |
| `anterior gradient homolog 2 (AGR2)` | `anterior…`, `Anterior…` | 6, 2 | Majority lower |
| `Egg white lysozyme` | `Egg white lysozyme`, `egg white lysozyme` | 3, 3 | Tie → title case more formal |
| `L-tyrosinamide` | `L-tyrosinamide`, `L-Tyrosinamide` | 3, 2 | Majority lower T |
| `Lysozyme` | `Lysozyme`, `lysozyme` | 22, 2 | Majority cap |
| `nucleolin` | `nucleolin`, `Nucleolin` | 13, 4 | Majority lower |
| `streptavidin` | `streptavidin`, `Streptavidin` | 81, 40 | Majority lower |
| `Thrombin` | `Thrombin`, `THROMBIN`, `thrombin` | 34, 5, 1 | Majority `Thrombin` |

**Total rows renamed:** 61  
**Unique target_name before:** 668 → **after:** 659  
**Full rename log:** `data/processed/target_name_normalization_log.csv`

---

## 2. Split-Contamination Impact (Critical Finding)

Because `assign_splits.py` grouped rows by `target_name` string for the Union-Find, collision variants were treated as **separate protein families** and could land in different splits. This violates the no-random-split / protein-family-split invariant.

| Protein (canonical) | Situation before fix | Impact |
|---|---|---|
| **streptavidin** | `streptavidin` (81): all train; `Streptavidin` (40): 28 train / 10 test / 2 val | ⚠ HIGH: 81 train streptavidin rows treated as different family from 40 that were split to test/val |
| **nucleolin** | `nucleolin` (13): all val; `Nucleolin` (4): 2 train / 2 val | ⚠ MEDIUM: 2 train rows incorrectly separated from the 13-row val family |
| **Egg white lysozyme** | `Egg white lysozyme` (3): all val; `egg white lysozyme` (3): all train | ⚠ MEDIUM: 3+3 rows split exactly 50/50 across train/val |
| **anterior gradient homolog 2 (AGR2)** | `anterior…` (6): 3 train / 3 test; `Anterior…` (2): 2 train | ⚠ MEDIUM: 2-row family not grouped with 3-row test members |
| **Thrombin** | `Thrombin` (34): 31 train / 3 val; `THROMBIN` (5): all train; `thrombin` (1): train | LOW: THROMBIN/thrombin rows all in train alongside most Thrombin; only 3 Thrombin rows in val |
| **Lysozyme** | `Lysozyme` (22): 21 train / 1 val; `lysozyme` (2): all train | LOW: 2 lysozyme rows should be in same split as 21-row Lysozyme train cluster (they are, both in train) |
| **(S)-ibuprofen** | Both variants all in train | NONE: no split separation |
| **L-tyrosinamide** | Both variants all in train | NONE: no split separation |

**Split re-assignment is required for at least streptavidin, nucleolin, Egg white lysozyme, and AGR2.**  
This is deferred to the next task per task scope.

---

## 3. Corrected Target Coverage Audit (Post-Normalization)

Queried live against post-normalization data (case-insensitive substring match).

| Target | Total | Train | Train Kd | Val | Val Kd | Test | Test Kd | Flags |
|---|---|---|---|---|---|---|---|---|
| insulin | 4 | 1 | 0 | 3 | 1 | 0 | 0 | ⚠ zero test |
| myoglobin | 5 | 5 | 4 | 0 | 0 | 0 | 0 | ⚠ zero val+test |
| NT-proBNP | 1 | 1 | 1 | 0 | 0 | 0 | 0 | ⚠ zero val+test (1 row total) |
| troponin I/T | 10 | 10 | 5 | 0 | 0 | 0 | 0 | ⚠ zero val+test |
| albumin | 52 | 0 | 0 | 2 | 2 | 50 | 0 | ⚠ zero train |

**albumin note:** 50 rows are `albumin` (lowercase), 1 is `Advanced glycation end products (AGE), Human serum albumin`, 1 is `Glycated Human Serum Albumin (GHSA)`. The 50-row `albumin` group is entirely in test. The previous audit (`data/finalize-splits` branch) used `case=False` substring search, so it correctly captured all 52 rows despite the capital-A discrepancy in the audit prose. Numbers are unchanged from the prior audit.

---

## 4. Leakage Detection Re-Run Results

**Script:** `scripts/data/detect_leakage.py`  
**Result:** 867 near-dupe pairs, 0 exact-label conflicts, **0 cross-split pairs**

The sequence-based Levenshtein comparison finds all pairs regardless of `target_name` casing, so no pairs were missed previously. However, the normalization revealed:

**2 same-protein near-dupe pairs previously mislabeled as cross-target:**

| Pair indices | Old target_a | Old target_b | New (canonical) | Split | Dist |
|---|---|---|---|---|---|
| (row 4495, row 4493) | `L-Tyrosinamide` | `L-tyrosinamide` | `L-tyrosinamide` | train–train | 1 |
| (row 2554, row 2289) | `Nucleolin` | `nucleolin` | `nucleolin` | val–val | 2 |

Both pairs are within the same split, so there is no active leakage. They were merely counted as cross-target near-dupes in the previous output file — now correctly counted as same-protein.

**No new leakage pairs were found.** The prior 867-pair count is unchanged.

---

## 5. Next Steps (Deferred)

1. Re-run `scripts/data/assign_splits.py` with the normalized target names so streptavidin (121 rows), nucleolin (17), Egg white lysozyme (6), and AGR2 (8) are properly co-located in a single split.
2. Re-run coverage audit after re-split to confirm correct distribution.
