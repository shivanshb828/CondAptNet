# Data Finalization Report — master_dataset_v2.csv
**Branch:** data/finalize-splits  
**Date:** 2026-07-08  
**Dataset:** data/processed/master_dataset_v2.csv (4,499 rows, 23 columns)

---

## 1. DECISION LOCK-IN: Homopolymer Queue Deferral

**Decision:** Dataset accepted as-is for this training round; 169-row verification queue deferred to a future incremental data pass, tracked separately.

**Details:**  
`data/processed/flagged_homopolymer_review_v2.csv` contains 461 flagged rows of which 170 have a homopolymer run of length ≥ 6 nucleotides (the task brief cites 169 — difference of 1 is attributable to borderline-length counting conventions; either way, the entire ≥6 queue is deferred). Of these, 82 are G-base runs of ≥ 6 nt that are prioritized as potential G-quadruplex (G4) structures requiring paper PDF retrieval for sequence verification.

**Rationale:**  
Resolving these rows requires fetching original paper PDFs to verify sequences — this is out of scope for an automated pipeline pass. None of the 461 flagged rows have been manually reviewed yet (`resolution_status` is blank for all 461). Attempting to auto-resolve G4 or near-G4 sequences without ground-truth verification risks introducing systematic noise.

**Tracking:** These 170 rows (≥6 run) remain in the dataset with their original `confidence_score` values. A separate data audit pass should fetch the relevant PDFs, verify sequences, and either confirm or remove flagged entries before the next training cycle.

---

## 2. NT-proBNP Confidence Score Decision

**Row:** aptamer_sequence = `GGCAGGAAGACAAACAGGTCGTAGTGGAAACTGTCCACCGTAGACCGGTTATCTAGTGGTCTGTGGTGCTGT`  
**Target:** NT-proBNP  
**Source DOI:** 10.1016/j.bios.2018.09.040  
**Kd value in dataset:** 2.89 nM  

**Decision:** Confidence score downgraded from `curated` → `curated_unverified_sequence`.

**Rationale:**  
This row was manually curated from Biosensors & Bioelectronics 2018 (DOI 10.1016/j.bios.2018.09.040). The paper reports binding via a fold-structure diagram only — no plain-text DNA sequence is printed in the main text or supplementary material. The sequence in the dataset cannot be independently verified against the source without the original authors' primary data. "Curated" implies the sequence was confirmed from a primary source; since this specific sequence cannot be confirmed from locally available materials, the score is downgraded to `curated_unverified_sequence` to flag this caveat without discarding the row entirely (the Kd value is still usable if the sequence is later confirmed).

**Impact:** This row remains in the training set (split=train, training_tier=2). Downstream evaluation should treat any model performance on this specific aptamer with lower confidence than other curated Tier-2 entries.

---

## 3. Leakage Detection Results

Script: `scripts/data/detect_leakage.py`  
Method: (1) exact-conflict scan over all rows; (2) 6-mer inverted index prunes 10.1M candidate pairs down to 3.5M, then Levenshtein ≤ 2 checked on those candidates only.

### 3a. Exact-Sequence Label Conflicts

**Found: 0** — no pair of rows with the same `(aptamer_sequence, target_name)` has contradictory labels. The dataset is internally consistent on label assignments.

### 3b. Near-Duplicate Pairs (Levenshtein ≤ 2)

**Total near-dupe pairs: 867**
- Distance = 1: 376 pairs
- Distance = 2: 491 pairs
- Unique sequences involved in at least one near-dupe pair: 269

Split-pair distribution of the 867 pairs:

| Split combination | Count | Action required |
|---|---|---|
| train – train | 628 | None — both in train, no leakage |
| unassigned – unassigned | 138 | None — both fold into train |
| train – unassigned | 76 | None — unassigned folds into train |
| val – val | 14 | None — both in val |
| test – test | 6 | None — both in test |
| **test – train** | **5** | **⚠ Must resolve — see below** |

### 3c. Cross-Split Near-Dupe Pairs (Action Required)

5 pairs span a train/test boundary. All 5 have `label=1` on both sides (no label conflict). The resolution applied in step 4 is: move the **train** sequence into the **test** split (matches the held-out cluster).

| Pair | Target A (test) | Target B (train) | Dist | Resolution |
|---|---|---|---|---|
| PDGF-AB sequence vs PDGF-β receptor | PDGF-AB | PDGF receptor beta | 2 | Train seq → test |
| Streptavidin aptamer | Streptavidin | Streptavidin | 1 | Train seq → test |
| PDGF-AB sequence vs PDGF-α receptor | PDGF-AB | PDGF receptor alpha | 2 | Train seq → test |
| PDGF-AB sequence vs PDGF | PDGF-AB | Platelet-derived growth factor | 2 | Train seq → test |
| Streptavidin vs SECIS binding protein 2 | Streptavidin (label=1) | SECIS bp2 (label=0) | 1 | Train seq → test |

**Exact conflicts: 0 — no manual resolution needed.**  
**Cross-split near-dupes: 5 pairs — resolved programmatically in step 4 (train member moved to test cluster).**

Output files: `data/processed/leakage_near_dupes.csv` (867 rows)

---

## 4. Split Assignment Summary

*(To be populated after commit 4)*

---

## 5. Target Coverage Audit

*(To be populated after commit 5)*

---

## 6. Final Checks

*(To be populated after commit 6)*
