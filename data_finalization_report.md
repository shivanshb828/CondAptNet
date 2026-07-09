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

*(To be populated after commit 3)*

---

## 4. Split Assignment Summary

*(To be populated after commit 4)*

---

## 5. Target Coverage Audit

*(To be populated after commit 5)*

---

## 6. Final Checks

*(To be populated after commit 6)*
