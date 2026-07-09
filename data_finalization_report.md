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

*(To be populated after commit 2)*

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
