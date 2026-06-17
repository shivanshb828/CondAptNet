# Aptamer Dataset Cleaning Report
Generated: 2026-06-17 15:11:08
Input: data/processed/master_dataset.csv

- Loaded 4643 rows, 19 columns from master_dataset.csv

## Phase 1 — Reverse-Complement & Exact Duplicate Removal
- Input rows: 4643
- Exact duplicates (same seq + target) removed: 77
- Reverse-complement pairs found: 5; rows removed: 5
- Output rows: 4561  (removed 82 total in Phase 1)

## Phase 3 — Split Assignment & Leakage Repair
- Split assignment from augmented files: train=2735, val=501, test=390, unassigned=935

## Phase 2 — Priority Target Coverage Audit
- Coverage by split ('_unassigned' = not in any split file):
- Target              total   train    val   test  unassigned
- ------------------ ------ ------- ------ ------ -----------
- Insulin                 1       0      1      0           0
- Myoglobin               1       1      0      0           0
- **WARNING**: Myoglobin: 1 rows but NONE in val or test — evaluation impossible
- Troponin                3       3      0      0           0
- **WARNING**: Troponin: 3 rows but NONE in val or test — evaluation impossible
- NT-proBNP               0       0      0      0           0
- **WARNING**: NT-proBNP: ABSENT from dataset — will be added via curated file in Phase 6
- Albumin                52       0      0     50           2
- 
Current split sizes: train=2735, val=501, test=390, unassigned=935

## Phase 3b — Leakage Detection and Repair
- Sequence overlap train∩val  : 267
- Sequence overlap train∩test : 209
- Sequence overlap val∩test   : 82
- Rows reassigned to train: 596
- Post-repair leakage — train∩val=0, train∩test=0, val∩test=0
- ✓ Zero sequence overlap across splits
- Final split sizes: train=3331, val=159, test=136, unassigned=935

## Phase 4 — Nucleic Acid Type & DNA Filter
- Nucleic acid type breakdown: {'ssDNA': 4494, 'unknown': 67}
- Non-ssDNA rows removed: 67  (saved to non_dna_entries.csv)
- Out-of-range length rows removed: 10 (too_short=6, too_long=4)
- Output rows after DNA filter: 4484

## Phase 5 — Target Type Classification & ID Re-mapping
- Target type distribution: {'protein': 4370, 'organism': 38, 'cell': 33, 'other': 26, 'small_molecule': 17}
- Non-protein rows with UniProt ID (mapping error to correct): 0
- Querying PubChem for 10 unique small molecule targets ...
- PubChem hits: 2 / 10  (misses → flagged_for_review.csv)

## Phase 6 — Column Restructuring & Curated Data Merge
- Column restructuring complete: 23 columns
- 20-column schema satisfied; training extras appended: protein_sequence, label, training_tier
- Curated file loaded: 13 rows, targets: ['Cardiac Troponin I', 'NT-proBNP', 'Insulin', 'Myoglobin']
- Curated rows: 13 total, 0 already in master (skipped), 13 new rows added
- Output rows: 4497  (+13 from curated merge)
- Backfilled protein_sequence for 11 curated rows from UniProt ID lookup

## Phase 7 — Final Validation & Export
- Final split sizes (assigned rows):
-   train   :  3344 rows (91.9%)
-   val     :   159 rows (4.4%)
-   test    :   136 rows (3.7%)
-   unassigned: 858
- 
Priority target final counts (full dataset):
-   Insulin       : 2
-   Myoglobin     : 5
-   Troponin      : 10
-   NT-proBNP     : 1
-   Albumin       : 52
-   Leakage check train∩val: 0 (clean)
-   Leakage check train∩test: 0 (clean)
-   Leakage check val∩test: 0 (clean)
- **WARNING**: 2 validation issues found:
- **WARNING**:   ↳ Split 'val' is < 5% of assigned data (4.4%)
- **WARNING**:   ↳ Split 'test' is < 5% of assigned data (3.7%)
- 
Exported: /Users/shivanshbansal/GitHub/continuitybioML/data/processed/master_dataset_cleaned.csv  (4497 rows, 23 columns)
- Exported: /Users/shivanshbansal/GitHub/continuitybioML/data/processed/non_dna_entries.csv  (77 rows)
- Exported: /Users/shivanshbansal/GitHub/continuitybioML/data/processed/flagged_for_review.csv  (609 rows)
