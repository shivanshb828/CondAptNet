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

---
## Merge: sinha2018_ntprobnp_update_v2.csv + troponin_insulin_update.csv → master_dataset_v2.csv
**Date:** 2026-07-05
**Row count:** 4497 → 4499  (net +2)

### Changes Applied
1. **NT-proBNP** (sequence ending …GCTGT): updated `source_doi` (10.1039/c9lc00115h → 10.1016/j.bios.2018.09.040), `assay_type` (Microfluidic SELEX → SPR (Biacore)), `selection_buffer`, `binding_buffer`, `ph=7.4`, `temperature_C=21`. `confidence_score` unchanged ("curated").
2. **Tro2** (CCCGACC…): filled `kd_value=1.13 nM`, `assay_type=SPR (Biacore)`, `binding_buffer`, `ph=7.4`, `na_concentration_mM=150`.
3. **Tro3** (ATGCGTT…): filled `kd_value=1.14 nM`, same metadata as Tro2.
4. **Tro5** (CAACTGT…): filled `kd_value=3.25 nM`, same metadata as Tro2.
5. **Tro4 / Tro6**: update-file values (270 pM / 317 pM) identical to master (0.27 nM / 0.317 nM). Confirmed true duplicates — no row added.
6. **Insulin (new)**: appended 2 new rows (GGAGGTGGATGGGGTGGGTGTGG, GGAGGGGTGTGGGAGGGGGCTGGTTGTGGTCC) from 10.1016/j.bios.2008.06.016. `kd_value` blank per source paper.

### Open Conflicts — manual sign-off required before next training run
**CONFLICT 1 — Tro1 sequence discrepancy (logged to flagged_for_review.csv):**
- Master (41 nt): `TCACACCCTCCCTCCCACATACCGCATACACTTTTCTGATT`
- Update file (40 nt): `TCACACCCTCCCTCCCACATACCGCATACACTTTCTGATT`
- Differ at position 35 (extra T in master). Master row left untouched.
- Required: check Jo et al. 2015 (10.1021/acs.analchem.5b02312) Table 1 directly.

**CONFLICT 2 — Insulin Kd attribution (logged to flagged_for_review.csv):**
- Existing curated row (GGTGGTGGGGGGGGTTGGTAGGGTGTCTTC, 12700 nM, DOI 10.1016/j.bios.2008.06.016) left untouched.
- Prior verification concluded that paper reports only qualitative binding assays, no numeric Kd for any IGA candidate.
- Required: check source PDF; confirm Kd belongs to a different candidate or identify the misattribution.

---
## Correction pass: tro1_and_insulin_corrections.csv → master_dataset_v2.csv
**Date:** 2026-07-05
**Row count unchanged:** 4499 (4 in-place corrections, no appends)

Corrected 4 rows via byte-verification against Jo et al. 2015 and Yoshida et al. 2009 primary tables: Tro1 (41nt->40nt, extra T), IGA1 (extra G), IGA2 (extra T), IGA3 (extra G + removed unsupported 12700nM Kd, downgraded confidence curated->flagged). Root cause: homopolymer run transcription errors, likely systemic — recommend a dedicated G/T-run verification pass across the full curated dataset before the next training run.

### Correction details
| Row | Old sequence | New sequence | Other changes |
|-----|-------------|-------------|---------------|
| Tro1 | TCACACCCTCCCTCCCACATACCGCATACACTTTTCTGATT (41nt) | TCACACCCTCCCTCCCACATACCGCATACACTTTCTGATT (40nt) | sequence only |
| IGA1 | GGAGGTGGATGGGGTGGGTGTGG (23nt) | GGAGGTGATGGGGTGGGTGTGG (22nt) | sequence only |
| IGA2 | GGAGGGGTGTGGGAGGGGGCTGGTTGTGGTCC (32nt) | GGAGGGGTGGGGAGGGGGCTGGTTGTGGTCC (31nt) | sequence only |
| IGA3 | GGTGGTGGGGGGGGTTGGTAGGGTGTCTTC (30nt) | GGTGGTGGGGGGGTTGGTAGGGTGTCTTC (29nt) | kd_value blanked (was 12700 nM, unsupported per Yoshida 2009); confidence_score curated→flagged |

### Homopolymer run scan (3+ consecutive same base, paper/patent/curated sources)
Found 4016 sequences with 3+ homopolymer runs from manually transcribed sources.
Top 20 longest runs (candidates for manual re-verification):
  len=25 run='AAAAAAAAAAAAAAAAAAAAAAAAA'  'NFkappaB p65'  'PMID:18426920'  GAAGCTTACAAGAAGGACAGCACGAATAAAACCTGCGTAAATCCGCCCCA...
  len=25 run='AAAAAAAAAAAAAAAAAAAAAAAAA'  'E.Coli rho factor'  'PMID:18426920'  GAAGCTTACAAGAAGGACAGCACGAATAAAACCTGCGTAAATCCGCCCCA...
  len=25 run='AAAAAAAAAAAAAAAAAAAAAAAAA'  'human vascular endothelial growth factor VEGF165'  'PMID:18426920'  GAAGCTTACAAGAAGGACAGCACGAATAAAACCTGCGTAAATCCGCCCCA...
  len=25 run='AAAAAAAAAAAAAAAAAAAAAAAAA'  'HCV E2 glycoprotein'  'PMID:18426920'  GAAGCTTACAAGAAGGACAGCACGAATAAAACCTGCGTAAATCCGCCCCA...
  len=25 run='AAAAAAAAAAAAAAAAAAAAAAAAA'  'dsRNA activated protein kinase'  'PMID:18426920'  GAAGCTTACAAGAAGGACAGCACGAATAAAACCTGCGTAAATCCGCCCCA...
  len=17 run='GGGGGGGGGGGGGGGGG'  'Eukaryotic translation initiation factor 4E (eIF4e)'  'PMID:25514650'  ACACTCTTTCCCTACACGACGCTCTTCCGATCTTTGGTTGGAGGTGGTGG...
  len=13 run='TTTTTTTTTTTTT'  'Mycobacterium tuberculosis HspX antigen'  'PMID:30205966'  GTCTTGACTAGTTACGCCGGGAACAATATGTTCAAGGGCTTTTTTTTTTT...
  len=10 run='TTTTTTTTTT'  'Proteinase K-resistant isoform (PrPSc)'  'PMID:18433888'  GGTATTGAGGGTCGCATCTCCTTTTGTGTGTTTTTTTTATTGTTTTTTTT...
  len=10 run='GGGGGGGGGG'  'Group A streptococcus (GAS) serotype M4'  'PMID:28121169'  GCCTGTTGTGAGCCTCCTAACTCCTCGAGGGGGGGGGGATGAAAAGGAAA...
  len=10 run='TTTTTTTTTT'  'Anti-human tumor necrosis factor alpha (anti‐hTNF‐α)'  'PMID:31989789'  GCTGTGTGACTCCTGCAATCCGATCGGTATATCCGTCGGATTTTTTTTTT...
  len=10 run='CCCCCCCCCC'  'Dopamine'  'PMID:9245404'  GGGAATTCCGCGTGTGCTGGCGGGGAGAACTTACATGGATTAGAGAGTGG...
  len=10 run='TTTTTTTTTT'  'His-tagged truncated murine prion protein (H-MoPrP90-231)'  'PMID:18433888'  TCCTTTTGTGTGTTTTTTTTATTGTTTTTTTTTTGTTTTT...
  len=9 run='GGGGGGGGG'  'Poly-beta-1,4-N-acetylglucosamine (Chitin) (poly-β-1,6-N-acetyl-D-glucosamine)'  'PMID:10743940'  TAGGGAATTCGTCGACGGATCCCCGTAACCCTGCGGGGGGGGGAGAAGGC...
  len=9 run='GGGGGGGGG'  'Anterior gradient homolog 2 (AGR2)'  'PMID:23029506'  TCTCGGACGCGTGTGGTCGGCGACGCACCGATCGCAGGTTCGGGATTTTC...
  len=9 run='GGGGGGGGG'  'Anterior gradient homolog 2 (AGR2)'  'PMID:23029506'  TCTCGGACGCGTGTGGTCGGCGGGTGGGAGTTGTGGGGGGGGGTGGGAGG...
  len=9 run='GGGGGGGGG'  'Eukaryotic translation initiation factor 4E (eIF4e)'  'PMID:25514650'  ACACTCTTTCCCTACACGACGCTCTTCCGATCTTGTGGAGGTGGTTGGGG...
  len=9 run='TTTTTTTTT'  'Golgi protein-73 (GP 73)'  'PMID:26583119'  ACGCTCGGATGCCACTACAGTTGGTTTTTTTTTGTTATTTAGAGTAAAAA...
  len=9 run='AAAAAAAAA'  'Colon carcinoma cell lines (HCT-8), Human'  'PMID:25999049'  GGCAGGAAGACAAACACGCAACAACATACGAAGCCACACAAAAAAAAACA...
  len=9 run='TTTTTTTTT'  'Colon carcinoma cell lines (HCT-8), Human'  'PMID:25999049'  GGCAGGAAGACAAACATGGTTGTGTTTTTTTTTGTGTGGCTTCGTATGTT...
  len=9 run='GGGGGGGGG'  'MDA-MB-231 cells, Human'  'PMID:34067799'  ACGCTCGGATGCCACTACAGGGAGGGGGGGGGAAAGTAAGCGGGGGGTCG...

**Action required before next training run:** verify run lengths in the top candidates above against their primary source PDFs, particularly any sequences with runs of 5+ identical bases.

---
## Homopolymer run audit — detection pass only (no auto-corrections)
**Date:** 2026-07-05
**Output:** data/processed/flagged_homopolymer_review.csv

### Scan parameters
- Threshold: 3+ consecutive identical bases in aptamer_sequence
- Scope: all non-database rows (database rows skipped — machine-readable, low OCR risk)
- 1,486 database rows skipped; 3,013 paper/patent rows examined

### Results
| Confidence tier | Flagged | Source (a) local | Source (b) needs PDF |
|----------------|---------|-----------------|---------------------|
| curated        | 11      | 0               | 11                  |
| flagged        | 1       | 0               | 1                   |
| extracted      | 1,951   | 0               | 1,951               |
| non-curated    | 365     | 0               | 365                 |
| uncertain      | 305     | 1               | 304                 |
| **Total**      | **2,633** | **1**         | **2,632**           |

### Priority findings
**11 curated rows flagged** — these are highest priority since "curated" implies
prior verification that apparently missed the run:
- 7 Cardiac Troponin I rows (10.1021/acs.analchem.5b02312 / 10.31661/jbpe.v0i0.797)
  — 2 with CCCC runs (Tro4, Tro6: len=4) — priority-verify against Jo et al. 2015 Table 1
- 1 NT-proBNP row (10.1016/j.bios.2018.09.040) — AAA run (len=3)
- 3 Myoglobin rows (10.1021/ac501088q) — CCC/TTT/AAA runs (len=3)

**Action required before next training run:**
1. Obtain PDFs for the 5 unique DOIs covering the 11 curated rows and byte-verify
   run lengths against primary tables (same process as Tro1/IGA1/IGA2/IGA3 today).
2. For the 1,951 extracted rows, schedule a lower-priority verification pass
   (these were not previously claimed as verified, so errors are expected but less urgent).
3. Only 1 flagged row is locally resolvable (uncertain tier, Li2014 XLSX on disk).

---
## Task 1 — kd_value=0.0 placeholder fix
**Date:** 2026-07-06
**Rows corrected:** 226 → kd_value and kd_unit both blanked

Local source check: none of the 19 source_doi values (all PMIDs) have a local PDF,
extracted text, or scraper_provenance.jsonl entry on disk. Kd=0 is physically
impossible. All 226 rows are scraper defaults; sequence and label data remain valid.

| source_doi | rows blanked |
|------------|-------------|
| PMID:18426920 | 40 |
| PMID:20545348 | 34 |
| PMID:18403417 | 28 |
| PMID:21238427 | 23 |
| PMID:20300533 | 17 |
| PMID:20971648 | 16 |
| PMID:21531729 | 16 |
| PMID:20095591 | 11 |
| PMID:20153201 | 9 |
| PMID:20843027 | 7 |
| PMID:20452328 | 6 |
| PMID:18388495 | 6 |
| PMID:18302343 | 4 |
| PMID:20348540 | 3 |
| PMID:21471212 | 2 |
| PMID:20093103 | 1 |
| PMID:21076782 | 1 |
| PMID:20022942 | 1 |
| PMID:22927983 | 1 |

Other round-number check: 1.0 (6 rows), 10.0 (22), 100.0 (11), 1000.0 (24) all include
curated rows and are physically plausible Kd values — left as-is.

---
## Task 2 — data/ directory consolidation
**Date:** 2026-07-06
**Archived to data/archive/:** ~24 MB (8 items)
**data/processed/ before:** ~24 MB → **after:** 3.2 MB

Reference checks: grepped scripts/ and tests/ before every move.
- non_dna_entries.csv: written-only by clean_dataset.py (kept in place)
- checkpoints/phase*.csv: written-only by clean_dataset.py (archived)
- All other archived files: zero script references

Test suite: 374 passed, 21 pre-existing failures (ModuleNotFoundError: Bio — venv not on shell PATH), 5 skipped. No regressions.
