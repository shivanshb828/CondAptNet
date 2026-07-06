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

---
## Task 1 — PMID:18187506 Kd copy bug (label=0 rows)
**Date:** 2026-07-06

**Pattern:** 33 label=0 rows from PMID:18187506 had kd_value copied from the
label=1 row with the same aptamer_sequence. Every sequence × target pair where
label=0 had the EXACT same Kd as the label=1 row (verified programmatically: 100%
match rate across all 6 sequences with co-occurring binder/non-binder rows).
Non-binder rows should never have a Kd — Kd is undefined for a non-interaction.

**Action:** Blanked kd_value + kd_unit for all 33 label=0 rows.
8 label=1 rows (all targeting Special AT-rich sequence-binding protein / SATB1)
retained their Kd values (10–1000 nM range). These are round-number order-of-magnitude
estimates consistent with ~2008 gel-shift methodology; cannot confirm exact values
without PDF. confidence_score='extracted', so no false-curated claim.

Note: 2 sequences (TATTAATAATAATATTAATAATAA, TATTAGCAATAATATTAGCAATAA) had only
label=0 rows with Kd values and no corresponding binder row — Kd was also wrong
for these; blanked with the same pass.

## Task 2 — Float-precision artifact fix (122 rows)
**Date:** 2026-07-06

**Root cause:** `kd_converter.py convert_kd()` and `kd_extractor.py _to_nM()`
performed unit-conversion multiplications (e.g. pM×1e-3, µM×1e3) without
rounding, producing IEEE 754 imprecision strings like '239.99999999999997' (should
be 240.0) and '14.999999999999998' (should be 15.0).

**Code fix:** Both functions now round to 4 significant figures via
`float(f"{value * factor:.4g}")` before returning. Affects future scraper runs;
does not retroactively affect rows already in master.

**Data fix:** 122 rows corrected to 4 sig-fig rounded values. Original
pre-conversion values are not recoverable from provenance (scraper_provenance.jsonl
does not retain pre-conversion raw values). Rounded in place — these rows should
be treated as "reconstructed precision, not re-derived" if used in precision-sensitive
downstream analysis.

**Curated convention check:** curated Kd values in this dataset use 2–3 sig figs
(0.27, 0.317, 1.13, 2.89, 3.25 nM). 4 sig figs is a safe maximum that won't
over-round any legitimate precision.

**Post-fix scan:** 0 remaining float-precision artifacts (5+ consecutive zeros
or nines pattern). Row count: 4499 (unchanged).

---
## Audit close-out: homopolymer/data-integrity open items
**Date:** 2026-07-06

### Item 1 — Curated-tier recheck at >=3 threshold

Total curated rows: 595. Flagged at >=3 (excluding 6 already-resolved sequences): 566 (95%).
The 95% flag rate at >=3 confirms this threshold is still noisy even for the curated tier;
the task of "manually verifying 562 curated rows needing PDFs" is not feasible without a
prioritization strategy.

**4 rows are checkable against locally available PDFs:**

| Sequence | Target | Run | Source | Status |
|----------|--------|-----|--------|--------|
| CCCGACCACGTCCCTGCCCTTTCCTAACCTGTTTGTTGAT | Cardiac Troponin I (Tro2) | CCC (len=3) | Jo et al. 2015 (10.1021/acs.analchem.5b02312) | Kd filled from correction file (byte-verified vs paper), but sequence itself not individually byte-checked |
| ATGCGTTGAACCCTCCTGACCGTTTATCACATACTCCAGA | Cardiac Troponin I (Tro3) | CCC (len=3) | Jo et al. 2015 | Same |
| CAACTGTAATGTACCCTCCTCGATCACGCACCACTTGCAT | Cardiac Troponin I (Tro5) | CCC (len=3) | Jo et al. 2015 | Same |
| GGCAGGAAGACAAACAGGTCGTAGTGGAAACTGTCCACCGTAGACCGGTTATCTAGTGGTCTGTGGTGCTGT | NT-proBNP | AAA (len=3) | Sinha et al. 2018 (10.1016/j.bios.2018.09.040) | Sequence established in prior metadata correction pass |

**Action required:** Verify Tro2/3/5 sequences character-by-character against Jo et al. 2015
Table 1 (same table where Tro1 had its 41→40nt correction). The CCC runs in these
three sequences are at positions that could plausibly have been miscounted during
transcription. NT-proBNP is lower priority — AAA appears in the middle of ACAAAC,
a low-risk context, but should be confirmed when Sinha 2018 is open anyway.

The remaining 562 curated rows needing PDFs are not individually actionable without
a PDF retrieval campaign. Prioritization: same DOI-frequency-first approach as the >=6
list (top curated DOIs: PMID:15606780, PMID:23387511, PMID:23403083 — 13 rows each).

### Item 2 — >=6 homopolymer list (170 rows, 90 unique sequences)

1 checkable row: IGA3 (GGGGGGG run, Yoshida 2009) — ALREADY RESOLVED this session.
The 7-G run is the correct byte-verified form. No further action needed.

169 needs-PDF rows remain. Prioritized with G4-forming sequences first (see Item 3):

| Rank | rows | uniq | G4 | max_run | Source |
|------|------|------|----|---------|--------|
| 1 | 8 | 8 | 5 | 6 | 020-211-642-872-093 (patent) ← G4 PRIORITY |
| 2 | 8 | 3 | 1 | 8 | PMID:21204783 ← G4 PRIORITY |
| 3 | 5 | 5 | 1 | 8 | 109-252-256-022-54X (patent) ← G4 PRIORITY |
| 4 | 5 | 1 | 1 | 6 | (blank DOI) ← G4 PRIORITY |
| 5 | 4 | 4 | 1 | 7 | KR101583578B1 ← G4 PRIORITY |
| 6 | 3 | 1 | 1 | 6 | PMID:19955232 ← G4 PRIORITY |
| 7 | 2 | 2 | 1 | 6 | 055-805-980-791-147 ← G4 PRIORITY |
| 8 | 1 | 1 | 1 | 8 | 148-901-319-254-80X ← G4 PRIORITY |
| 9 | 12 | 3 | 0 | 9 | PMID:7678562 (highest non-G4 row count) |
| 10 | 12 | 12 | 0 | 8 | KR101200553B1 |
| 11 | 11 | 4 | 0 | 7 | PMID:18637731 |
| 12 | 8 | 2 | 0 | 7 | PMID:16397296 |
| 13–44 | ... | | 0 | | see flagged_homopolymer_review_v2.csv |

### Item 3 — G4 prioritization correction (methodology)

**REVERSAL OF PRIOR RECOMMENDATION.** The earlier audit (>=5 pass) recommended
deprioritizing G4-forming aptamers on the theory that long G-runs are "expected
biology" and therefore less likely to represent transcription errors. This recommendation
was wrong and is retracted.

Evidence from this session: IGA2 and IGA3 are both G4-forming insulin aptamers. Both
had real transcription errors (extra nucleotides in G-runs) that were caught precisely
because they were manually checked. The long G-run did not protect them from errors —
if anything, a long homopolymer run is a harder target for manual transcription,
making errors more likely, not less.

**New rule:** G4-forming sequences (defined as: aptamer_sequence containing 2+ separate
runs of 4+ consecutive G's) are sorted to the TOP of any homopolymer verification queue,
not the bottom. The risk of a miscounted G in a GGGG+ context is higher than average,
not lower.

This correction is logged here so it cannot be silently reintroduced in future audit passes.
