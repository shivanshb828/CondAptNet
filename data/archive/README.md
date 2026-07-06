# data/archive/

Files moved here are not actively used but have provenance value.
Do not delete without confirming nothing references them.
Archived: 2026-07-06

## Contents

| File | Original location | Reason archived |
|------|------------------|-----------------|
| `master_dataset_v1.csv` | `data/processed/master_dataset_v1.csv` or `data/raw/master_dataset_v1.csv` | Superseded by master_dataset_v2.csv (cleaning pipeline phases 1-7 applied) |
| `master_dataset_v2.csv.bak` | `data/processed/master_dataset_v2.csv.bak` or `data/raw/master_dataset_v2.csv.bak` | Pre-merge snapshot (before sinha2018/troponin/insulin metadata merge) |
| `master_dataset_v2.csv.bak2` | `data/processed/master_dataset_v2.csv.bak2` or `data/raw/master_dataset_v2.csv.bak2` | Pre-correction snapshot (before Tro1/IGA1/IGA2/IGA3 byte-verified sequence fixes) |
| `flagged_homopolymer_review_v1.csv` | `data/processed/flagged_homopolymer_review.csv` or `data/raw/flagged_homopolymer_review.csv` | Superseded by flagged_homopolymer_review_v2.csv (>=3 threshold too noisy at 87% base rate) |
| `scraper_run_20260617_135540.log` | `data/processed/scraper_run_20260617_135540.log` or `data/raw/scraper_run_20260617_135540.log` | Old scraper run log (2026-06-17); scraper has since been re-run |
| `train_run_20260617_144600.log` | `data/processed/train_run_20260617_144600.log` or `data/raw/train_run_20260617_144600.log` | Old training run log (2026-06-17) |
| `enrich_run.log` | `data/processed/enrich_run.log` or `data/raw/enrich_run.log` | Old protein enrichment run log |
| `checkpoints/` | `data/processed/checkpoints/` or `data/raw/checkpoints/` | Intermediate cleaning pipeline snapshots (phase1-6.csv); clean_dataset.py writes these, no script reads them as input |

## What is NOT here (and why)

| File | Status |
|------|--------|
| `data/processed/master_dataset_v2.csv` | Current live dataset |
| `data/processed/flagged_for_review.csv` | Open conflicts — 2 unresolved entries remain |
| `data/processed/flagged_homopolymer_review_v2.csv` | Current homopolymer audit (>=5 threshold) |
| `data/processed/non_dna_entries.csv` | Written by clean_dataset.py; kept in place |
| `data/augmented/tier1_train.csv`, `val.csv`, `test.csv` | Active training splits |
| `data/raw/scraped_dataset.csv` etc. | Original source data, never intermediate output |

## Reference checks performed before archiving

- Grepped `scripts/` and `tests/` for every filename above.
- `non_dna_entries.csv` and `checkpoints/phase*.csv` are referenced only in
  `clean_dataset.py` as **output paths** (written, never read as input by any script).
  They are safe to archive; re-running `clean_dataset.py` regenerates them.
- All log files and `.bak` files had zero script references.
- No test files reference any archived path.
