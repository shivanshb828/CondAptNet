"""
Merge scraped_dataset.csv → master_dataset.csv.

Translates the scraper's 20-column scientific schema into the training schema
used by master_dataset.csv, then appends unique rows (dedup on sequence +
target_protein, case-insensitive).

Filters applied before merge:
  - nucleic_acid_type == 'ssDNA'   (native DNA only — no T→U)
  - target_type == 'protein'       (skip small molecules, cells, organisms)
  - sequence 20–120 nt, A/T/G/C only

Schema mapping (scraper → master):
  aptamer_sequence     → sequence
  target_name          → target_protein
  kd_value             → Kd_nM       (already in nM per scraper schema)
  ph                   → pH          (default 7.4)
  na_concentration_mM  → salt_mM     (default 150.0)
  mg_concentration_mM  → mg_mM       (default 2.0)
  temperature_C        → temp_C      (default 37.0)
  binding_buffer|selection_buffer → buffer_type (PBS=0, HEPES=1, Tris=2, other=3)
  target_id (UniProt)  → uniprot_id
  source_doi           → source_pmid
  source_type          → source      (paper→pubmed, patent→patent, ...)

All merged rows: label=1, training_tier=1, augmented=False,
                 needs_sequence_enrichment=False, protein_sequence=None

Usage:
    python scripts/data/merge_scraped_to_master.py
    python scripts/data/merge_scraped_to_master.py --dry-run
    python scripts/data/merge_scraped_to_master.py --scraped path/to/scraped.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_PROCESSED, DATA_RAW,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
    SEQ_MIN_LEN, SEQ_MAX_LEN, VALID_BASES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MASTER_PATH  = Path(DATA_PROCESSED) / "master_dataset.csv"
SCRAPED_PATH = Path(DATA_RAW)       / "scraped_dataset.csv"

MASTER_COLS = [
    "sequence", "target_protein", "uniprot_id", "protein_sequence",
    "Kd_nM", "pH", "salt_mM", "temp_C", "buffer_type", "mg_mM",
    "label", "source_pmid", "training_tier", "augmented", "aug_method",
    "needs_sequence_enrichment", "source",
]

_BUFFER_MAP = {
    "pbs":   0, "phosphate":  0,
    "hepes": 1,
    "tris":  2,
}


def _map_buffer(raw: str | None) -> int:
    if not raw or pd.isna(raw):
        return DEFAULT_BUFFER
    s = str(raw).lower()
    for key, val in _BUFFER_MAP.items():
        if key in s:
            return val
    return 3  # other


def _map_source(source_type: str | None) -> str:
    mapping = {
        "paper":    "pubmed",
        "patent":   "patent",
        "database": "database",
        "preprint": "biorxiv",
    }
    return mapping.get(str(source_type).lower().strip(), "scraped")


def _is_valid_dna(seq: str) -> bool:
    if not isinstance(seq, str) or not seq.strip():
        return False
    s = seq.strip().upper()
    if not (SEQ_MIN_LEN <= len(s) <= SEQ_MAX_LEN):
        return False
    return set(s) <= VALID_BASES


def translate_row(row: pd.Series) -> dict:
    """Convert one scraped row into a master_dataset row dict."""
    seq = str(row.get("aptamer_sequence", "") or "").strip().upper()

    # Buffer: prefer binding_buffer, fall back to selection_buffer
    buf_raw = row.get("binding_buffer") or row.get("selection_buffer")
    buffer_type = _map_buffer(buf_raw)

    # UniProt ID: only use if source confirms it
    uniprot_id = None
    if str(row.get("target_id_source", "")).strip() == "UniProt":
        raw_id = str(row.get("target_id", "") or "").strip()
        if raw_id:
            uniprot_id = raw_id

    # Kd: already in nM per scraper schema contract
    kd_raw = row.get("kd_value")
    try:
        kd = float(kd_raw) if kd_raw not in (None, "", float("nan")) else None
    except (ValueError, TypeError):
        kd = None
    if kd is not None and kd < 0:
        kd = None

    # Condition fields with physiological defaults
    def _fval(col: str, default: float) -> float:
        v = row.get(col)
        try:
            return float(v) if v not in (None, "", float("nan")) else default
        except (ValueError, TypeError):
            return default

    return {
        "sequence":                  seq,
        "target_protein":            str(row.get("target_name", "") or "").strip(),
        "uniprot_id":                uniprot_id,
        "protein_sequence":          None,
        "Kd_nM":                     kd,
        "pH":                        _fval("ph",                  DEFAULT_PH),
        "salt_mM":                   _fval("na_concentration_mM", DEFAULT_SALT_MM),
        "temp_C":                    _fval("temperature_C",       DEFAULT_TEMP_C),
        "buffer_type":               buffer_type,
        "mg_mM":                     _fval("mg_concentration_mM", DEFAULT_MG_MM),
        "label":                     1,       # SELEX-selected → binder
        "source_pmid":               str(row.get("source_doi", "") or "").strip() or None,
        "training_tier":             1,
        "augmented":                 False,
        "aug_method":                None,
        "needs_sequence_enrichment": False,
        "source":                    _map_source(row.get("source_type")),
    }


def merge(
    scraped_path: Path = SCRAPED_PATH,
    master_path:  Path = MASTER_PATH,
    dry_run: bool = False,
) -> dict:
    if not scraped_path.exists():
        log.error("scraped_dataset.csv not found at %s — run the scraper first.", scraped_path)
        return {}

    scraped = pd.read_csv(scraped_path, dtype=str)
    master  = pd.read_csv(master_path)

    log.info("Scraped rows:  %d", len(scraped))
    log.info("Master rows:   %d", len(master))

    # ── Filter to ssDNA protein targets ──────────────────────────────────────
    dna_mask     = scraped["nucleic_acid_type"].str.strip() == "ssDNA"
    protein_mask = scraped["target_type"].str.strip() == "protein"
    filtered     = scraped[dna_mask & protein_mask].copy()
    log.info("After ssDNA+protein filter: %d rows", len(filtered))

    # ── Translate to master schema ────────────────────────────────────────────
    translated_rows = [translate_row(r) for _, r in filtered.iterrows()]

    # ── Validate sequences ────────────────────────────────────────────────────
    valid_rows = [r for r in translated_rows if _is_valid_dna(r["sequence"]) and r["target_protein"]]
    log.info("After sequence QC: %d rows", len(valid_rows))

    if not valid_rows:
        log.warning("No valid rows to merge.")
        return {"new_rows": 0}

    new_df = pd.DataFrame(valid_rows, columns=MASTER_COLS)

    # ── Deduplicate against master ─────────────────────────────────────────────
    master_keys = set(
        zip(
            master["sequence"].str.upper().str.strip(),
            master["target_protein"].str.lower().str.strip(),
        )
    )
    new_keys = list(
        zip(
            new_df["sequence"].str.upper().str.strip(),
            new_df["target_protein"].str.lower().str.strip(),
        )
    )
    unique_mask = [k not in master_keys for k in new_keys]
    unique_df   = new_df[unique_mask].drop_duplicates(
        subset=["sequence", "target_protein"]
    )
    log.info("After dedup vs master: %d new unique rows", len(unique_df))

    if dry_run:
        log.info("[DRY RUN] Would append %d rows to master_dataset.csv", len(unique_df))
        if len(unique_df) > 0:
            log.info("Sample new targets: %s", unique_df["target_protein"].value_counts().head(10).to_dict())
        return {"new_rows": len(unique_df), "dry_run": True}

    # ── Append to master ───────────────────────────────────────────────────────
    updated = pd.concat([master, unique_df], ignore_index=True)
    updated.to_csv(master_path, index=False)

    stats = {
        "new_rows":           len(unique_df),
        "master_before":      len(master),
        "master_after":       len(updated),
        "sources":            unique_df["source"].value_counts().to_dict(),
        "new_proteins":       unique_df["target_protein"].nunique(),
    }
    log.info("=" * 50)
    log.info("Merged %d new rows into master_dataset.csv", stats["new_rows"])
    log.info("Master: %d → %d rows", stats["master_before"], stats["master_after"])
    log.info("New proteins: %d unique targets", stats["new_proteins"])
    log.info("By source: %s", stats["sources"])
    log.info("=" * 50)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge scraped_dataset.csv into master_dataset.csv")
    parser.add_argument("--scraped",  default=str(SCRAPED_PATH), help="Path to scraped_dataset.csv")
    parser.add_argument("--master",   default=str(MASTER_PATH),  help="Path to master_dataset.csv")
    parser.add_argument("--dry-run",  action="store_true",        help="Preview only — do not write")
    args = parser.parse_args()

    stats = merge(
        scraped_path=Path(args.scraped),
        master_path=Path(args.master),
        dry_run=args.dry_run,
    )
    if not stats:
        sys.exit(1)


if __name__ == "__main__":
    main()
