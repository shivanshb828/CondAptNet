"""
Aptamer database adapter.

Handles structured aptamer databases — sources with curated, high-confidence
data. Returns records with confidence_score="curated".

Sources:
  1. UTexas Aptamer DB (local CSV — already at data/raw/utexas_aptamer_db/)
     Zenodo: doi.org/10.5281/zenodo.8264921 — 1,495 entries, ssDNA filter → ~896 rows
     Already partially processed in master_dataset.csv (training_tier=1).
     This adapter re-reads the raw file to catch any rows filtered differently
     in the main build_dataset.py pass.

  2. Aptamer Database Zenodo dump (if present at data/raw/aptamerbase/)
     Pre-2016 AptamerBase dump from github.com/micheldumontier/aptamerbase.
     Parsed as CSV/JSON depending on which format was downloaded.

  3. Apta-MCTS benchmark data (data/raw/li2014_benchmark/)
     Li et al. 2014 ssDNA entries with known sequences.

All local — no HTTP calls needed. Rate limiter is still called (no-op at 2 req/s)
for consistency.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper import config as cfg
from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.schema import (
    make_empty_record, validate_record, BLANK, SCHEMA_COLUMNS,
)
from scripts.data.scraper.extractors.target_resolver import classify_target_type
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

# Column name aliases across different database CSV formats
_SEQ_ALIASES   = {"sequence", "aptamer_sequence", "Sequence", "DNA_sequence", "seq"}
_TARGET_ALIASES = {"target", "target_name", "Target", "target_protein", "protein"}
_KD_ALIASES     = {"kd", "kd_nm", "Kd_nM", "kd_value", "Kd", "affinity_nm"}

# UTexas DB column map (from build_dataset.py precedent)
_UTEXAS_COL_MAP = {
    "Sequence":          "aptamer_sequence",
    "Target":            "target_name",
    "Kd (nM)":           "kd_value",
    "Type":              "nucleic_acid_type",
    "pH":                "ph",
    "Temperature (°C)":  "temperature_C",
    "NaCl (mM)":         "na_concentration_mM",
    "MgCl2 (mM)":        "mg_concentration_mM",
    "Buffer":            "selection_buffer",
}


class DatabasesAdapter(BaseAdapter):
    """
    Parse local aptamer database files and return curated records.
    """

    source_name = "databases"
    source_type = "database"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)

    # ── UTexas Aptamer DB ──────────────────────────────────────────────────────

    def _load_utexas(self) -> list[dict]:
        db_dir = cfg.DATA_RAW / "utexas_aptamer_db"
        csvs   = list(db_dir.glob("*.csv")) if db_dir.exists() else []
        if not csvs:
            log.debug("UTexas Aptamer DB not found at %s", db_dir)
            return []

        all_records: list[dict] = []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
            except Exception as exc:
                log.warning("Failed to load UTexas CSV %s: %s", csv_path.name, exc)
                continue

            # Rename columns using the map if headers match
            df = df.rename(columns={k: v for k, v in _UTEXAS_COL_MAP.items() if k in df.columns})

            for _, row in df.iterrows():
                seq = _find_col(row, _SEQ_ALIASES)
                tgt = _find_col(row, _TARGET_ALIASES)
                if not seq or not tgt:
                    continue

                # Only ssDNA rows (filter T→U conversion artefacts)
                na_type = str(row.get("nucleic_acid_type", "ssDNA")).strip()
                if "rna" in na_type.lower():
                    continue
                if not re.match(r"^[ATGCatgc]+$", seq.replace(" ", "")):
                    continue
                seq = seq.upper().replace(" ", "")
                if not (20 <= len(seq) <= 120):
                    continue

                rec = make_empty_record()
                rec.update({
                    "aptamer_sequence":    seq,
                    "nucleic_acid_type":   "ssDNA",
                    "modifications":       "none",
                    "target_name":         tgt.strip(),
                    "target_type":         classify_target_type(tgt),
                    "kd_value":            _safe_float(row.get("kd_value", "")),
                    "kd_unit":             "nM" if _safe_float(row.get("kd_value", "")) else BLANK,
                    "selection_buffer":    str(row.get("selection_buffer", "")).strip() or BLANK,
                    "ph":                  _safe_float(row.get("ph", "")),
                    "na_concentration_mM": _safe_float(row.get("na_concentration_mM", "")),
                    "mg_concentration_mM": _safe_float(row.get("mg_concentration_mM", "")),
                    "temperature_C":       _safe_float(row.get("temperature_C", "")),
                    "source_doi":          "10.5281/zenodo.8264921",
                    "source_type":         "database",
                    "confidence_score":    "curated",
                    "split":               "train",
                })
                ok, _ = validate_record(rec)
                if ok:
                    all_records.append(rec)

        log.info("UTexas DB: %d curated records", len(all_records))
        return all_records

    # ── AptamerBase dump ───────────────────────────────────────────────────────

    def _load_aptamerbase(self) -> list[dict]:
        ab_dir = cfg.DATA_RAW / "aptamerbase"
        csvs   = list(ab_dir.glob("*.csv")) if ab_dir.exists() else []
        if not csvs:
            log.debug("AptamerBase dump not found at %s", ab_dir)
            return []

        all_records: list[dict] = []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
            except Exception as exc:
                log.warning("AptamerBase CSV %s failed: %s", csv_path.name, exc)
                continue

            for _, row in df.iterrows():
                seq = _find_col(row, _SEQ_ALIASES)
                tgt = _find_col(row, _TARGET_ALIASES)
                if not seq or not tgt:
                    continue
                seq = seq.upper().replace(" ", "").replace("-", "")
                if not re.match(r"^[ATGC]+$", seq) or not (20 <= len(seq) <= 120):
                    continue

                rec = make_empty_record()
                rec.update({
                    "aptamer_sequence":  seq,
                    "nucleic_acid_type": "ssDNA",
                    "modifications":     "none",
                    "target_name":       tgt.strip(),
                    "target_type":       classify_target_type(tgt),
                    "kd_value":          _safe_float(_find_col(row, _KD_ALIASES)),
                    "kd_unit":           "nM" if _safe_float(_find_col(row, _KD_ALIASES)) else BLANK,
                    "source_doi":        "github.com/micheldumontier/aptamerbase",
                    "source_type":       "database",
                    "confidence_score":  "non-curated",
                    "split":             "train",
                })
                ok, _ = validate_record(rec)
                if ok:
                    all_records.append(rec)

        log.info("AptamerBase: %d records", len(all_records))
        return all_records

    # ── Li2014 benchmark ───────────────────────────────────────────────────────

    def _load_li2014(self) -> list[dict]:
        li_dir = cfg.DATA_RAW / "li2014_benchmark"
        csvs   = list(li_dir.glob("*.csv")) if li_dir.exists() else []
        if not csvs:
            return []

        all_records: list[dict] = []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, dtype=str).fillna("")
            except Exception:
                continue

            for _, row in df.iterrows():
                seq = _find_col(row, _SEQ_ALIASES)
                tgt = _find_col(row, _TARGET_ALIASES)
                if not seq or not tgt:
                    continue
                seq = seq.upper().replace(" ", "")
                if not re.match(r"^[ATGC]+$", seq) or not (20 <= len(seq) <= 120):
                    continue

                rec = make_empty_record()
                rec.update({
                    "aptamer_sequence":  seq,
                    "nucleic_acid_type": "ssDNA",
                    "modifications":     "none",
                    "target_name":       tgt.strip(),
                    "target_type":       classify_target_type(tgt),
                    "kd_value":          _safe_float(_find_col(row, _KD_ALIASES)),
                    "kd_unit":           "nM" if _safe_float(_find_col(row, _KD_ALIASES)) else BLANK,
                    "source_doi":        "10.1371/journal.pone.0086729",
                    "source_type":       "database",
                    "confidence_score":  "curated",
                    "split":             "train",
                })
                ok, _ = validate_record(rec)
                if ok:
                    all_records.append(rec)

        log.info("Li2014: %d records", len(all_records))
        return all_records

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, max_results: int = 5000) -> list[dict]:
        self._limiter.wait()   # consistency
        records = (
            self._load_utexas()
            + self._load_aptamerbase()
            + self._load_li2014()
        )
        log.info("DatabasesAdapter: %d total records", len(records))
        return records[:max_results]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_col(row, aliases: set) -> str:
    """Return the first non-empty value from a row dict matching any alias."""
    for col in aliases:
        val = row.get(col, "")
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _safe_float(val) -> object:
    """Return float if parseable and > 0, else BLANK sentinel."""
    if val is None or str(val).strip() in ("", "N/A", "NA", "nan"):
        return BLANK
    try:
        f = float(str(val).strip())
        return f if f > 0 else BLANK
    except (ValueError, TypeError):
        return BLANK
