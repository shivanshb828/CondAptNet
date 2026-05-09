"""
Unified dataset builder for CondAptNet.

Parses all raw data sources and outputs master_dataset.csv with the canonical
schema. Sources processed:

  1. UTexas Aptamer Database (Zenodo: doi.org/10.5281/zenodo.8264921)
     - 1,495 entries; filter to ssDNA only (~896 rows)
     - Has: sequence, target, Kd, pH, buffer_type
     - Missing: salt_mM, temp_C, mg_mM, uniprot_id, protein_sequence
       → imputed with physiological defaults / flagged for enrichment

  2. Li et al. 2014 (PLOS ONE doi:10.1371/journal.pone.0086729) File S1
     - 2,320 entries (580 positive + 1,740 negative), 164 proteins
     - Has: label (positive/negative), target_protein, source_pmid
     - Missing: sequence (buried in source papers — needs PubMed lookup)
       → sequence=None, needs_sequence_enrichment=True

Both sources get training_tier=1 (general pretraining).
Sequences marked None are excluded from training until enriched.

Output: data/processed/master_dataset.csv

Usage:
    python scripts/data/build_dataset.py
    python scripts/data/build_dataset.py --utexas-only
    python scripts/data/build_dataset.py --li2014-only
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_RAW, DATA_PROCESSED,
    BUFFER_TYPES,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
    VALIDATION_TARGETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────
SCHEMA_COLS = [
    "sequence", "target_protein", "uniprot_id", "protein_sequence",
    "Kd_nM", "pH", "salt_mM", "temp_C", "buffer_type", "mg_mM",
    "label", "source_pmid", "training_tier",
    "augmented", "aug_method",
    "needs_sequence_enrichment", "source",
]

# ── Buffer type mapping ───────────────────────────────────────────────────────
_BUFFER_MAP = {
    "pbs":             0,
    "phosphate":       0,
    "hepes":           1,
    "tris":            2,
    "other":           3,
    "not reported":    3,
    "not available":   3,
}

def _map_buffer(raw: str | float) -> int:
    if pd.isna(raw):
        return DEFAULT_BUFFER
    s = str(raw).lower()
    for key, code in _BUFFER_MAP.items():
        if key in s:
            return code
    return 3  # other


def _extract_pmid_from_url(url: str | float) -> str | None:
    if pd.isna(url):
        return None
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", str(url))
    return m.group(1) if m else None


def _clean_sequence(seq: str | float) -> str | None:
    if pd.isna(seq):
        return None
    s = str(seq).strip().upper()
    # Strip primer flanks: keep only the core random region if delimited by spaces
    # (some entries have fixed flanks separated by spaces or special chars)
    s = re.sub(r"[^ATGCU]", "", s)   # remove non-nucleotide chars
    s = s.replace("U", "T")          # normalise any stray U→T (UTexas stores DNA)
    return s if s else None


def _assign_tier(target: str) -> int:
    """Tier 2 for validation targets, Tier 1 for everything else."""
    if pd.isna(target):
        return 1
    t = str(target).lower()
    for vt in VALIDATION_TARGETS:
        if vt.lower() in t:
            return 2
    return 1


# ── Parser: UTexas Aptamer Database ──────────────────────────────────────────

def parse_utexas(xlsx_path: str) -> pd.DataFrame:
    log.info("Parsing UTexas Aptamer Database: %s", xlsx_path)
    raw = pd.read_excel(xlsx_path)
    log.info("  Raw shape: %s", raw.shape)

    # Filter to ssDNA only — our model requires native DNA
    dna = raw[raw["Type of Nucleic Acid"] == "ssDNA"].copy()
    log.info("  After ssDNA filter: %d rows (dropped %d RNA/modified)",
             len(dna), len(raw) - len(dna))

    records = []
    for _, row in dna.iterrows():
        seq = _clean_sequence(row.get("Aptamer Sequence"))

        # Kd: already numeric in this file
        kd_raw = row.get("Kd (nM)")
        kd = float(kd_raw) if pd.notna(kd_raw) else None

        # pH
        ph_raw = row.get("pH")
        ph = float(ph_raw) if pd.notna(ph_raw) else DEFAULT_PH

        # salt_mM: not a clean column — use physiological default
        salt = DEFAULT_SALT_MM

        # temp_C: not in dataset — use physiological default
        temp = DEFAULT_TEMP_C

        # mg_mM: divalent salt column exists but is free text; use physiological default
        mg = DEFAULT_MG_MM

        buffer_type = _map_buffer(row.get("Type of the buffer"))
        target = str(row.get("Target ", "")).strip()
        pmid = _extract_pmid_from_url(row.get("Link to PubMed Entry"))

        records.append({
            "sequence":                   seq,
            "target_protein":             target,
            "uniprot_id":                 None,    # enrichment pass needed
            "protein_sequence":           None,    # enrichment pass needed
            "Kd_nM":                      kd,
            "pH":                         ph,
            "salt_mM":                    salt,
            "temp_C":                     temp,
            "buffer_type":                buffer_type,
            "mg_mM":                      mg,
            "label":                      1,       # all UTexas entries are confirmed binders
            "source_pmid":                pmid,
            "training_tier":              _assign_tier(target),
            "augmented":                  False,
            "aug_method":                 None,
            "needs_sequence_enrichment":  False,
            "source":                     "utexas",
        })

    df = pd.DataFrame(records)
    log.info("  UTexas parsed: %d rows (%d have Kd, %d have pH)",
             len(df),
             df["Kd_nM"].notna().sum(),
             (df["pH"] != DEFAULT_PH).sum())
    return df


# ── Parser: Li et al. 2014 ────────────────────────────────────────────────────

def parse_li2014(s1_path: str) -> pd.DataFrame:
    """
    Li et al. 2014 File S1 has labels + aptamer/target IDs but NO sequences.
    Sequences must be fetched from PubMed later. Rows are flagged accordingly.

    aptamer_id format: '{PMID}-{protein_name}-{index}'
    class: 'positive' (1) or 'negative' (0)
    """
    log.info("Parsing Li et al. 2014 S1: %s", s1_path)
    raw = pd.read_excel(s1_path)
    log.info("  Raw shape: %s", raw.shape)

    records = []
    for _, row in raw.iterrows():
        apt_id   = str(row.get("aptamer_id", ""))
        target   = str(row.get("target_id", "")).replace("_", " ").strip()
        label    = 1 if str(row.get("class", "")).strip().lower() == "positive" else 0

        # Extract PMID from aptamer_id (first token before first hyphen)
        pmid_match = re.match(r"^(\d+)-", apt_id)
        pmid = pmid_match.group(1) if pmid_match else None

        records.append({
            "sequence":                   None,    # not in dataset — needs PubMed lookup
            "target_protein":             target,
            "uniprot_id":                 None,
            "protein_sequence":           None,
            "Kd_nM":                      None,    # not reported in this dataset
            "pH":                         DEFAULT_PH,
            "salt_mM":                    DEFAULT_SALT_MM,
            "temp_C":                     DEFAULT_TEMP_C,
            "buffer_type":                DEFAULT_BUFFER,
            "mg_mM":                      DEFAULT_MG_MM,
            "label":                      label,
            "source_pmid":                pmid,
            "training_tier":              _assign_tier(target),
            "augmented":                  False,
            "aug_method":                 None,
            "needs_sequence_enrichment":  True,    # sequence must be fetched before use
            "source":                     "li2014",
        })

    df = pd.DataFrame(records)
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    log.info("  Li2014 parsed: %d rows (%d positive, %d negative) — sequences pending",
             len(df), pos, neg)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def build_master(include_utexas: bool = True, include_li2014: bool = True) -> pd.DataFrame:
    parts = []

    if include_utexas:
        utexas_path = os.path.join(DATA_RAW, "utexas_aptamer_db", "aptamer_database.xlsx")
        if not os.path.exists(utexas_path):
            log.warning("UTexas file not found: %s", utexas_path)
        else:
            parts.append(parse_utexas(utexas_path))

    if include_li2014:
        li_path = os.path.join(DATA_RAW, "li2014_benchmark", "pone.0086729.s001.xlsx")
        if not os.path.exists(li_path):
            log.warning("Li2014 S1 file not found: %s", li_path)
        else:
            parts.append(parse_li2014(li_path))

    if not parts:
        raise RuntimeError("No data sources found.")

    master = pd.concat(parts, ignore_index=True)

    # Enforce schema column order
    for col in SCHEMA_COLS:
        if col not in master.columns:
            master[col] = None
    master = master[SCHEMA_COLS]

    # Summary
    total      = len(master)
    with_seq   = master["sequence"].notna().sum()
    pending    = master["needs_sequence_enrichment"].sum()
    tier1      = (master["training_tier"] == 1).sum()
    tier2      = (master["training_tier"] == 2).sum()
    positive   = (master["label"] == 1).sum()
    negative   = (master["label"] == 0).sum()

    log.info("=" * 50)
    log.info("Master dataset: %d total rows", total)
    log.info("  With sequence (ready to train): %d", with_seq)
    log.info("  Needs sequence enrichment:      %d", pending)
    log.info("  Tier 1 (general):               %d", tier1)
    log.info("  Tier 2 (validation targets):    %d", tier2)
    log.info("  Positive labels:                %d", positive)
    log.info("  Negative labels:                %d", negative)
    log.info("=" * 50)

    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CondAptNet master dataset")
    parser.add_argument("--utexas-only",  action="store_true")
    parser.add_argument("--li2014-only",  action="store_true")
    parser.add_argument("--output", default=os.path.join(DATA_PROCESSED, "master_dataset.csv"))
    args = parser.parse_args()

    include_utexas = not args.li2014_only
    include_li2014 = not args.utexas_only

    master = build_master(include_utexas=include_utexas, include_li2014=include_li2014)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    master.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
