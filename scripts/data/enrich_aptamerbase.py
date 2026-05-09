"""
AptamerBase enrichment pipeline.

Joins the three AptamerBase GitHub dump files, recovers sequences for Li2014 rows,
and adds novel unmodified DNA aptamer-protein pairs to master_dataset.csv.

Sources (from ~/Downloads or path provided via --ab-dir):
    aptamerbase_aptamers.csv
    aptamerbase_interactions.csv
    aptamerbase_experiments.csv

Effects on master_dataset.csv:
    1. Li2014 rows whose aptamer_id matches an AptamerBase label get their
       sequence filled in and needs_sequence_enrichment set to False.
    2. New unmodified, ATGC-only, 20-120 nt aptamer-protein pairs are appended
       as source='aptamerbase', label=1, training_tier=1.

Usage:
    python scripts/data/enrich_aptamerbase.py
    python scripts/data/enrich_aptamerbase.py --ab-dir /path/to/csvs
    python scripts/data/enrich_aptamerbase.py --dry-run
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_PROCESSED, DATA_RAW,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
    VALIDATION_TARGETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCHEMA_COLS = [
    "sequence", "target_protein", "uniprot_id", "protein_sequence",
    "Kd_nM", "pH", "salt_mM", "temp_C", "buffer_type", "mg_mM",
    "label", "source_pmid", "training_tier",
    "augmented", "aug_method",
    "needs_sequence_enrichment", "source",
]


def _extract_pmid(s: str) -> str | None:
    m = re.match(r"^(\d+)", str(s).strip())
    return m.group(1) if m else None


def _clean_seq(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    s = re.sub(r"[^ATGCU]", "", s)
    s = s.replace("U", "T")
    return s if s else None


def _assign_tier(target) -> int:
    if pd.isna(target):
        return 1
    t = str(target).lower()
    for vt in VALIDATION_TARGETS:
        if vt.lower() in t:
            return 2
    return 1


def load_and_join(ab_dir: str) -> pd.DataFrame:
    """Join the three AptamerBase files into aptamer-protein pairs."""
    apt    = pd.read_csv(os.path.join(ab_dir, "aptamerbase_aptamers.csv"))
    int_df = pd.read_csv(os.path.join(ab_dir, "aptamerbase_interactions.csv"))

    aptamer_uris = set(apt["id"])

    apt_rows = int_df[int_df["has_participant"].isin(aptamer_uris)][
        ["int", "has_participant", "dissociation_constant_value"]
    ].rename(columns={"has_participant": "aptamer_uri",
                      "dissociation_constant_value": "kd_molar"})

    tgt_rows = int_df[~int_df["has_participant"].isin(aptamer_uris)][
        ["int", "participant_label"]
    ].rename(columns={"participant_label": "target_name"})

    paired = apt_rows.merge(tgt_rows, on="int", how="inner")
    paired = paired.merge(
        apt[["id", "label", "sequence", "has_modification_details"]],
        left_on="aptamer_uri", right_on="id", how="left",
    )

    paired["pmid"]           = paired["label"].apply(_extract_pmid)
    paired["sequence_clean"] = paired["sequence"].apply(_clean_seq)
    paired = paired[paired["sequence_clean"].notna()].copy()

    log.info("AptamerBase joined: %d rows with usable sequence", len(paired))
    return paired


def build_lookup(paired: pd.DataFrame) -> dict:
    """label (lowercase) → {sequence_clean, pmid, target_name, kd_molar}."""
    lookup: dict = {}
    for _, row in paired.iterrows():
        key = str(row["label"]).lower().strip()
        if key not in lookup:
            lookup[key] = {
                "sequence_clean": row["sequence_clean"],
                "pmid":           row["pmid"],
                "target_name":    row["target_name"],
                "kd_molar":       row["kd_molar"],
            }
    log.info("Lookup table: %d unique aptamer labels", len(lookup))
    return lookup


def enrich_li2014(master: pd.DataFrame, lookup: dict) -> tuple[pd.DataFrame, int]:
    """
    Fill sequence for Li2014 rows in master whose aptamer_id matches AptamerBase.
    Returns (updated master, n_recovered).
    """
    li_path = os.path.join(DATA_RAW, "li2014_benchmark", "pone.0086729.s001.xlsx")
    if not os.path.exists(li_path):
        log.warning("Li2014 S1 not found at %s — skipping enrichment", li_path)
        return master, 0

    li_raw = pd.read_excel(li_path)
    li_raw["label_key"] = li_raw["aptamer_id"].str.lower().str.strip()

    master_li_idx = master[master["source"] == "li2014"].index.tolist()
    assert len(master_li_idx) == len(li_raw), (
        f"Row count mismatch: master={len(master_li_idx)}, li_raw={len(li_raw)}"
    )

    recovered = 0
    for i, master_idx in enumerate(master_li_idx):
        if not master.at[master_idx, "needs_sequence_enrichment"]:
            continue
        key = li_raw.at[i, "label_key"]
        if key not in lookup:
            continue
        hit = lookup[key]
        seq = hit["sequence_clean"]
        if not (seq and re.fullmatch(r"[ATGC]+", seq) and 20 <= len(seq) <= 120):
            continue
        master.at[master_idx, "sequence"]                  = seq
        master.at[master_idx, "needs_sequence_enrichment"] = False
        kd = hit["kd_molar"]
        if pd.notna(kd):
            master.at[master_idx, "Kd_nM"] = float(kd) * 1e9
        recovered += 1

    log.info("Li2014 sequences recovered: %d", recovered)
    return master, recovered


def build_new_rows(paired: pd.DataFrame, existing_seqs: set) -> pd.DataFrame:
    """
    Return new master rows from AptamerBase: unmodified, ATGC-only, 20-120 nt,
    not already in master.
    """
    clean = paired[paired["has_modification_details"].isna()].copy()
    clean = clean[clean["sequence_clean"].str.fullmatch(r"[ATGC]+", na=False)]
    clean["seq_len"] = clean["sequence_clean"].str.len()
    clean = clean[(clean["seq_len"] >= 20) & (clean["seq_len"] <= 120)]
    clean = clean.drop_duplicates(subset=["sequence_clean", "target_name"])
    clean = clean[~clean["sequence_clean"].isin(existing_seqs)]

    rows = []
    for _, row in clean.iterrows():
        kd = row["kd_molar"]
        rows.append({
            "sequence":                  row["sequence_clean"],
            "target_protein":            row["target_name"],
            "uniprot_id":                None,
            "protein_sequence":          None,
            "Kd_nM":                     float(kd) * 1e9 if pd.notna(kd) else None,
            "pH":                        DEFAULT_PH,
            "salt_mM":                   DEFAULT_SALT_MM,
            "temp_C":                    DEFAULT_TEMP_C,
            "buffer_type":               DEFAULT_BUFFER,
            "mg_mM":                     DEFAULT_MG_MM,
            "label":                     1,
            "source_pmid":               row["pmid"],
            "training_tier":             _assign_tier(row["target_name"]),
            "augmented":                 False,
            "aug_method":                None,
            "needs_sequence_enrichment": False,
            "source":                    "aptamerbase",
        })

    log.info("New AptamerBase rows: %d", len(rows))
    return pd.DataFrame(rows, columns=SCHEMA_COLS) if rows else pd.DataFrame(columns=SCHEMA_COLS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab-dir", default=str(Path.home() / "Downloads"),
                        help="Directory containing the three AptamerBase CSVs")
    parser.add_argument("--output", default=os.path.join(DATA_PROCESSED, "master_dataset.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    master_path = os.path.join(DATA_PROCESSED, "master_dataset.csv")
    if not os.path.exists(master_path):
        log.error("master_dataset.csv not found — run build_dataset.py first")
        sys.exit(1)

    master = pd.read_csv(master_path)
    log.info("Loaded master: %d rows", len(master))

    paired = load_and_join(args.ab_dir)
    lookup = build_lookup(paired)

    master, n_recovered = enrich_li2014(master, lookup)

    existing_seqs = set(master["sequence"].dropna().str.upper().str.strip())
    new_df = build_new_rows(paired, existing_seqs)

    master_updated = pd.concat([master, new_df], ignore_index=True)[SCHEMA_COLS]

    total_ready = (
        master_updated["sequence"].notna() &
        (master_updated["needs_sequence_enrichment"] == False)
    ).sum()
    still_pending = (master_updated["needs_sequence_enrichment"] == True).sum()

    log.info("=" * 50)
    log.info("Li2014 sequences recovered : %d", n_recovered)
    log.info("New AptamerBase rows added : %d", len(new_df))
    log.info("Total master rows          : %d", len(master_updated))
    log.info("Training-ready rows        : %d", total_ready)
    log.info("Still pending enrichment   : %d", still_pending)
    log.info("=" * 50)

    if args.dry_run:
        log.info("Dry-run — not writing output")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    master_updated.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
