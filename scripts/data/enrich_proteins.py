"""
UniProt protein sequence enrichment for master_dataset.csv.

For each unique target_protein name that has no protein_sequence, queries the
UniProt REST API (Swiss-Prot reviewed entries first, then any) and fills in
uniprot_id + protein_sequence for all matching rows.

Usage:
    python scripts/data/enrich_proteins.py              # all proteins
    python scripts/data/enrich_proteins.py --top 30     # top-30 by row count
    python scripts/data/enrich_proteins.py --dry-run    # print matches, no write
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
REQUEST_DELAY  = 0.4   # seconds between API calls (stay within rate limit)


def search_uniprot(protein_name: str, organism: str | None = "9606") -> tuple[str, str] | None:
    """
    Search UniProt for a protein by name.
    Returns (accession, sequence) for the best reviewed match, or None.

    organism='9606' tries human first; pass None to allow any organism.
    """
    query = f'"{protein_name}" AND reviewed:true'
    if organism:
        query += f" AND organism_id:{organism}"

    params = {
        "query":  query,
        "format": "json",
        "size":   "1",
        "fields": "accession,sequence",
    }
    try:
        resp = requests.get(UNIPROT_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            acc = results[0]["primaryAccession"]
            seq = results[0]["sequence"]["value"]
            return acc, seq
    except Exception as exc:
        log.debug("UniProt query failed ('%s'): %s", protein_name, exc)
    return None


def find_sequence(protein_name: str) -> tuple[str, str] | None:
    """
    Try human first (organism=9606), then any reviewed organism, then unrestricted.
    """
    for org in ("9606", None):
        hit = search_uniprot(protein_name, organism=org)
        if hit:
            return hit
    return None


def enrich(
    master: pd.DataFrame,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Enrich master_dataset.csv with protein_sequence from UniProt.

    Args:
        master : loaded master_dataset.csv
        top_n  : if set, only enrich the top-N proteins (by row count)

    Returns:
        (updated master, n_proteins_found)
    """
    # Unique proteins that still need enrichment
    need = master[master["protein_sequence"].isna()]["target_protein"]
    proteins_by_count = need.value_counts()

    if top_n is not None:
        proteins_by_count = proteins_by_count.head(top_n)

    # Ensure object dtype so string values can be assigned into all-NaN columns
    master["uniprot_id"]       = master["uniprot_id"].astype(object)
    master["protein_sequence"] = master["protein_sequence"].astype(object)

    targets = proteins_by_count.index.tolist()
    log.info("Proteins to enrich: %d (covering %d rows)",
             len(targets), proteins_by_count.sum())

    found = 0
    for i, name in enumerate(targets, 1):
        log.info("[%d/%d] Searching: %s", i, len(targets), name)
        hit = find_sequence(name)
        time.sleep(REQUEST_DELAY)

        if hit is None:
            log.warning("  Not found: %s", name)
            continue

        acc, seq = hit
        mask = master["target_protein"] == name
        master.loc[mask, "uniprot_id"]       = acc
        master.loc[mask, "protein_sequence"] = seq
        n = mask.sum()
        log.info("  Found %s (%d aa) → updated %d rows", acc, len(seq), n)
        found += 1

    return master, found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",     type=int, default=None,
                        help="Enrich only the top-N proteins by row count")
    parser.add_argument("--output",  default=os.path.join(DATA_PROCESSED, "master_dataset.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    master_path = os.path.join(DATA_PROCESSED, "master_dataset.csv")
    if not os.path.exists(master_path):
        log.error("master_dataset.csv not found — run build_dataset.py first")
        sys.exit(1)

    master = pd.read_csv(master_path)
    log.info("Loaded master: %d rows, %d already have protein_sequence",
             len(master), master["protein_sequence"].notna().sum())

    master, n_found = enrich(master, top_n=args.top)

    still_missing = master["protein_sequence"].isna().sum()
    total_ready = (
        master["sequence"].notna() &
        (master["needs_sequence_enrichment"] == False) &
        master["protein_sequence"].notna()
    ).sum()

    log.info("=" * 50)
    log.info("Proteins found         : %d", n_found)
    log.info("Proteins still missing : %d", still_missing)
    log.info("Training-ready rows    : %d", total_ready)
    log.info("=" * 50)

    if args.dry_run:
        log.info("Dry-run — not writing")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    master.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
