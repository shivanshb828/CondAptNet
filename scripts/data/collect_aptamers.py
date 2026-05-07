"""
PubMed literature collection for aptamer-protein binding data (2012–2025).

Searches PubMed via Entrez API for SELEX studies and extracts metadata.
Output: data/raw/pubmed_results.csv

Usage:
    python scripts/data/collect_aptamers.py
"""

import os
import sys
import time
import logging
import csv
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_RAW, RANDOM_SEED

try:
    from Bio import Entrez
except ImportError:
    sys.exit("BioPython not found. Run: pip install biopython")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

Entrez.email = "coolshivansh7@gmail.com"

# ── Search queries ────────────────────────────────────────────────────────────
QUERIES = [
    # Core SELEX / aptamer binding
    '("DNA aptamer"[Title/Abstract] AND "SELEX"[Title/Abstract] AND "binding"[Title/Abstract])',
    '("aptamer selection"[Title/Abstract] AND "dissociation constant"[Title/Abstract])',
    '("DNA aptamer"[Title/Abstract] AND "Kd"[Title/Abstract] AND "protein"[Title/Abstract])',
    # Target-specific
    '("insulin aptamer"[Title/Abstract])',
    '("myoglobin aptamer"[Title/Abstract])',
    '("troponin aptamer"[Title/Abstract])',
    '("NT-proBNP aptamer"[Title/Abstract])',
    '("albumin aptamer"[Title/Abstract])',
]

DATE_RANGE = "2012/01/01:2025/12/31[PDAT]"
MAX_PER_QUERY = 500


def search_pubmed(query: str, date_range: str, retmax: int = MAX_PER_QUERY) -> list[str]:
    """Return list of PMIDs matching query."""
    full_query = f"({query}) AND {date_range}"
    handle = Entrez.esearch(db="pubmed", term=full_query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    log.info("  Query returned %d PMIDs", len(pmids))
    return pmids


def fetch_summaries(pmids: list[str]) -> list[dict]:
    """Fetch title, abstract, journal, year for a list of PMIDs."""
    if not pmids:
        return []
    id_str = ",".join(pmids)
    handle = Entrez.efetch(db="pubmed", id=id_str, rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    results = []
    for article in records.get("PubmedArticle", []):
        medline = article["MedlineCitation"]
        art = medline["Article"]

        pmid = str(medline["PMID"])
        title = str(art.get("ArticleTitle", ""))
        abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(t) for t in abstract_texts)

        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

        results.append({"pmid": pmid, "title": title, "abstract": abstract, "year": year})
    return results


def collect_all() -> list[dict]:
    all_pmids: set[str] = set()
    for q in QUERIES:
        log.info("Searching: %s", q[:80])
        try:
            pmids = search_pubmed(q, DATE_RANGE)
            all_pmids.update(pmids)
        except Exception as exc:
            log.warning("Query failed: %s", exc)
        time.sleep(0.4)  # NCBI rate limit: 3 req/s without API key

    log.info("Total unique PMIDs: %d", len(all_pmids))
    pmid_list = sorted(all_pmids)

    # Fetch in batches of 100
    all_records: list[dict] = []
    batch_size = 100
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i : i + batch_size]
        log.info("Fetching batch %d/%d", i // batch_size + 1, -(-len(pmid_list) // batch_size))
        try:
            records = fetch_summaries(batch)
            all_records.extend(records)
        except Exception as exc:
            log.warning("Batch fetch failed: %s", exc)
        time.sleep(0.4)

    return all_records


def save_results(records: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["pmid", "title", "abstract", "year"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    log.info("Saved %d records to %s", len(records), out_path)


if __name__ == "__main__":
    out_path = os.path.join(DATA_RAW, "pubmed_results.csv")
    log.info("Starting PubMed collection (this takes ~20 min for large result sets)")
    records = collect_all()
    save_results(records, out_path)
    log.info("Done. %d papers collected.", len(records))
