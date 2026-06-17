"""
Scraper configuration — API credentials, rate limits, output paths.

All secrets come from environment variables. Never hardcode credentials here.
"""

import os
from pathlib import Path

# ── Credentials (set via env vars before running) ────────────────────────────
ENTREZ_EMAIL          = os.environ.get("ENTREZ_EMAIL",          "coolshivansh7@gmail.com")
NCBI_API_KEY          = os.environ.get("NCBI_API_KEY",          "")   # 10 req/s vs 3 req/s
EPO_CLIENT_KEY        = os.environ.get("EPO_CLIENT_KEY",        "")   # EPO OPS OAuth2
EPO_CLIENT_SECRET     = os.environ.get("EPO_CLIENT_SECRET",     "")
LENS_API_TOKEN        = os.environ.get("LENS_API_TOKEN",        "")
SEMANTIC_SCHOLAR_KEY  = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# ── Rate limits (requests/second per source) ─────────────────────────────────
# Match published API limits. Exceeding them risks IP bans that halt the run.
RATE_LIMITS: dict[str, float] = {
    "pubmed":           3.0,    # 3/s without API key; 10/s with NCBI_API_KEY
    "pmc":              3.0,
    "biorxiv":          1.0,
    "openalex":         5.0,    # polite pool — no hard limit published
    "patents_us":       2.0,    # PatentsView — no published limit; be conservative
    "patents_epo":      2.0,    # EPO OPS free tier: 4 GB/week, ~2/s is safe
    "patents_wipo":     1.0,
    "lens":             0.167,  # 10 requests/minute
    "google_patents":   0.2,    # 1 request/5s — aggressive scraping = ban
    "databases":        2.0,
    "semantic_scholar": 0.333,  # 100 requests/5 min = 0.333/s
}

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[3]
DATA_RAW       = _ROOT / "data" / "raw"
DATA_PROCESSED = _ROOT / "data" / "processed"

SCRAPER_OUTPUT   = DATA_RAW / "scraped_dataset.csv"
PROVENANCE_LOG   = DATA_RAW / "scraper_provenance.jsonl"
COVERAGE_REPORT  = DATA_RAW / "scraper_coverage_report.txt"
ERROR_LOG        = DATA_RAW / "scraper_errors.log"
EXISTING_MASTER  = DATA_PROCESSED / "master_dataset.csv"

# ── PubMed search queries ─────────────────────────────────────────────────────
PUBMED_QUERIES: list[str] = [
    '("DNA aptamer"[Title/Abstract] AND "SELEX"[Title/Abstract])',
    '("aptamer"[Title/Abstract] AND "dissociation constant"[Title/Abstract])',
    '("aptamer"[Title/Abstract] AND "Kd"[Title/Abstract])',
    '("aptamer selection"[Title/Abstract] AND "binding affinity"[Title/Abstract])',
    '("ssDNA aptamer"[Title/Abstract])',
    '("RNA aptamer"[Title/Abstract] AND "SELEX"[Title/Abstract])',
    '("CE-SELEX"[Title/Abstract] AND "aptamer"[Title/Abstract])',
    '("Capture-SELEX"[Title/Abstract])',
    '("Cell-SELEX"[Title/Abstract] AND "aptamer"[Title/Abstract])',
    '("Microfluidic SELEX"[Title/Abstract])',
    '("aptamer"[Title/Abstract] AND "SPR"[Title/Abstract] AND "binding"[Title/Abstract])',
    '("aptamer"[Title/Abstract] AND "ITC"[Title/Abstract])',
    '("aptamer"[Title/Abstract] AND "EMSA"[Title/Abstract])',
    '("systematic evolution of ligands"[Title/Abstract])',
    '("in vitro selection"[Title/Abstract] AND "oligonucleotide"[Title/Abstract])',
]

# ── Validation bounds (mirror of top-level config.py) ────────────────────────
SEQ_MIN_LEN     = 20
SEQ_MAX_LEN     = 120
GC_MIN          = 0.20
GC_MAX          = 0.80
MAX_HOMOPOLYMER = 8
