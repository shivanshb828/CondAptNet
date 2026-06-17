"""
Lens.org adapter.

Lens.org is a free scholarly/patent search platform that aggregates PubMed,
Crossref, Microsoft Academic, and WIPO patent data. It has a JSON REST API
requiring a Bearer token (free to register at access.lens.org).

Without LENS_API_TOKEN this adapter returns [] with a warning.

Endpoints:
  Scholarly: POST https://api.lens.org/scholarly/search
  Patent:    POST https://api.lens.org/patent/search

Both return JSON with title, abstract, DOI, and (for patents) claims.

Rate limit: 10 requests/minute = 0.167 req/s — enforced in RATE_LIMITS["lens"].
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper import config as cfg
from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_SCHOLARLY_URL = "https://api.lens.org/scholarly/search"
_PATENT_URL    = "https://api.lens.org/patent/search"

_SCHOLARLY_QUERIES = [
    {"query_string": {"query": "DNA aptamer SELEX binding affinity protein", "default_operator": "AND"}},
    {"query_string": {"query": "ssDNA aptamer Kd nM dissociation constant",  "default_operator": "AND"}},
    {"query_string": {"query": "aptamer selection oligonucleotide binding",   "default_operator": "AND"}},
]

# Lens patent API: search for PCT/international aptamer patents.
# Valid top-level fields: lens_id, jurisdiction, doc_number, abstract, biblio,
#   sequence_listing, legal_status, date_published, publication_type.
_PATENT_QUERIES = [
    {"query_string": {"query": "DNA aptamer SELEX nucleotide binding protein",  "default_operator": "AND"}},
    {"query_string": {"query": "aptamer dissociation constant Kd nM sequence", "default_operator": "AND"}},
]


def _doi_from_external_ids(external_ids: list) -> str:
    """Extract DOI string from Lens scholarly external_ids list."""
    for entry in (external_ids or []):
        if isinstance(entry, dict) and entry.get("type") == "doi":
            return entry.get("value", "")
    return ""


def _scholarly_payload(query: dict, from_: int = 0, size: int = 100) -> dict:
    return {
        "query": query,
        "from": from_,
        "size": size,
        "include": ["title", "abstract", "external_ids", "year_published"],
        "sort": [{"_score": "desc"}],
    }


def _patent_payload(query: dict, from_: int = 0, size: int = 100) -> dict:
    return {
        "query": query,
        "from": from_,
        "size": size,
        "include": ["lens_id", "abstract", "biblio", "claims", "doc_number", "jurisdiction"],
        "sort": [{"_score": "desc"}],
    }


class LensAdapter(BaseAdapter):
    """
    Search Lens.org for aptamer papers + patents via the JSON API.
    """

    source_name = "lens"
    source_type = "paper"   # overridden per-record below

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        if cfg.LENS_API_TOKEN:
            self._session.headers["Authorization"] = f"Bearer {cfg.LENS_API_TOKEN}"

    def _post_search(self, url: str, payload: dict) -> dict:
        resp = self._post(url, json=payload)
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def _iter_scholarly(self, query: dict, max_hits: int = 200):
        from_ = 0
        size  = min(100, max_hits)
        while from_ < max_hits:
            data = self._post_search(_SCHOLARLY_URL, _scholarly_payload(query, from_, size))
            hits = data.get("data", [])
            if not hits:
                break
            for hit in hits:
                yield hit
            from_ += len(hits)
            if len(hits) < size:
                break

    def _iter_patents(self, query: dict, max_hits: int = 200):
        from_ = 0
        size  = min(100, max_hits)
        while from_ < max_hits:
            data = self._post_search(_PATENT_URL, _patent_payload(query, from_, size))
            hits = data.get("data", [])
            if not hits:
                break
            for hit in hits:
                yield hit
            from_ += len(hits)
            if len(hits) < size:
                break

    def run(self, max_results: int = 500) -> list[dict]:
        if not cfg.LENS_API_TOKEN:
            log.warning("LENS_API_TOKEN not set; skipping Lens adapter")
            return []

        all_records: list[dict] = []
        seen_ids: set[str] = set()

        from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract

        # ── Scholarly papers ──────────────────────────────────────────────────
        for query in _SCHOLARLY_QUERIES:
            for hit in self._iter_scholarly(query, max_hits=100):
                lid = hit.get("lens_id", "") or hit.get("title", "")[:40]
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)

                title    = hit.get("title") or ""
                abstract = hit.get("abstract") or ""
                text     = f"{title}\n{abstract}"
                doi      = _doi_from_external_ids(hit.get("external_ids", []))
                url      = f"https://doi.org/{doi}" if doi else f"https://lens.org/search"

                target = _guess_target_from_abstract(text)
                recs   = self._extract_records_from_text(
                    text=text, target_name=target, source_url=url,
                    doi=doi, confidence="extracted",
                )
                all_records.extend(recs)
                if len(all_records) >= max_results:
                    break

        # ── Patents ───────────────────────────────────────────────────────────
        for query in _PATENT_QUERIES:
            for hit in self._iter_patents(query, max_hits=100):
                lid = hit.get("lens_id", "") or hit.get("doc_number", "")
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)

                abstract = hit.get("abstract") or ""
                # Title is nested inside biblio.invention_title[0].text
                biblio = hit.get("biblio") or {}
                inv_titles = biblio.get("invention_title", [])
                title = ""
                if isinstance(inv_titles, list) and inv_titles:
                    title = inv_titles[0].get("text", "") if isinstance(inv_titles[0], dict) else str(inv_titles[0])
                # Claims contain explicit sequence text — include for extraction
                claims_raw = hit.get("claims") or []
                claims_text = ""
                if isinstance(claims_raw, list):
                    for claim_group in claims_raw:
                        if isinstance(claim_group, dict):
                            for c in claim_group.get("claims", []):
                                if isinstance(c, dict):
                                    claims_text += " ".join(c.get("claim_text", [])) + " "
                text = f"{title}\n{abstract}\n{claims_text}".strip()
                jur  = hit.get("jurisdiction", "")
                doc  = hit.get("doc_number", "")
                url  = f"https://lens.org/lens/patent/{jur}_{doc}" if jur and doc else f"https://lens.org/{lid}"

                target = _guess_target_from_abstract(text)
                recs   = self._extract_records_from_text(
                    text=text, target_name=target, source_url=url,
                    doi=lid, confidence="extracted",
                    extra_fields={"source_type": "patent"},
                )
                all_records.extend(recs)
                if len(all_records) >= max_results:
                    break

        log.info("LensAdapter: %d records", len(all_records))
        return all_records[:max_results]
