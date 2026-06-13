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
    {"match": {"title_abstract_text": "DNA aptamer SELEX binding affinity"}},
    {"match": {"title_abstract_text": "ssDNA aptamer Kd nM protein"}},
    {"match": {"title_abstract_text": "aptamer selection dissociation constant"}},
]

_PATENT_QUERIES = [
    {"match": {"title_claims_abstract": "DNA aptamer SELEX nucleotide binding"}},
    {"match": {"title_claims_abstract": "aptamer protein binding Kd dissociation"}},
]


def _scholarly_payload(query: dict, from_: int = 0, size: int = 100) -> dict:
    return {
        "query": query,
        "from": from_,
        "size": size,
        "include": ["title", "abstract", "doi", "external_ids", "year_published"],
        "sort": [{"_score": "desc"}],
    }


def _patent_payload(query: dict, from_: int = 0, size: int = 100) -> dict:
    return {
        "query": query,
        "from": from_,
        "size": size,
        "include": ["lens_id", "title", "abstract", "claims", "publication_number"],
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

        # ── Scholarly papers ──────────────────────────────────────────────────
        for query in _SCHOLARLY_QUERIES:
            for hit in self._iter_scholarly(query, max_hits=100):
                lid = hit.get("lens_id", "")
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)

                title    = hit.get("title") or ""
                abstract = hit.get("abstract") or ""
                text     = f"{title}\n{abstract}"
                doi      = hit.get("doi") or ""
                url      = f"https://doi.org/{doi}" if doi else f"https://lens.org/{lid}"

                from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract
                target = _guess_target_from_abstract(text)

                recs = self._extract_records_from_text(
                    text=text,
                    target_name=target,
                    source_url=url,
                    doi=doi,
                    confidence="extracted",
                )
                all_records.extend(recs)
                if len(all_records) >= max_results:
                    break

        # ── Patents ───────────────────────────────────────────────────────────
        for query in _PATENT_QUERIES:
            for hit in self._iter_patents(query, max_hits=100):
                lid = hit.get("lens_id", "") or hit.get("publication_number", "")
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)

                title   = hit.get("title") or ""
                abstract = hit.get("abstract") or ""
                claims   = hit.get("claims") or ""
                if isinstance(claims, list):
                    claims = " ".join(str(c) for c in claims)
                text = f"{title}\n{abstract}\n{claims}"
                url  = f"https://lens.org/{lid}"

                from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract
                target = _guess_target_from_abstract(text)

                recs = self._extract_records_from_text(
                    text=text,
                    target_name=target,
                    source_url=url,
                    doi=lid,
                    confidence="extracted",
                    extra_fields={"source_type": "patent"},
                )
                all_records.extend(recs)
                if len(all_records) >= max_results:
                    break

        log.info("LensAdapter: %d records", len(all_records))
        return all_records[:max_results]
