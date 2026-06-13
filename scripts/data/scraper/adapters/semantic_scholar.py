"""
Semantic Scholar adapter.

Semantic Scholar (semanticscholar.org) provides a free academic search API
with optional API key for higher rate limits.

Without SEMANTIC_SCHOLAR_API_KEY: 100 requests per 5 minutes (0.333 req/s).
With key: 1 req/s or higher (depends on tier).

Endpoints:
  Search: GET https://api.semanticscholar.org/graph/v1/paper/search
    ?query={q}
    &fields=paperId,title,abstract,year,externalIds,openAccessPdf
    &limit=100

  Paper detail (for open-access PDF link):
    GET https://api.semanticscholar.org/graph/v1/paper/{paperId}
    ?fields=title,abstract,openAccessPdf

Rate limit key: "semantic_scholar" in RATE_LIMITS config (0.333/s).
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

_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_PAPER_URL  = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

_FIELDS     = "paperId,title,abstract,year,externalIds,openAccessPdf"

_QUERIES = [
    "DNA aptamer SELEX binding protein",
    "ssDNA aptamer dissociation constant Kd",
    "aptamer selection in vitro evolution",
    "aptamer binding affinity nM protein target",
    "CE-SELEX aptamer selection binding",
    "Capture-SELEX aptamer sequence",
    "cell-SELEX DNA aptamer",
    "aptamer SELEX SPR ITC binding",
]


class SemanticScholarAdapter(BaseAdapter):
    """
    Search Semantic Scholar for aptamer papers; extract records from abstracts.
    """

    source_name = "semantic_scholar"
    source_type = "paper"

    def __init__(
        self,
        prov_logger: Optional[ProvenanceLogger] = None,
        queries: Optional[list[str]] = None,
    ) -> None:
        super().__init__(prov_logger)
        self.queries = queries if queries is not None else _QUERIES
        if cfg.SEMANTIC_SCHOLAR_KEY:
            self._session.headers["x-api-key"] = cfg.SEMANTIC_SCHOLAR_KEY

    def _search(self, query: str, offset: int = 0, limit: int = 100) -> dict:
        resp = self._get(
            _SEARCH_URL,
            params={
                "query":  query,
                "fields": _FIELDS,
                "offset": offset,
                "limit":  limit,
            },
        )
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def _iter_papers(self, query: str, max_papers: int = 200):
        """Yield paper dicts for one query."""
        offset = 0
        limit  = min(100, max_papers)
        seen   = 0
        while seen < max_papers:
            data   = self._search(query, offset=offset, limit=limit)
            papers = data.get("data", [])
            if not papers:
                break
            for paper in papers:
                yield paper
                seen += 1
            total = data.get("total", 0)
            offset += len(papers)
            if offset >= total:
                break

    def run(self, max_results: int = 500) -> list[dict]:
        all_records: list[dict] = []
        seen_ids: set[str] = set()

        per_query = max(50, max_results // max(len(self.queries), 1))

        for query in self.queries:
            for paper in self._iter_papers(query, max_papers=per_query):
                pid = paper.get("paperId", "")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                title    = paper.get("title") or ""
                abstract = paper.get("abstract") or ""
                text     = f"{title}\n{abstract}"

                if not text.strip():
                    continue

                ext_ids = paper.get("externalIds") or {}
                doi     = ext_ids.get("DOI") or ""
                url     = (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://www.semanticscholar.org/paper/{pid}"
                )

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

            if len(all_records) >= max_results:
                break

        log.info("SemanticScholarAdapter: %d records", len(all_records))
        return all_records[:max_results]
