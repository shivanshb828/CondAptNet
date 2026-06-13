"""
OpenAlex adapter.

OpenAlex (openalex.org) is a fully open bibliographic index of 250M+ scholarly
works. No API key required; email in the User-Agent enables the "polite pool"
with higher rate limits (10 req/s vs 1 req/s unauthenticated).

Endpoints used:
  GET https://api.openalex.org/works
    ?search={query}
    &filter=concepts.id:C12345  (optional concept filter)
    &per-page=100
    &cursor=*

Returns: abstract + title for each work. No full text, but enough for
sequence/Kd extraction in review/methods papers that quote sequences.

Rate limit: 5 req/s (polite pool), set in config.RATE_LIMITS["openalex"].
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org/works"

# OpenAlex concept IDs relevant to aptamers
# C1276952 = "DNA aptamer", C2776844 = "SELEX", C95457728 = "Biochemistry"
_APTAMER_QUERIES = [
    "DNA aptamer SELEX",
    "aptamer binding affinity dissociation constant",
    "aptamer selection protein target",
    "ssDNA aptamer SELEX Kd",
    "RNA aptamer in vitro selection",
]

_APTAMER_KW = re.compile(r"\baptamer\b|\bSELEX\b", re.IGNORECASE)


class OpenAlexAdapter(BaseAdapter):
    """
    Query OpenAlex for aptamer papers; extract sequences from abstracts.
    """

    source_name = "openalex"
    source_type = "paper"

    def __init__(
        self,
        prov_logger: Optional[ProvenanceLogger] = None,
        queries: Optional[list[str]] = None,
    ) -> None:
        super().__init__(prov_logger)
        self.queries = queries or _APTAMER_QUERIES
        # Email in polite-pool header (no API key required)
        from scripts.data.scraper import config as cfg
        self._session.headers["From"] = cfg.ENTREZ_EMAIL

    def _search(self, query: str, cursor: str = "*", per_page: int = 100) -> dict:
        resp = self._get(
            _BASE_URL,
            params={
                "search":   query,
                "per-page": per_page,
                "cursor":   cursor,
                "select":   "id,doi,title,abstract_inverted_index,open_access",
            },
        )
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
        """
        OpenAlex stores abstracts as inverted index: {word: [positions...]}.
        Reconstruct the plain text.
        """
        if not inverted_index:
            return ""
        words: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[i] for i in sorted(words))

    def run(self, max_results: int = 500) -> list[dict]:
        all_records: list[dict] = []
        seen_ids: set[str] = set()

        for query in self.queries:
            cursor = "*"
            while len(all_records) < max_results:
                data   = self._search(query, cursor=cursor)
                works  = data.get("results", [])
                if not works:
                    break

                for work in works:
                    work_id = work.get("id", "")
                    if work_id in seen_ids:
                        continue
                    seen_ids.add(work_id)

                    title    = work.get("title") or ""
                    abstract = self._reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    )
                    text = f"{title}\n{abstract}"

                    if not _APTAMER_KW.search(text):
                        continue

                    doi    = work.get("doi") or ""
                    url    = doi if doi.startswith("http") else f"https://doi.org/{doi}" if doi else work_id

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

                # Advance cursor
                meta   = data.get("meta", {})
                cursor = meta.get("next_cursor", "")
                if not cursor:
                    break

        log.info("OpenAlexAdapter: %d records", len(all_records))
        return all_records[:max_results]
