"""
bioRxiv preprint adapter.

Strategy:
  1. Fetch recent bioRxiv papers in the `bioinformatics` and `biochemistry`
     categories using the bioRxiv Content API (date-range based).
  2. Filter locally by keyword to keep only aptamer-related papers.
  3. For matching preprints, fetch the full abstract JSON via the content API.
  4. Extract sequences / Kd / conditions from the abstract + title text.

bioRxiv Content API:
  GET https://api.biorxiv.org/details/{server}/{interval}/{cursor}
  server   : biorxiv | medrxiv
  interval : YYYY-MM-DD/YYYY-MM-DD  (date range, max 30 days recommended)
  cursor   : 0-based offset for pagination

No API key required — but adhere to the 1 req/s rate limit set in config.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_BASE_URL    = "https://api.biorxiv.org/details"
_SERVER      = "biorxiv"
_CATEGORIES  = {"bioinformatics", "biochemistry", "molecular-biology", "synthetic-biology"}
_APTAMER_KW  = re.compile(
    r"\baptamer\b|\bSELEX\b|\bDNA\s+binding\b|\bRNA\s+aptamer\b",
    re.IGNORECASE,
)


class BioRxivAdapter(BaseAdapter):
    """
    Scrape bioRxiv for aptamer-related preprints.
    """

    source_name = "biorxiv"
    source_type = "preprint"

    def __init__(
        self,
        prov_logger: Optional[ProvenanceLogger] = None,
        lookback_days: int = 365,
    ) -> None:
        super().__init__(prov_logger)
        self.lookback_days = lookback_days

    def _fetch_window(self, start: date, end: date, cursor: int = 0) -> dict:
        """Fetch one page of results for a date window."""
        url = f"{_BASE_URL}/{_SERVER}/{start.isoformat()}/{end.isoformat()}/{cursor}"
        resp = self._get(url)
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def _iter_papers(self, max_results: int):
        """Yield paper dicts from the API, newest first, up to max_results."""
        end   = date.today()
        start = end - timedelta(days=self.lookback_days)
        # API returns up to 100 per page
        cursor = 0
        seen   = 0
        while seen < max_results:
            data = self._fetch_window(start, end, cursor)
            collection = data.get("collection", [])
            if not collection:
                break
            for paper in collection:
                yield paper
                seen += 1
                if seen >= max_results:
                    break
            # Paginate
            messages = data.get("messages", [{}])
            total = int(messages[0].get("total", 0)) if messages else 0
            cursor += len(collection)
            if cursor >= total:
                break

    def run(self, max_results: int = 500) -> list[dict]:
        all_records: list[dict] = []
        for paper in self._iter_papers(max_results * 5):   # over-fetch, filter locally
            category = paper.get("category", "").lower().replace(" ", "-")
            if category not in _CATEGORIES:
                continue

            title    = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text     = f"{title}\n{abstract}"

            if not _APTAMER_KW.search(text):
                continue

            doi     = paper.get("doi", "")
            source  = f"https://www.biorxiv.org/content/{doi}" if doi else "https://www.biorxiv.org"

            from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract
            target = _guess_target_from_abstract(text)

            recs = self._extract_records_from_text(
                text=text,
                target_name=target,
                source_url=source,
                doi=doi,
                confidence="extracted",
            )
            all_records.extend(recs)
            if len(all_records) >= max_results:
                break

        log.info("BioRxivAdapter: %d records", len(all_records))
        return all_records[:max_results]
