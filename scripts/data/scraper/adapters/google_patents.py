"""
Google Patents adapter.

Google Patents has no official public API. This adapter scrapes the
Google Patents search results page using requests + BeautifulSoup.

IMPORTANT: Google rate-limits aggressively. The limiter is set to 0.2 req/s
(1 request every 5 seconds). Exceeding this WILL result in IP bans that
halt the entire pipeline. Never reduce this rate limit.

Approach:
  1. GET https://patents.google.com/xhr/query?url={encoded_params} (JSON endpoint
     used by the patents.google.com web client — returns structured JSON).
  2. Parse patent numbers + snippets from the JSON response.
  3. For each result, fetch the individual patent page and extract claims.

The XHR endpoint returns a JSON object with a `results` array containing
`patent` objects with `patent_id`, `title`, and `snippet`.

Robots.txt: Google allows scraping with reasonable rate limits per
https://patents.google.com/robots.txt (no `/xhr/` disallow as of 2025).
"""

from __future__ import annotations

import logging
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.parsers.text_parser import parse_html
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_XHR_URL     = "https://patents.google.com/xhr/query"
_PATENT_URL  = "https://patents.google.com/patent/{patent_id}"

_SEARCH_QUERIES = [
    "DNA aptamer SELEX binding protein",
    "ssDNA aptamer dissociation constant Kd nM",
    "aptamer selection oligonucleotide binding affinity",
    "in vitro SELEX DNA binding target protein",
]

_APTAMER_KW = re.compile(r"\baptamer\b|\bSELEX\b|\bssDNA\b", re.IGNORECASE)

# Patent claims often contain sequences in 5'→3' notation or as plain ATGC runs
_PATENT_ID_RE = re.compile(r'\b(US\d{7,10}[AB]\d?|EP\d{7,10}[AB]\d?|WO\d{4}/\d{6,})\b')


class GooglePatentsAdapter(BaseAdapter):
    """
    Scrape Google Patents search results for aptamer patents.
    """

    source_name = "google_patents"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        # Mimic a real browser to avoid 403
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _xhr_search(self, query: str, page: int = 0) -> dict:
        """
        Call the Google Patents XHR/JSON endpoint.
        Returns parsed JSON dict or {}.
        """
        params = urllib.parse.urlencode({
            "url": f"q={urllib.parse.quote(query)}&num=10&start={page*10}",
            "exp": "",
        })
        resp = self._get(f"{_XHR_URL}?{params}")
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def _fetch_patent_page(self, patent_id: str) -> str:
        """Fetch individual patent page and return cleaned text."""
        url  = _PATENT_URL.format(patent_id=patent_id)
        resp = self._get(url)
        if resp is None:
            return ""
        doc = parse_html(resp.text, source_path=url)
        return doc.text if doc.parse_ok else ""

    def _parse_xhr_results(self, data: dict) -> list[dict]:
        """Extract patent IDs and snippets from XHR JSON response."""
        items: list[dict] = []
        results = data.get("results", {})
        if isinstance(results, dict):
            cluster = results.get("cluster", [])
            for cl in cluster:
                for result in cl.get("result", []):
                    pat  = result.get("patent", {})
                    pid  = pat.get("publication_number", "")
                    if not pid:
                        continue
                    items.append({
                        "patent_id": pid,
                        "title":     pat.get("title", ""),
                        "snippet":   result.get("snippet", {}).get("text", ""),
                    })
        return items

    def run(self, max_results: int = 200) -> list[dict]:
        # Conservative cap — Google bans aggressive scrapers
        max_results = min(max_results, 200)
        all_records: list[dict] = []
        seen_ids: set[str] = set()

        for query in _SEARCH_QUERIES:
            for page in range(5):    # max 5 pages = 50 results per query
                data  = self._xhr_search(query, page=page)
                items = self._parse_xhr_results(data)
                if not items:
                    break

                for item in items:
                    pid = item.get("patent_id", "").replace(" ", "")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    # Quick scan of snippet first
                    snippet = item.get("snippet", "")
                    title   = item.get("title", "")
                    quick   = f"{title} {snippet}"

                    if not _APTAMER_KW.search(quick):
                        continue

                    # Fetch full patent page for claims (where sequences appear)
                    full_text = self._fetch_patent_page(pid)
                    text = full_text if full_text else quick

                    url = _PATENT_URL.format(patent_id=pid)
                    from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract
                    target = _guess_target_from_abstract(text)

                    recs = self._extract_records_from_text(
                        text=text,
                        target_name=target,
                        source_url=url,
                        doi=pid,
                        confidence="extracted",
                    )
                    all_records.extend(recs)
                    if len(all_records) >= max_results:
                        break

                if len(all_records) >= max_results:
                    break

        log.info("GooglePatentsAdapter: %d records", len(all_records))
        return all_records[:max_results]
