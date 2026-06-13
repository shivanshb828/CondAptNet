"""
WIPO PatentScope adapter.

WIPO PatentScope provides international (PCT) patent data. The REST API
endpoint allows keyword search and returns patent metadata + abstracts.

API endpoint:
  GET https://patentscope.wipo.int/search/en/query.jsf (web interface)

For programmatic access, WIPO provides an API at:
  GET https://api.lens.org/patent/search  (Lens.org wraps WIPO data)

Because WIPO's own REST API requires account registration and has limited
public documentation, this adapter uses the WIPO PatentScope search URL
and scrapes the results as structured JSON from their search endpoint.

Alternative approach used here: scrape
  POST https://patentscope.wipo.int/search/en/search.jsf
with the form-encoded query, using BeautifulSoup to parse results.

Rate limit: 1 req/s (very conservative — WIPO has no published limit).

If WIPO access is unavailable or returns empty, the adapter silently
returns []. The main pipeline has 9 other sources.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.parsers.text_parser import parse_html
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_SEARCH_URL = "https://patentscope.wipo.int/search/en/query.jsf"
_RESULT_URL = "https://patentscope.wipo.int/search/en/result.jsf"

_WIPO_QUERIES = [
    "aptamer SELEX DNA binding protein",
    "ssDNA aptamer dissociation constant",
    "DNA aptamer nucleic acid binding affinity",
]

_APTAMER_KW = re.compile(r"\baptamer\b|\bSELEX\b|\bssDNA\b", re.IGNORECASE)

# WIPO result page: patent titles are in <span class="trans-title"> or similar
_TITLE_RE    = re.compile(r'class="[^"]*title[^"]*"[^>]*>([^<]{5,200})<', re.IGNORECASE)
_ABSTRACT_RE = re.compile(r'class="[^"]*abstract[^"]*"[^>]*>([^<]{10,2000})<', re.IGNORECASE)
_WO_NUM_RE   = re.compile(r'\bWO\s*\d{4}/\d+\b', re.IGNORECASE)


class WIPOAdapter(BaseAdapter):
    """
    Scrape WIPO PatentScope for international aptamer patents.
    """

    source_name = "patents_wipo"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        # WIPO rejects requests without a browser-like User-Agent
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; CondAptNet/1.0; aptamer ML research; "
            "coolshivansh7@gmail.com)"
        )

    def _search(self, query: str, rows: int = 25) -> Optional[str]:
        """POST a search and return the HTML results page."""
        # First GET to establish session cookies
        self._get(_SEARCH_URL)

        resp = self._post(
            _RESULT_URL,
            data={
                "query":         query,
                "office":        "",
                "dateRangeField": "PD",
                "rows":          str(rows),
                "sortOption":    "Relevance",
            },
        )
        if resp is None:
            return None
        return resp.text

    def _parse_results(self, html: Optional[str]) -> list[dict]:
        """Extract patent numbers + text snippets from WIPO search HTML."""
        if not html:
            return []
        doc    = parse_html(html, source_path="wipo_results")
        text   = doc.text
        titles = _TITLE_RE.findall(html)
        wo_nums = _WO_NUM_RE.findall(text)
        abstracts = _ABSTRACT_RE.findall(html)

        items: list[dict] = []
        for i, wo in enumerate(wo_nums):
            items.append({
                "wo_number": wo.replace(" ", "").upper(),
                "title":     titles[i] if i < len(titles) else "",
                "abstract":  abstracts[i] if i < len(abstracts) else "",
            })
        return items

    def run(self, max_results: int = 500) -> list[dict]:
        all_records: list[dict] = []
        seen_ids: set[str] = set()

        for query in _WIPO_QUERIES:
            html  = self._search(query)
            items = self._parse_results(html)

            for item in items:
                wo = item.get("wo_number", "")
                if not wo or wo in seen_ids:
                    continue
                seen_ids.add(wo)

                text = f"{item.get('title','')} {item.get('abstract','')}".strip()
                if not _APTAMER_KW.search(text):
                    continue

                url = f"https://patentscope.wipo.int/search/en/detail.jsf?docId={wo}"

                from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract
                target = _guess_target_from_abstract(text)

                recs = self._extract_records_from_text(
                    text=text,
                    target_name=target,
                    source_url=url,
                    doi=wo,
                    confidence="extracted",
                )
                all_records.extend(recs)
                if len(all_records) >= max_results:
                    break

        log.info("WIPOAdapter: %d records", len(all_records))
        return all_records[:max_results]
