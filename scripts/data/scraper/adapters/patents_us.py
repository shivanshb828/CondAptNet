"""
USPTO PatentsView adapter.

PatentsView (patentsview.org) provides a REST JSON API for US patent data.
No API key required. Returns patent title, abstract, and claim text.

API endpoint:
  POST https://search.patentsview.org/api/v1/patent/
  Body: {
    "q": {"_text_any": {"patent_abstract": "aptamer SELEX"}},
    "f": ["patent_id","patent_title","patent_abstract","patent_date"],
    "o": {"per_page": 100, "page": 1}
  }

Rate limit: 2 req/s (conservative) — no published limit.

Patents are a rich source of aptamer sequences — inventors must disclose
sequences in claims, which are quoted verbatim (no paraphrasing).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_API_URL = "https://search.patentsview.org/api/v1/patent/"

_QUERIES = [
    {"_text_any": {"patent_abstract": "DNA aptamer SELEX binding"}},
    {"_text_any": {"patent_abstract": "aptamer dissociation constant Kd"}},
    {"_text_any": {"patent_abstract": "ssDNA aptamer protein target"}},
    {"_text_any": {"patent_abstract": "in vitro selection oligonucleotide binding"}},
]

_FIELDS = [
    "patent_id",
    "patent_title",
    "patent_abstract",
    "patent_date",
    "patent_number",
]


class PatentsUSAdapter(BaseAdapter):
    """
    Query PatentsView for US aptamer patents and extract sequences from claims.
    """

    source_name = "patents_us"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)

    def _fetch_page(self, query: dict, page: int = 1, per_page: int = 100) -> dict:
        payload = {
            "q": query,
            "f": _FIELDS,
            "o": {"per_page": per_page, "page": page},
        }
        resp = self._post(_API_URL, json=payload)
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def run(self, max_results: int = 500) -> list[dict]:
        all_records: list[dict] = []
        seen_ids: set[str] = set()

        for query in _QUERIES:
            page = 1
            while len(all_records) < max_results:
                data    = self._fetch_page(query, page=page)
                patents = data.get("patents", [])
                if not patents:
                    break

                for pat in patents:
                    pid = pat.get("patent_id") or pat.get("patent_number", "")
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    title    = pat.get("patent_title", "") or ""
                    abstract = pat.get("patent_abstract", "") or ""
                    text     = f"{title}\n{abstract}"

                    if not text.strip():
                        continue

                    url = f"https://patents.google.com/patent/US{pid}"
                    doi = f"US{pid}"   # patent number as DOI proxy

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

                # Pagination: check if more pages exist
                total    = data.get("total_patent_count", 0)
                per_page = data.get("count", 100)
                if page * per_page >= int(total or 0):
                    break
                page += 1

        log.info("PatentsUSAdapter: %d records", len(all_records))
        return all_records[:max_results]
