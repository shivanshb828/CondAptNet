"""
WIPO PCT patent adapter — via Lens.org patent API.

The WIPO PatentScope web interface (patentscope.wipo.int) changed its URL
structure and times out unreliably. Lens.org already indexes all PCT (WO)
international patents from WIPO, so we query Lens with jurisdiction="WO"
to get WIPO coverage without hitting patentscope.wipo.int directly.

Requires: LENS_API_TOKEN in environment (same token as the Lens adapter).
Without the token, returns [] with a warning.

Rate limit: shared 10 req/min Lens limit. Runs after the main Lens adapter,
so uses the same token bucket — rate limiting is handled in base.BaseAdapter.
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

_PATENT_URL = "https://api.lens.org/patent/search"

_PCT_QUERIES = [
    {"bool": {"must": [
        {"query_string": {"query": "aptamer SELEX DNA binding", "default_operator": "AND"}},
        {"term": {"jurisdiction": "WO"}},
    ]}},
    {"bool": {"must": [
        {"query_string": {"query": "ssDNA aptamer dissociation constant protein", "default_operator": "AND"}},
        {"term": {"jurisdiction": "WO"}},
    ]}},
]


class WIPOAdapter(BaseAdapter):
    """
    Pull PCT (WO) aptamer patents from Lens.org as a WIPO data source.
    Falls back cleanly if LENS_API_TOKEN is missing.
    """

    source_name = "patents_wipo"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        if cfg.LENS_API_TOKEN:
            self._session.headers["Authorization"] = f"Bearer {cfg.LENS_API_TOKEN}"

    def _post_search(self, payload: dict) -> dict:
        resp = self._post(_PATENT_URL, json=payload)
        if resp is None:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}

    def run(self, max_results: int = 500) -> list[dict]:
        if not cfg.LENS_API_TOKEN:
            log.warning("LENS_API_TOKEN not set; skipping WIPO (Lens PCT) adapter")
            return []

        from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract

        all_records: list[dict] = []
        seen_ids:    set[str]   = set()

        for query in _PCT_QUERIES:
            from_ = 0
            size  = min(100, max_results)
            while from_ < max_results:
                payload = {
                    "query":   query,
                    "from":    from_,
                    "size":    size,
                    "include": ["lens_id", "abstract", "biblio", "claims", "doc_number", "jurisdiction"],
                    "sort":    [{"_score": "desc"}],
                }
                data = self._post_search(payload)
                hits = data.get("data", [])
                if not hits:
                    break

                for hit in hits:
                    lid = hit.get("lens_id", "") or hit.get("doc_number", "")
                    if lid in seen_ids:
                        continue
                    seen_ids.add(lid)

                    abstract   = hit.get("abstract") or ""
                    biblio     = hit.get("biblio") or {}
                    inv_titles = biblio.get("invention_title", [])
                    title = ""
                    if isinstance(inv_titles, list) and inv_titles:
                        title = (inv_titles[0].get("text", "")
                                 if isinstance(inv_titles[0], dict) else str(inv_titles[0]))
                    claims_raw  = hit.get("claims") or []
                    claims_text = ""
                    if isinstance(claims_raw, list):
                        for cg in claims_raw:
                            if isinstance(cg, dict):
                                for c in cg.get("claims", []):
                                    if isinstance(c, dict):
                                        claims_text += " ".join(c.get("claim_text", [])) + " "
                    text = f"{title}\n{abstract}\n{claims_text}".strip()
                    doc  = hit.get("doc_number", "")
                    url  = f"https://lens.org/lens/patent/WO_{doc}" if doc else f"https://lens.org/{lid}"

                    target = _guess_target_from_abstract(text)
                    recs   = self._extract_records_from_text(
                        text=text, target_name=target, source_url=url,
                        doi=lid, confidence="extracted",
                        extra_fields={"source_type": "patent"},
                    )
                    all_records.extend(recs)
                    if len(all_records) >= max_results:
                        break

                from_ += len(hits)
                if len(hits) < size:
                    break

            if len(all_records) >= max_results:
                break

        log.info("WIPOAdapter (via Lens PCT): %d records", len(all_records))
        return all_records[:max_results]
