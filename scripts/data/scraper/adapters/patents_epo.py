"""
EPO OPS (Open Patent Services) adapter.

EPO OPS provides programmatic access to the European Patent Office database.
It uses OAuth2 client credentials flow:
  1. POST to token endpoint with client_key + client_secret
  2. Attach Bearer token to all subsequent requests
  3. Token expires in 20 minutes — refresh when expired

Credentials (required):
  EPO_CLIENT_KEY    — from https://developers.epo.org
  EPO_CLIENT_SECRET — from https://developers.epo.org

Without credentials, run() returns [] with a warning (not a crash).

API:
  Token:  POST https://ops.epo.org/3.2/auth/accesstoken
  Search: GET  https://ops.epo.org/3.2/rest-services/published-data/search
            ?q={cql_query}&Range=1-25

Response: XML (OPS format). Parsed with lxml.

Rate limit: 2 req/s (EPO free tier: 4 GB/week).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper import config as cfg
from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

_TOKEN_URL  = "https://ops.epo.org/3.2/auth/accesstoken"
_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"
_FETCH_URL  = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{epodoc}/full-cycle"

_CQL_QUERIES = [
    'txt="aptamer" AND txt="SELEX" AND txt="binding"',
    'txt="DNA aptamer" AND txt="dissociation constant"',
    'txt="ssDNA" AND txt="SELEX" AND txt="protein"',
    'txt="aptamer" AND txt="Kd" AND txt="nM"',
]


class EPOAdapter(BaseAdapter):
    """
    Search EPO OPS for European aptamer patents.
    """

    source_name = "patents_epo"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        self._token: Optional[str] = None
        self._token_expiry: float  = 0.0

    # ── OAuth2 ────────────────────────────────────────────────────────────────

    def _refresh_token(self) -> bool:
        """Obtain a new OAuth2 token. Returns True on success."""
        if not cfg.EPO_CLIENT_KEY or not cfg.EPO_CLIENT_SECRET:
            return False
        self._limiter.wait()
        try:
            resp = self._session.post(
                _TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     cfg.EPO_CLIENT_KEY,
                    "client_secret": cfg.EPO_CLIENT_SECRET,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token        = data.get("access_token")
            expires_in         = int(data.get("expires_in", 1200))
            self._token_expiry = time.monotonic() + expires_in - 30
            self._session.headers["Authorization"] = f"Bearer {self._token}"
            return True
        except Exception as exc:
            log.warning("EPO token refresh failed: %s", exc)
            return False

    def _ensure_auth(self) -> bool:
        """Ensure we have a valid token. Returns False if credentials missing."""
        if not cfg.EPO_CLIENT_KEY:
            return False
        if self._token is None or time.monotonic() > self._token_expiry:
            return self._refresh_token()
        return True

    # ── Search + fetch ────────────────────────────────────────────────────────

    def _search(self, cql: str, start: int = 1, count: int = 25) -> Optional[str]:
        resp = self._get(
            _SEARCH_URL,
            params={"q": cql, "Range": f"{start}-{start+count-1}"},
            headers={"Accept": "application/json"},
        )
        return resp.text if resp else None

    def _parse_search_results(self, json_text: Optional[str]) -> list[dict]:
        """Parse OPS search response; returns list of {doc_number, title, abstract}."""
        if not json_text:
            return []
        import json
        try:
            data = json.loads(json_text)
        except Exception:
            return []
        docs: list[dict] = []
        try:
            results = (
                data["ops:world-patent-data"]["ops:biblio-search"]
                ["ops:search-result"]["ops:publication-reference"]
            )
            if isinstance(results, dict):
                results = [results]
            for ref in results:
                doc_id = ref.get("@document-id", {})
                number = doc_id.get("doc-number", {}).get("$", "")
                docs.append({"doc_number": number, "title": "", "abstract": ""})
        except (KeyError, TypeError):
            pass
        return docs

    def run(self, max_results: int = 500) -> list[dict]:
        if not self._ensure_auth():
            log.warning("EPO_CLIENT_KEY / EPO_CLIENT_SECRET not set; skipping EPO adapter")
            return []

        all_records: list[dict] = []
        seen_ids: set[str] = set()

        for cql in _CQL_QUERIES:
            start = 1
            while len(all_records) < max_results:
                raw   = self._search(cql, start=start, count=25)
                items = self._parse_search_results(raw)
                if not items:
                    break

                for item in items:
                    doc_num = item.get("doc_number", "")
                    if not doc_num or doc_num in seen_ids:
                        continue
                    seen_ids.add(doc_num)

                    text   = f"{item.get('title','')} {item.get('abstract','')}".strip()
                    url    = f"https://worldwide.espacenet.com/patent/search/family/{doc_num}"
                    doi    = doc_num

                    if not text:
                        continue

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

                start += 25
                if len(items) < 25:
                    break

        log.info("EPOAdapter: %d records", len(all_records))
        return all_records[:max_results]
