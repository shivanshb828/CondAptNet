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
  Token:    POST https://ops.epo.org/3.2/auth/accesstoken
  Search:   GET  https://ops.epo.org/3.2/rest-services/published-data/search/biblio
                   ?q={cql_query}&Range=1-25
            (uses /biblio constituent to return titles inline)
  Abstract: GET  https://ops.epo.org/3.2/rest-services/published-data/
                   publication/epodoc/{docnum}/abstract

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

_TOKEN_URL      = "https://ops.epo.org/3.2/auth/accesstoken"
_SEARCH_URL     = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
_ABSTRACT_URL   = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{}/abstract"

_CQL_QUERIES = [
    'txt="aptamer" AND txt="SELEX" AND txt="binding"',
    'txt="DNA aptamer" AND txt="dissociation constant"',
    'txt="ssDNA" AND txt="SELEX" AND txt="protein"',
    'txt="aptamer" AND txt="Kd" AND txt="nM"',
]

# /search/biblio returns abstract inline — no per-patent calls needed.
# Each page is 1 API call, so 10 pages × 4 queries = 40 calls maximum.
_MAX_PAGES_PER_QUERY = 10


def _extract_title(inv_title) -> str:
    """Pull English title from EPO invention-title field (list or dict)."""
    if isinstance(inv_title, list):
        for t in inv_title:
            if isinstance(t, dict) and t.get("@lang", "").lower() in ("en", ""):
                return t.get("$", "")
        if inv_title:
            first = inv_title[0]
            return first.get("$", "") if isinstance(first, dict) else str(first)
    elif isinstance(inv_title, dict):
        return inv_title.get("$", "")
    return ""


def _extract_abstract(doc: dict) -> str:
    """
    Pull abstract text from an EPO exchange-document dict.

    EPO returns abstract inline in /search/biblio as:
      {"@lang": "en", "p": {"$": "text..."}}  (single paragraph)
      {"@lang": "en", "p": [{"$": "p1"}, {"$": "p2"}]}  (multiple)
    or as a list of the above for multi-language abstracts.
    """
    abstract_field = doc.get("abstract")
    if not abstract_field:
        return ""
    # Normalize to single-language dict (prefer English)
    if isinstance(abstract_field, list):
        en = next((a for a in abstract_field
                   if isinstance(a, dict) and a.get("@lang", "").lower() in ("en", "")), None)
        abstract_field = en or (abstract_field[0] if abstract_field else {})
    if not isinstance(abstract_field, dict):
        return str(abstract_field)
    paragraphs = abstract_field.get("p", [])
    if isinstance(paragraphs, str):
        return paragraphs
    if isinstance(paragraphs, dict):
        return paragraphs.get("$", "")
    return " ".join(p.get("$", "") for p in paragraphs if isinstance(p, dict))


class EPOAdapter(BaseAdapter):
    """
    Search EPO OPS for European aptamer patents.

    Two-step per patent:
      1. /search/biblio → doc number + title (inline, no extra call)
      2. /publication/epodoc/{num}/abstract → abstract text

    The abstract frequently contains explicit aptamer sequences (ACGT…) and
    Kd values, making it the best source for extraction.
    """

    source_name = "patents_epo"
    source_type = "patent"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        super().__init__(prov_logger)
        self._token: Optional[str] = None
        self._token_expiry: float  = 0.0

    # ── OAuth2 ────────────────────────────────────────────────────────────────

    def _refresh_token(self) -> bool:
        """Obtain a new OAuth2 token. Returns True on success.

        EPO OPS uses non-standard OAuth2: credentials go in the Authorization
        header as Basic base64(key:secret), NOT as form-body client_id/secret.
        """
        if not cfg.EPO_CLIENT_KEY or not cfg.EPO_CLIENT_SECRET:
            return False
        import base64
        creds = base64.b64encode(
            f"{cfg.EPO_CLIENT_KEY}:{cfg.EPO_CLIENT_SECRET}".encode()
        ).decode()
        self._limiter.wait()
        try:
            resp = self._session.post(
                _TOKEN_URL,
                headers={
                    "Authorization":  f"Basic {creds}",
                    "Content-Type":   "application/x-www-form-urlencoded",
                },
                data="grant_type=client_credentials",
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
        if not cfg.EPO_CLIENT_KEY:
            return False
        if self._token is None or time.monotonic() > self._token_expiry:
            return self._refresh_token()
        return True

    # ── Search ───────────────────────────────────────────────────────────────

    def _search(self, cql: str, start: int = 1, count: int = 25) -> Optional[str]:
        """GET /search/biblio — returns titles inline in JSON."""
        if start > 2000:
            return None
        resp = self._get(
            _SEARCH_URL,
            params={"q": cql, "Range": f"{start}-{min(start + count - 1, 2000)}"},
            headers={"Accept": "application/json"},
        )
        if resp is None:
            return None
        if resp.status_code == 403:
            log.debug("EPO 403 on Range=%d — end of results for this query", start)
            return None
        return resp.text

    def _parse_search_results(self, json_text: Optional[str]) -> list[dict]:
        """
        Parse /search/biblio JSON response.
        Returns list of {doc_number, title, abstract}.

        EPO OPS structure (confirmed live):
          search-result.exchange-documents → LIST of {"exchange-document": {doc}}
          doc.bibliographic-data.invention-title → dict OR list of lang dicts
          doc.abstract → dict OR list of lang dicts (inline — no per-patent call needed)
        """
        if not json_text:
            return []
        import json
        try:
            data = json.loads(json_text)
        except Exception:
            return []

        docs: list[dict] = []
        try:
            search_result = (
                data["ops:world-patent-data"]["ops:biblio-search"]["ops:search-result"]
            )

            if "exchange-documents" in search_result:
                # exchange-documents is a LIST; each item is {"exchange-document": doc}
                items = search_result["exchange-documents"]
                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    doc = item.get("exchange-document", {})
                    # doc may itself be a list in edge cases
                    doc_list = doc if isinstance(doc, list) else [doc]

                    for d in doc_list:
                        country = d.get("@country", "")
                        doc_num = d.get("@doc-number", "")
                        kind    = d.get("@kind", "")
                        epodoc  = f"{country}{doc_num}{kind}" if country and doc_num else doc_num

                        biblio = d.get("bibliographic-data", {})
                        title  = _extract_title(biblio.get("invention-title") or d.get("invention-title"))
                        # Abstract is returned inline by /search/biblio
                        abstract = _extract_abstract(d)

                        if epodoc:
                            docs.append({"doc_number": epodoc, "title": title, "abstract": abstract})

            elif "ops:publication-reference" in search_result:
                # Fallback: old /search endpoint — no titles or abstracts
                refs = search_result["ops:publication-reference"]
                if isinstance(refs, dict):
                    refs = [refs]
                for ref in refs:
                    doc_id = ref.get("@document-id", {})
                    num    = doc_id.get("doc-number", {}).get("$", "")
                    if num:
                        docs.append({"doc_number": num, "title": "", "abstract": ""})

        except (KeyError, TypeError):
            pass
        return docs

    # ── Abstract fetch ───────────────────────────────────────────────────────

    def _fetch_abstract(self, doc_number: str) -> str:
        """
        Fetch abstract text for one patent via the individual endpoint.
        Returns empty string on any error.
        """
        url  = _ABSTRACT_URL.format(doc_number)
        resp = self._get(url, headers={"Accept": "application/json"})
        if resp is None:
            return ""
        try:
            data = resp.json()
            doc  = data["ops:world-patent-data"]["exchange-documents"]["exchange-document"]
            if isinstance(doc, list):
                doc = doc[0]
            return _extract_abstract(doc)
        except (KeyError, TypeError, AttributeError, ValueError):
            return ""

    # ── Main ─────────────────────────────────────────────────────────────────

    def run(self, max_results: int = 500) -> list[dict]:
        if not self._ensure_auth():
            log.warning("EPO_CLIENT_KEY / EPO_CLIENT_SECRET not set; skipping EPO adapter")
            return []

        from scripts.data.scraper.adapters.pubmed_pmc import _guess_target_from_abstract

        all_records: list[dict] = []
        seen_ids:    set[str]   = set()

        for cql in _CQL_QUERIES:
            start = 1
            pages_this_query = 0

            while len(all_records) < max_results and pages_this_query < _MAX_PAGES_PER_QUERY:
                raw   = self._search(cql, start=start, count=25)
                items = self._parse_search_results(raw)
                if not items:
                    break
                pages_this_query += 1

                for item in items:
                    doc_num = item.get("doc_number", "")
                    if not doc_num or doc_num in seen_ids:
                        continue
                    seen_ids.add(doc_num)

                    title    = item.get("title", "")
                    abstract = item.get("abstract", "")   # inline from /search/biblio
                    text     = f"{title}\n{abstract}".strip()

                    if not text:
                        continue

                    url    = f"https://worldwide.espacenet.com/patent/search/family/{doc_num}"
                    target = _guess_target_from_abstract(text)
                    recs   = self._extract_records_from_text(
                        text=text, target_name=target, source_url=url,
                        doi=doc_num, confidence="extracted",
                    )
                    all_records.extend(recs)
                    if len(all_records) >= max_results:
                        break

                start += 25
                if len(items) < 25:
                    break

            if len(all_records) >= max_results:
                break

        log.info("EPOAdapter: %d records", len(all_records))
        return all_records[:max_results]
