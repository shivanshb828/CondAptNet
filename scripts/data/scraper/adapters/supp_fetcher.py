"""
Supplementary-file fetcher for PMC open-access papers.

PMC stores supplementary materials at:
  https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}/bin/{filename}

The xml_parser.ParsedXML.supplementary_urls list contains xlink:href values
from <supplementary-material> elements.  These can be:
  - Bare filenames:   "pone.0086729.s001.xlsx"
  - Relative paths:   "supplementary/table1.csv"
  - Full HTTPS URLs:  "https://..."

This module:
  1. Resolves each href against a base URL
  2. Filters to supported types (Excel / CSV / PDF / plain text)
  3. Downloads the file (respects the caller's rate-limiter and session)
  4. Parses with the appropriate parser
  5. Returns a flat list of text strings, one per successful file,
     ready to be passed to BaseAdapter._extract_records_from_text()

Usage:
    from scripts.data.scraper.adapters.supp_fetcher import fetch_supplementary_texts

    texts = fetch_supplementary_texts(
        supp_urls  = doc.supplementary_urls,
        base_url   = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/",
        session    = self._session,
        limiter    = self._limiter,
    )
    for text in texts:
        recs = self._extract_records_from_text(text, ...)
"""

from __future__ import annotations

import logging
import os.path
import urllib.parse
from typing import Optional

import requests

log = logging.getLogger(__name__)

# File types worth downloading
_SUPPORTED_EXTS = {".xlsx", ".xls", ".csv", ".tsv", ".pdf", ".txt"}

# Skip anything obviously not a data file
_SKIP_EXTS = {".zip", ".gz", ".tar", ".docx", ".doc", ".pptx", ".ppt",
              ".mp4", ".avi", ".png", ".jpg", ".jpeg", ".gif", ".tif",
              ".tiff", ".bmp", ".svg"}

_DEFAULT_MAX_BYTES = 20 * 1024 * 1024   # 20 MB per file
_DEFAULT_MAX_FILES = 8                   # files per paper


def _resolve_url(href: str, base_url: str) -> str:
    """
    Resolve a supplementary href to an absolute URL.

    If href is already absolute, return it unchanged.
    Otherwise join against base_url.
    """
    if href.startswith("http://") or href.startswith("https://"):
        return href
    # urllib.parse.urljoin handles both bare filenames and relative paths
    return urllib.parse.urljoin(base_url, href)


def _file_ext(url: str) -> str:
    """Return lowercase file extension from URL path, e.g. '.xlsx'."""
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower()


def _download(
    url: str,
    session: requests.Session,
    limiter,
    max_bytes: int,
) -> Optional[bytes]:
    """
    Download a supplementary file.

    Returns raw bytes on success, None on any failure or if the file is
    too large (checked via Content-Length header before downloading body).
    """
    # HEAD first to check size without downloading
    try:
        limiter.wait()
        head = session.head(url, timeout=15, allow_redirects=True)
        cl = int(head.headers.get("Content-Length", 0))
        if cl > max_bytes:
            log.debug("Skipping %s — Content-Length %d > %d", url, cl, max_bytes)
            return None
    except Exception as exc:
        log.debug("HEAD failed for %s: %s", url, exc)
        # HEAD might not be supported; proceed to GET anyway

    try:
        limiter.wait()
        resp = session.get(url, timeout=30, stream=True, allow_redirects=True)
        resp.raise_for_status()

        # Guard against unexpectedly large files when Content-Length was absent
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(65_536):
            total += len(chunk)
            if total > max_bytes:
                log.debug("Aborting download of %s — exceeded %d bytes", url, max_bytes)
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    except requests.RequestException as exc:
        log.debug("GET failed for %s: %s", url, exc)
        return None


def _parse_to_text(raw: bytes, ext: str, url: str) -> Optional[str]:
    """
    Dispatch raw bytes to the appropriate parser and return text.

    Returns None when the file cannot be parsed or yields no useful text.
    """
    try:
        if ext in {".xlsx", ".xls"}:
            from scripts.data.scraper.parsers.excel_parser import parse_excel
            doc = parse_excel(raw)
            return doc.full_text if doc.parse_ok and doc.full_text.strip() else None

        if ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            from scripts.data.scraper.parsers.excel_parser import parse_csv
            doc = parse_csv(raw, sep=sep)
            return doc.full_text if doc.parse_ok and doc.full_text.strip() else None

        if ext == ".pdf":
            from scripts.data.scraper.parsers.pdf_parser import parse_pdf, tables_to_text
            doc = parse_pdf(raw)
            if not doc.parse_ok:
                return None
            # Prefer table text (sequences live in tables, not prose)
            table_text = tables_to_text(doc.tables) if doc.tables else ""
            body_text  = "\n".join(p.text for p in doc.pages)
            combined   = (table_text + "\n" + body_text).strip()
            return combined or None

        if ext == ".txt":
            from scripts.data.scraper.parsers.text_parser import parse_plain_text
            doc = parse_plain_text(raw.decode("utf-8", errors="replace"))
            return doc.text if doc.text.strip() else None

    except Exception as exc:
        log.debug("Parser error for %s (%s): %s", url, ext, exc)

    return None


def fetch_supplementary_texts(
    supp_urls:     list[str],
    base_url:      str,
    session:       requests.Session,
    limiter,
    max_file_bytes: int = _DEFAULT_MAX_BYTES,
    max_files:      int = _DEFAULT_MAX_FILES,
) -> list[str]:
    """
    Fetch and parse supplementary files, returning a list of text strings.

    Args:
        supp_urls:      hrefs from ParsedXML.supplementary_urls
        base_url:       Used to resolve relative hrefs — typically
                        "https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}/bin/"
        session:        Caller's requests.Session (already has User-Agent set)
        limiter:        Caller's rate-limiter (has a .wait() method)
        max_file_bytes: Skip files larger than this (default 20 MB)
        max_files:      Process at most this many files per paper (default 8)

    Returns:
        List of text strings (one per successfully parsed file).
        Empty list if nothing useful was found.
    """
    texts: list[str] = []
    processed = 0

    for href in supp_urls:
        if processed >= max_files:
            log.debug("Reached max_files=%d, stopping supplementary fetch", max_files)
            break

        ext = _file_ext(href)
        if ext in _SKIP_EXTS:
            log.debug("Skipping unsupported type: %s", href)
            continue
        if ext not in _SUPPORTED_EXTS:
            log.debug("Unknown extension %r, skipping: %s", ext, href)
            continue

        url = _resolve_url(href, base_url)
        log.debug("Fetching supplementary file: %s", url)

        raw = _download(url, session, limiter, max_file_bytes)
        if raw is None:
            continue

        try:
            text = _parse_to_text(raw, ext, url)
        except Exception as exc:
            log.debug("Parse error for %s: %s", url, exc)
            continue
        if text:
            log.info("Supplementary file parsed: %s (%d chars)", url, len(text))
            texts.append(text)
            processed += 1
        else:
            log.debug("No usable text from supplementary file: %s", url)

    return texts
