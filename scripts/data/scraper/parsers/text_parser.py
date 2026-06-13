"""
Plain text and HTML parser.

Handles two input types:
  1. HTML — uses BeautifulSoup4 to strip tags, extract tables, and clean text.
  2. Plain text — minimal cleaning (whitespace normalisation, encoding fix).

Returns a ParsedText with the cleaned text body, any HTML tables found,
and basic metadata.

Design:
  - HTML tables are extracted as lists-of-lists (same structure as pdf_parser)
    so the same downstream table-to-text utilities work for all document types.
  - Script, style, nav, header, footer elements are stripped before text
    extraction — they add noise, not aptamer data.
  - No character encoding guessing: callers must provide str or decoded bytes.
    If bytes are passed, UTF-8 with replacement is attempted.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup, Tag
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# Tags whose content should be removed entirely before text extraction
_NOISE_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}

# Multiple-whitespace normalisation
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedText:
    """
    Result of parsing an HTML or plain-text document.

    Attributes:
        text        Cleaned body text.
        tables      HTML tables as list[list[list[str]]] (rows x cells).
        title       Page <title> (empty string for plain text).
        source_path File path or "<string>" / "<bytes>".
        is_html     True if the source was parsed as HTML.
        parse_ok    False if parsing failed.
        error       Error message when parse_ok is False.
    """
    text:        str                       = ""
    tables:      list[list[list[str]]]     = field(default_factory=list)
    title:       str                       = ""
    source_path: str                       = ""
    is_html:     bool                      = False
    parse_ok:    bool                      = True
    error:       str                       = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Normalise whitespace in a text block."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_html_tables(soup: "BeautifulSoup") -> list[list[list[str]]]:
    """Extract all <table> elements as lists-of-lists."""
    tables: list[list[list[str]]] = []
    for tbl in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in tbl.find_all("tr"):
            cells = [
                cell.get_text(separator=" ", strip=True)
                for cell in tr.find_all(["td", "th"])
            ]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _html_to_text(soup: "BeautifulSoup") -> str:
    """Strip HTML to readable text, preserving line breaks at block elements."""
    # Remove noise elements in-place
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Replace <br> with newline before get_text
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Block-level elements: insert newlines around them
    for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                                  "li", "tr", "section", "article"]):
        block.insert_before("\n")
        block.insert_after("\n")

    return soup.get_text(separator="")


# ── Public API ────────────────────────────────────────────────────────────────

def parse_html(
    source:      Union[str, bytes],
    source_path: str = "<string>",
) -> ParsedText:
    """
    Parse an HTML document and return a ParsedText.

    Args:
        source:      HTML as string or bytes (decoded with utf-8/replace).
        source_path: Label for the source (URL or file path, for provenance).

    Returns:
        ParsedText. parse_ok=False with error message on failure.
    """
    if not _HAS_BS4:
        return ParsedText(
            parse_ok=False,
            error="beautifulsoup4 is not installed. Run: pip install beautifulsoup4",
            source_path=source_path,
        )

    if isinstance(source, bytes):
        source = source.decode("utf-8", errors="replace")

    try:
        soup = BeautifulSoup(source, "html.parser")
    except Exception as exc:
        log.warning("HTML parse failed (%s): %s", source_path, exc)
        return ParsedText(parse_ok=False, error=str(exc), source_path=source_path)

    title_el = soup.find("title")
    title    = title_el.get_text(strip=True) if title_el else ""

    tables = _extract_html_tables(soup)
    raw    = _html_to_text(soup)
    text   = _norm(raw)

    return ParsedText(
        text=text,
        tables=tables,
        title=title,
        source_path=source_path,
        is_html=True,
        parse_ok=True,
    )


def parse_plain_text(
    source:      Union[str, bytes],
    source_path: str = "<string>",
) -> ParsedText:
    """
    Parse a plain-text document and return a ParsedText.

    Args:
        source:      Text as string or bytes.
        source_path: Label for the source.

    Returns:
        ParsedText with no tables and empty title.
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8", errors="replace")

    try:
        text = _norm(source)
    except Exception as exc:
        return ParsedText(parse_ok=False, error=str(exc), source_path=source_path)

    return ParsedText(
        text=text,
        tables=[],
        title="",
        source_path=source_path,
        is_html=False,
        parse_ok=True,
    )


def parse_file(path: Union[str, Path]) -> ParsedText:
    """
    Auto-detect HTML vs plain text by file extension and parse accordingly.

    Recognises .html, .htm as HTML; everything else as plain text.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except Exception as exc:
        return ParsedText(parse_ok=False, error=str(exc), source_path=str(path))

    if path.suffix.lower() in (".html", ".htm"):
        return parse_html(raw, source_path=str(path))
    return parse_plain_text(raw, source_path=str(path))


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        doc = parse_file(sys.argv[1])
        print(f"Title:   {doc.title[:80]}")
        print(f"Tables:  {len(doc.tables)}")
        print(f"HTML:    {doc.is_html}")
        print(f"Preview: {doc.text[:500]}")
    else:
        _HTML = """
        <html>
        <head><title>Aptamer binding test</title></head>
        <body>
        <p>The Kd was 5.2 nM for aptamer ATCGATCGATCGATCGATCGATCG against thrombin.</p>
        <table>
          <tr><th>Sequence</th><th>Target</th><th>Kd</th></tr>
          <tr><td>ATCGATCGATCGATCGATCGATCG</td><td>thrombin</td><td>5.2 nM</td></tr>
        </table>
        </body>
        </html>
        """
        doc = parse_html(_HTML, source_path="test")
        assert doc.parse_ok
        assert "thrombin" in doc.text
        assert len(doc.tables) == 1
        assert doc.title == "Aptamer binding test"
        print("text_parser self-test passed.")
        print(f"  Title:   {doc.title}")
        print(f"  Tables:  {len(doc.tables)}")
        print(f"  First table row: {doc.tables[0][0]}")
