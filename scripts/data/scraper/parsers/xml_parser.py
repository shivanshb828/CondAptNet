"""
PMC NXML full-text parser.

Parses PubMed Central NXML (journal article XML) and returns structured text
extracted from:
  - Abstract
  - Body text (paragraphs, section titles)
  - Tables (header + data rows as tab-separated text)
  - Figure captions
  - Supplementary material references (URLs / filenames)

Uses lxml for parsing — robust against malformed XML via recover=True.

Design decisions:
  - We never parse XML with Python's stdlib xml.etree (not secure against
    malformed input). We use lxml with recover=True as the safe default.
  - No heuristic text filtering — all paragraphs are returned. The sequence
    and Kd extractors filter what's relevant.
  - Supplementary material URLs are extracted separately because they often
    contain supplement PDFs / Excel files that hold the actual aptamer tables.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

try:
    from lxml import etree
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

# PMC NXML namespace (may or may not be present)
_NS = {"xlink": "http://www.w3.org/1999/xlink"}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedTable:
    label:   str              # e.g. "Table 1" or ""
    caption: str
    rows:    list[list[str]]  # header row(s) + data rows


@dataclass
class ParsedXML:
    """
    Structured content extracted from a PMC NXML article.

    Attributes:
        title               Article title.
        abstract            Abstract text (joined paragraphs).
        body_text           All body paragraphs joined by newlines.
        tables              List of ParsedTable objects.
        figure_captions     List of figure caption strings.
        supplementary_urls  URLs/hrefs from <supplementary-material> elements.
        full_text           Concatenation of all text (abstract + body + tables
                            + captions) — ready for regex extractors.
        parse_ok            False if the document could not be parsed.
        error               Error message when parse_ok is False.
    """
    title:              str  = ""
    abstract:           str  = ""
    body_text:          str  = ""
    tables:             list[ParsedTable] = field(default_factory=list)
    figure_captions:    list[str]         = field(default_factory=list)
    supplementary_urls: list[str]         = field(default_factory=list)
    full_text:          str               = ""
    doi:                str               = ""
    parse_ok:           bool              = True
    error:              str               = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text(elem) -> str:
    """Recursively collect all text under an lxml element."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p.strip() for p in parts if p.strip())


def _extract_doi(root) -> str:
    """Extract DOI from <article-id pub-id-type='doi'>."""
    for el in root.findall(".//article-id"):
        if el.get("pub-id-type", "").lower() == "doi":
            doi = _text(el).strip()
            if doi:
                return doi
    return ""


def _extract_title(root) -> str:
    for tag in ("article-title", "title"):
        el = root.find(f".//{tag}")
        if el is not None:
            return _text(el)
    return ""


def _extract_abstract(root) -> str:
    parts: list[str] = []
    for ab in root.findall(".//abstract"):
        for p in ab.findall(".//p"):
            t = _text(p)
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_body(root) -> str:
    parts: list[str] = []
    body = root.find(".//body")
    if body is None:
        return ""
    for el in body.iter():
        if el.tag in ("p", "title"):
            t = _text(el)
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_tables(root) -> list[ParsedTable]:
    tables: list[ParsedTable] = []

    for tbl in root.findall(".//table-wrap"):
        label_el   = tbl.find("label")
        caption_el = tbl.find("caption")
        label   = _text(label_el)   if label_el   is not None else ""
        caption = _text(caption_el) if caption_el is not None else ""

        rows: list[list[str]] = []
        for tr in tbl.findall(".//tr"):
            cells = [_text(td) for td in tr.findall(".//*") if td.tag in ("td", "th")]
            if cells:
                rows.append(cells)

        tables.append(ParsedTable(label=label, caption=caption, rows=rows))

    return tables


def _extract_figure_captions(root) -> list[str]:
    caps: list[str] = []
    for fig in root.findall(".//fig"):
        cap_el = fig.find(".//caption")
        if cap_el is not None:
            t = _text(cap_el)
            if t:
                caps.append(t)
    return caps


def _extract_supplementary_urls(root) -> list[str]:
    urls: list[str] = []
    for sup in root.findall(".//supplementary-material"):
        # xlink:href is the canonical attribute for URLs in NXML
        href = sup.get("{http://www.w3.org/1999/xlink}href", "")
        if not href:
            href = sup.get("href", "")
        if href:
            urls.append(href)
    return urls


def _tables_to_text(tables: list[ParsedTable]) -> str:
    parts: list[str] = []
    for t in tables:
        header = f"[{t.label}] {t.caption}" if (t.label or t.caption) else "[Table]"
        parts.append(header)
        for row in t.rows:
            parts.append("\t".join(row))
        parts.append("")
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_nxml(
    source: Union[str, Path, bytes],
) -> ParsedXML:
    """
    Parse a PMC NXML file and return a ParsedXML.

    Args:
        source: File path (str or Path) or raw XML bytes.

    Returns:
        ParsedXML. parse_ok=False with error message on failure.
    """
    if not _HAS_LXML:
        return ParsedXML(
            parse_ok=False,
            error="lxml is not installed. Run: pip install lxml",
        )

    try:
        parser = etree.XMLParser(recover=True, no_network=True)

        if isinstance(source, bytes):
            root = etree.fromstring(source, parser=parser)
        else:
            root = etree.parse(str(source), parser=parser).getroot()

    except Exception as exc:
        log.warning("NXML parse failed: %s", exc)
        return ParsedXML(parse_ok=False, error=str(exc))

    title    = _extract_title(root)
    abstract = _extract_abstract(root)
    body     = _extract_body(root)
    tables   = _extract_tables(root)
    captions = _extract_figure_captions(root)
    sup_urls = _extract_supplementary_urls(root)
    doi      = _extract_doi(root)

    table_text   = _tables_to_text(tables)
    caption_text = "\n".join(captions)

    full_text = "\n\n".join(
        part for part in [title, abstract, body, table_text, caption_text]
        if part.strip()
    )

    return ParsedXML(
        title=title,
        abstract=abstract,
        body_text=body,
        tables=tables,
        figure_captions=captions,
        supplementary_urls=sup_urls,
        full_text=full_text,
        doi=doi,
        parse_ok=True,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python xml_parser.py <path_to_nxml>")
        sys.exit(0)
    doc = parse_nxml(sys.argv[1])
    print(f"Title:       {doc.title[:80]}")
    print(f"Tables:      {len(doc.tables)}")
    print(f"Sup URLs:    {doc.supplementary_urls}")
    print(f"Full text preview (first 500 chars):\n{doc.full_text[:500]}")
