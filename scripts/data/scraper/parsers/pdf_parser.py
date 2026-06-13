"""
PDF parser for aptamer supplement files and preprint PDFs.

Uses pdfplumber for text and table extraction. Returns a ParsedDocument
with page-by-page text, all tables as lists-of-lists, and metadata.

Design decisions:
  - Text is extracted page-by-page; page numbers are preserved so byte-offset
    approximations can be computed by callers.
  - Tables are extracted as raw cell strings (no type coercion). The sequence
    extractor decides what is a sequence.
  - Multi-column PDF layouts are handled by pdfplumber's bbox-cropping strategy:
    we try full-page first; if it looks garbled (many interleaved single chars)
    we fall back to column detection.
  - On import error (pdfplumber not installed), the module raises ImportError
    immediately so the adapter can skip PDF parsing gracefully.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedPage:
    page_number: int        # 1-indexed
    text:        str        # full extracted text (spaces/newlines preserved)
    tables:      list[list[list[str]]]   # list of tables; each table is rows of cells
    char_offset: int        # cumulative character offset at start of this page


@dataclass
class ParsedDocument:
    """
    Result of parsing a PDF file.

    Attributes:
        pages       List of ParsedPage objects (one per PDF page).
        full_text   Concatenated text of all pages (newline between pages).
        all_tables  Flat list of all tables across all pages.
        n_pages     Total page count.
        source_path File path or "<bytes>" if parsed from bytes.
        parse_ok    False if the file could not be opened at all.
        error       Error message when parse_ok is False.
    """
    pages:       list[ParsedPage] = field(default_factory=list)
    full_text:   str              = ""
    all_tables:  list[list[list[str]]] = field(default_factory=list)
    n_pages:     int              = 0
    source_path: str              = ""
    parse_ok:    bool             = True
    error:       str              = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _looks_garbled(text: str, threshold: float = 0.3) -> bool:
    """
    Return True if text looks like interleaved columns (garbled extraction).
    Heuristic: unusually high ratio of single-character words.
    """
    words = text.split()
    if len(words) < 20:
        return False
    single = sum(1 for w in words if len(w) == 1)
    return (single / len(words)) > threshold


def _extract_page_text(page) -> str:
    """
    Extract text from a pdfplumber page, with multi-column fallback.
    """
    text = page.extract_text() or ""
    if _looks_garbled(text):
        # Try left/right column split at page midpoint
        mid = page.width / 2
        left  = page.within_bbox((0,            0, mid,          page.height))
        right = page.within_bbox((mid,           0, page.width,  page.height))
        left_text  = left.extract_text()  or ""
        right_text = right.extract_text() or ""
        combined = left_text + "\n" + right_text
        if not _looks_garbled(combined):
            text = combined
    return text


def _clean_cells(table: list[list]) -> list[list[str]]:
    """Normalise table cells to strings, replacing None with empty string."""
    return [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(
    source: Union[str, Path, bytes],
    max_pages: Optional[int] = None,
) -> ParsedDocument:
    """
    Parse a PDF and return a ParsedDocument.

    Args:
        source:     File path (str or Path) or raw bytes.
        max_pages:  Cap on pages to parse (None = all). Useful for large PDFs.

    Returns:
        ParsedDocument. parse_ok=False with error message if the file cannot
        be opened or pdfplumber is not installed.
    """
    if not _HAS_PDFPLUMBER:
        return ParsedDocument(
            parse_ok=False,
            error="pdfplumber is not installed. Run: pip install pdfplumber",
        )

    source_path = str(source) if not isinstance(source, bytes) else "<bytes>"

    try:
        if isinstance(source, bytes):
            pdf_file = pdfplumber.open(io.BytesIO(source))
        else:
            pdf_file = pdfplumber.open(source)
    except Exception as exc:
        log.warning("PDF open failed (%s): %s", source_path, exc)
        return ParsedDocument(parse_ok=False, error=str(exc), source_path=source_path)

    pages: list[ParsedPage] = []
    all_tables: list[list[list[str]]] = []
    offset = 0

    try:
        with pdf_file:
            n_pages = len(pdf_file.pages)
            limit   = min(n_pages, max_pages) if max_pages else n_pages

            for i, page in enumerate(pdf_file.pages[:limit], start=1):
                text   = _extract_page_text(page)
                raw_tables = page.extract_tables() or []
                tables = [_clean_cells(t) for t in raw_tables]

                pages.append(ParsedPage(
                    page_number=i,
                    text=text,
                    tables=tables,
                    char_offset=offset,
                ))
                all_tables.extend(tables)
                offset += len(text) + 1   # +1 for the page separator newline

    except Exception as exc:
        log.warning("PDF parse error (%s): %s", source_path, exc)
        # Return whatever pages were collected before the error
        if not pages:
            return ParsedDocument(parse_ok=False, error=str(exc), source_path=source_path)

    full_text = "\n".join(p.text for p in pages)

    return ParsedDocument(
        pages=pages,
        full_text=full_text,
        all_tables=all_tables,
        n_pages=len(pages),
        source_path=source_path,
        parse_ok=True,
    )


def tables_to_text(tables: list[list[list[str]]], separator: str = "\t") -> str:
    """
    Flatten a list of tables into tab-separated text (one row per line).
    Useful for passing table content to the sequence/Kd extractors.
    """
    lines: list[str] = []
    for table in tables:
        for row in table:
            lines.append(separator.join(row))
        lines.append("")   # blank line between tables
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(0)
    doc = parse_pdf(sys.argv[1])
    print(f"Pages:  {doc.n_pages}")
    print(f"Tables: {len(doc.all_tables)}")
    print(f"Text preview (first 500 chars):\n{doc.full_text[:500]}")
