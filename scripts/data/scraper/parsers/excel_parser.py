"""
Excel / CSV parser for aptamer supplement spreadsheets.

Handles .xlsx, .xls, .csv, and .tsv files. Returns a ParsedSpreadsheet
with all sheets as named DataFrames, plus a text rendering for passing
to sequence/Kd extractors.

Design decisions:
  - Multi-sheet workbooks: all sheets returned, named by sheet title.
  - Sequence column detection: scans headers and first data rows for
    patterns consistent with DNA/RNA sequences (length heuristic + ATGCU).
  - No type coercion — all cells returned as strings. Extractors decide
    what is a sequence or number.
  - Empty rows and entirely-blank columns are dropped.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

log = logging.getLogger(__name__)

# ── Sequence column heuristics ────────────────────────────────────────────────

_SEQ_HEADER_KEYWORDS = re.compile(
    r"sequence|aptamer|oligo|primer|probe|nucleotide|dna|rna",
    re.IGNORECASE,
)
_SEQ_CELL_PATTERN = re.compile(
    r"^[ATGCU]{20,120}$",
    re.IGNORECASE,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedSheet:
    name:     str
    df:       pd.DataFrame   # all cells as strings, NaN → ""
    n_rows:   int
    n_cols:   int
    seq_cols: list[str]      # column names likely containing sequences


@dataclass
class ParsedSpreadsheet:
    """
    Result of parsing an Excel or CSV file.

    Attributes:
        sheets      List of ParsedSheet objects (one per sheet/tab).
        full_text   All cells concatenated as tab-separated rows, sheets
                    separated by double newline — ready for regex extractors.
        n_sheets    Total sheet count.
        source_path File path or "<bytes>".
        parse_ok    False if the file could not be opened.
        error       Error message when parse_ok is False.
    """
    sheets:      list[ParsedSheet] = field(default_factory=list)
    full_text:   str               = ""
    n_sheets:    int               = 0
    source_path: str               = ""
    parse_ok:    bool              = True
    error:       str               = ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_seq_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that likely contain DNA/RNA sequences."""
    seq_cols: list[str] = []

    for col in df.columns:
        # Header keyword match
        if _SEQ_HEADER_KEYWORDS.search(str(col)):
            seq_cols.append(str(col))
            continue

        # Check first 5 non-empty data cells
        sample = df[col].dropna().head(5).astype(str)
        hits = sum(1 for v in sample if _SEQ_CELL_PATTERN.match(v.strip()))
        if hits >= 2:
            seq_cols.append(str(col))

    return seq_cols


def _df_to_text(df: pd.DataFrame, sep: str = "\t") -> str:
    """Flatten a DataFrame to tab-separated text (header + rows)."""
    lines: list[str] = []
    lines.append(sep.join(str(c) for c in df.columns))
    for _, row in df.iterrows():
        lines.append(sep.join(str(v) for v in row))
    return "\n".join(lines)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop entirely-empty rows and columns; fill remaining NaN with ''."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")
    df.columns = [str(c) for c in df.columns]
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def parse_excel(
    source:     Union[str, Path, bytes],
    sheet_name: Optional[Union[str, int]] = None,
) -> ParsedSpreadsheet:
    """
    Parse an Excel (.xlsx / .xls) file and return a ParsedSpreadsheet.

    Args:
        source:     File path, URL-like path, or raw bytes.
        sheet_name: Specific sheet to load (None = all sheets).

    Returns:
        ParsedSpreadsheet. parse_ok=False with error message on failure.
    """
    source_path = str(source) if not isinstance(source, bytes) else "<bytes>"

    try:
        if isinstance(source, bytes):
            buf = io.BytesIO(source)
        else:
            buf = source

        raw = pd.read_excel(
            buf,
            sheet_name=sheet_name if sheet_name is not None else None,
            dtype=str,
            header=0,
        )

    except Exception as exc:
        log.warning("Excel open failed (%s): %s", source_path, exc)
        return ParsedSpreadsheet(parse_ok=False, error=str(exc), source_path=source_path)

    return _assemble(raw, source_path)


def parse_csv(
    source: Union[str, Path, bytes],
    sep:    str = ",",
) -> ParsedSpreadsheet:
    """
    Parse a CSV or TSV file.

    Args:
        source: File path or raw bytes.
        sep:    Delimiter (",", "\\t", etc.). Pass sep="\\t" for TSV files.

    Returns:
        ParsedSpreadsheet with a single sheet named "sheet1".
    """
    source_path = str(source) if not isinstance(source, bytes) else "<bytes>"

    try:
        if isinstance(source, bytes):
            buf = io.StringIO(source.decode("utf-8", errors="replace"))
        else:
            buf = source

        df_raw = pd.read_csv(buf, sep=sep, dtype=str)

    except Exception as exc:
        log.warning("CSV open failed (%s): %s", source_path, exc)
        return ParsedSpreadsheet(parse_ok=False, error=str(exc), source_path=source_path)

    # Wrap single DataFrame as a dict to reuse _assemble
    return _assemble({"sheet1": df_raw}, source_path)


def _assemble(
    raw:         Union[pd.DataFrame, dict],
    source_path: str,
) -> ParsedSpreadsheet:
    """Build ParsedSpreadsheet from a raw pd.read_excel result."""
    # Normalise: pd.read_excel returns either a df (single sheet) or dict
    if isinstance(raw, pd.DataFrame):
        sheet_map: dict = {"Sheet1": raw}
    else:
        sheet_map = raw

    sheets: list[ParsedSheet] = []
    text_parts: list[str] = []

    for name, df in sheet_map.items():
        df = _clean_df(df)
        seq_cols = _detect_seq_cols(df)
        sheets.append(ParsedSheet(
            name=str(name),
            df=df,
            n_rows=len(df),
            n_cols=len(df.columns),
            seq_cols=seq_cols,
        ))
        text_parts.append(f"=== Sheet: {name} ===\n" + _df_to_text(df))

    return ParsedSpreadsheet(
        sheets=sheets,
        full_text="\n\n".join(text_parts),
        n_sheets=len(sheets),
        source_path=source_path,
        parse_ok=True,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python excel_parser.py <path_to_xlsx_or_csv>")
        sys.exit(0)
    path = sys.argv[1]
    doc = parse_csv(path) if path.endswith((".csv", ".tsv")) else parse_excel(path)
    print(f"Sheets: {doc.n_sheets}")
    for s in doc.sheets:
        print(f"  {s.name!r}: {s.n_rows} rows x {s.n_cols} cols — seq cols: {s.seq_cols}")
    print(f"Text preview (first 500 chars):\n{doc.full_text[:500]}")
