"""
Row-level aptamer-Kd-target pair extraction from parsed tables.

The critical gap in the original pipeline: _extract_records_from_text()
assigned the document's single best Kd to every sequence found. Papers
routinely publish tables with one aptamer per row and its own Kd. This
module fixes that by parsing table structure directly.

Algorithm:
  1. Detect which column holds aptamer sequences (ATGC pattern match).
  2. Detect which column holds Kd values (numeric + unit suffix in header or cells).
  3. Detect which column holds target name (optional; falls back to caller-provided name).
  4. For each data row: pair the sequence with its own Kd and return a structured record.

Returns a list of TableRecord dataclasses. Each TableRecord maps cleanly to
one schema row. The caller (pubmed_pmc.py) merges these with metadata.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.parsers.xml_parser import ParsedTable
from scripts.data.scraper.extractors.sequence_extractor import extract_sequences
from scripts.data.scraper.extractors.kd_extractor import extract_kd_from_text, best_kd
from scripts.data.scraper.utils.kd_converter import convert_kd

# ── Column header patterns ─────────────────────────────────────────────────────

# "aptamer" alone is ambiguous — "Aptamer ID" or "Aptamer Name" is NOT a sequence column.
# Only match it when combined with sequence-context words.
_SEQ_HEADER = re.compile(
    r"\b(?:sequence|oligo(?:nucleotide)?|ssDNA|nucleotide\s*sequence|"
    r"aptamer\s+(?:sequence|seq)|DNA\s+(?:sequence|seq)|"
    r"(?:selected|final|consensus)\s+(?:sequence|seq)|"
    r"5[''ʼ].{0,10}3[''ʼ])\b",        # e.g. "5'→3'" notation in header
    re.IGNORECASE,
)

_KD_HEADER = re.compile(
    r"\b(?:Kd|KD|K_d|K_D|dissociation|affinity|binding\s*constant|"
    r"IC50|EC50|Ki|Kd\s*\(?nM\)?|Kd\s*\(?µM\)?|Kd\s*\(?pM\)?)\b",
    re.IGNORECASE,
)

_TARGET_HEADER = re.compile(
    r"\b(?:target|protein|molecule|analyte|ligand|antigen)\b",
    re.IGNORECASE,
)

_NAME_HEADER = re.compile(
    r"\b(?:name|id|identifier|clone|aptamer\s*#|apt\s*#|label|no\.?)\b",
    re.IGNORECASE,
)

# Inline Kd pattern for cell-level detection when headers are absent
_KD_CELL = re.compile(
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*"
    r"(?:[±\+\-]\s*\d+\.?\d*\s*)?"
    r"(pM|nM|[µu]M|mM)\b",
    re.IGNORECASE,
)

# Contiguous DNA sequence in a cell (≥18 nt; slightly relaxed from the 20-nt minimum
# because table cells occasionally omit the primer regions)
_SEQ_CELL = re.compile(r"\b([ATGCatgc]{18,120})\b")


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class TableRecord:
    """
    One aptamer-Kd pair extracted from a table row.

    Attributes:
        sequence        Normalised aptamer sequence (uppercase ATGC).
        kd_nM           Kd in nanomolar, or None if not found in this row.
        kd_unit_orig    Original Kd unit string before conversion, or None.
        target_name     Target name from the table (may be empty string).
        row_index       0-based index of the data row in the table.
        table_label     Label of the source table (e.g. "Table 1").
        confidence      "extracted" always for table-based extraction.
    """
    sequence:     str
    kd_nM:        Optional[float]
    kd_unit_orig: Optional[str]
    target_name:  str
    row_index:    int
    table_label:  str
    confidence:   str = "extracted"


# ── Column detection ───────────────────────────────────────────────────────────

def _detect_seq_col(rows: list[list[str]], header_row: list[str]) -> Optional[int]:
    """
    Return the column index most likely to contain aptamer sequences, or None.

    Strategy (in priority order):
      1. Header keyword match AND cells in that column contain DNA sequences.
      2. Pure cell-content scoring when no header match is found.
    """
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)

    # Score each column by fraction of cells containing a DNA sequence
    col_hits = [0] * n_cols
    for row in rows:
        for j, cell in enumerate(row):
            if j < n_cols and _SEQ_CELL.search(cell.strip()):
                col_hits[j] += 1

    # Collect header-matching columns and break ties with cell-content score
    header_candidates: list[tuple[int, int]] = []  # (col_index, cell_hit_count)
    for i, h in enumerate(header_row):
        if _SEQ_HEADER.search(h) and i < n_cols:
            header_candidates.append((i, col_hits[i]))

    if header_candidates:
        # Prefer the header-matching column that also has the most sequence cells
        best = max(header_candidates, key=lambda x: x[1])
        # Accept it even if cell count is 0 (header match is strong enough)
        return best[0]

    # No header match — fall back to pure cell-content scoring
    best_col = max(range(n_cols), key=lambda j: col_hits[j])
    if col_hits[best_col] > 0 and col_hits[best_col] >= len(rows) * 0.3:
        return best_col
    return None


def _detect_kd_col(rows: list[list[str]], header_row: list[str]) -> Optional[int]:
    """
    Return the column index most likely to hold Kd values, or None.
    """
    # Header match
    for i, h in enumerate(header_row):
        if _KD_HEADER.search(h):
            return i

    # Fallback: look for cells with numeric + unit pattern
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)
    col_hits = [0] * n_cols
    for row in rows:
        for j, cell in enumerate(row):
            if j < n_cols and _KD_CELL.search(cell.strip()):
                col_hits[j] += 1

    best_col = max(range(n_cols), key=lambda j: col_hits[j])
    if col_hits[best_col] > 0 and col_hits[best_col] >= len(rows) * 0.3:
        return best_col
    return None


def _detect_target_col(rows: list[list[str]], header_row: list[str]) -> Optional[int]:
    """Return column index for target name, or None."""
    for i, h in enumerate(header_row):
        if _TARGET_HEADER.search(h):
            return i
    return None


def _parse_kd_cell(cell: str) -> tuple[Optional[float], Optional[str]]:
    """
    Extract (kd_nM, original_unit) from a single table cell string.
    Handles: "5.2 nM", "0.27 ± 0.03 nM", "12.4", ">100 nM" (strips >/<).
    Returns (None, None) if no Kd found.
    """
    # Strip inequality signs and whitespace
    cell = re.sub(r"^[<>≤≥~≈]\s*", "", cell.strip())

    m = _KD_CELL.search(cell)
    if not m:
        # Try bare number — may be in a column where unit is in the header
        bare = re.match(r"^\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$", cell)
        if bare:
            try:
                return float(bare.group(1)), None  # unit unknown
            except ValueError:
                pass
        return None, None

    try:
        value = float(m.group(1))
        unit  = m.group(2)
        nM    = convert_kd(value, unit).value_nM
        return nM, unit
    except (ValueError, TypeError):
        return None, None


def _extract_unit_from_header(header: str) -> Optional[str]:
    """Pull unit out of header text like 'Kd (nM)' → 'nM'."""
    m = re.search(r"\(?(pM|nM|[µu]M|mM)\)?", header, re.IGNORECASE)
    return m.group(1) if m else None


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_from_table(
    table: ParsedTable,
    fallback_target: str = "",
) -> list[TableRecord]:
    """
    Extract aptamer-Kd pairs from one ParsedTable.

    Args:
        table:           A ParsedTable from xml_parser.parse_nxml().
        fallback_target: Target name to use when no target column is found.
                         Typically the paper title/abstract-derived guess.

    Returns:
        List of TableRecord objects (one per data row with a valid sequence).
        Empty list if the table has no aptamer sequence column.
    """
    if not table.rows or len(table.rows) < 2:
        return []

    # Separate header row from data rows
    header_row = [str(c).strip() for c in table.rows[0]]
    data_rows  = table.rows[1:]

    seq_col    = _detect_seq_col(data_rows, header_row)
    kd_col     = _detect_kd_col(data_rows, header_row)
    target_col = _detect_target_col(data_rows, header_row)

    if seq_col is None:
        return []

    # If Kd column header contains a unit, use it as fallback for bare numbers
    kd_header_unit: Optional[str] = None
    if kd_col is not None:
        kd_header_unit = _extract_unit_from_header(header_row[kd_col] if kd_col < len(header_row) else "")

    records: list[TableRecord] = []

    for row_i, row in enumerate(data_rows):
        if seq_col >= len(row):
            continue

        raw_cell = str(row[seq_col]).strip()

        # Extract sequence(s) from this cell (usually just one)
        seq_hits = extract_sequences(raw_cell, valid_only=True)
        if not seq_hits:
            # Relaxed: try without length validation for very short cells
            m = _SEQ_CELL.search(raw_cell)
            if not m:
                continue
            # Must be at least 20 nt after joining to matter for training
            candidate = m.group(1).upper()
            if len(candidate) < 20:
                continue
            # Validate manually
            from scripts.data.validate_sequences import validate_sequence
            ok, _ = validate_sequence(candidate)
            if not ok:
                continue
            seq_hits_seqs = [candidate]
        else:
            seq_hits_seqs = [s.sequence for s in seq_hits]

        # Kd for this row
        kd_nM: Optional[float] = None
        kd_unit_orig: Optional[str] = None

        if kd_col is not None and kd_col < len(row):
            kd_cell = str(row[kd_col]).strip()
            kd_nM, kd_unit_orig = _parse_kd_cell(kd_cell)

            # If we got a bare number and know the unit from the header, convert now
            if kd_nM is None:
                bare = re.match(r"^\s*[<>≤≥~≈]?\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$", kd_cell)
                if bare and kd_header_unit:
                    try:
                        kd_nM = convert_kd(float(bare.group(1)), kd_header_unit).value_nM
                        kd_unit_orig = kd_header_unit
                    except (ValueError, TypeError):
                        pass

        # Target for this row
        target = fallback_target
        if target_col is not None and target_col < len(row):
            cell_target = str(row[target_col]).strip()
            if cell_target and cell_target.lower() not in ("", "n/a", "na", "-"):
                target = cell_target

        for seq in seq_hits_seqs:
            records.append(TableRecord(
                sequence=seq,
                kd_nM=kd_nM,
                kd_unit_orig=kd_unit_orig,
                target_name=target,
                row_index=row_i,
                table_label=table.label,
            ))

    return records


def extract_from_all_tables(
    tables: list[ParsedTable],
    fallback_target: str = "",
) -> list[TableRecord]:
    """
    Apply extract_from_table to every table in a document.

    Deduplicates by sequence — if the same sequence appears in multiple tables,
    the one with a Kd value wins; otherwise the first occurrence is kept.

    Args:
        tables:          All tables from a parsed PMC article.
        fallback_target: Paper-level target guess for tables without a target column.

    Returns:
        Deduplicated list of TableRecord objects.
    """
    all_records: list[TableRecord] = []
    for table in tables:
        all_records.extend(extract_from_table(table, fallback_target))

    # Deduplicate: prefer records with Kd over those without
    seen: dict[str, TableRecord] = {}
    for rec in all_records:
        key = rec.sequence
        if key not in seen:
            seen[key] = rec
        elif rec.kd_nM is not None and seen[key].kd_nM is None:
            seen[key] = rec  # upgrade to the version with a Kd

    return list(seen.values())


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.data.scraper.parsers.xml_parser import ParsedTable

    _TABLES = [
        ParsedTable(
            label="Table 1",
            caption="Selected DNA aptamers against thrombin",
            rows=[
                # Header
                ["Name", "Sequence", "Kd (nM)", "Notes"],
                # Data rows
                ["Apt-1", "CCCCTGCAGGTGATTTTGCTCAAGTCAGAAGGATAAACTGTCCAGAACTTGGAATATATCAGTATCGCTAATCAGGCGGAT", "1.0", "high affinity"],
                ["Apt-2", "CCCCTGCAGGTGATTTTGCTCAAGTCAGGCGTTAGGGAAGGGCGTCGAAAGCAGGGTGGGAGTATCGCTAATCAGGCGGAT", "5.0", ""],
                ["Apt-3", "GCAATAGCGGTTACCAGTTTTAATCAGTTGGTCATTAGCAATAGCAGGCGTTTGCAATCAGGCGGATGAT", "12.4", ""],
            ],
        ),
        ParsedTable(
            label="Table 2",
            caption="Negative control sequences — no Kd measured",
            rows=[
                ["Sequence", "Target"],
                ["ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC", "Thrombin"],
            ],
        ),
        ParsedTable(
            label="Table 3",
            caption="Short table — too few rows",
            rows=[["Sequence"]],
        ),
    ]

    print("table_extractor self-test:")
    results = extract_from_all_tables(_TABLES, fallback_target="Thrombin")
    print(f"  Records extracted: {len(results)}")
    for r in results:
        kd_str = f"{r.kd_nM:.2f} nM" if r.kd_nM else "no Kd"
        print(f"    [{r.table_label}] seq={r.sequence[:30]}... | Kd={kd_str} | target={r.target_name}")

    assert len(results) == 4, f"Expected 4 records, got {len(results)}"
    apt1 = next((r for r in results if r.kd_nM and abs(r.kd_nM - 1.0) < 0.01), None)
    assert apt1 is not None, "Apt-1 1.0 nM not found"
    apt2 = next((r for r in results if r.kd_nM and abs(r.kd_nM - 5.0) < 0.01), None)
    assert apt2 is not None, "Apt-2 5.0 nM not found"
    no_kd = [r for r in results if r.kd_nM is None]
    assert len(no_kd) == 1, f"Expected 1 record with no Kd, got {len(no_kd)}"
    print("\nAll assertions passed.")
