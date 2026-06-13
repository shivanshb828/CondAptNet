"""
20-column output schema for the CondAptNet aptamer mining pipeline.

Every adapter returns a list of dicts. Every dict must pass validate_record()
before being written to scraped_dataset.csv.

Column spec is authoritative — do not modify without updating dataset_column_spec.md.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

# ── Column definitions ────────────────────────────────────────────────────────

SCHEMA_COLUMNS: list[str] = [
    "aptamer_sequence",       # 1
    "nucleic_acid_type",      # 2
    "modifications",          # 3
    "target_name",            # 4
    "target_type",            # 5
    "target_id",              # 6  (allow blank)
    "target_id_source",       # 7  (allow blank)
    "kd_value",               # 8  (allow blank) — always in nM
    "kd_unit",                # 9  (allow blank) — original unit before conversion
    "assay_type",             # 10 (allow blank)
    "selection_buffer",       # 11 (allow blank)
    "binding_buffer",         # 12 (allow blank)
    "ph",                     # 13 (allow blank)
    "na_concentration_mM",    # 14 (allow blank)
    "mg_concentration_mM",    # 15 (allow blank)
    "temperature_C",          # 16 (allow blank)
    "source_doi",             # 17 (allow blank)
    "source_type",            # 18 (allow blank)
    "confidence_score",       # 19 required
    "split",                  # 20 required
]

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "aptamer_sequence",
    "nucleic_acid_type",
    "modifications",
    "target_name",
    "target_type",
    "confidence_score",
    "split",
})

CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "nucleic_acid_type": frozenset({
        "ssDNA", "ssRNA", "LNA", "2'-OMe RNA", "2'-F RNA", "other",
    }),
    "target_type": frozenset({
        "protein", "peptide", "small_molecule", "cell", "organism",
        "ion", "toxin", "other",
    }),
    "target_id_source": frozenset({
        "UniProt", "PDB", "PubChem", "KEGG", "ChEBI",
        "BRENDA", "Cellosaurus", "NCBI_Taxonomy", "other",
    }),
    "kd_unit": frozenset({"pM", "nM", "µM", "uM", "mM"}),
    "source_type": frozenset({"paper", "patent", "database", "preprint"}),
    "confidence_score": frozenset({"curated", "extracted", "non-curated", "uncertain"}),
    "split": frozenset({"train", "val", "test"}),
}

# ── Blank sentinel (stored in CSV as empty string, not NaN) ──────────────────
BLANK = ""


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a single record dict against the schema.

    Returns (is_valid, errors). errors is empty when valid.
    Never raises — always returns a result.
    """
    errors: list[str] = []

    # Required fields must be present and non-empty
    for col in REQUIRED_COLUMNS:
        val = record.get(col)
        if val is None or str(val).strip() == "":
            errors.append(f"required field missing or empty: {col!r}")

    # Categorical fields — check allowed values when present
    for col, allowed in CATEGORICAL_VALUES.items():
        val = record.get(col)
        if val is not None and str(val).strip() != "":
            if str(val).strip() not in allowed:
                errors.append(
                    f"invalid value for {col!r}: {val!r}. "
                    f"Allowed: {sorted(allowed)}"
                )

    # kd_value must be numeric when present
    kd = record.get("kd_value")
    if kd is not None and str(kd).strip() != "":
        try:
            v = float(kd)
            if v < 0:
                errors.append(f"kd_value must be ≥ 0, got {v}")
        except (ValueError, TypeError):
            errors.append(f"kd_value is not numeric: {kd!r}")

    # kd_value and kd_unit must be both present or both absent
    has_kd   = kd is not None and str(kd).strip() != ""
    has_unit = (record.get("kd_unit") or "") != ""
    if has_kd and not has_unit:
        errors.append("kd_value present but kd_unit is missing")
    if has_unit and not has_kd:
        errors.append("kd_unit present but kd_value is missing")

    # Numeric range checks for optional float fields
    _float_bounds: dict[str, tuple[float, float]] = {
        "ph":                  (0.0, 14.0),
        "na_concentration_mM": (0.0, 10_000.0),
        "mg_concentration_mM": (0.0, 1_000.0),
        "temperature_C":       (-80.0, 200.0),
    }
    for col, (lo, hi) in _float_bounds.items():
        val = record.get(col)
        if val is not None and str(val).strip() != "":
            try:
                v = float(val)
                if not (lo <= v <= hi):
                    errors.append(f"{col} = {v} out of expected range [{lo}, {hi}]")
            except (ValueError, TypeError):
                errors.append(f"{col} is not numeric: {val!r}")

    return len(errors) == 0, errors


def make_empty_record() -> dict[str, Any]:
    """Return a dict with all 20 columns set to BLANK / default values."""
    return {col: BLANK for col in SCHEMA_COLUMNS}


def empty_dataframe() -> pd.DataFrame:
    """Return a properly typed empty DataFrame with all 20 columns."""
    dtypes: dict[str, str] = {
        "aptamer_sequence":    "object",
        "nucleic_acid_type":   "object",
        "modifications":       "object",
        "target_name":         "object",
        "target_type":         "object",
        "target_id":           "object",
        "target_id_source":    "object",
        "kd_value":            "float64",
        "kd_unit":             "object",
        "assay_type":          "object",
        "selection_buffer":    "object",
        "binding_buffer":      "object",
        "ph":                  "float64",
        "na_concentration_mM": "float64",
        "mg_concentration_mM": "float64",
        "temperature_C":       "float64",
        "source_doi":          "object",
        "source_type":         "object",
        "confidence_score":    "object",
        "split":               "object",
    }
    df = pd.DataFrame(columns=SCHEMA_COLUMNS)
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    return df


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of validated record dicts to a DataFrame.
    Columns are enforced to SCHEMA_COLUMNS order.
    Extra keys in records are silently dropped.
    """
    rows = []
    for rec in records:
        row = {col: rec.get(col, BLANK) for col in SCHEMA_COLUMNS}
        rows.append(row)
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)

    # Cast numeric columns
    for col in ("kd_value", "ph", "na_concentration_mM", "mg_concentration_mM", "temperature_C"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
