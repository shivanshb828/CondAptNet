"""
Kd value extractor for aptamer binding affinity papers.

Wraps kd_converter.py to add:
  - Additional regex patterns (more Kd/KD notation variants)
  - Molar-unit support (M / nM scientific notation in some papers)
  - IC50 / EC50 / Ki detection — extracted separately, never conflated with Kd
  - Structured ExtractedKd dataclass with position + context
  - Per-document deduplication (same value+unit reported twice = one entry)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.data.scraper.utils.kd_converter import convert_kd, KdMeasurement
from scripts.data.scraper.utils.provenance import text_context

# ── Patterns ──────────────────────────────────────────────────────────────────

# Primary Kd pattern — handles Kd, KD, kd and common separator phrasings.
# Separator group covers: = | of | : | was | is | (plain whitespace)
_KD_PATTERN = re.compile(
    r"""
    \b[Kk][Dd]\b                                   # Kd or KD
    \s*(?:=|:\s*|\bof\b|\bwas\b|\bis\b|\s)\s*      # separator
    (\d+\.?\d*(?:[eE][+-]?\d+)?)                   # numeric value (optional sci-notation)
    \s*
    (?:[±\+\-]\s*\d+\.?\d*\s*)?                    # optional ± error term
    (pM|nM|[µu]M|mM|M\b)                          # unit (M = molar, handled specially)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Scientific notation molar form: "Kd = 2.3 × 10⁻⁹ M" or "Kd = 2.3e-9 M"
_KD_SCI_M = re.compile(
    r"\b[Kk][Dd]\b\s*(?:=|of|:|\s)\s*"
    r"(\d+\.?\d*)\s*[×x\*]\s*10\s*\^?\s*(-\d+)\s*M\b",
    re.IGNORECASE,
)

# IC50 / EC50 / Ki — not Kd, but important to capture and distinguish
_IC50_PATTERN = re.compile(
    r"\b(IC50|EC50|Ki)\b\s*(?:=|of|:|\s)\s*"
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*"
    r"(?:[±\+\-]\s*\d+\.?\d*\s*)?"
    r"(pM|nM|[µu]M|mM)",
    re.IGNORECASE,
)

# ── Conversion factor for molar units ────────────────────────────────────────
# "M" alone = 1 mol/L → multiply by 1e9 to get nM
_MOLAR_FACTORS: dict[str, float] = {
    "m":  1e9,    # 1 M = 1e9 nM
    "pm": 1e-3,
    "nm": 1.0,
    "µm": 1e3,
    "um": 1e3,
    "mm": 1e6,
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtractedKd:
    """
    A Kd measurement extracted from source text with full provenance.

    Attributes:
        value_nM        Converted value in nM (always).
        original_value  Numeric value as it appeared in text.
        original_unit   Unit as it appeared in text.
        canonical_unit  Normalised unit for kd_unit schema column.
        start           Byte offset of match start in source text.
        end             Byte offset of match end.
        context         ±200 chars around the match.
        measure_type    "kd" | "ic50" | "ec50" | "ki"
    """
    value_nM:       float
    original_value: float
    original_unit:  str
    canonical_unit: str
    start:          int
    end:            int
    context:        str
    measure_type:   str  # "kd" | "ic50" | "ec50" | "ki"


# ── Internal helpers ──────────────────────────────────────────────────────────

_CANONICAL_UNIT: dict[str, str] = {
    "pm": "pM", "nm": "nM", "µm": "µM", "um": "µM", "mm": "mM", "m": "M",
}


def _to_nM(value: float, unit: str) -> Optional[float]:
    """Convert any unit to nM. Returns None on unrecognised unit."""
    key = unit.strip().lower()
    factor = _MOLAR_FACTORS.get(key)
    if factor is None:
        return None
    # Round to 4 significant figures to prevent IEEE 754 imprecision artifacts.
    return float(f"{value * factor:.4g}")


# ── Public API ────────────────────────────────────────────────────────────────

def extract_kd_from_text(
    text:       str,
    deduplicate: bool = True,
) -> list[ExtractedKd]:
    """
    Extract all Kd measurements from source text.

    Patterns recognised:
      - "Kd = 5.2 nM", "KD of 47 ± 3 nM", "kd: 0.5 µM"
      - "Kd = 2.3 × 10^-9 M"  (scientific notation molar)
      - "Kd = 2.3e-9 M"        (E-notation molar)

    IC50/EC50/Ki are also extracted but tagged as measure_type != "kd".

    Args:
        text:        Source document as a string.
        deduplicate: If True, remove duplicate (value_nM, measure_type) pairs.

    Returns:
        List of ExtractedKd objects (may be empty).
    """
    results: list[ExtractedKd] = []

    # ── Standard Kd pattern ───────────────────────────────────────────────────
    for m in _KD_PATTERN.finditer(text):
        raw_val  = m.group(1)
        raw_unit = m.group(2)
        try:
            value = float(raw_val)
            nM = _to_nM(value, raw_unit)
            if nM is None or nM < 0:
                continue
            canonical = _CANONICAL_UNIT.get(raw_unit.strip().lower(), raw_unit)
            results.append(ExtractedKd(
                value_nM=nM,
                original_value=value,
                original_unit=raw_unit,
                canonical_unit=canonical,
                start=m.start(),
                end=m.end(),
                context=text_context(text, m.start(), m.end()),
                measure_type="kd",
            ))
        except (ValueError, OverflowError):
            continue

    # ── Scientific-notation molar: "Kd = 2.3 × 10^-9 M" ─────────────────────
    for m in _KD_SCI_M.finditer(text):
        try:
            mantissa = float(m.group(1))
            exponent = int(m.group(2))
            value_M  = mantissa * (10 ** exponent)    # value in Molar
            nM       = value_M * 1e9
            if nM < 0:
                continue
            results.append(ExtractedKd(
                value_nM=nM,
                original_value=mantissa,
                original_unit=f"×10^{exponent} M",
                canonical_unit="nM",
                start=m.start(),
                end=m.end(),
                context=text_context(text, m.start(), m.end()),
                measure_type="kd",
            ))
        except (ValueError, OverflowError):
            continue

    # ── IC50 / EC50 / Ki ──────────────────────────────────────────────────────
    for m in _IC50_PATTERN.finditer(text):
        metric_type = m.group(1).lower()   # "ic50", "ec50", or "ki"
        raw_val     = m.group(2)
        raw_unit    = m.group(3)
        try:
            value = float(raw_val)
            nM = _to_nM(value, raw_unit)
            if nM is None or nM < 0:
                continue
            canonical = _CANONICAL_UNIT.get(raw_unit.strip().lower(), raw_unit)
            results.append(ExtractedKd(
                value_nM=nM,
                original_value=value,
                original_unit=raw_unit,
                canonical_unit=canonical,
                start=m.start(),
                end=m.end(),
                context=text_context(text, m.start(), m.end()),
                measure_type=metric_type,
            ))
        except (ValueError, OverflowError):
            continue

    if deduplicate:
        seen: set[tuple[float, str]] = set()
        deduped: list[ExtractedKd] = []
        for r in results:
            key = (round(r.value_nM, 6), r.measure_type)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped

    return results


def best_kd(extractions: list[ExtractedKd]) -> Optional[ExtractedKd]:
    """
    Return the single most likely Kd from a list of extractions.

    Preference order:
      1. measure_type == "kd"  (IC50/Ki are less reliable proxies)
      2. Smallest value (highest affinity)
    """
    kd_only = [e for e in extractions if e.measure_type == "kd"]
    pool = kd_only if kd_only else extractions
    if not pool:
        return None
    return min(pool, key=lambda e: e.value_nM)


if __name__ == "__main__":
    _TESTS = [
        ("Kd = 5.2 nM",                    "kd",   5.2),
        ("KD of 47 ± 3 nM",                "kd",   47.0),
        ("Kd: 0.5 µM",                     "kd",   500.0),
        ("Kd = 0.3 uM",                    "kd",   300.0),
        ("Kd = 500 pM",                    "kd",   0.5),
        ("Kd = 1 mM",                      "kd",   1_000_000.0),
        ("Kd = 2.3 × 10^-9 M",            "kd",   2.3),
        ("IC50 = 120 nM",                  "ic50", 120.0),
        ("Ki of 75 nM",                    "ki",   75.0),
    ]

    print("kd_extractor self-test:")
    all_ok = True
    for text, expected_type, expected_nM in _TESTS:
        results = extract_kd_from_text(text)
        match = next((r for r in results if r.measure_type == expected_type), None)
        ok = match is not None and abs(match.value_nM - expected_nM) < 1e-3
        status = "OK" if ok else "FAIL"
        got = f"{match.value_nM:.4g} nM ({match.measure_type})" if match else "None"
        print(f"  [{status}] {text!r:40s} → {got}")
        if not ok:
            all_ok = False
    print(f"\n{'All tests passed.' if all_ok else 'SOME TESTS FAILED.'}")
