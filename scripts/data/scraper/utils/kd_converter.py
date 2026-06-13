"""
Kd unit conversion with full audit trail.

Rules (from dataset spec):
    pM  → value / 1000      (÷ 1,000)
    nM  → value (unchanged)
    µM  → value × 1000      (× 1,000)
    mM  → value × 1,000,000 (× 1,000,000)

All kd_value entries in the output schema are in nM.
Original value and unit are always preserved for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Conversion factors to nM ──────────────────────────────────────────────────
_FACTORS: dict[str, float] = {
    "pm":  1e-3,
    "nm":  1.0,
    "µm":  1e3,
    "um":  1e3,   # ASCII fallback for µ
    "mm":  1e6,
}

# Canonical unit names for storage in kd_unit column
_CANONICAL: dict[str, str] = {
    "pm": "pM",
    "nm": "nM",
    "µm": "µM",
    "um": "µM",   # normalise to µM in output
    "mm": "mM",
}

# ── Kd extraction regex ───────────────────────────────────────────────────────
# Matches: "Kd = 5.2 nM", "KD of 47 ± 3 nM", "kd: 0.5 µM", "Kd 112.5nM"
KD_PATTERN = re.compile(
    r'[Kk][Dd]\s*(?:=|of|:|\s)\s*'
    r'(\d+\.?\d*(?:e[+-]?\d+)?)\s*'      # value (optional scientific notation)
    r'(?:[±\+\-]\s*\d+\.?\d*\s*)?'        # optional ± error term
    r'(pM|nM|[µu]M|mM)',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KdMeasurement:
    """Immutable Kd measurement with full provenance."""
    value_nM:       float   # converted value, always in nM
    original_value: float   # value as it appeared in the source
    original_unit:  str     # unit as it appeared in the source (e.g. "µM")
    canonical_unit: str     # normalised unit stored in kd_unit column

    def __str__(self) -> str:
        return (f"{self.value_nM:.4g} nM  "
                f"(source: {self.original_value} {self.original_unit})")


def convert_kd(value: float, unit: str) -> KdMeasurement:
    """
    Convert a Kd measurement to nM.

    Args:
        value: Numeric Kd value (must be ≥ 0).
        unit:  Unit string. Accepted: pM, nM, µM, uM, mM (case-insensitive).

    Returns:
        KdMeasurement with value_nM, original_value, original_unit, canonical_unit.

    Raises:
        ValueError: if unit is unrecognised or value is negative.
    """
    if value < 0:
        raise ValueError(f"Kd value must be ≥ 0, got {value}")

    unit_key = unit.strip().lower()
    factor = _FACTORS.get(unit_key)
    if factor is None:
        raise ValueError(
            f"Unrecognised Kd unit {unit!r}. "
            f"Accepted: pM, nM, µM/uM, mM (case-insensitive)."
        )

    return KdMeasurement(
        value_nM=value * factor,
        original_value=value,
        original_unit=unit.strip(),
        canonical_unit=_CANONICAL[unit_key],
    )


def extract_kd_from_text(text: str) -> list[KdMeasurement]:
    """
    Extract all Kd measurements from free text.

    Does NOT generate or modify values — pure regex extraction.
    Ambiguous matches are silently skipped (logged at caller level).

    Returns list of KdMeasurement objects (may be empty).
    """
    results: list[KdMeasurement] = []
    for m in KD_PATTERN.finditer(text):
        raw_value = m.group(1)
        raw_unit  = m.group(2)
        try:
            meas = convert_kd(float(raw_value), raw_unit)
            results.append(meas)
        except (ValueError, OverflowError):
            continue
    return results


def parse_kd_string(s: str) -> Optional[KdMeasurement]:
    """
    Parse a standalone Kd string like "5.2 nM" or "0.3 µM".

    Tries to extract a (value, unit) pair directly without requiring the
    "Kd =" prefix. Returns None if parsing fails.
    """
    # Standalone pattern: number followed immediately by unit
    standalone = re.match(
        r'^\s*(\d+\.?\d*(?:e[+-]?\d+)?)\s*(pM|nM|[µu]M|mM)\s*$',
        s.strip(),
        re.IGNORECASE,
    )
    if standalone:
        try:
            return convert_kd(float(standalone.group(1)), standalone.group(2))
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    # Self-test
    cases: list[tuple[float, str, float]] = [
        (1000.0, "pM",  1.0),
        (1.0,    "nM",  1.0),
        (1.0,    "µM",  1000.0),
        (1.0,    "uM",  1000.0),
        (1.0,    "mM",  1_000_000.0),
        (112.5,  "nM",  112.5),        # thrombin aptamer benchmark Kd
        (47.0,   "nM",  47.0),
        (0.3,    "µM",  300.0),
    ]
    print("KdConverter self-test:")
    all_ok = True
    for val, unit, expected_nM in cases:
        m = convert_kd(val, unit)
        ok = abs(m.value_nM - expected_nM) < 1e-9
        status = "OK" if ok else f"FAIL (got {m.value_nM}, expected {expected_nM})"
        print(f"  {val} {unit} → {m.value_nM} nM  [{status}]")
        if not ok:
            all_ok = False

    # Regex extraction test
    text = "The aptamer showed a Kd of 5.2 nM and another study reported KD = 0.3 µM."
    extracted = extract_kd_from_text(text)
    print(f"\nExtracted from text: {[str(e) for e in extracted]}")
    assert len(extracted) == 2
    assert abs(extracted[0].value_nM - 5.2)   < 1e-9
    assert abs(extracted[1].value_nM - 300.0) < 1e-9

    print(f"\n{'All tests passed.' if all_ok else 'SOME TESTS FAILED.'}")
