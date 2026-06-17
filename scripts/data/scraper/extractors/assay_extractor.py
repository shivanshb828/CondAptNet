"""
SELEX method / assay type extractor.

Detects the selection or binding assay method from free text.
Returns a single canonical assay_type string (or empty string if none found).

Recognized assay types (in priority order — more specific first):
  CE-SELEX, Capillary Electrophoresis SELEX
  Microfluidic SELEX
  Cell-SELEX
  Capture SELEX
  Toggle SELEX
  Magnetic bead SELEX
  SPR (Surface Plasmon Resonance)
  ITC (Isothermal Titration Calorimetry)
  EMSA (Electrophoretic Mobility Shift Assay)
  MST (MicroScale Thermophoresis)
  BLI (BioLayer Interferometry)
  Nitrocellulose filter binding
  Fluorescence anisotropy
  Flow cytometry
  SELEX (generic fallback)
"""

from __future__ import annotations

import re
from typing import Optional


# ── Patterns in priority order ─────────────────────────────────────────────────
# Each entry: (canonical_name, compiled_regex)
# First match wins.

_ASSAY_RULES: list[tuple[str, re.Pattern]] = [
    ("CE-SELEX", re.compile(
        r"\b(?:CE-SELEX|capillary\s+electrophoresis\s+(?:SELEX|selection))\b",
        re.IGNORECASE,
    )),
    ("Microfluidic SELEX", re.compile(
        r"\b(?:microfluidic\s+SELEX|micro-?fluidic\s+(?:selection|aptamer))\b",
        re.IGNORECASE,
    )),
    ("Cell-SELEX", re.compile(
        r"\b(?:cell-SELEX|cell\s+SELEX|whole.cell\s+SELEX|Cell-Systematic)\b",
        re.IGNORECASE,
    )),
    ("Capture SELEX", re.compile(
        r"\b(?:capture-SELEX|capture\s+SELEX)\b",
        re.IGNORECASE,
    )),
    ("Toggle SELEX", re.compile(
        r"\b(?:toggle-SELEX|toggle\s+SELEX)\b",
        re.IGNORECASE,
    )),
    ("Magnetic bead SELEX", re.compile(
        r"\b(?:magnetic\s+bead\s+SELEX|bead-based\s+SELEX|MACS-SELEX|"
        r"magnetic\s+(?:bead|particle)\s+(?:selection|aptamer))\b",
        re.IGNORECASE,
    )),
    ("SPR", re.compile(
        r"\b(?:SPR|surface\s+plasmon\s+resonance|Biacore)\b",
        re.IGNORECASE,
    )),
    ("ITC", re.compile(
        r"\b(?:ITC|isothermal\s+titration\s+calorimetry)\b",
        re.IGNORECASE,
    )),
    ("EMSA", re.compile(
        r"\b(?:EMSA|electrophoretic\s+mobility\s+shift|gel\s+shift\s+assay|"
        r"band\s+shift\s+assay)\b",
        re.IGNORECASE,
    )),
    ("MST", re.compile(
        r"\b(?:MST|microscale\s+thermophoresis|MicroScale\s+Thermophoresis)\b",
        re.IGNORECASE,
    )),
    ("BLI", re.compile(
        r"\b(?:BLI|bio-?layer\s+interferometry|Octet)\b",
        re.IGNORECASE,
    )),
    ("Nitrocellulose filter", re.compile(
        r"\b(?:nitrocellulose\s+filter|filter\s+binding\s+assay|membrane\s+filter\s+assay)\b",
        re.IGNORECASE,
    )),
    ("Fluorescence anisotropy", re.compile(
        r"\b(?:fluorescence\s+(?:anisotropy|polarization)|FA\s+(?:binding|assay))\b",
        re.IGNORECASE,
    )),
    ("Flow cytometry", re.compile(
        r"\b(?:flow\s+cytometry|FACS\s+(?:assay|binding|screening))\b",
        re.IGNORECASE,
    )),
    ("SELEX", re.compile(
        r"\b(?:SELEX|systematic\s+evolution\s+of\s+ligands)\b",
        re.IGNORECASE,
    )),
]


def extract_assay_type(text: str) -> str:
    """
    Extract the SELEX method or binding assay type from document text.

    Applies patterns in priority order (most-specific first).
    Returns the canonical assay name string, or "" if nothing matches.

    Args:
        text: Document text (full-text or abstract).

    Returns:
        Canonical assay type string, e.g. "CE-SELEX", "SPR", "SELEX".
        Empty string if no assay type is found.
    """
    if not text:
        return ""
    for canonical, pattern in _ASSAY_RULES:
        if pattern.search(text):
            return canonical
    return ""


if __name__ == "__main__":
    _TESTS = [
        ("We performed CE-SELEX to select aptamers against thrombin.", "CE-SELEX"),
        ("Aptamers were identified using microfluidic SELEX in a chip-based format.", "Microfluidic SELEX"),
        ("Cell-SELEX was used to target MUC1 on cancer cells.", "Cell-SELEX"),
        ("Binding was measured by SPR on a Biacore instrument.", "SPR"),
        ("ITC was used to determine the thermodynamic binding parameters.", "ITC"),
        ("EMSA confirmed aptamer binding at 37°C.", "EMSA"),
        ("Binding was assessed by microscale thermophoresis (MST).", "MST"),
        ("Biolayer interferometry (BLI) was performed on the Octet system.", "BLI"),
        ("The aptamer was selected by the SELEX procedure.", "SELEX"),
        ("No specific method mentioned in this abstract.", ""),
    ]

    print("assay_extractor self-test:")
    all_ok = True
    for text, expected in _TESTS:
        got = extract_assay_type(text)
        ok  = got == expected
        status = "OK" if ok else f"FAIL (expected {expected!r}, got {got!r})"
        print(f"  [{status[:2]}] {text[:60]!r}")
        if not ok:
            all_ok = False
    print(f"\n{'All tests passed.' if all_ok else 'SOME TESTS FAILED.'}")
