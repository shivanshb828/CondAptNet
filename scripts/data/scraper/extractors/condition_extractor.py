"""
Experimental condition extractor.

Extracts pH, Na⁺ concentration, Mg²⁺ concentration, temperature, and buffer
type from free text. All values are Optional — None means not found in the
source; the training pipeline applies physiological defaults later.

Design:
  - Multiple regex patterns per field to handle common phrasings
  - When multiple values found for the same field, returns the FIRST occurrence
    (earliest in the document — typically the methods section comes before results)
  - Buffer type captures both selection and binding/assay buffers separately
  - No defaults applied here — defaults belong in train.py and augment.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# ── Regex patterns ────────────────────────────────────────────────────────────

# pH: "pH 7.4", "pH = 7.4", "pH of 7.4", "adjusted to pH 7.4"
_PH = re.compile(
    r"pH\s*(?:=|of|:|\s)\s*(\d+\.?\d*)",
    re.IGNORECASE,
)

# Sodium / NaCl / KCl concentration in mM
# Note: KCl is sometimes used as a Na surrogate in buffers; stored under na_concentration_mM
_NA = re.compile(
    r"(\d+\.?\d*)\s*mM\s*(?:NaCl|KCl|Na\s*Cl|Na\+|K\+|sodium\s*chloride|potassium\s*chloride|NaCl/KCl)",
    re.IGNORECASE,
)
# Also catch "150 mM salt" or "50 mM ionic strength" as Na proxy
_IONIC = re.compile(
    r"(\d+\.?\d*)\s*mM\s*(?:salt|ionic\s*strength)",
    re.IGNORECASE,
)

# Magnesium: MgCl2, Mg2+, MgSO4, magnesium
_MG = re.compile(
    r"(\d+\.?\d*)\s*mM\s*(?:MgCl\s*2|MgCl₂|Mg\s*2\+|Mg\s*²\+|MgSO\s*4|MgSO₄|magnesium)",
    re.IGNORECASE,
)

# Temperature: "37°C", "37 °C", "37 C", "37 degrees C", "at 37°C"
_TEMP = re.compile(
    r"(\d+\.?\d*)\s*°?\s*(?:degrees?\s*)?C(?:elsius)?(?=\s|,|\.|;|\))",
    re.IGNORECASE,
)

# Buffer keywords — ordered by specificity (more specific first)
_BUFFER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("PBS",    re.compile(r"\bPBS\b|phosphate[- ]buffered\s+saline|phosphate\s+buffered\s+saline", re.IGNORECASE)),
    ("HEPES",  re.compile(r"\bHEPES\b", re.IGNORECASE)),
    ("Tris",   re.compile(r"\bTris(?:-HCl|-EDTA|-base)?\b", re.IGNORECASE)),
    ("SELEX",  re.compile(r"\b(?:selection|binding|SELEX)\s+buffer\b", re.IGNORECASE)),
    ("Sodium phosphate", re.compile(r"\bsodium\s+phosphate\b", re.IGNORECASE)),
    ("other",  re.compile(r"\bbuffer\b", re.IGNORECASE)),
]

# Distinguish "selection buffer" (SELEX conditions) from "binding/assay buffer" (Kd measurement)
_SELECTION_BUFFER_CONTEXT = re.compile(r"\bselection\s+buffer\b", re.IGNORECASE)
_BINDING_BUFFER_CONTEXT   = re.compile(
    r"\b(?:binding|assay|measurement|incubation|SPR|ITC|EMSA)\s+buffer\b",
    re.IGNORECASE,
)

# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ConditionExtraction:
    """
    Experimental conditions extracted from a paper.
    All fields are Optional — None = not found in source.

    Attributes:
        ph                  pH of binding/selection buffer.
        na_concentration_mM Na⁺ (or K⁺) concentration in mM.
        mg_concentration_mM Mg²⁺ concentration in mM.
        temperature_C       Reaction temperature in °C.
        selection_buffer    Buffer used during SELEX selection.
        binding_buffer      Buffer used during Kd measurement / binding assay.
        all_buffers         All buffer mentions (list of strings), for audit.
    """
    ph:                  Optional[float]
    na_concentration_mM: Optional[float]
    mg_concentration_mM: Optional[float]
    temperature_C:       Optional[float]
    selection_buffer:    Optional[str]
    binding_buffer:      Optional[str]
    all_buffers:         list[str]


def _first_float(pattern: re.Pattern, text: str) -> Optional[float]:
    """Return float from first match of pattern in text, or None."""
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return None


def _detect_buffers(text: str) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Find selection_buffer, binding_buffer, and all buffer mentions.
    """
    all_bufs: list[str] = []
    selection: Optional[str] = None
    binding:   Optional[str] = None

    # Scan sentence-level context for buffer mentions
    # Split into sentences for context detection
    sentences = re.split(r"[.!?;]\s+", text)

    for sent in sentences:
        buf_type = None
        for name, pat in _BUFFER_PATTERNS:
            if pat.search(sent):
                buf_type = name
                break
        if buf_type is None:
            continue

        all_bufs.append(buf_type)

        if _SELECTION_BUFFER_CONTEXT.search(sent):
            if selection is None:
                selection = buf_type
        if _BINDING_BUFFER_CONTEXT.search(sent):
            if binding is None:
                binding = buf_type

    # If we found a buffer but couldn't distinguish context, use for both
    if all_bufs and selection is None and binding is None:
        selection = all_bufs[0]
        binding   = all_bufs[0]

    # Remove duplicate labels in all_bufs, preserve order
    seen: set[str] = set()
    deduped: list[str] = []
    for b in all_bufs:
        if b not in seen:
            seen.add(b)
            deduped.append(b)

    return selection, binding, deduped


# ── Public API ────────────────────────────────────────────────────────────────

def extract_conditions(text: str) -> ConditionExtraction:
    """
    Extract all experimental conditions from source text.

    Returns a ConditionExtraction with Optional fields; None means not found.
    Never raises — problematic matches are silently skipped.
    """
    ph   = _first_float(_PH,   text)
    mg   = _first_float(_MG,   text)
    temp = _first_float(_TEMP, text)

    # Na: try explicit NaCl/KCl first, fall back to generic salt/ionic-strength
    na = _first_float(_NA, text)
    if na is None:
        na = _first_float(_IONIC, text)

    # Temperature sanity guard — ignore body-of-text mentions like "37°C is body temp"
    # Only accept biologically plausible values (0–100°C)
    if temp is not None and not (0.0 <= temp <= 100.0):
        temp = None

    # pH sanity guard
    if ph is not None and not (0.0 <= ph <= 14.0):
        ph = None

    sel_buf, bind_buf, all_bufs = _detect_buffers(text)

    return ConditionExtraction(
        ph=ph,
        na_concentration_mM=na,
        mg_concentration_mM=mg,
        temperature_C=temp,
        selection_buffer=sel_buf,
        binding_buffer=bind_buf,
        all_buffers=all_bufs,
    )


if __name__ == "__main__":
    _sample = (
        "Aptamers were selected in PBS (pH 7.4) containing 150 mM NaCl "
        "and 2 mM MgCl2 at 37°C. Binding experiments were performed in "
        "binding buffer (50 mM HEPES pH 7.2, 150 mM NaCl, 2 mM MgCl2)."
    )
    c = extract_conditions(_sample)
    print(f"pH              : {c.ph}")
    print(f"Na (mM)         : {c.na_concentration_mM}")
    print(f"Mg (mM)         : {c.mg_concentration_mM}")
    print(f"Temp (°C)       : {c.temperature_C}")
    print(f"Selection buffer: {c.selection_buffer}")
    print(f"Binding buffer  : {c.binding_buffer}")
    print(f"All buffers     : {c.all_buffers}")
    assert c.ph   == 7.4
    assert c.na_concentration_mM == 150.0
    assert c.mg_concentration_mM == 2.0
    assert c.temperature_C == 37.0
    print("\ncondition_extractor self-test passed.")
