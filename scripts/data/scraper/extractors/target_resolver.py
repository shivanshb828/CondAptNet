"""
Target name → target_type classification + optional UniProt resolution.

Two independent functions:

  classify_target_type(name)
      Pure keyword matching. No network calls. Returns one of the seven
      target_type categorical values from the schema. Fast — call it for every row.

  resolve_target(name, use_network=True)
      Full resolution: classify type, then for proteins attempt UniProt lookup
      by delegating to the existing enrich_proteins.py logic (reused, not duplicated).
      Returns a ResolvedTarget dataclass.

The LLM is ONLY used (when explicitly requested) to map highly ambiguous target
names to standard names before keyword matching or UniProt search. It never
generates or modifies sequences or Kd values.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# ── Target type keyword rules ─────────────────────────────────────────────────
# Each entry: (target_type, compiled_regex)
# First match wins → order matters (more specific patterns first)

_TYPE_RULES: list[tuple[str, re.Pattern]] = [
    # Cell lines / whole cells
    ("cell", re.compile(
        r"\bcell\s*(?:line|surface|type)\b"
        r"|\bHeLa\b|\bHEK\s*293\b|\bPC12\b|\bJurkat\b|\bRamos\b"
        r"|\bMCF-?7\b|\bA549\b|\bHT-?29\b|\bU87\b|\bK562\b"
        r"|\bwhole\s*cell\b|\bliving\s*cell\b|\bcell-SELEX\b",
        re.IGNORECASE,
    )),

    # Organisms / bacteria / viruses
    # Viral names (HIV, SARS-CoV, etc.) require "virus/virion" to avoid matching
    # protein names like "HIV-1 reverse transcriptase".
    # Bacterial genus names are specific enough to stand alone.
    ("organism", re.compile(
        r"\b(?:Escherichia|Salmonella|Staphylococcus|Listeria|Bacillus|Clostridium|"
        r"Mycobacterium|Pseudomonas|Klebsiella|Vibrio|Campylobacter)\b"
        r"|\b(?:HIV|SARS-CoV|SARS-CoV-2|influenza|herpes|adenovirus|hepatitis|"
        r"coronavirus|parvovirus)\s+(?:virus|virion|particle)\b"
        r"|\bbacterium\b|\bbacteria\b|\b(?<!\w)virus\b|\bvirion\b"
        r"|\bfungus\b|\byeast\b|\bparasite\b",
        re.IGNORECASE,
    )),

    # Ions / heavy metals
    # Symbols like "Cd2+", "Pb2+", "Hg2+" use ASCII digit before + or unicode superscript
    ("ion", re.compile(
        r"\b(?:lead|mercury|cadmium|arsenic|copper|zinc|iron|cobalt|nickel|chromium|"
        r"manganese|silver|gold|platinum)\s*(?:ion|ions|[²³]?\+)?\b"
        r"|\b(?:Pb|Hg|Cd|As|Cu|Zn|Fe|Co|Ni|Cr|Mn|Ag|Au)\s*\d?[²³]?\+"
        r"|\bheavy\s*metal\b",
        re.IGNORECASE,
    )),

    # Toxins (mycotoxins, bacterial toxins)
    ("toxin", re.compile(
        r"\b(?:ochratoxin|aflatoxin|zearalenone|fumonisin|deoxynivalenol|"
        r"botulinum|ricin|abrin|tetrodotoxin|saxitoxin|microcystin)\b"
        r"|\btoxin\b|\bmycotoxin\b",
        re.IGNORECASE,
    )),

    # Small molecules (drugs, metabolites, dyes, nucleotides)
    ("small_molecule", re.compile(
        # Drugs
        r"\b(?:cocaine|codeine|morphine|heroin|amphetamine|methamphetamine|"
        r"ampicillin|kanamycin|tetracycline|streptomycin|doxorubicin|gentamicin|"
        r"chloramphenicol|penicillin|vancomycin|ciprofloxacin)\b"
        # Alkaloids / stimulants
        r"|\b(?:theophylline|caffeine|adenosine|dopamine|serotonin|epinephrine|"
        r"norepinephrine)\b"
        # Nucleotides (free form)
        r"|\b(?:ATP|ADP|AMP|GTP|GDP|NAD\+?|FAD|cAMP|cGMP)\b"
        # Dyes
        r"|\b(?:sulforhodamine|rhodamine|fluorescein|malachite\s*green|crystal\s*violet)\b"
        # Vitamins / metabolites
        r"|\b(?:riboflavin|folic\s*acid|vitamin\s*[A-EK]\d?|biotin)\b"
        # Other small molecules
        r"|\b(?:glucose|sucrose|fructose|lactose|cholesterol|cortisol|estradiol|"
        r"progesterone|testosterone|bisphenol|PFAS|PCB)\b",
        re.IGNORECASE,
    )),

    # Peptides (short, non-full-protein targets)
    ("peptide", re.compile(
        r"\b(?:peptide|epitope|fragment\s*\d+[-–]\d+|aa\s*\d+[-–]\d+)\b"
        r"|\b(?:angiotensin|bradykinin|substance\s*P|oxytocin|vasopressin|"
        r"glucagon|insulin\s*B\s*chain)\b",
        re.IGNORECASE,
    )),
]

# ── UniProt integration (import from existing enrich_proteins.py) ─────────────

def _import_enrich():
    """Lazy import to avoid loading UniProt code when only classification is needed."""
    import importlib, sys
    # enrich_proteins.py is at scripts/data/enrich_proteins.py
    # It's already on sys.path since we inserted project root above
    from scripts.data.enrich_proteins import (
        find_sequence, is_non_protein, load_overrides
    )
    return find_sequence, is_non_protein, load_overrides


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ResolvedTarget:
    """
    Resolution result for a target name.

    Attributes:
        name            Original target name as given.
        target_type     Schema categorical: protein | peptide | small_molecule | ...
        target_id       UniProt accession (proteins), PubChem CID (small molecules),
                        or empty string if unresolved.
        target_id_source  Schema categorical: UniProt | PubChem | ... or empty.
        protein_sequence  Full amino acid sequence (for proteins), or None.
        resolved        True if a database ID was found.
        confidence      "high" (exact DB match), "medium" (fuzzy match), "low" (keyword only)
    """
    name:             str
    target_type:      str
    target_id:        str
    target_id_source: str
    protein_sequence: Optional[str]
    resolved:         bool
    confidence:       str   # "high" | "medium" | "low"


# ── Public API ────────────────────────────────────────────────────────────────

def classify_target_type(name: str) -> str:
    """
    Classify a target name into one of the schema's target_type categories.

    Pure keyword matching — no network calls. Fast.

    Returns one of: "protein" | "peptide" | "small_molecule" | "cell" |
                    "organism" | "ion" | "toxin" | "other"
    """
    if not isinstance(name, str) or not name.strip():
        return "other"

    for target_type, pattern in _TYPE_RULES:
        if pattern.search(name):
            return target_type

    # Default: protein (the most common aptamer target type)
    return "protein"


def resolve_target(
    name:          str,
    use_network:   bool = True,
    overrides_path: Optional[str] = None,
) -> ResolvedTarget:
    """
    Full target resolution: classify type + optional UniProt/DB lookup.

    Args:
        name:           Target name as it appears in the source document.
        use_network:    If True (default), attempt UniProt lookup for proteins.
                        Set False for offline/test use.
        overrides_path: Path to protein_name_overrides.csv, or None for default.

    Returns:
        ResolvedTarget dataclass.
    """
    target_type = classify_target_type(name)

    # Non-protein targets — return classification only (no UniProt lookup)
    if target_type != "protein":
        return ResolvedTarget(
            name=name,
            target_type=target_type,
            target_id="",
            target_id_source="",
            protein_sequence=None,
            resolved=False,
            confidence="low",
        )

    # Protein — attempt UniProt resolution
    if not use_network:
        return ResolvedTarget(
            name=name,
            target_type="protein",
            target_id="",
            target_id_source="",
            protein_sequence=None,
            resolved=False,
            confidence="low",
        )

    try:
        find_sequence, is_non_protein, load_overrides = _import_enrich()

        # Double-check: if enrich_proteins marks it as non-protein, reclassify
        if is_non_protein(name):
            new_type = _reclassify_non_protein(name)
            return ResolvedTarget(
                name=name,
                target_type=new_type,
                target_id="",
                target_id_source="",
                protein_sequence=None,
                resolved=False,
                confidence="low",
            )

        overrides = load_overrides(overrides_path) if overrides_path else load_overrides()
        hit = find_sequence(name, overrides=overrides)

        if hit:
            acc, seq = hit
            return ResolvedTarget(
                name=name,
                target_type="protein",
                target_id=acc,
                target_id_source="UniProt",
                protein_sequence=seq,
                resolved=True,
                confidence="high",
            )

    except Exception:
        pass

    # Protein but UniProt lookup failed
    return ResolvedTarget(
        name=name,
        target_type="protein",
        target_id="",
        target_id_source="",
        protein_sequence=None,
        resolved=False,
        confidence="low",
    )


def _reclassify_non_protein(name: str) -> str:
    """
    When enrich_proteins.is_non_protein() flags a name, map it to the right
    schema category.
    """
    name_l = name.lower()
    if any(w in name_l for w in ("cell line", "hela", "hek", "pc12")):
        return "cell"
    if any(w in name_l for w in ("toxin", "mycotoxin", "aflatoxin", "ochratoxin")):
        return "toxin"
    if any(w in name_l for w in ("ion", "lead", "mercury", "cadmium")):
        return "ion"
    return "small_molecule"


if __name__ == "__main__":
    _CASES: list[tuple[str, str]] = [
        ("Thrombin",                       "protein"),
        ("VEGF",                           "protein"),
        ("Insulin",                        "protein"),
        ("HeLa cell line",                 "cell"),
        ("RAMOS B cells",                  "cell"),
        ("cocaine",                        "small_molecule"),
        ("ATP",                            "small_molecule"),
        ("theophylline",                   "small_molecule"),
        ("Ochratoxin A",                   "toxin"),
        ("lead ion",                       "ion"),
        ("Cd2+",                           "ion"),
        ("HIV virus",                      "organism"),
        ("Salmonella typhimurium",         "organism"),
        ("angiotensin II peptide",         "peptide"),
    ]

    print("target_resolver classify_target_type self-test:")
    all_ok = True
    for name, expected in _CASES:
        got = classify_target_type(name)
        ok  = got == expected
        status = "OK" if ok else f"FAIL (expected {expected!r}, got {got!r})"
        print(f"  [{status[:2]}] {name!r:40s} → {got}")
        if not ok:
            all_ok = False
    print(f"\n{'All tests passed.' if all_ok else 'SOME TESTS FAILED.'}")
