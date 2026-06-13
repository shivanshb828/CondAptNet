"""
DNA/RNA aptamer sequence extractor.

Applies four regex strategies in priority order and validates every candidate
through the same QC rules used for the training dataset.

Strategies (tried in order; first match for a given sequence wins):
  1. prime   — 5'→3' notation: "5'-ATGCATGC...-3'"
  2. primary — contiguous ATGC{20-120} word-boundary anchored
  3. spaced  — ATGC blocks separated by whitespace (formatted blocks in papers)
  4. rna     — contiguous AUGC{20-120}; stored as ssRNA, U→T for validation

ZERO-TOLERANCE RULE: sequences are returned exactly as matched then normalised
(strip whitespace, uppercase, U→T for DNA storage). An LLM never touches them.
Byte offsets are stored for provenance verification.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.data.validate_sequences import validate_sequence
from scripts.data.scraper.utils.provenance import text_context

# ── Regex patterns ────────────────────────────────────────────────────────────

# Priority 1: 5'→3' notation — most explicit, highest confidence
_SEQ_PRIME = re.compile(
    r"5['’ʼ]\s*-?\s*([ATGCUatgcu]{20,120})\s*-?\s*3['’ʼ]"
)

# Priority 2: contiguous ATGC run, word-boundary anchored
_SEQ_PRIMARY = re.compile(r"\b([ATGCatgc]{20,120})\b")

# Priority 3: space-separated blocks (some papers typeset long sequences in chunks)
# e.g. "ATGCATGC GCATGCAT ATGCATGC" — joined they must be 20-120 nt
_SEQ_SPACED = re.compile(
    r"\b([ATGCatgc]{5,}(?:[ \t]+[ATGCatgc]{5,}){1,20})\b"
)

# Priority 4: RNA sequences (allow U)
_SEQ_RNA_PRIMARY = re.compile(r"\b([AUGCaugc]{20,120})\b")
_SEQ_RNA_PRIME   = re.compile(
    r"5['’ʼ]\s*-?\s*([AUGCaugc]{20,120})\s*-?\s*3['’ʼ]"
)

# ── Data class ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtractedSequence:
    """
    A validated aptamer sequence extracted from source text.

    Attributes:
        sequence          Normalised sequence stored in database
                          (uppercase, spaces stripped, U→T for DNA entries).
        raw_match         Exactly what was matched in the source text.
        start             Start byte offset in the source text string.
        end               End byte offset.
        pattern           Which strategy found it: "prime" | "primary" | "spaced" | "rna_prime" | "rna"
        nucleic_acid_type "ssDNA" or "ssRNA"
        context           ±200 characters surrounding the match (for provenance).
        valid             Whether the sequence passed QC validation.
        fail_reason       Why it failed (empty string when valid=True).
    """
    sequence:          str
    raw_match:         str
    start:             int
    end:               int
    pattern:           str
    nucleic_acid_type: str
    context:           str
    valid:             bool
    fail_reason:       str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise(raw: str, is_rna: bool = False) -> str:
    """Strip whitespace, uppercase. For DNA storage: U→T."""
    s = raw.replace(" ", "").replace("\t", "").upper()
    if not is_rna:
        s = s.replace("U", "T")
    return s


def _validate_rna(seq: str) -> tuple[bool, str]:
    """
    Validate an RNA sequence (AUGC alphabet) with the same QC bounds.
    Mirrors validate_sequence() but allows U in place of T.
    """
    from config import SEQ_MIN_LEN, SEQ_MAX_LEN, GC_MIN, GC_MAX, MAX_HOMOPOLYMER
    import re as _re

    if len(seq) < SEQ_MIN_LEN:
        return False, f"too_short ({len(seq)} < {SEQ_MIN_LEN})"
    if len(seq) > SEQ_MAX_LEN:
        return False, f"too_long ({len(seq)} > {SEQ_MAX_LEN})"

    illegal = set(seq) - set("AUGC")
    if illegal:
        return False, f"invalid_bases ({','.join(sorted(illegal))})"

    gc = (seq.count("G") + seq.count("C")) / len(seq)
    if gc < GC_MIN:
        return False, f"gc_too_low ({gc:.2f} < {GC_MIN})"
    if gc > GC_MAX:
        return False, f"gc_too_high ({gc:.2f} > {GC_MAX})"

    hp = _re.compile(rf"(A{{{MAX_HOMOPOLYMER+1},}}|U{{{MAX_HOMOPOLYMER+1},}}|G{{{MAX_HOMOPOLYMER+1},}}|C{{{MAX_HOMOPOLYMER+1},}})")
    if hp.search(seq):
        return False, f"homopolymer_run (>{MAX_HOMOPOLYMER} identical bases)"

    return True, ""


def _make(
    text:    str,
    raw:     str,
    start:   int,
    end:     int,
    pattern: str,
    is_rna:  bool,
) -> ExtractedSequence:
    norm = _normalise(raw, is_rna=is_rna)
    na_type = "ssRNA" if is_rna else "ssDNA"
    ctx = text_context(text, start, end)

    if is_rna:
        # Validate the RNA form (AUGC); store ssRNA sequence as-is (with U)
        ok, reason = _validate_rna(norm.replace("T", "U"))
        stored_seq = norm.replace("T", "U")   # preserve U in stored RNA sequence
    else:
        ok, reason = validate_sequence(norm)
        stored_seq = norm

    return ExtractedSequence(
        sequence=stored_seq,
        raw_match=raw,
        start=start,
        end=end,
        pattern=pattern,
        nucleic_acid_type=na_type,
        context=ctx,
        valid=ok,
        fail_reason=reason,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_sequences(
    text:        str,
    valid_only:  bool = True,
    source_url:  str  = "",
) -> list[ExtractedSequence]:
    """
    Extract and validate all aptamer sequences from a text string.

    Args:
        text:       Source document text (UTF-8 string).
        valid_only: If True (default), return only QC-passing sequences.
        source_url: Passed through to logs; not stored on the dataclass.

    Returns:
        List of ExtractedSequence objects, deduplicated by normalised sequence.
        Within a deduplicated set, the highest-priority pattern match is kept.
    """
    candidates: list[ExtractedSequence] = []

    # Apply patterns in priority order (prime > primary > spaced > rna_prime > rna).
    # Tracking seen ranges prevents one sequence from being double-counted by
    # overlapping patterns.
    seen_ranges: set[tuple[int, int]] = set()

    def _add(m: re.Match, pattern: str, is_rna: bool) -> None:
        span = m.span(1)
        if span in seen_ranges:
            return
        seen_ranges.add(span)
        raw = m.group(1)
        candidates.append(_make(text, raw, span[0], span[1], pattern, is_rna))

    for m in _SEQ_RNA_PRIME.finditer(text):
        _add(m, "rna_prime", is_rna=True)

    for m in _SEQ_PRIME.finditer(text):
        # Check if any U present → treat as RNA prime
        is_rna = "U" in m.group(1).upper() or "u" in m.group(1)
        _add(m, "prime", is_rna=is_rna)

    for m in _SEQ_PRIMARY.finditer(text):
        _add(m, "primary", is_rna=False)

    for m in _SEQ_SPACED.finditer(text):
        raw_joined = m.group(1).replace(" ", "").replace("\t", "")
        if 20 <= len(raw_joined) <= 120:
            _add(m, "spaced", is_rna=False)

    for m in _SEQ_RNA_PRIMARY.finditer(text):
        raw = m.group(1).upper()
        # Skip if all bases are ATGC (already captured by primary pattern)
        if not set(raw) - set("ATGC"):
            continue
        _add(m, "rna", is_rna=True)

    # Deduplicate by normalised sequence — keep first (highest-priority) occurrence
    seen_seqs: dict[str, ExtractedSequence] = {}
    for c in candidates:
        key = c.sequence
        if key not in seen_seqs:
            seen_seqs[key] = c

    results = list(seen_seqs.values())
    if valid_only:
        results = [r for r in results if r.valid]
    return results


if __name__ == "__main__":
    _TESTS = [
        # (description, text, expected_sequences, expected_na_types)
        (
            "prime notation ssDNA",
            "The aptamer 5'-AGTCCGTGGTAGGGCAGGTTGGGGTGACT-3' was synthesised.",
            ["AGTCCGTGGTAGGGCAGGTTGGGGTGACT"],
            ["ssDNA"],
        ),
        (
            "inline primary",
            "Selected aptamer AGTCCGTGGTAGGGCAGGTTGGGGTGACT showed high specificity.",
            ["AGTCCGTGGTAGGGCAGGTTGGGGTGACT"],
            ["ssDNA"],
        ),
        (
            "space-separated blocks",
            "Sequence: ATGCATGCATGCATGC GCATGCATGCATGCAT ATGCATGCATGCATGC",
            ["ATGCATGCATGCATGCGCATGCATGCATGCATATGCATGCATGCATGC"],
            ["ssDNA"],
        ),
        (
            "RNA prime notation",
            "The RNA aptamer 5'-AUGCAUGCAUGCAUGCAUGCAUGCAUGC-3' was selected.",
            ["AUGCAUGCAUGCAUGCAUGCAUGCAUGC"],
            ["ssRNA"],
        ),
        (
            "too short — rejected",
            "Short seq GGTTGG was ignored.",
            [],
            [],
        ),
        (
            "two sequences in same text",
            "Aptamer A: AGTCCGTGGTAGGGCAGGTTGGGGTGACT "
            "Aptamer B: ATGCATGCATGCATGCATGCATGCATGCATGC",
            ["AGTCCGTGGTAGGGCAGGTTGGGGTGACT", "ATGCATGCATGCATGCATGCATGCATGCATGC"],
            ["ssDNA", "ssDNA"],
        ),
    ]

    print("sequence_extractor self-test:")
    all_ok = True
    for desc, text, expected_seqs, expected_na in _TESTS:
        results = extract_sequences(text, valid_only=True)
        got_seqs = [r.sequence for r in results]
        got_na   = [r.nucleic_acid_type for r in results]
        ok = set(got_seqs) == set(expected_seqs) and set(got_na) == set(expected_na)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         expected: {expected_seqs}")
            print(f"         got:      {got_seqs}")
    print(f"\n{'All tests passed.' if all_ok else 'SOME TESTS FAILED.'}")
