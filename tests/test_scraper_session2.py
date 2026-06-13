"""
Session 2 unit tests — extractors for the aptamer scraper.

Run with:
    cd continuitybioML
    source condaptnet_env/bin/activate
    python -m pytest tests/test_scraper_session2.py -v

Test cases use real-world examples from published aptamer papers.
No network calls — all UniProt-dependent tests use use_network=False.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═════════════════════════════════════════════════════════════════════════════
# 1. SequenceExtractor
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.extractors.sequence_extractor import (
    extract_sequences,
    ExtractedSequence,
    _normalise,
)

# Known real aptamer sequences used in tests
THROMBIN_29MER  = "AGTCCGTGGTAGGGCAGGTTGGGGTGACT"      # Macaya 1993, 29 nt
THROMBIN_15MER  = "GGTTGGTGTGGTTGG"                     # 15 nt — too short, rejected
VEGF_APTAMER    = "TGTGGGGGTGGACGGGCCGGGTAGA"           # 25 nt VEGF aptamer
INSULIN_APT     = "GCAATGGTACGGTACTTCCGGTACATGGTACGGTACTTCCAGCTTA"  # 46 nt
RNA_APTAMER     = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"  # tRNA-like, 76 nt

# Short invalid variants (for rejection tests)
TOO_SHORT       = "ATGCATGC"                             # 8 nt
LOW_GC          = "ATATATATATATATATATATATATATAT"         # 28 nt, 0% GC
HIGH_GC         = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"  # 87 nt, 100% GC


class TestSequenceExtractor:

    # ── Prime notation ────────────────────────────────────────────────────────

    def test_prime_notation_basic(self):
        text = f"The aptamer 5'-{THROMBIN_29MER}-3' was synthesised."
        results = extract_sequences(text)
        assert len(results) == 1
        assert results[0].sequence   == THROMBIN_29MER
        assert results[0].pattern    == "prime"
        assert results[0].nucleic_acid_type == "ssDNA"

    def test_prime_notation_curly_quotes(self):
        text = f"Sequence: 5’-{THROMBIN_29MER}-3’"
        results = extract_sequences(text)
        seqs = [r.sequence for r in results]
        assert THROMBIN_29MER in seqs

    def test_prime_notation_no_dash(self):
        text = f"The aptamer 5' {VEGF_APTAMER} 3' was tested."
        results = extract_sequences(text)
        seqs = [r.sequence for r in results]
        assert VEGF_APTAMER in seqs

    def test_rna_prime_notation(self):
        rna_seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"   # 31 nt ssRNA
        text = f"The RNA aptamer 5'-{rna_seq}-3' was selected."
        results = extract_sequences(text, valid_only=False)
        rna = [r for r in results if r.nucleic_acid_type == "ssRNA"]
        assert len(rna) >= 1
        assert rna[0].sequence.replace("T", "U") == rna_seq or rna[0].sequence == rna_seq

    # ── Primary (contiguous) ──────────────────────────────────────────────────

    def test_primary_inline(self):
        text = f"Selected aptamer {THROMBIN_29MER} showed high specificity."
        results = extract_sequences(text)
        assert any(r.sequence == THROMBIN_29MER for r in results)

    def test_primary_multiple_in_text(self):
        text = (f"Aptamer A ({THROMBIN_29MER}) competed with "
                f"aptamer B ({VEGF_APTAMER}) for binding.")
        results = extract_sequences(text)
        seqs = {r.sequence for r in results}
        assert THROMBIN_29MER in seqs
        assert VEGF_APTAMER   in seqs

    def test_too_short_rejected_valid_only(self):
        text = f"Short sequence {TOO_SHORT} was excluded."
        results = extract_sequences(text, valid_only=True)
        seqs = [r.sequence for r in results]
        assert TOO_SHORT not in seqs

    def test_too_short_below_regex_minimum(self):
        # Sequences <20 nt are below the regex {20,120} minimum — they are never
        # captured at all, even with valid_only=False. This is intentional: the
        # regex floor and the QC floor are both 20 nt, so there is nothing to flag.
        text = f"Short sequence {TOO_SHORT} was excluded."
        results = extract_sequences(text, valid_only=False)
        assert not any(r.sequence == TOO_SHORT for r in results)

    def test_15mer_thrombin_rejected(self):
        text = f"Classic 15-mer {THROMBIN_15MER} is too short."
        results = extract_sequences(text, valid_only=True)
        assert all(r.sequence != THROMBIN_15MER for r in results)

    def test_low_gc_rejected(self):
        results = extract_sequences(f"Seq: {LOW_GC}", valid_only=True)
        assert not any(r.sequence == LOW_GC for r in results)

    def test_high_gc_rejected(self):
        results = extract_sequences(f"Seq: {HIGH_GC}", valid_only=True)
        assert not any(r.sequence == HIGH_GC for r in results)

    def test_no_false_positive_from_lorem(self):
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        results = extract_sequences(text, valid_only=True)
        assert results == []

    # ── Spaced blocks ─────────────────────────────────────────────────────────

    def test_spaced_blocks(self):
        # 48-nt sequence split into 3 × 16-nt blocks
        block = "ATGCATGCATGCATGC"
        joined = block * 3
        text = f"Sequence: {block} {block} {block} was tested."
        results = extract_sequences(text, valid_only=True)
        seqs = [r.sequence for r in results]
        assert joined in seqs or any(len(s) == 48 for s in seqs)

    # ── Deduplication ─────────────────────────────────────────────────────────

    def test_same_sequence_two_patterns(self):
        # Same sequence in both prime notation and inline — should appear once
        text = (f"5'-{THROMBIN_29MER}-3' was the selected aptamer. "
                f"The sequence {THROMBIN_29MER} showed high affinity.")
        results = extract_sequences(text, valid_only=True)
        seqs = [r.sequence for r in results]
        assert seqs.count(THROMBIN_29MER) == 1

    # ── Context extraction ────────────────────────────────────────────────────

    def test_context_present(self):
        text = f"Selected aptamer {THROMBIN_29MER} showed high specificity."
        results = extract_sequences(text)
        assert results
        assert THROMBIN_29MER in results[0].context

    def test_start_end_offsets(self):
        text = f"___{THROMBIN_29MER}___"
        results = extract_sequences(text)
        if results:
            r = results[0]
            assert text[r.start:r.end] == THROMBIN_29MER

    # ── Normalisation ─────────────────────────────────────────────────────────

    def test_lowercase_input_normalised(self):
        text = f"aptamer {THROMBIN_29MER.lower()} was found."
        results = extract_sequences(text, valid_only=True)
        assert any(r.sequence == THROMBIN_29MER for r in results)

    def test_normalise_strips_spaces(self):
        assert _normalise("ATG CAT GC") == "ATGCATGC"

    def test_normalise_u_to_t(self):
        assert _normalise("AUGCAUGC") == "ATGCATGC"

    def test_normalise_rna_preserves_u(self):
        assert _normalise("AUGCAUGC", is_rna=True) == "AUGCAUGC"

    # ── Long sequence boundary ────────────────────────────────────────────────

    def test_120nt_accepted(self):
        seq = "ATGCATGCATGCATGCATGC" * 6        # 120 nt, 50% GC
        results = extract_sequences(f"Full-length: {seq}", valid_only=True)
        assert any(r.sequence == seq for r in results)

    def test_121nt_rejected(self):
        seq = "ATGCATGCATGCATGCATGC" * 6 + "A"  # 121 nt
        results = extract_sequences(f"Overlong: {seq}", valid_only=True)
        # The 120-nt prefix might be extracted; the 121-nt exact string should not pass
        assert not any(r.sequence == seq for r in results)


# ═════════════════════════════════════════════════════════════════════════════
# 2. KdExtractor
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.extractors.kd_extractor import (
    extract_kd_from_text,
    best_kd,
    ExtractedKd,
)


class TestKdExtractor:

    # ── Basic extraction ──────────────────────────────────────────────────────

    def test_kd_nM_equals(self):
        results = extract_kd_from_text("Kd = 5.2 nM.")
        assert len(results) == 1
        assert results[0].measure_type == "kd"
        assert abs(results[0].value_nM - 5.2) < 1e-6

    def test_kd_was_phrasing(self):
        results = extract_kd_from_text("The Kd was 5.2 nM.")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 5.2) < 1e-6

    def test_kd_capital(self):
        results = extract_kd_from_text("KD = 47 nM measured by SPR.")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 47.0) < 1e-6

    def test_kd_of_phrasing(self):
        results = extract_kd_from_text("A Kd of 112.5 nM was determined.")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 112.5) < 1e-6

    def test_kd_with_error_term(self):
        results = extract_kd_from_text("Kd = 47 ± 3 nM")
        assert results and abs(results[0].value_nM - 47.0) < 1e-6

    def test_kd_pM(self):
        results = extract_kd_from_text("Kd = 500 pM")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 0.5) < 1e-9

    def test_kd_uM(self):
        results = extract_kd_from_text("Kd = 0.5 µM")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 500.0) < 1e-6

    def test_kd_uM_ascii(self):
        results = extract_kd_from_text("Kd = 0.3 uM was measured.")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 300.0) < 1e-6

    def test_kd_mM(self):
        results = extract_kd_from_text("Kd = 0.001 mM")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 1000.0) < 1e-3

    # ── Scientific notation molar ─────────────────────────────────────────────

    def test_kd_sci_notation_molar(self):
        results = extract_kd_from_text("Kd = 2.3 × 10^-9 M")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 2.3) < 1e-6

    def test_kd_sci_notation_x(self):
        results = extract_kd_from_text("Kd = 5 x 10^-8 M (50 nM equivalent)")
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 50.0) < 1e-3

    # ── IC50 / EC50 / Ki are tagged separately ────────────────────────────────

    def test_ic50_extracted_with_correct_type(self):
        results = extract_kd_from_text("IC50 = 120 nM for this compound.")
        ic50 = [r for r in results if r.measure_type == "ic50"]
        assert ic50 and abs(ic50[0].value_nM - 120.0) < 1e-6

    def test_ec50_extracted(self):
        results = extract_kd_from_text("EC50 of 45 nM was observed.")
        ec50 = [r for r in results if r.measure_type == "ec50"]
        assert ec50 and abs(ec50[0].value_nM - 45.0) < 1e-6

    def test_ki_extracted(self):
        results = extract_kd_from_text("Ki of 75 nM for competitive binding.")
        ki = [r for r in results if r.measure_type == "ki"]
        assert ki and abs(ki[0].value_nM - 75.0) < 1e-6

    def test_kd_preferred_over_ic50(self):
        text = "Kd = 10 nM; IC50 = 300 nM"
        results = extract_kd_from_text(text)
        b = best_kd(results)
        assert b is not None
        assert b.measure_type == "kd"
        assert abs(b.value_nM - 10.0) < 1e-6

    def test_best_kd_no_ic50_uses_smallest(self):
        text = "Variant A: Kd = 100 nM. Variant B: Kd = 5 nM."
        results = extract_kd_from_text(text)
        b = best_kd(results)
        assert b and abs(b.value_nM - 5.0) < 1e-6

    # ── Multi-value texts ─────────────────────────────────────────────────────

    def test_multiple_kd_values(self):
        text = "Kd = 5.2 nM for aptamer A; KD of 47 ± 3 nM for aptamer B."
        results = extract_kd_from_text(text)
        kd_vals = sorted([r.value_nM for r in results if r.measure_type == "kd"])
        assert len(kd_vals) >= 2
        assert abs(kd_vals[0] - 5.2) < 1e-6
        assert abs(kd_vals[1] - 47.0) < 1e-6

    def test_deduplication_same_value(self):
        text = "Kd = 5 nM as determined by SPR. The Kd (5 nM) was confirmed by ITC."
        results = extract_kd_from_text(text, deduplicate=True)
        kd = [r for r in results if r.measure_type == "kd"]
        assert len(kd) == 1

    # ── Provenance / context ──────────────────────────────────────────────────

    def test_context_stored(self):
        text = "The aptamer had Kd = 5.2 nM in binding buffer."
        results = extract_kd_from_text(text)
        assert results and results[0].context != ""

    def test_start_end_offsets(self):
        text = "xxx Kd = 10 nM yyy"
        results = extract_kd_from_text(text)
        kd = [r for r in results if r.measure_type == "kd"]
        if kd:
            assert kd[0].start >= 0
            assert kd[0].end   > kd[0].start

    # ── No match ─────────────────────────────────────────────────────────────

    def test_no_kd_in_text(self):
        text = "No binding data was reported in this preliminary study."
        assert extract_kd_from_text(text) == []

    def test_partial_kd_not_matched(self):
        # "Kd" without a numeric value + unit → should not match
        text = "The Kd for this aptamer was not determined."
        assert extract_kd_from_text(text) == []

    # ── Thrombin benchmark ────────────────────────────────────────────────────

    def test_thrombin_benchmark_kd(self):
        text = "The thrombin-binding aptamer had a Kd of 112.5 nM (PMID 1741036)."
        results = extract_kd_from_text(text)
        kd = [r for r in results if r.measure_type == "kd"]
        assert kd and abs(kd[0].value_nM - 112.5) < 1e-6
        assert kd[0].original_unit == "nM"


# ═════════════════════════════════════════════════════════════════════════════
# 3. ConditionExtractor
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.extractors.condition_extractor import (
    extract_conditions,
    ConditionExtraction,
)


class TestConditionExtractor:

    def _text(self, **kwargs) -> str:
        """Build a realistic methods-section sentence from provided values."""
        parts = []
        if "ph"   in kwargs: parts.append(f"pH {kwargs['ph']}")
        if "na"   in kwargs: parts.append(f"{kwargs['na']} mM NaCl")
        if "mg"   in kwargs: parts.append(f"{kwargs['mg']} mM MgCl2")
        if "temp" in kwargs: parts.append(f"{kwargs['temp']}°C")
        if "buf"  in kwargs: parts.append(kwargs["buf"])
        return "Experiments were performed in " + ", ".join(parts) + "."

    def test_physiological_all_fields(self):
        text = ("Aptamers were selected in PBS (pH 7.4) containing 150 mM NaCl "
                "and 2 mM MgCl2 at 37°C.")
        c = extract_conditions(text)
        assert c.ph                  == pytest.approx(7.4)
        assert c.na_concentration_mM == pytest.approx(150.0)
        assert c.mg_concentration_mM == pytest.approx(2.0)
        assert c.temperature_C       == pytest.approx(37.0)

    def test_ph_equals_notation(self):
        c = extract_conditions("Buffer was adjusted to pH = 7.2.")
        assert c.ph == pytest.approx(7.2)

    def test_ph_of_notation(self):
        c = extract_conditions("At a pH of 6.8 the aptamer folded correctly.")
        assert c.ph == pytest.approx(6.8)

    def test_nacl_extraction(self):
        c = extract_conditions("Buffer: 50 mM NaCl, 10 mM Tris.")
        assert c.na_concentration_mM == pytest.approx(50.0)

    def test_kcl_counted_as_salt(self):
        c = extract_conditions("100 mM KCl was added to the binding buffer.")
        assert c.na_concentration_mM == pytest.approx(100.0)

    def test_mgcl2_extraction(self):
        c = extract_conditions("5 mM MgCl2 was required for folding.")
        assert c.mg_concentration_mM == pytest.approx(5.0)

    def test_mg2plus_notation(self):
        c = extract_conditions("1.5 mM Mg2+ was present.")
        assert c.mg_concentration_mM == pytest.approx(1.5)

    def test_temperature_celsius(self):
        c = extract_conditions("Incubation was at 25°C for 30 minutes.")
        assert c.temperature_C == pytest.approx(25.0)

    def test_temperature_degrees_c(self):
        c = extract_conditions("Selection was performed at 37 degrees C.")
        assert c.temperature_C == pytest.approx(37.0)

    def test_temperature_without_degree_symbol(self):
        c = extract_conditions("The reaction was incubated at 37 C for 1 h.")
        assert c.temperature_C == pytest.approx(37.0)

    def test_implausible_temperature_rejected(self):
        c = extract_conditions("The experiment was at 200°C (impossible).")
        assert c.temperature_C is None

    def test_implausible_ph_rejected(self):
        c = extract_conditions("The pH was 15.0 (out of range).")
        assert c.ph is None

    def test_buffer_pbs_detected(self):
        c = extract_conditions("Binding was measured in PBS buffer.")
        assert c.all_buffers and "PBS" in c.all_buffers

    def test_buffer_hepes_detected(self):
        c = extract_conditions("50 mM HEPES was used as the buffer.")
        assert c.all_buffers and "HEPES" in c.all_buffers

    def test_buffer_tris_detected(self):
        c = extract_conditions("10 mM Tris-HCl pH 8.0 was used.")
        assert c.all_buffers and "Tris" in c.all_buffers

    def test_no_conditions_found(self):
        c = extract_conditions("The aptamer sequence was AGTCCGTGGTAGGGCAGGTTGGGGTGACT.")
        assert c.ph is None
        assert c.na_concentration_mM is None
        assert c.mg_concentration_mM is None
        assert c.temperature_C is None

    def test_selection_vs_binding_buffer(self):
        text = ("SELEX selection buffer contained PBS. "
                "Binding assay buffer was HEPES.")
        c = extract_conditions(text)
        assert c.selection_buffer == "PBS"
        assert c.binding_buffer   == "HEPES"

    def test_all_buffers_deduplicated(self):
        text = "PBS was used. Another PBS experiment confirmed the result."
        c = extract_conditions(text)
        assert c.all_buffers.count("PBS") == 1

    def test_missing_fields_are_none(self):
        c = extract_conditions("No conditions were reported.")
        assert c.ph is None
        assert c.na_concentration_mM is None
        assert c.mg_concentration_mM is None
        assert c.temperature_C is None

    def test_returns_conditionextraction_type(self):
        c = extract_conditions("pH 7.4")
        assert isinstance(c, ConditionExtraction)


# ═════════════════════════════════════════════════════════════════════════════
# 4. TargetResolver
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.extractors.target_resolver import (
    classify_target_type,
    resolve_target,
    ResolvedTarget,
)


class TestTargetResolver:

    # ── Proteins ──────────────────────────────────────────────────────────────

    def test_thrombin_is_protein(self):
        assert classify_target_type("Thrombin") == "protein"

    def test_vegf_is_protein(self):
        assert classify_target_type("VEGF") == "protein"

    def test_insulin_is_protein(self):
        assert classify_target_type("Insulin") == "protein"

    def test_troponin_is_protein(self):
        assert classify_target_type("Troponin I") == "protein"

    def test_egfr_is_protein(self):
        assert classify_target_type("EGFR") == "protein"

    def test_hiv_rt_is_protein(self):
        assert classify_target_type("HIV-1 reverse transcriptase") == "protein"

    # ── Small molecules ───────────────────────────────────────────────────────

    def test_cocaine_is_small_molecule(self):
        assert classify_target_type("cocaine") == "small_molecule"

    def test_atp_is_small_molecule(self):
        assert classify_target_type("ATP") == "small_molecule"

    def test_theophylline_is_small_molecule(self):
        assert classify_target_type("theophylline") == "small_molecule"

    def test_kanamycin_is_small_molecule(self):
        assert classify_target_type("kanamycin") == "small_molecule"

    def test_dopamine_is_small_molecule(self):
        assert classify_target_type("dopamine") == "small_molecule"

    def test_doxorubicin_is_small_molecule(self):
        assert classify_target_type("doxorubicin") == "small_molecule"

    # ── Cells ─────────────────────────────────────────────────────────────────

    def test_hela_cell_line(self):
        assert classify_target_type("HeLa cell line") == "cell"

    def test_hek293(self):
        assert classify_target_type("HEK293 cells") == "cell"

    def test_ramos_cells(self):
        assert classify_target_type("RAMOS B cell line") == "cell"

    def test_whole_cell(self):
        assert classify_target_type("whole cell SELEX target") == "cell"

    # ── Organisms ─────────────────────────────────────────────────────────────

    def test_salmonella(self):
        assert classify_target_type("Salmonella typhimurium") == "organism"

    def test_hiv_virus(self):
        assert classify_target_type("HIV virus") == "organism"

    def test_sars_cov(self):
        assert classify_target_type("SARS-CoV-2 virus") == "organism"

    # ── Ions ─────────────────────────────────────────────────────────────────

    def test_lead_ion(self):
        assert classify_target_type("lead ion") == "ion"

    def test_mercury_ion(self):
        assert classify_target_type("mercury ion") == "ion"

    def test_cadmium_symbol(self):
        assert classify_target_type("Cd2+") == "ion"

    # ── Toxins ────────────────────────────────────────────────────────────────

    def test_ochratoxin(self):
        assert classify_target_type("Ochratoxin A") == "toxin"

    def test_aflatoxin(self):
        assert classify_target_type("Aflatoxin B1") == "toxin"

    # ── Peptides ─────────────────────────────────────────────────────────────

    def test_angiotensin_peptide(self):
        assert classify_target_type("angiotensin II peptide") == "peptide"

    def test_epitope_fragment(self):
        assert classify_target_type("fragment 1-100 peptide") == "peptide"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_string(self):
        assert classify_target_type("") == "other"

    def test_none_input(self):
        assert classify_target_type(None) == "other"  # type: ignore[arg-type]

    def test_unknown_defaults_to_protein(self):
        assert classify_target_type("XYZ-mystery-protein-42") == "protein"

    # ── resolve_target (offline) ──────────────────────────────────────────────

    def test_resolve_small_molecule_no_network(self):
        r = resolve_target("cocaine", use_network=False)
        assert isinstance(r, ResolvedTarget)
        assert r.target_type == "small_molecule"
        assert r.resolved    is False
        assert r.target_id   == ""

    def test_resolve_protein_offline(self):
        r = resolve_target("Thrombin", use_network=False)
        assert r.target_type == "protein"
        assert r.resolved    is False

    def test_resolve_cell_no_network(self):
        r = resolve_target("HeLa cell line", use_network=False)
        assert r.target_type == "cell"
        assert r.resolved    is False

    def test_resolve_returns_resolved_target_type(self):
        r = resolve_target("Unknown", use_network=False)
        assert isinstance(r, ResolvedTarget)

    def test_resolved_target_has_all_fields(self):
        r = resolve_target("Thrombin", use_network=False)
        assert hasattr(r, "name")
        assert hasattr(r, "target_type")
        assert hasattr(r, "target_id")
        assert hasattr(r, "target_id_source")
        assert hasattr(r, "protein_sequence")
        assert hasattr(r, "resolved")
        assert hasattr(r, "confidence")
