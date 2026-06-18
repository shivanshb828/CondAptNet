"""
Session 1 unit tests — foundation utilities for the aptamer scraper.

Run with:
    cd continuitybioML
    source condaptnet_env/bin/activate
    python -m pytest tests/test_scraper_session1.py -v

Tests cover:
    - KdConverter: unit conversion, regex extraction, error cases
    - Schema: validation logic, required fields, categoricals
    - Deduplicator: new vs duplicate, normalisation, batch filter
    - RateLimiter: interval enforcement, context manager
    - ProvenanceLogger: write/read round-trip, threading
    - validate_sequence (existing): length, GC, homopolymer, duplicates
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import threading
from pathlib import Path

import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═════════════════════════════════════════════════════════════════════════════
# 1. KdConverter
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.utils.kd_converter import (
    convert_kd,
    extract_kd_from_text,
    parse_kd_string,
    KdMeasurement,
)


class TestKdConverter:
    def test_pm_to_nM(self):
        m = convert_kd(1000.0, "pM")
        assert abs(m.value_nM - 1.0) < 1e-9
        assert m.original_value == 1000.0
        assert m.original_unit  == "pM"
        assert m.canonical_unit == "pM"

    def test_nM_unchanged(self):
        m = convert_kd(112.5, "nM")
        assert abs(m.value_nM - 112.5) < 1e-9
        assert m.canonical_unit == "nM"

    def test_uM_to_nM(self):
        m = convert_kd(0.3, "µM")
        assert abs(m.value_nM - 300.0) < 1e-9
        assert m.canonical_unit == "µM"

    def test_uM_ascii_to_nM(self):
        """ASCII 'uM' should normalise to µM and convert correctly."""
        m = convert_kd(1.0, "uM")
        assert abs(m.value_nM - 1000.0) < 1e-9
        assert m.canonical_unit == "µM"

    def test_mM_to_nM(self):
        m = convert_kd(0.001, "mM")
        assert abs(m.value_nM - 1000.0) < 1e-9

    def test_case_insensitive_nm(self):
        m = convert_kd(5.0, "NM")
        assert abs(m.value_nM - 5.0) < 1e-9

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            convert_kd(-1.0, "nM")

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            convert_kd(1.0, "ng/mL")

    def test_thrombin_benchmark_kd(self):
        """Thrombin aptamer GGTTGGTGTGGTTGG: Kd = 112.5 nM (PMID 1741036)."""
        m = convert_kd(112.5, "nM")
        assert abs(m.value_nM - 112.5) < 1e-9

    def test_zero_value(self):
        m = convert_kd(0.0, "nM")
        assert m.value_nM == 0.0

    # ── Regex extraction ──────────────────────────────────────────────────────

    def test_extract_single_nM(self):
        text = "The aptamer had a Kd of 5.2 nM under physiological conditions."
        results = extract_kd_from_text(text)
        assert len(results) == 1
        assert abs(results[0].value_nM - 5.2) < 1e-9

    def test_extract_multiple(self):
        text = "Kd = 5.2 nM for variant A and KD = 0.3 µM for variant B."
        results = extract_kd_from_text(text)
        assert len(results) == 2
        assert abs(results[0].value_nM - 5.2)   < 1e-9
        assert abs(results[1].value_nM - 300.0) < 1e-9

    def test_extract_with_error_term(self):
        text = "Kd = 47 ± 3 nM measured by SPR."
        results = extract_kd_from_text(text)
        assert len(results) == 1
        assert abs(results[0].value_nM - 47.0) < 1e-9

    def test_extract_pM(self):
        text = "The high-affinity aptamer showed Kd = 500 pM."
        results = extract_kd_from_text(text)
        assert len(results) == 1
        assert abs(results[0].value_nM - 0.5) < 1e-9

    def test_extract_no_match(self):
        text = "No binding data was reported in this study."
        results = extract_kd_from_text(text)
        assert results == []

    def test_parse_kd_string_nM(self):
        m = parse_kd_string("112.5 nM")
        assert m is not None
        assert abs(m.value_nM - 112.5) < 1e-9

    def test_parse_kd_string_uM(self):
        m = parse_kd_string("0.5 µM")
        assert m is not None
        assert abs(m.value_nM - 500.0) < 1e-9

    def test_parse_kd_string_invalid(self):
        m = parse_kd_string("strong binding")
        assert m is None

    def test_immutability(self):
        m = convert_kd(1.0, "nM")
        with pytest.raises((AttributeError, TypeError)):
            m.value_nM = 999.0  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Schema validation
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.schema import (
    validate_record,
    make_empty_record,
    empty_dataframe,
    records_to_dataframe,
    SCHEMA_COLUMNS,
    REQUIRED_COLUMNS,
)


def _valid_record(**overrides) -> dict:
    """Minimal valid record with all required fields filled."""
    base = {
        "aptamer_sequence": "GGTTGGTGTGGTTGG",
        "nucleic_acid_type": "ssDNA",
        "modifications": "none",
        "target_name": "thrombin",
        "target_type": "protein",
        "confidence_score": "curated",
        "split": "train",
    }
    base.update(overrides)
    return base


class TestSchema:
    def test_minimal_valid_record(self):
        ok, errors = validate_record(_valid_record())
        assert ok, f"Expected valid, got errors: {errors}"

    def test_full_record_valid(self):
        rec = _valid_record(
            kd_value=112.5,
            kd_unit="nM",
            ph=7.4,
            na_concentration_mM=150.0,
            mg_concentration_mM=2.0,
            temperature_C=37.0,
            source_doi="10.1234/test",
            source_type="paper",
            target_id="P00734",
            target_id_source="UniProt",
        )
        ok, errors = validate_record(rec)
        assert ok, errors

    def test_missing_required_field(self):
        for col in REQUIRED_COLUMNS:
            rec = _valid_record()
            del rec[col]
            ok, errors = validate_record(rec)
            assert not ok
            assert any(col in e for e in errors)

    def test_empty_required_field(self):
        rec = _valid_record(target_name="")
        ok, errors = validate_record(rec)
        assert not ok
        assert any("target_name" in e for e in errors)

    def test_invalid_nucleic_acid_type(self):
        rec = _valid_record(nucleic_acid_type="DNA")   # not in allowed set
        ok, errors = validate_record(rec)
        assert not ok
        assert any("nucleic_acid_type" in e for e in errors)

    def test_invalid_confidence_score(self):
        rec = _valid_record(confidence_score="verified")
        ok, errors = validate_record(rec)
        assert not ok

    def test_invalid_split_value(self):
        rec = _valid_record(split="training")
        ok, errors = validate_record(rec)
        assert not ok

    def test_negative_kd_value(self):
        rec = _valid_record(kd_value=-5.0, kd_unit="nM")
        ok, errors = validate_record(rec)
        assert not ok
        assert any("kd_value" in e for e in errors)

    def test_kd_value_without_unit(self):
        rec = _valid_record(kd_value=5.0)
        ok, errors = validate_record(rec)
        assert not ok
        assert any("kd_unit" in e for e in errors)

    def test_kd_unit_without_value(self):
        rec = _valid_record(kd_unit="nM")
        ok, errors = validate_record(rec)
        assert not ok

    def test_ph_out_of_range(self):
        rec = _valid_record(ph=15.0)
        ok, errors = validate_record(rec)
        assert not ok
        assert any("ph" in e for e in errors)

    def test_non_numeric_temperature(self):
        rec = _valid_record(temperature_C="room temperature")
        ok, errors = validate_record(rec)
        assert not ok

    def test_blank_optional_fields_ok(self):
        rec = _valid_record(target_id="", kd_value="", kd_unit="", source_doi="")
        ok, errors = validate_record(rec)
        assert ok, errors

    def test_schema_has_20_columns(self):
        assert len(SCHEMA_COLUMNS) == 20

    def test_empty_dataframe_columns(self):
        df = empty_dataframe()
        assert list(df.columns) == SCHEMA_COLUMNS
        assert len(df) == 0

    def test_records_to_dataframe(self):
        recs = [_valid_record(kd_value=5.0, kd_unit="nM"),
                _valid_record(target_name="insulin")]
        df = records_to_dataframe(recs)
        assert len(df) == 2
        assert list(df.columns) == SCHEMA_COLUMNS
        assert df["kd_value"].iloc[0] == 5.0

    def test_make_empty_record(self):
        rec = make_empty_record()
        assert set(rec.keys()) == set(SCHEMA_COLUMNS)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Deduplicator
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.utils.deduplication import Deduplicator
import pandas as pd


class TestDeduplicator:
    def test_new_entry_not_duplicate(self):
        d = Deduplicator()
        assert not d.is_duplicate("GGTTGGTGTGGTTGG", "thrombin")

    def test_registered_entry_is_duplicate(self):
        d = Deduplicator()
        d.register("GGTTGGTGTGGTTGG", "thrombin")
        assert d.is_duplicate("GGTTGGTGTGGTTGG", "thrombin")

    def test_sequence_normalisation(self):
        d = Deduplicator()
        d.register("GGTTGGTGTGGTTGG", "Thrombin")
        # lowercase sequence, mixed-case target → still duplicate
        assert d.is_duplicate("ggttggtgtggttgg", "THROMBIN")

    def test_same_seq_different_target_not_duplicate(self):
        d = Deduplicator()
        d.register("GGTTGGTGTGGTTGG", "thrombin")
        assert not d.is_duplicate("GGTTGGTGTGGTTGG", "VEGF")

    def test_register_returns_dup_flag(self):
        d = Deduplicator()
        assert not d.register("AAATTTGGGCCCAAATTTGGG", "target1")   # first → False
        assert     d.register("AAATTTGGGCCCAAATTTGGG", "target1")   # second → True

    def test_size_tracks_unique_entries(self):
        d = Deduplicator()
        d.register("AAATTTGGGCCCAAATTTGGG", "t1")
        d.register("AAATTTGGGCCCAAATTTGGG", "t1")  # duplicate — not counted again
        d.register("GCGCGCGCGCGCGCGCGCGC", "t1")
        assert d.size == 2

    def test_from_master_old_schema(self):
        csv = (
            "sequence,target_protein\n"
            "GGTTGGTGTGGTTGG,Thrombin\n"
            "ATCGATCGATCGATCGATCG,Insulin\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(csv)
            tmp = f.name
        try:
            d = Deduplicator.from_master(tmp)
            assert d.size == 2
            assert d.is_duplicate("GGTTGGTGTGGTTGG", "Thrombin")
            assert d.is_duplicate("ATCGATCGATCGATCGATCG", "Insulin")
            assert not d.is_duplicate("ATCGATCGATCGATCGATCG", "VEGF")
        finally:
            os.unlink(tmp)

    def test_from_master_missing_file(self):
        d = Deduplicator.from_master("/nonexistent/path.csv")
        assert d.size == 0

    def test_filter_dataframe(self):
        d = Deduplicator()
        d.register("GGTTGGTGTGGTTGG", "thrombin")   # pre-seed

        df = pd.DataFrame({
            "aptamer_sequence": ["GGTTGGTGTGGTTGG", "ATCGATCGATCGATCGATCG", "GCGCGCGCGCGCGCGCGCGC"],
            "target_name":      ["thrombin",         "insulin",               "thrombin"],
        })
        unique, dups = d.filter_dataframe(df)
        assert len(unique) == 2
        assert len(dups)   == 1
        assert dups.iloc[0]["aptamer_sequence"] == "GGTTGGTGTGGTTGG"

    def test_intra_batch_dedup(self):
        d = Deduplicator()
        df = pd.DataFrame({
            "aptamer_sequence": ["GGTTGGTGTGGTTGG", "GGTTGGTGTGGTTGG"],
            "target_name":      ["thrombin",         "thrombin"],
        })
        unique, dups = d.filter_dataframe(df)
        assert len(unique) == 1
        assert len(dups)   == 1


# ═════════════════════════════════════════════════════════════════════════════
# 4. RateLimiter
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.utils.rate_limiter import RateLimiter, get_limiter, reset_all


class TestRateLimiter:
    def test_first_call_is_instant(self):
        lim = RateLimiter(100.0)   # 100 req/s → 10ms gap
        t0 = time.monotonic()
        lim.wait()
        assert time.monotonic() - t0 < 0.05

    def test_interval_enforced(self):
        lim = RateLimiter(5.0)   # 5 req/s → 0.2s gap
        lim.wait()               # burn the instant first call
        t0 = time.monotonic()
        lim.wait()
        elapsed = time.monotonic() - t0
        assert 0.15 <= elapsed <= 0.40, f"Expected ~0.2s, got {elapsed:.3f}s"

    def test_context_manager(self):
        lim = RateLimiter(1000.0)
        with lim:
            pass   # should not raise

    def test_invalid_rps_raises(self):
        with pytest.raises(ValueError):
            RateLimiter(0.0)
        with pytest.raises(ValueError):
            RateLimiter(-1.0)

    def test_get_limiter_registry(self):
        reset_all()
        lim1 = get_limiter("pubmed")
        lim2 = get_limiter("pubmed")
        assert lim1 is lim2   # same instance

    def test_different_sources_independent(self):
        reset_all()
        lim_pub = get_limiter("pubmed")
        lim_oax = get_limiter("openalex")
        assert lim_pub is not lim_oax

    def test_thread_safety(self):
        """10 threads share one limiter at 50 req/s — no deadlock."""
        lim = RateLimiter(50.0)
        results = []

        def call():
            lim.wait()
            results.append(time.monotonic())

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(results) == 10
        assert all(t is not None for t in results)


# ═════════════════════════════════════════════════════════════════════════════
# 5. ProvenanceLogger
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.scraper.utils.provenance import ProvenanceLogger, load_provenance, text_context
import json


class TestProvenance:
    def test_write_and_read(self, tmp_path):
        log_path = tmp_path / "prov.jsonl"
        with ProvenanceLogger(log_path) as plog:
            plog.record(
                aptamer_sequence="GGTTGGTGTGGTTGG",
                target_name="thrombin",
                source_url="https://pubmed.ncbi.nlm.nih.gov/1741036/",
                source_type="paper",
                extraction_method="regex",
                raw_text_context="...aptamer GGTTGGTGTGGTTGG binds...",
                byte_offset=12345,
                source_file_hash="deadbeef",
            )
        records = load_provenance(log_path)
        assert len(records) == 1
        r = records[0]
        assert r["aptamer_sequence"]  == "GGTTGGTGTGGTTGG"
        assert r["target_name"]       == "thrombin"
        assert r["byte_offset"]       == 12345
        assert r["extraction_method"] == "regex"
        assert "extraction_timestamp" in r

    def test_multiple_records(self, tmp_path):
        log_path = tmp_path / "prov.jsonl"
        with ProvenanceLogger(log_path) as plog:
            for i in range(5):
                plog.record(
                    aptamer_sequence=f"ATGCATGCATGCATGCATGC{i}",
                    target_name=f"protein{i}",
                    source_url=f"https://example.com/{i}",
                    source_type="database",
                    extraction_method="table_parse",
                    raw_text_context=f"row {i}",
                )
            assert plog.records_written == 5
        assert len(load_provenance(log_path)) == 5

    def test_none_byte_offset_stored(self, tmp_path):
        log_path = tmp_path / "prov.jsonl"
        with ProvenanceLogger(log_path) as plog:
            plog.record(
                aptamer_sequence="GCGCGCGCGCGCGCGCGCGC",
                target_name="target",
                source_url="https://example.com",
                source_type="preprint",
                extraction_method="pdf_parse",
                raw_text_context="...",
            )
        records = load_provenance(log_path)
        assert records[0]["byte_offset"] is None

    def test_missing_log_returns_empty(self):
        records = load_provenance("/nonexistent/path.jsonl")
        assert records == []

    def test_thread_safe_writes(self, tmp_path):
        log_path = tmp_path / "prov.jsonl"
        with ProvenanceLogger(log_path) as plog:
            def write_10():
                for _ in range(10):
                    plog.record(
                        aptamer_sequence="ATGCATGCATGCATGCATGC",
                        target_name="t",
                        source_url="u",
                        source_type="paper",
                        extraction_method="regex",
                        raw_text_context="ctx",
                    )
            threads = [threading.Thread(target=write_10) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        records = load_provenance(log_path)
        assert len(records) == 40

    def test_text_context_window(self):
        text   = "A" * 100 + "TARGET" + "B" * 100
        ctx    = text_context(text, 100, 106, window=10)
        assert "TARGET" in ctx
        assert len(ctx) <= 30   # 10 + 6 + 10 + 2 ellipsis chars

    def test_text_context_at_start(self):
        text = "START" + "X" * 300
        ctx  = text_context(text, 0, 5, window=20)
        assert "START" in ctx
        assert not ctx.startswith("…")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Sequence validation (existing validate_sequences.py)
# ═════════════════════════════════════════════════════════════════════════════

from scripts.data.validate_sequences import validate_sequence


class TestSequenceValidation:
    def test_valid_sequence(self):
        ok, reason = validate_sequence("ATGCATGCATGCATGCATGCATGC")
        assert ok
        assert reason == ""

    def test_thrombin_15mer_too_short(self):
        # The classic 15-mer G-quad aptamer is below SEQ_MIN_LEN=20 — correctly rejected.
        ok, reason = validate_sequence("GGTTGGTGTGGTTGG")
        assert not ok
        assert "too_short" in reason

    def test_thrombin_29mer_valid(self):
        # 29-mer thrombin aptamer from Macaya et al. 1993
        ok, reason = validate_sequence("AGTCCGTGGTAGGGCAGGTTGGGGTGACT")
        assert ok, f"29-mer thrombin aptamer should be valid, got: {reason}"

    def test_too_short(self):
        ok, reason = validate_sequence("ATGCATGC")
        assert not ok
        assert "too_short" in reason

    def test_too_long(self):
        ok, reason = validate_sequence("ATGC" * 31)   # 124 nt
        assert not ok
        assert "too_long" in reason

    def test_invalid_bases(self):
        ok, reason = validate_sequence("ATGCATGCATGCATGCATGCRZXY")
        assert not ok
        assert "invalid_bases" in reason

    def test_rna_u_invalid(self):
        ok, reason = validate_sequence("AUGCAUGCAUGCAUGCAUGCAUGC")
        assert not ok
        assert "invalid_bases" in reason

    def test_gc_too_low(self):
        ok, reason = validate_sequence("ATATATATATATATATATATAT")
        assert not ok
        assert "gc_too_low" in reason

    def test_gc_too_high(self):
        ok, reason = validate_sequence("GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC")
        assert not ok
        assert "gc_too_high" in reason

    def test_homopolymer_run(self):
        # 9 A's in a row triggers MAX_HOMOPOLYMER=8 rule
        ok, reason = validate_sequence("AAAAAAAAA" + "GCATGCATGCATGCATGCG")
        assert not ok
        assert "homopolymer" in reason

    def test_boundary_homopolymer_ok(self):
        # Exactly 8 A's → should pass
        ok, reason = validate_sequence("AAAAAAAA" + "GCGCGCGCGCGCGCGCGCGC")
        # 8 A's + 20 GC = 28 nt, GC = 20/28 = 0.71 → valid
        assert ok, f"8 A's should not trigger homopolymer rule, got: {reason}"

    def test_duplicate_detection(self):
        seq  = "ATGCATGCATGCATGCATGCATGC"   # valid 24-mer
        key  = seq + "|Thrombin"
        seen = {key}
        ok, reason = validate_sequence(seq, existing_keys=seen, key=key)
        assert not ok
        assert "duplicate" in reason

    def test_non_string_input(self):
        ok, reason = validate_sequence(None)  # type: ignore[arg-type]
        assert not ok
        assert "not_a_string" in reason

    def test_lowercase_normalised(self):
        # validate_sequence strips and uppercases internally
        ok, reason = validate_sequence("atgcatgcatgcatgcatgcatgc")
        assert ok, reason

    def test_120nt_boundary_valid(self):
        seq = "ATGCATGCATGCATGCATGC" * 6   # 120 nt, GC = 50%
        ok, reason = validate_sequence(seq)
        assert ok, reason

    def test_121nt_boundary_invalid(self):
        seq = "ATGCATGCATGCATGCATGCA" * 5 + "ATGCATGCATGCATGCATGCA"  # > 120
        # build exactly 121 nt
        seq = "ATGCATGCATGCATGCATGC" * 6 + "A"   # 121 nt
        ok, reason = validate_sequence(seq)
        assert not ok
        assert "too_long" in reason
