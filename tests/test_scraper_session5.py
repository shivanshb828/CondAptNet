"""
Session 5 unit tests: merge.py and main.py orchestrator.

All disk I/O uses tmp_path (pytest fixture). No real adapters are called —
they are replaced with lightweight stubs that return fixture records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.scraper.schema import SCHEMA_COLUMNS, validate_record, make_empty_record

# ── Fixtures ───────────────────────────────────────────────────────────────────

_SEQ1 = "ATCGATCGATCGATCGATCGATCG"   # 24-nt thrombin
_SEQ2 = "GCTAGCTAGCTAGCTAGCTAGCTA"   # 24-nt insulin
_SEQ3 = "ATGATGATGATGATGATGATGATG"   # 24-nt VEGF


def _make_record(seq: str, target: str, source_type: str = "paper") -> dict:
    """Build a valid 20-column record."""
    rec = make_empty_record()
    rec.update({
        "aptamer_sequence":  seq,
        "nucleic_acid_type": "ssDNA",
        "modifications":     "none",
        "target_name":       target,
        "target_type":       "protein",
        "confidence_score":  "extracted",
        "split":             "train",
        "source_type":       source_type,
        "kd_value":          "",
        "kd_unit":           "",
    })
    return rec


def _master_csv(path: Path, rows: list[tuple[str, str]] | None = None) -> Path:
    """Write a minimal master_dataset.csv (old schema) to path."""
    lines = ["sequence,target_protein"]
    if rows:
        for seq, tgt in rows:
            lines.append(f"{seq},{tgt}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# merge.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestMergeStats:

    def test_import(self):
        from scripts.data.scraper.merge import MergeStats
        s = MergeStats()
        assert s.total_incoming == 0

    def test_summary_string(self):
        from scripts.data.scraper.merge import MergeStats
        s = MergeStats(total_incoming=100, valid_incoming=90, new_rows_written=70)
        summary = s.summary()
        assert "100" in summary
        assert "90" in summary
        assert "70" in summary

    def test_summary_per_source(self):
        from scripts.data.scraper.merge import MergeStats
        s = MergeStats(per_source={"paper": 50, "patent": 20})
        summary = s.summary()
        assert "paper" in summary
        assert "patent" in summary


class TestMergeRecords:

    def test_import(self):
        from scripts.data.scraper.merge import merge_records
        assert callable(merge_records)

    def test_safety_guard_raises_if_output_equals_master(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        same = tmp_path / "same.csv"
        with pytest.raises(ValueError, match="must not be the same"):
            merge_records([], master_path=same, output_path=same)

    def test_empty_input_creates_header_only_output(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        stats  = merge_records([], master_path=master, output_path=output)
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) == 0
        assert stats.new_rows_written == 0

    def test_new_unique_rows_written(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        records = [
            _make_record(_SEQ1, "thrombin"),
            _make_record(_SEQ2, "insulin"),
        ]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.new_rows_written == 2
        assert stats.duplicates_vs_master == 0
        df = pd.read_csv(output)
        assert len(df) == 2

    def test_deduplicates_against_master(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        # SEQ1/thrombin is already in master
        master = _master_csv(tmp_path / "master.csv", rows=[(_SEQ1, "thrombin")])
        output = tmp_path / "scraped.csv"
        records = [
            _make_record(_SEQ1, "thrombin"),   # duplicate
            _make_record(_SEQ2, "insulin"),    # new
        ]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.new_rows_written == 1
        assert stats.duplicates_vs_master == 1
        df = pd.read_csv(output)
        assert len(df) == 1
        assert df["aptamer_sequence"].iloc[0] == _SEQ2

    def test_intra_batch_deduplication(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        # SEQ1 appears twice in the same batch
        records = [
            _make_record(_SEQ1, "thrombin"),
            _make_record(_SEQ1, "thrombin"),   # intra-batch dup
            _make_record(_SEQ2, "insulin"),
        ]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.new_rows_written == 2    # only one of the two _SEQ1 rows

    def test_all_duplicates_returns_zero_new_rows(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv", rows=[(_SEQ1, "thrombin")])
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin")]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.new_rows_written == 0

    def test_output_csv_has_schema_columns(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master  = _master_csv(tmp_path / "master.csv")
        output  = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin")]
        merge_records(records, master_path=master, output_path=output)
        df = pd.read_csv(output)
        for col in SCHEMA_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_append_mode_does_not_duplicate_prior_output(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"

        # First run: write SEQ1
        merge_records([_make_record(_SEQ1, "thrombin")],
                      master_path=master, output_path=output)

        # Second run: SEQ1 (dup) + SEQ2 (new)
        stats2 = merge_records(
            [_make_record(_SEQ1, "thrombin"), _make_record(_SEQ2, "insulin")],
            master_path=master, output_path=output,
        )
        assert stats2.new_rows_written == 1   # only SEQ2 is new
        df = pd.read_csv(output)
        assert len(df) == 2                   # SEQ1 from run1, SEQ2 from run2

    def test_invalid_records_dropped(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        bad_rec = {"aptamer_sequence": "ATCG", "target_name": "thrombin"}  # too short, missing fields
        good_rec = _make_record(_SEQ1, "thrombin")
        stats = merge_records([bad_rec, good_rec], master_path=master, output_path=output)
        assert stats.invalid_incoming == 1
        assert stats.valid_incoming == 1
        assert stats.new_rows_written == 1

    def test_master_missing_starts_fresh(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = tmp_path / "nonexistent_master.csv"
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin")]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.new_rows_written == 1

    def test_per_source_stats(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        records = [
            _make_record(_SEQ1, "thrombin", source_type="paper"),
            _make_record(_SEQ2, "insulin",  source_type="patent"),
        ]
        stats = merge_records(records, master_path=master, output_path=output)
        assert "paper"  in stats.per_source
        assert "patent" in stats.per_source

    def test_normalization_case_insensitive_dedup(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        # Master has uppercase sequence + lowercase target
        master = _master_csv(tmp_path / "master.csv", rows=[(_SEQ1.upper(), "thrombin")])
        output = tmp_path / "scraped.csv"
        # Incoming has lowercase sequence + mixed-case target — should still be a dup
        rec = _make_record(_SEQ1.lower(), "Thrombin")
        # normalise seq to uppercase before building record
        rec["aptamer_sequence"] = _SEQ1.upper()  # extractor always uppercases
        stats = merge_records([rec], master_path=master, output_path=output)
        assert stats.duplicates_vs_master == 1

    def test_output_total_rows_count(self, tmp_path):
        from scripts.data.scraper.merge import merge_records
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin"), _make_record(_SEQ2, "insulin")]
        stats = merge_records(records, master_path=master, output_path=output)
        assert stats.output_total_rows == 2


class TestCoverageReport:

    def test_writes_report_file(self, tmp_path):
        from scripts.data.scraper.merge import write_coverage_report, MergeStats
        report_path = tmp_path / "report.txt"
        stats = MergeStats(total_incoming=100, new_rows_written=80)
        write_coverage_report(stats, report_path=report_path)
        assert report_path.exists()

    def test_report_contains_stats(self, tmp_path):
        from scripts.data.scraper.merge import write_coverage_report, MergeStats
        report_path = tmp_path / "report.txt"
        stats = MergeStats(total_incoming=42, new_rows_written=30)
        write_coverage_report(stats, report_path=report_path)
        text = report_path.read_text()
        assert "42" in text
        assert "30" in text

    def test_report_has_timestamp(self, tmp_path):
        from scripts.data.scraper.merge import write_coverage_report, MergeStats
        report_path = tmp_path / "report.txt"
        write_coverage_report(MergeStats(), report_path=report_path)
        text = report_path.read_text()
        assert "Run at" in text or "2" in text   # year digit

    def test_report_creates_parent_dir(self, tmp_path):
        from scripts.data.scraper.merge import write_coverage_report, MergeStats
        deep = tmp_path / "a" / "b" / "c" / "report.txt"
        write_coverage_report(MergeStats(), report_path=deep)
        assert deep.exists()


class TestHelpers:

    def test_count_rows_missing_file(self, tmp_path):
        from scripts.data.scraper.merge import _count_rows
        assert _count_rows(tmp_path / "nonexistent.csv") == 0

    def test_count_rows_header_only(self, tmp_path):
        from scripts.data.scraper.merge import _count_rows
        f = tmp_path / "empty.csv"
        f.write_text("col1,col2\n")
        assert _count_rows(f) == 0

    def test_count_rows_with_data(self, tmp_path):
        from scripts.data.scraper.merge import _count_rows
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\na,b\nc,d\n")
        assert _count_rows(f) == 2

    def test_write_empty_if_missing_creates_file(self, tmp_path):
        from scripts.data.scraper.merge import _write_empty_if_missing
        p = tmp_path / "new.csv"
        _write_empty_if_missing(p)
        assert p.exists()
        df = pd.read_csv(p)
        assert list(df.columns) == SCHEMA_COLUMNS

    def test_write_empty_if_missing_does_not_overwrite(self, tmp_path):
        from scripts.data.scraper.merge import _write_empty_if_missing
        p = tmp_path / "existing.csv"
        p.write_text("important content\n")
        _write_empty_if_missing(p)
        assert p.read_text() == "important content\n"


# ═══════════════════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainImports:

    def test_import(self):
        from scripts.data.scraper.main import run_pipeline, main, ALL_SOURCES
        assert callable(run_pipeline)
        assert callable(main)
        assert isinstance(ALL_SOURCES, list)

    def test_all_sources_contains_ten(self):
        from scripts.data.scraper.main import ALL_SOURCES
        assert len(ALL_SOURCES) == 10

    def test_all_sources_are_unique(self):
        from scripts.data.scraper.main import ALL_SOURCES
        assert len(ALL_SOURCES) == len(set(ALL_SOURCES))


class TestLoadAdapter:

    def test_load_known_adapter(self):
        from scripts.data.scraper.main import _load_adapter
        adapter = _load_adapter("databases", prov_logger=None)
        assert adapter.source_name == "databases"

    def test_load_unknown_raises(self):
        from scripts.data.scraper.main import _load_adapter
        with pytest.raises(ValueError, match="Unknown source"):
            _load_adapter("nonexistent_source", prov_logger=None)

    def test_all_adapters_loadable(self):
        from scripts.data.scraper.main import _load_adapter, ALL_SOURCES
        for source in ALL_SOURCES:
            adapter = _load_adapter(source, prov_logger=None)
            assert adapter.source_name == source


class TestRunPipeline:

    def _stub_adapter(self, records: list[dict]):
        """Return a minimal adapter stub that returns given records."""
        mock = MagicMock()
        mock.run.return_value = records
        return mock

    def test_dry_run_returns_stats(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=self._stub_adapter([])):
            stats = run_pipeline(
                sources=["databases"],
                max_per_source=5,
                dry_run=True,
                master_path=master,
                output_path=output,
            )
        assert not output.exists()   # dry run → no write
        assert stats.total_incoming == 0

    def test_dry_run_does_not_write_output(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin")]
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=self._stub_adapter(records)):
            run_pipeline(
                sources=["databases"],
                max_per_source=5,
                dry_run=True,
                master_path=master,
                output_path=output,
            )
        assert not output.exists()

    def test_full_run_writes_output(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin"), _make_record(_SEQ2, "insulin")]
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=self._stub_adapter(records)):
            stats = run_pipeline(
                sources=["databases"],
                max_per_source=5,
                dry_run=False,
                master_path=master,
                output_path=output,
            )
        assert output.exists()
        assert stats.new_rows_written == 2

    def test_adapter_exception_logged_not_raised(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        failing_adapter = MagicMock()
        failing_adapter.run.side_effect = RuntimeError("network down")
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=failing_adapter):
            # Should NOT raise — errors are caught and logged
            stats = run_pipeline(
                sources=["databases"],
                max_per_source=5,
                dry_run=False,
                master_path=master,
                output_path=output,
            )
        assert stats.total_incoming == 0

    def test_multiple_sources_combined(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"

        call_count = [0]
        source_records = {
            "databases":        [_make_record(_SEQ1, "thrombin")],
            "semantic_scholar": [_make_record(_SEQ2, "insulin")],
            "openalex":         [_make_record(_SEQ3, "VEGF")],
        }

        def load_side_effect(source_name, prov_logger):
            mock = MagicMock()
            mock.run.return_value = source_records.get(source_name, [])
            call_count[0] += 1
            return mock

        with patch("scripts.data.scraper.main._load_adapter", side_effect=load_side_effect):
            stats = run_pipeline(
                sources=["databases", "semantic_scholar", "openalex"],
                max_per_source=5,
                dry_run=False,
                master_path=master,
                output_path=output,
            )

        assert stats.new_rows_written == 3
        assert call_count[0] == 3

    def test_dedup_against_master_during_pipeline(self, tmp_path):
        from scripts.data.scraper.main import run_pipeline
        # SEQ1/thrombin is already in master
        master = _master_csv(tmp_path / "master.csv", rows=[(_SEQ1, "thrombin")])
        output = tmp_path / "scraped.csv"
        records = [_make_record(_SEQ1, "thrombin"), _make_record(_SEQ2, "insulin")]
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=self._stub_adapter(records)):
            stats = run_pipeline(
                sources=["databases"],
                max_per_source=5,
                dry_run=False,
                master_path=master,
                output_path=output,
            )
        assert stats.new_rows_written == 1


class TestMainCLI:

    def test_main_unknown_source_returns_1(self):
        from scripts.data.scraper.main import main
        exit_code = main(["--sources", "nonexistent_source", "--dry-run"])
        assert exit_code == 1

    def test_main_dry_run_exits_0(self, tmp_path):
        from scripts.data.scraper.main import main
        master = _master_csv(tmp_path / "master.csv")
        output = tmp_path / "scraped.csv"
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=MagicMock(run=MagicMock(return_value=[]))):
            exit_code = main([
                "--sources", "databases",
                "--dry-run",
                "--master", str(master),
                "--output", str(output),
            ])
        assert exit_code == 0

    def test_main_safety_guard_exits_1(self, tmp_path):
        from scripts.data.scraper.main import main
        same = str(tmp_path / "same.csv")
        with patch("scripts.data.scraper.main._load_adapter",
                   return_value=MagicMock(run=MagicMock(return_value=[]))):
            exit_code = main([
                "--sources", "databases",
                "--master", same,
                "--output", same,   # same path → should fail
            ])
        assert exit_code == 1

    def test_default_sources_are_all(self):
        from scripts.data.scraper.main import _parse_args, ALL_SOURCES
        args = _parse_args([])
        parsed = [s.strip() for s in args.sources.split(",")]
        assert set(parsed) == set(ALL_SOURCES)

    def test_parse_sources_subset(self):
        from scripts.data.scraper.main import _parse_args
        args = _parse_args(["--sources", "pubmed,databases"])
        sources = [s.strip() for s in args.sources.split(",")]
        assert sources == ["pubmed", "databases"]

    def test_parse_max_per_source(self):
        from scripts.data.scraper.main import _parse_args
        args = _parse_args(["--max-per-source", "1000"])
        assert args.max_per_source == 1000

    def test_parse_dry_run_flag(self):
        from scripts.data.scraper.main import _parse_args
        args = _parse_args(["--dry-run"])
        assert args.dry_run is True
