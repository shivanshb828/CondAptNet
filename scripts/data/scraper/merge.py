"""
Merge scraped records into scraped_dataset.csv.

CRITICAL INVARIANTS — never violate:
  1. NEVER overwrite master_dataset.csv. Output is always scraped_dataset.csv
     (cfg.SCRAPER_OUTPUT). The two files must remain separate.
  2. ALWAYS deduplicate against master_dataset.csv before writing — even if
     the output file is empty or doesn't exist yet.
  3. Intra-batch duplicates (same seq+target appearing twice in new records)
     are also removed.

Deduplication key: (normalised_sequence, normalised_target_name)
  - Sequence: strip whitespace, uppercase
  - Target: strip whitespace, lowercase
  - Exact match only — no fuzzy matching on sequences

Output:
  - If scraped_dataset.csv does not exist: created fresh.
  - If it already exists: NEW unique rows are APPENDED to it. Rows already in
    the file are treated as duplicates and not re-added.

Usage (programmatic):
    from scripts.data.scraper.merge import merge_records
    stats = merge_records(new_records)

Usage (CLI):
    python -m scripts.data.scraper.merge --input scraped_raw.jsonl
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.scraper import config as cfg
from scripts.data.scraper.schema import (
    records_to_dataframe, validate_record, SCHEMA_COLUMNS,
)
from scripts.data.scraper.utils.deduplication import Deduplicator

log = logging.getLogger(__name__)


# ── Stats dataclass ────────────────────────────────────────────────────────────

@dataclass
class MergeStats:
    """Summary of a merge operation."""
    total_incoming:    int = 0
    valid_incoming:    int = 0
    invalid_incoming:  int = 0
    duplicates_vs_master: int = 0
    intra_batch_dups:  int = 0
    new_rows_written:  int = 0
    output_total_rows: int = 0
    master_rows:       int = 0
    per_source:        dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Incoming records   : {self.total_incoming:>6}",
            f"  valid (schema)   : {self.valid_incoming:>6}",
            f"  invalid (dropped): {self.invalid_incoming:>6}",
            f"Duplicates vs master: {self.duplicates_vs_master:>5}",
            f"Intra-batch dups   : {self.intra_batch_dups:>6}",
            f"New rows written   : {self.new_rows_written:>6}",
            f"Output total rows  : {self.output_total_rows:>6}",
            f"Master rows        : {self.master_rows:>6}",
        ]
        if self.per_source:
            lines.append("Per source:")
            for src, cnt in sorted(self.per_source.items()):
                lines.append(f"  {src:<22}: {cnt:>5}")
        return "\n".join(lines)


# ── Merge function ─────────────────────────────────────────────────────────────

def merge_records(
    new_records: list[dict],
    master_path:  Optional[Path] = None,
    output_path:  Optional[Path] = None,
) -> MergeStats:
    """
    Deduplicate new_records against master_dataset.csv and append unique rows
    to scraped_dataset.csv.

    Args:
        new_records:  List of record dicts (20-column schema) from the adapters.
        master_path:  Path to master_dataset.csv. Defaults to cfg.EXISTING_MASTER.
        output_path:  Path to write/append scraped_dataset.csv. Defaults to
                      cfg.SCRAPER_OUTPUT. MUST NOT equal master_path.

    Returns:
        MergeStats dataclass with full accounting of what happened.

    Raises:
        ValueError: If output_path == master_path (safety guard).
    """
    master_path = Path(master_path or cfg.EXISTING_MASTER)
    output_path = Path(output_path or cfg.SCRAPER_OUTPUT)

    if output_path.resolve() == master_path.resolve():
        raise ValueError(
            f"output_path ({output_path}) must not be the same as master_path "
            f"({master_path}). The master dataset must never be overwritten."
        )

    stats = MergeStats()
    stats.total_incoming = len(new_records)

    # ── Step 1: validate incoming records ─────────────────────────────────────
    valid_records: list[dict] = []
    for rec in new_records:
        ok, _ = validate_record(rec)
        if ok:
            valid_records.append(rec)
        else:
            stats.invalid_incoming += 1
    stats.valid_incoming = len(valid_records)

    log.info(
        "Merge: %d incoming → %d valid, %d invalid",
        stats.total_incoming, stats.valid_incoming, stats.invalid_incoming,
    )

    if not valid_records:
        log.warning("No valid records to merge.")
        _write_empty_if_missing(output_path)
        stats.output_total_rows = _count_rows(output_path)
        return stats

    # ── Step 2: load deduplicator from master ─────────────────────────────────
    dedup = Deduplicator.from_master(master_path)
    stats.master_rows = dedup.size
    log.info("Deduplicator loaded %d pairs from master", stats.master_rows)

    # Also load existing output file into deduplicator to avoid re-adding rows
    # that were written in a previous run
    if output_path.exists():
        dedup_output = Deduplicator.from_master(output_path)
        # Merge output's seen set into master dedup
        dedup._seen.update(dedup_output._seen)
        log.debug("Also loaded %d pairs from existing output file", dedup_output.size)

    # ── Step 3: convert to DataFrame ──────────────────────────────────────────
    new_df = records_to_dataframe(valid_records)

    # ── Step 4: deduplicate (master + existing output + intra-batch) ─────────
    unique_df, dups_df = dedup.filter_dataframe(
        new_df,
        seq_col="aptamer_sequence",
        tgt_col="target_name",
    )

    total_dups        = len(dups_df)
    # We can't easily split master vs intra-batch dups post-hoc; report combined
    stats.duplicates_vs_master = total_dups
    stats.new_rows_written     = len(unique_df)

    log.info(
        "Merge: %d unique new rows, %d duplicates (combined master + intra-batch)",
        stats.new_rows_written, total_dups,
    )

    # ── Step 5: per-source counts (on unique rows only) ───────────────────────
    if "source_type" in unique_df.columns and not unique_df.empty:
        for src, cnt in unique_df["source_type"].value_counts().items():
            stats.per_source[str(src)] = int(cnt)

    # ── Step 6: write output ──────────────────────────────────────────────────
    if not unique_df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists()
        unique_df.to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False,
        )
        log.info("Wrote %d rows to %s", len(unique_df), output_path)
    else:
        _write_empty_if_missing(output_path)
        log.info("No new unique rows; output file unchanged.")

    stats.output_total_rows = _count_rows(output_path)
    return stats


# ── Coverage report ────────────────────────────────────────────────────────────

def write_coverage_report(
    stats:       MergeStats,
    report_path: Optional[Path] = None,
) -> Path:
    """
    Write a plain-text coverage/run summary to cfg.COVERAGE_REPORT (default).
    Returns the path written.
    """
    report_path = Path(report_path or cfg.COVERAGE_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    import time
    lines = [
        "=" * 55,
        "CondAptNet Scraper — Run Summary",
        time.strftime("Run at %Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "=" * 55,
        stats.summary(),
        "=" * 55,
    ]
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    log.info("Coverage report written to %s", report_path)
    return report_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_empty_if_missing(path: Path) -> None:
    """Create an empty (header-only) output CSV if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SCHEMA_COLUMNS).to_csv(path, index=False)


def _count_rows(path: Path) -> int:
    """Return number of data rows in a CSV (header not counted)."""
    if not path.exists():
        return 0
    try:
        return max(0, sum(1 for _ in path.open()) - 1)
    except Exception:
        return 0


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Merge a JSONL file of scraped records into scraped_dataset.csv"
    )
    parser.add_argument(
        "--input", required=True, help="JSONL file of records to merge"
    )
    parser.add_argument(
        "--output", default=str(cfg.SCRAPER_OUTPUT), help="Output CSV path"
    )
    parser.add_argument(
        "--master", default=str(cfg.EXISTING_MASTER), help="Master dataset path"
    )
    args = parser.parse_args()

    records: list[dict] = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed JSONL line: %s", e)

    stats = merge_records(records, master_path=Path(args.master), output_path=Path(args.output))
    print(stats.summary())
    write_coverage_report(stats)
