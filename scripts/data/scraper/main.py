"""
CondAptNet aptamer mining pipeline — main orchestrator.

Runs all configured source adapters, collects records, deduplicates against
master_dataset.csv, and writes new rows to scraped_dataset.csv.

Usage:
    # Run all sources, max 500 records per source
    python -m scripts.data.scraper.main

    # Specific sources only
    python -m scripts.data.scraper.main --sources pubmed,databases,semantic_scholar

    # Dry run: collect records but don't write output
    python -m scripts.data.scraper.main --dry-run

    # Larger scrape
    python -m scripts.data.scraper.main --max-per-source 2000

Output:
    data/raw/scraped_dataset.csv     — new unique rows (NEVER master_dataset.csv)
    data/raw/scraper_provenance.jsonl — byte-level provenance log
    data/raw/scraper_coverage_report.txt — run summary
    data/raw/scraper_errors.log       — warnings/errors

Environment variables (set before running):
    ENTREZ_EMAIL            (required by NCBI TOS)
    NCBI_API_KEY            (optional: 10 req/s vs 3 req/s)
    EPO_CLIENT_KEY          (required for EPO adapter)
    EPO_CLIENT_SECRET       (required for EPO adapter)
    LENS_API_TOKEN          (required for Lens adapter)
    SEMANTIC_SCHOLAR_API_KEY (optional: higher rate limit)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.scraper import config as cfg
from scripts.data.scraper.merge import merge_records, write_coverage_report, MergeStats
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

# ── Adapter registry ──────────────────────────────────────────────────────────
# Keys match RATE_LIMITS in config.py. Lazy imports avoid paying startup cost
# for adapters that are skipped.

ALL_SOURCES: list[str] = [
    "databases",         # local files — always fast, run first
    "pubmed",            # PubMed + PMC (Bio.Entrez)
    "semantic_scholar",  # Semantic Scholar API
    "openalex",          # OpenAlex open index
    "biorxiv",           # bioRxiv preprints
    "patents_us",        # PatentsView REST
    "patents_epo",       # EPO OPS (needs credentials)
    "patents_wipo",      # WIPO PatentScope
    "lens",              # Lens.org (needs token)
    "google_patents",    # Google Patents (conservative scraping)
]


def _load_adapter(source_name: str, prov_logger: Optional[ProvenanceLogger]):
    """Lazy-import and instantiate an adapter by source name."""
    mapping = {
        "pubmed":            ("scripts.data.scraper.adapters.pubmed_pmc",        "PubMedPMCAdapter"),
        "biorxiv":           ("scripts.data.scraper.adapters.biorxiv",           "BioRxivAdapter"),
        "openalex":          ("scripts.data.scraper.adapters.openalex",          "OpenAlexAdapter"),
        "patents_us":        ("scripts.data.scraper.adapters.patents_us",        "PatentsUSAdapter"),
        "patents_epo":       ("scripts.data.scraper.adapters.patents_epo",       "EPOAdapter"),
        "patents_wipo":      ("scripts.data.scraper.adapters.patents_wipo",      "WIPOAdapter"),
        "lens":              ("scripts.data.scraper.adapters.lens",              "LensAdapter"),
        "google_patents":    ("scripts.data.scraper.adapters.google_patents",    "GooglePatentsAdapter"),
        "databases":         ("scripts.data.scraper.adapters.databases",         "DatabasesAdapter"),
        "semantic_scholar":  ("scripts.data.scraper.adapters.semantic_scholar",  "SemanticScholarAdapter"),
    }
    if source_name not in mapping:
        raise ValueError(f"Unknown source: {source_name!r}. Available: {list(mapping)}")

    module_path, class_name = mapping[source_name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(prov_logger=prov_logger)


# ── Core run ──────────────────────────────────────────────────────────────────

def run_pipeline(
    sources:        Optional[list[str]] = None,
    max_per_source: int                  = 500,
    dry_run:        bool                 = False,
    master_path:    Optional[Path]       = None,
    output_path:    Optional[Path]       = None,
    prov_path:      Optional[Path]       = None,
    workers:        int                  = 6,
) -> MergeStats:
    """
    Execute the full scraping pipeline.

    Args:
        sources:        List of source names to run. None = ALL_SOURCES.
        max_per_source: Max records requested from each adapter.
        dry_run:        If True, collect and validate records but do NOT write
                        to disk. Useful for testing or capacity estimation.
        master_path:    Path to master_dataset.csv (default: cfg.EXISTING_MASTER).
        output_path:    Path to write scraped_dataset.csv (default: cfg.SCRAPER_OUTPUT).
        prov_path:      Path to provenance JSONL log (default: cfg.PROVENANCE_LOG).
        workers:        Max parallel adapter threads (default: 6).

    Returns:
        MergeStats with full accounting of the run.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sources     = sources or ALL_SOURCES
    master_path = Path(master_path or cfg.EXISTING_MASTER)
    output_path = Path(output_path or cfg.SCRAPER_OUTPUT)
    prov_path   = Path(prov_path   or cfg.PROVENANCE_LOG)

    n_workers = min(len(sources), workers)
    log.info(
        "Pipeline starting — sources: %s, max_per_source: %d, workers: %d, dry_run: %s",
        sources, max_per_source, n_workers, dry_run,
    )

    t_start = time.monotonic()
    all_records:   list[dict] = []
    source_errors: list[str]  = []
    records_lock = threading.Lock()
    errors_lock  = threading.Lock()

    def _run_source(source_name: str) -> tuple[str, list[dict], float]:
        t0      = time.monotonic()
        adapter = _load_adapter(source_name, prov_logger=prov)
        records = adapter.run(max_results=max_per_source)
        return source_name, records, time.monotonic() - t0

    with ProvenanceLogger(prov_path) as prov:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_source, name): name for name in sources}
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    src, records, elapsed = future.result()
                    log.info("[%s] %d records in %.1fs", src, len(records), elapsed)
                    with records_lock:
                        all_records.extend(records)
                except Exception as exc:
                    msg = f"[{source_name}] adapter failed: {exc}"
                    log.error(msg)
                    with errors_lock:
                        source_errors.append(msg)

    total_elapsed = time.monotonic() - t_start
    log.info(
        "All adapters done in %.1fs — %d records collected from %d sources",
        total_elapsed, len(all_records), len(sources),
    )

    if dry_run:
        log.info("Dry run — skipping write to disk.")
        # Still compute stats for reporting
        from scripts.data.scraper.schema import validate_record
        valid   = sum(1 for r in all_records if validate_record(r)[0])
        invalid = len(all_records) - valid
        stats = MergeStats(
            total_incoming=len(all_records),
            valid_incoming=valid,
            invalid_incoming=invalid,
        )
        return stats

    # ── Merge + deduplicate + write ───────────────────────────────────────────
    stats = merge_records(
        new_records=all_records,
        master_path=master_path,
        output_path=output_path,
    )

    # Write coverage report
    report_path = write_coverage_report(stats)

    # Append any adapter errors to the error log
    if source_errors:
        error_log = Path(cfg.ERROR_LOG)
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, "a", encoding="utf-8") as fh:
            import time as _time
            stamp = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
            for msg in source_errors:
                fh.write(f"{stamp} ERROR {msg}\n")

    log.info(
        "Pipeline complete — %d new rows written to %s",
        stats.new_rows_written, output_path,
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level  = logging.DEBUG if verbose else logging.INFO
    format_ = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.ERROR_LOG, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=format_, handlers=handlers, force=True)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.data.scraper.main",
        description="CondAptNet aptamer mining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        default=",".join(ALL_SOURCES),
        help=f"Comma-separated adapter names to run (default: all). "
             f"Available: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=500,
        metavar="N",
        help="Max records to request from each adapter (default: 500)",
    )
    parser.add_argument(
        "--output",
        default=str(cfg.SCRAPER_OUTPUT),
        help="Output CSV path (default: data/raw/scraped_dataset.csv)",
    )
    parser.add_argument(
        "--master",
        default=str(cfg.EXISTING_MASTER),
        help="Master dataset path for deduplication (default: data/processed/master_dataset.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        metavar="N",
        help="Number of parallel adapter threads (default: 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and validate records but do NOT write to disk",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns exit code (0 = success)."""
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in ALL_SOURCES]
    if unknown:
        log.error("Unknown source(s): %s. Available: %s", unknown, ALL_SOURCES)
        return 1

    try:
        stats = run_pipeline(
            sources=sources,
            max_per_source=args.max_per_source,
            dry_run=args.dry_run,
            master_path=Path(args.master),
            output_path=Path(args.output),
            workers=args.workers,
        )
    except ValueError as exc:
        log.error("Pipeline error: %s", exc)
        return 1

    print("\n" + stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
