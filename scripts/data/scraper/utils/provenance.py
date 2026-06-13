"""
Byte-level provenance logging.

Every extracted row gets a JSONL record that answers:
  - WHERE was the sequence found? (source_url, byte_offset)
  - HOW was it extracted? (extraction_method)
  - WHAT was the surrounding text? (raw_text_context, ±200 chars)
  - WHEN? (extraction_timestamp UTC ISO 8601)
  - INTEGRITY: SHA-256 of the source file at extraction time

JSONL format: one JSON object per line, UTF-8, no BOM.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional


def file_sha256(path: str | Path) -> str:
    """Return hex SHA-256 of a file. Used to detect source drift post-scrape."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def text_context(text: str, match_start: int, match_end: int, window: int = 200) -> str:
    """
    Extract ±window characters around a match position for provenance logging.
    Ellipsis added at truncation boundaries.
    """
    lo  = max(0, match_start - window)
    hi  = min(len(text), match_end + window)
    ctx = text[lo:hi]
    if lo > 0:
        ctx = "…" + ctx
    if hi < len(text):
        ctx = ctx + "…"
    return ctx


class ProvenanceLogger:
    """
    Append-only JSONL log. Thread-safe via a threading.Lock.

    Usage:
        with ProvenanceLogger("data/raw/scraper_provenance.jsonl") as log:
            log.record(
                aptamer_sequence="GGTTGGTGTGGTTGG",
                target_name="thrombin",
                source_url="https://pubmed.ncbi.nlm.nih.gov/1741036/",
                source_type="paper",
                extraction_method="regex",
                raw_text_context="...the aptamer GGTTGGTGTGGTTGG was found...",
                byte_offset=12345,
                source_file_hash="abc123...",
            )
    """

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self._path, "a", encoding="utf-8")
        self._count  = 0

        import threading
        self._lock = threading.Lock()

    def record(
        self,
        aptamer_sequence:   str,
        target_name:        str,
        source_url:         str,
        source_type:        str,
        extraction_method:  str,       # "regex" | "table_parse" | "pdf_parse" | "xml_parse"
        raw_text_context:   str,
        byte_offset:        Optional[int]  = None,
        source_file_hash:   Optional[str]  = None,
        extra:              Optional[dict] = None,
    ) -> None:
        """Append one provenance record to the JSONL file."""
        entry: dict = {
            "aptamer_sequence":    aptamer_sequence,
            "target_name":         target_name,
            "source_url":          source_url,
            "source_type":         source_type,
            "extraction_method":   extraction_method,
            "byte_offset":         byte_offset,
            "source_file_hash":    source_file_hash,
            "raw_text_context":    raw_text_context,
            "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            entry.update(extra)

        with self._lock:
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
            self._count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "ProvenanceLogger":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def records_written(self) -> int:
        return self._count


def load_provenance(log_path: str | Path) -> list[dict]:
    """Read all records from a JSONL provenance log. Returns empty list if file missing."""
    path = Path(log_path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # malformed line — skip, don't crash
    return records


if __name__ == "__main__":
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = tmp.name

    try:
        with ProvenanceLogger(tmp_path) as plog:
            plog.record(
                aptamer_sequence="GGTTGGTGTGGTTGG",
                target_name="thrombin",
                source_url="https://pubmed.ncbi.nlm.nih.gov/1741036/",
                source_type="paper",
                extraction_method="regex",
                raw_text_context="...the aptamer GGTTGGTGTGGTTGG binds thrombin...",
                byte_offset=12345,
                source_file_hash="deadbeef",
            )
            plog.record(
                aptamer_sequence="ATCGATCGATCGATCGATCG",
                target_name="insulin",
                source_url="https://doi.org/10.1234/test",
                source_type="database",
                extraction_method="table_parse",
                raw_text_context="Table 1: aptamer sequences",
            )
            assert plog.records_written == 2

        records = load_provenance(tmp_path)
        assert len(records) == 2
        assert records[0]["aptamer_sequence"] == "GGTTGGTGTGGTTGG"
        assert records[1]["target_name"]      == "insulin"
        assert "extraction_timestamp" in records[0]
        print(f"ProvenanceLogger self-test passed. Wrote {len(records)} records.")
    finally:
        os.unlink(tmp_path)
