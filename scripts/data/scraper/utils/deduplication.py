"""
Deduplication against the existing master_dataset.csv.

Deduplication key: (normalized_sequence, normalized_target_name).
  - Sequence normalized: strip whitespace, uppercase.
  - Target name normalized: strip whitespace, lowercase.

Two identical sequences binding DIFFERENT targets are kept as separate entries.
Fuzzy matching on sequences is NEVER performed — exact match only.

The Deduplicator is stateful: registering entries as they are processed
prevents intra-batch duplicates as well as cross-batch duplicates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Both old schema (sequence / target_protein) and new (aptamer_sequence / target_name)
_SEQ_COLS = ("aptamer_sequence", "sequence")
_TGT_COLS = ("target_name", "target_protein")


def _norm_seq(s: object) -> str:
    return str(s).strip().upper()


def _norm_tgt(s: object) -> str:
    return str(s).strip().lower()


class Deduplicator:
    """
    Tracks (sequence, target) pairs seen so far to detect duplicates.

    Load from the existing master_dataset.csv first, then call
    register() / is_duplicate() as new records arrive.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_master(cls, master_path: str | Path) -> "Deduplicator":
        """
        Build a Deduplicator pre-loaded with all (sequence, target) pairs
        from the existing master_dataset.csv (old or new schema).
        """
        dedup = cls()
        path  = Path(master_path)
        if not path.exists():
            log.warning("Master dataset not found at %s — starting with empty dedup set", path)
            return dedup

        df = pd.read_csv(path, usecols=lambda c: c in set(_SEQ_COLS + _TGT_COLS))

        seq_col = next((c for c in _SEQ_COLS if c in df.columns), None)
        tgt_col = next((c for c in _TGT_COLS if c in df.columns), None)

        if seq_col is None or tgt_col is None:
            log.error("master_dataset.csv has neither expected sequence nor target column. "
                      "Columns found: %s", list(df.columns))
            return dedup

        before = len(dedup._seen)
        for _, row in df.iterrows():
            seq = _norm_seq(row[seq_col])
            tgt = _norm_tgt(row[tgt_col])
            if seq and tgt and seq != "NAN":
                dedup._seen.add((seq, tgt))

        log.info("Deduplicator loaded %d existing (sequence, target) pairs from %s",
                 len(dedup._seen) - before, path)
        return dedup

    # ── Per-row operations ────────────────────────────────────────────────────

    def is_duplicate(self, sequence: str, target_name: str) -> bool:
        """Return True if this (sequence, target) pair was already seen."""
        return (_norm_seq(sequence), _norm_tgt(target_name)) in self._seen

    def register(self, sequence: str, target_name: str) -> bool:
        """
        Register a new entry. Returns True if it WAS a duplicate (already seen).
        The entry is NOT added to the seen set when it is a duplicate.
        """
        key = (_norm_seq(sequence), _norm_tgt(target_name))
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    # ── Batch operations ──────────────────────────────────────────────────────

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        seq_col: str = "aptamer_sequence",
        tgt_col: str = "target_name",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split df into (unique_df, duplicates_df).

        Registers each unique row so subsequent calls also de-dup against them.
        """
        keep_idx: list[int] = []
        dup_idx:  list[int] = []

        for idx in df.index:
            seq = str(df.at[idx, seq_col]) if seq_col in df.columns else ""
            tgt = str(df.at[idx, tgt_col]) if tgt_col in df.columns else ""
            if self.register(seq, tgt):
                dup_idx.append(idx)
            else:
                keep_idx.append(idx)

        unique = df.loc[keep_idx].reset_index(drop=True)
        dups   = df.loc[dup_idx].reset_index(drop=True)
        return unique, dups

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of (sequence, target) pairs currently tracked."""
        return len(self._seen)


if __name__ == "__main__":
    import tempfile, os

    # Build a tiny fake master CSV with old schema
    import io
    csv_content = (
        "sequence,target_protein\n"
        "GGTTGGTGTGGTTGG,Thrombin\n"
        "ATCGATCGATCGATCGATCG,Insulin\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write(csv_content)
        tmp = f.name

    try:
        dedup = Deduplicator.from_master(tmp)
        assert dedup.size == 2

        # Known duplicate
        assert dedup.is_duplicate("GGTTGGTGTGGTTGG", "Thrombin")
        assert dedup.is_duplicate("ggttggtgtggttgg", "thrombin")   # normalisation

        # Same sequence, different target → NOT a duplicate
        assert not dedup.is_duplicate("GGTTGGTGTGGTTGG", "VEGF")

        # New sequence
        assert not dedup.is_duplicate("ATGCATGCATGCATGCATGC", "Unknown")

        # register() adds and de-dups correctly
        assert not dedup.register("ATGCATGCATGCATGCATGC", "Unknown")   # first time → False
        assert     dedup.register("ATGCATGCATGCATGCATGC", "Unknown")   # second time → True

        print(f"Deduplicator self-test passed. Tracked {dedup.size} pairs.")
    finally:
        os.unlink(tmp)
