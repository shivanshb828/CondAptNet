"""
BaseAdapter — shared infrastructure for all 10 source adapters.

Every adapter inherits this class and:
  1. Declares source_name (matches RATE_LIMITS key in scraper/config.py)
  2. Declares source_type ("paper" | "patent" | "database" | "preprint")
  3. Implements run(max_results) → list[dict]

The base class provides:
  - _get() / _post()  rate-limited HTTP with automatic retry on 429
  - _extract_records_from_text()  applies all four extractors and builds records
  - _validate_and_filter()  drops records that fail schema validation
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.data.scraper.utils.rate_limiter   import get_limiter
from scripts.data.scraper.utils.provenance      import ProvenanceLogger, text_context
from scripts.data.scraper.schema                import validate_record, make_empty_record, BLANK
from scripts.data.scraper.extractors.sequence_extractor  import extract_sequences
from scripts.data.scraper.extractors.kd_extractor        import extract_kd_from_text, best_kd
from scripts.data.scraper.extractors.condition_extractor import extract_conditions
from scripts.data.scraper.extractors.target_resolver     import classify_target_type
from scripts.data.scraper.extractors.assay_extractor     import extract_assay_type

log = logging.getLogger(__name__)

_USER_AGENT = "CondAptNet/1.0 (aptamer ML research; coolshivansh7@gmail.com)"

# Retry budget for 429 / 503 transient errors
_RETRY_WAITS = (5, 15, 30)   # seconds between retries


class BaseAdapter:
    """
    Abstract base for all aptamer scraping adapters.

    Subclasses must set class-level:
        source_name : str   (key in RATE_LIMITS config)
        source_type : str   one of {"paper","patent","database","preprint"}

    And must implement:
        run(max_results=500) -> list[dict]
    """

    source_name: str = ""
    source_type: str = "paper"

    def __init__(self, prov_logger: Optional[ProvenanceLogger] = None) -> None:
        if not self.source_name:
            raise ValueError(f"{type(self).__name__} must set source_name")
        self._limiter   = get_limiter(self.source_name)
        self._prov      = prov_logger
        self._session   = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, url: str, params: Optional[dict] = None, **kwargs) -> Optional[requests.Response]:
        """Rate-limited GET with automatic retry on 429/503."""
        return self._request("GET", url, params=params, **kwargs)

    def _post(self, url: str, json: Optional[dict] = None,
              data=None, **kwargs) -> Optional[requests.Response]:
        """Rate-limited POST with automatic retry on 429/503."""
        return self._request("POST", url, json=json, data=data, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        kwargs.setdefault("timeout", 30)
        self._limiter.wait()
        for attempt, wait in enumerate((*_RETRY_WAITS, None), start=1):
            try:
                resp = self._session.request(method, url, **kwargs)
                if resp.status_code == 429 and wait is not None:
                    log.warning("%s rate-limited (429); sleeping %ss", self.source_name, wait)
                    time.sleep(wait)
                    self._limiter.wait()
                    continue
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                if wait is None:
                    log.warning("%s %s failed after retries: %s", self.source_name, url, exc)
                    return None
                log.debug("%s %s HTTP error attempt %d: %s", self.source_name, url, attempt, exc)
            except requests.RequestException as exc:
                log.warning("%s %s request error: %s", self.source_name, url, exc)
                return None
        return None

    # ── Core extraction ───────────────────────────────────────────────────────

    def _extract_records_from_text(
        self,
        text:        str,
        target_name: str,
        source_url:  str,
        doi:         str         = "",
        confidence:  str         = "extracted",
        extra_fields: Optional[dict] = None,
    ) -> list[dict]:
        """
        Apply all four extractors to text and build schema-compliant records.

        One record per unique aptamer sequence found. If a single Kd is found
        in the document it is assigned to every sequence extracted from that
        document (common case: one aptamer highlighted per paper).

        Args:
            text        :  Document text (full-text or abstract).
            target_name :  Target protein/molecule name (from title/metadata).
            source_url  :  URL or identifier of the source document.
            doi         :  DOI string (may be empty).
            confidence  :  Schema confidence_score value.
            extra_fields:  Additional column overrides (e.g. assay_type).

        Returns:
            List of validated record dicts (invalid records dropped with warning).
        """
        if not text or not target_name:
            return []

        seqs       = extract_sequences(text, valid_only=True)
        kd_list    = extract_kd_from_text(text)
        kd_best    = best_kd(kd_list)
        conditions = extract_conditions(text)
        tgt_type   = classify_target_type(target_name)
        assay_type = extract_assay_type(text)

        records: list[dict] = []

        for seq_hit in seqs:
            rec = make_empty_record()
            rec.update({
                "aptamer_sequence":    seq_hit.sequence,
                "nucleic_acid_type":   seq_hit.nucleic_acid_type,
                "modifications":       "none",
                "target_name":         target_name.strip(),
                "target_type":         tgt_type,
                "kd_value":            kd_best.value_nM       if kd_best else BLANK,
                "kd_unit":             kd_best.original_unit  if kd_best else BLANK,
                "assay_type":          assay_type             or BLANK,
                "selection_buffer":    conditions.selection_buffer    or BLANK,
                "binding_buffer":      conditions.binding_buffer      or BLANK,
                "ph":                  conditions.ph                  if conditions.ph is not None else BLANK,
                "na_concentration_mM": conditions.na_concentration_mM if conditions.na_concentration_mM is not None else BLANK,
                "mg_concentration_mM": conditions.mg_concentration_mM if conditions.mg_concentration_mM is not None else BLANK,
                "temperature_C":       conditions.temperature_C        if conditions.temperature_C is not None else BLANK,
                "source_doi":          doi,
                "source_type":         self.source_type,
                "confidence_score":    confidence,
                "split":               "train",
            })
            if extra_fields:
                rec.update(extra_fields)

            ok, errors = validate_record(rec)
            if ok:
                records.append(rec)
                if self._prov:
                    self._prov.record(
                        aptamer_sequence  = seq_hit.sequence,
                        target_name       = target_name,
                        source_url        = source_url,
                        source_type       = self.source_type,
                        extraction_method = seq_hit.pattern,
                        raw_text_context  = seq_hit.context,
                        byte_offset       = seq_hit.start,
                    )
            else:
                log.debug("Record dropped (%s | %s): %s", target_name, source_url, errors)

        return records

    def _validate_and_filter(self, records: list[dict]) -> list[dict]:
        """Drop records that fail schema validation (with warning log)."""
        valid = []
        for rec in records:
            ok, errors = validate_record(rec)
            if ok:
                valid.append(rec)
            else:
                log.warning("Schema validation failed: %s", errors)
        return valid

    # ── Interface ─────────────────────────────────────────────────────────────

    def run(self, max_results: int = 500) -> list[dict]:
        """
        Execute this adapter and return a list of validated aptamer records.
        Must be implemented by each subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() not implemented")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(source={self.source_name!r})"
