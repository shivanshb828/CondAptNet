"""
PubMed / PMC adapter.

Two-phase scrape:
  Phase 1 — PubMed: search → fetch abstracts for all matching PMIDs.
  Phase 2 — PMC:    for papers with a PMCID, fetch full NXML via Entrez efetch
            and pass through xml_parser → richer text + tables + supplement URLs.

Uses Bio.Entrez (Biopython) which is already installed in condaptnet_env.
Credentials: ENTREZ_EMAIL (required by NCBI TOS), NCBI_API_KEY (optional, raises
limit from 3 to 10 req/s — update RATE_LIMITS["pubmed"] accordingly).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Bio import Entrez

from scripts.data.scraper import config as cfg
from scripts.data.scraper.adapters.base import BaseAdapter
from scripts.data.scraper.adapters.supp_fetcher import fetch_supplementary_texts
from scripts.data.scraper.parsers.xml_parser import parse_nxml
from scripts.data.scraper.extractors.table_extractor import extract_from_all_tables
from scripts.data.scraper.schema import make_empty_record, validate_record, BLANK
from scripts.data.scraper.utils.provenance import ProvenanceLogger

log = logging.getLogger(__name__)

# Configure Entrez once at module level (safe to re-set)
Entrez.email   = cfg.ENTREZ_EMAIL
Entrez.api_key = cfg.NCBI_API_KEY or None
Entrez.tool    = "CondAptNet"


class PubMedPMCAdapter(BaseAdapter):
    """
    Search PubMed for aptamer papers; fetch full XML from PMC when available.
    """

    source_name = "pubmed"
    source_type = "paper"

    def __init__(
        self,
        prov_logger: Optional[ProvenanceLogger] = None,
        queries: Optional[list[str]] = None,
    ) -> None:
        super().__init__(prov_logger)
        self.queries = queries or cfg.PUBMED_QUERIES

    # ── Entrez helpers ────────────────────────────────────────────────────────

    def _entrez_search(self, query: str, retmax: int = 200) -> list[str]:
        """Return list of PMIDs for a query."""
        self._limiter.wait()
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, usehistory="n")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as exc:
            log.warning("PubMed esearch failed (%s): %s", query[:60], exc)
            return []

    def _entrez_fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """Return {pmid: abstract_text} for a batch of PMIDs."""
        if not pmids:
            return {}
        self._limiter.wait()
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="abstract",
                retmode="text",
            )
            raw = handle.read()
            handle.close()
        except Exception as exc:
            log.warning("PubMed efetch abstracts failed: %s", exc)
            return {}

        # Each PMID abstract is separated by a blank line + PMID line
        # Simple heuristic: return the whole block keyed by first PMID
        # (adequate for sequence extraction; context is always present)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return {pmid: raw for pmid in pmids}

    def _entrez_fetch_pmc_xml(self, pmcid: str) -> Optional[bytes]:
        """Fetch full NXML for a PMCID (e.g. 'PMC1234567' or '1234567')."""
        clean_id = pmcid.replace("PMC", "")
        self._limiter.wait()
        try:
            handle = Entrez.efetch(
                db="pmc",
                id=clean_id,
                rettype="full",
                retmode="xml",
            )
            raw = handle.read()
            handle.close()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return raw
        except Exception as exc:
            log.debug("PMC efetch failed (%s): %s", pmcid, exc)
            return None

    def _pmids_to_pmcids(self, pmids: list[str]) -> dict[str, str]:
        """Return {pmid: pmcid} for PMIDs that have a PMC full-text record."""
        if not pmids:
            return {}
        self._limiter.wait()
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=",".join(pmids),
                cmd="neighbor_history",
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception as exc:
            log.debug("elink pmid→pmc failed: %s", exc)
            return {}

        mapping: dict[str, str] = {}
        for link_set in records:
            pmid = link_set.get("IdList", [""])[0]
            for db_links in link_set.get("LinkSetDb", []):
                if db_links.get("DbTo") == "pmc":
                    for link in db_links.get("Link", []):
                        mapping[pmid] = "PMC" + link["Id"]
                        break
        return mapping

    # ── Table-structured extraction ───────────────────────────────────────────

    def _extract_from_tables(
        self,
        tables: list,
        fallback_target: str,
        source_url: str,
        doi: str,
    ) -> list[dict]:
        """
        Row-level extraction: each table row with a sequence becomes its own record
        with its own Kd — not the document-level best Kd assigned to every sequence.
        """
        from scripts.data.scraper.extractors.assay_extractor import extract_assay_type
        from scripts.data.scraper.extractors.target_resolver import classify_target_type

        table_records = extract_from_all_tables(tables, fallback_target=fallback_target)
        if not table_records:
            return []

        records: list[dict] = []
        for tr in table_records:
            target = tr.target_name or fallback_target
            rec = make_empty_record()
            rec.update({
                "aptamer_sequence":  tr.sequence,
                "nucleic_acid_type": "ssDNA",
                "modifications":     "none",
                "target_name":       target.strip(),
                "target_type":       classify_target_type(target),
                "kd_value":          tr.kd_nM if tr.kd_nM is not None else BLANK,
                "kd_unit":           tr.kd_unit_orig or ("nM" if tr.kd_nM is not None else BLANK),
                "source_doi":        doi,
                "source_type":       self.source_type,
                "confidence_score":  "extracted",
                "split":             "train",
            })
            ok, _ = validate_record(rec)
            if ok:
                records.append(rec)
                if self._prov:
                    self._prov.record(
                        aptamer_sequence  = tr.sequence,
                        target_name       = target,
                        source_url        = source_url,
                        source_type       = self.source_type,
                        extraction_method = f"table:{tr.table_label}:row{tr.row_index}",
                        raw_text_context  = "",
                        byte_offset       = 0,
                    )
        return records

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, max_results: int = 500) -> list[dict]:
        """
        Search all configured queries and extract aptamer records.

        Phase 1: abstract text for every PMID
        Phase 2: full PMC XML for PMIDs that have a PMC record
        """
        all_records: list[dict] = []
        seen_pmids: set[str] = set()
        per_query = max(50, max_results // max(len(self.queries), 1))

        for query in self.queries:
            pmids = self._entrez_search(query, retmax=per_query)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            if not new_pmids:
                continue
            seen_pmids.update(new_pmids)
            log.info("PubMed query %r → %d new PMIDs", query[:60], len(new_pmids))

            # Phase 1: abstracts in batches of 50
            for i in range(0, len(new_pmids), 50):
                batch = new_pmids[i:i+50]
                abstract_map = self._entrez_fetch_abstracts(batch)
                for pmid, text in abstract_map.items():
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    doi = ""   # DOI lookup would require extra elink call
                    recs = self._extract_records_from_text(
                        text=text,
                        target_name=_guess_target_from_abstract(text),
                        source_url=url,
                        doi=doi,
                        confidence="extracted",
                    )
                    all_records.extend(recs)

            # Phase 2: full PMC XML + supplementary files — no cap on PMIDs
            pmc_map = self._pmids_to_pmcids(new_pmids)
            for pmid, pmcid in pmc_map.items():
                xml_bytes = self._entrez_fetch_pmc_xml(pmcid)
                if not xml_bytes:
                    continue
                doc = parse_nxml(xml_bytes)
                if not doc.parse_ok:
                    continue

                pmc_url  = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                base_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/"
                doi      = doc.doi  # extracted from <article-id pub-id-type="doi">
                target   = _guess_target_from_abstract(doc.abstract or doc.full_text)

                # Phase 2a: structured table extraction (sequence+Kd per row)
                table_recs = self._extract_from_tables(
                    tables=doc.tables,
                    fallback_target=target,
                    source_url=pmc_url,
                    doi=doi,
                )
                all_records.extend(table_recs)
                if table_recs:
                    log.info("PMC %s: %d records from tables", pmcid, len(table_recs))

                # Phase 2b: full-text fallback for sequences not in tables
                text_recs = self._extract_records_from_text(
                    text=doc.full_text,
                    target_name=target,
                    source_url=pmc_url,
                    doi=doi,
                    confidence="extracted",
                )
                # Deduplicate against table records: skip sequences already captured
                table_seqs = {r["aptamer_sequence"] for r in table_recs}
                for rec in text_recs:
                    if rec["aptamer_sequence"] not in table_seqs:
                        all_records.append(rec)

                # Phase 2c: supplementary files (Excel/CSV/PDF)
                if doc.supplementary_urls:
                    supp_texts = fetch_supplementary_texts(
                        supp_urls=doc.supplementary_urls,
                        base_url=base_url,
                        session=self._session,
                        limiter=self._limiter,
                    )
                    for supp_text in supp_texts:
                        supp_recs = self._extract_records_from_text(
                            text=supp_text,
                            target_name=target,
                            source_url=pmc_url,
                            doi=doi,
                            confidence="extracted",
                        )
                        all_records.extend(supp_recs)
                    if supp_texts:
                        log.info(
                            "PMC %s: %d supplementary file(s), %d total records so far",
                            pmcid, len(supp_texts), len(all_records),
                        )

            if len(all_records) >= max_results:
                break

        log.info("PubMedPMCAdapter: extracted %d records", len(all_records))
        return all_records[:max_results]


# ── Target name heuristic ──────────────────────────────────────────────────────

# Common protein targets in aptamer literature — match earliest mention in abstract
import re

_TARGET_HINTS = re.compile(
    r"\b(thrombin|VEGF|insulin|lysozyme|myoglobin|troponin|albumin|"
    r"PSA|IgE|MUC1|HIV|influenza|HER2|PDGF|epidermal\s+growth\s+factor|"
    r"NT-proBNP|proBNP|C-reactive\s+protein|CRP|streptavidin|fibrinogen|"
    r"nucleolin|tenascin|osteopontin|transferrin|fibronectin)\b",
    re.IGNORECASE,
)

_APT_TARGET_RE = re.compile(
    r"(?:aptamer|SELEX)\s+(?:for|against|targeting|binding|to)\s+([A-Za-z0-9\-\s]+?)(?:\s*(?:,|\.|\band\b|with\b|in\b))",
    re.IGNORECASE,
)


def _guess_target_from_abstract(text: str) -> str:
    """
    Extract target protein name from abstract text.
    Falls back to 'unknown' if nothing found.
    """
    if not text:
        return "unknown"

    # Try explicit phrasing: "aptamer for/against X"
    m = _APT_TARGET_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if 3 < len(candidate) < 80:
            return candidate

    # Try known-protein name list
    m = _TARGET_HINTS.search(text)
    if m:
        return m.group(1)

    return "unknown"
