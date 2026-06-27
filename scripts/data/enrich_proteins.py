"""
UniProt protein sequence enrichment — v3.

Supports two dataset schemas:
  - Legacy  (master_dataset.csv):       target_protein / sequence / uniprot_id /
                                         needs_sequence_enrichment
  - Cleaned (master_dataset_cleaned.csv): target_name / aptamer_sequence /
                                         target_id / target_type

The schema is auto-detected from the input columns. On the cleaned schema only
rows with target_type == "protein" and a missing protein_sequence are enriched
(non-protein targets are already classified by the cleaning pipeline, so the
regex non-protein filter is skipped).

Changes from v2:
  - --input / --output flags (CLAUDE.md documents enriching the cleaned CSV)
  - Cleaned-schema enrichment path (writes protein_sequence + target_id)

v2 features retained:
  - Non-protein filtering (small molecules, cell lines) for the legacy schema
  - Fuzzy matching: strip noise words + 80% token-overlap scoring
  - Manual override table: data/raw/protein_name_overrides.csv
  - Synonym search: strips "human", "recombinant", etc. for retry

Usage:
    # cleaned schema (current canonical training file)
    python scripts/data/enrich_proteins.py --input data/processed/master_dataset_cleaned.csv
    python scripts/data/enrich_proteins.py --input data/processed/master_dataset_cleaned.csv --dry-run

    # legacy schema (defaults to master_dataset.csv in place)
    python scripts/data/enrich_proteins.py              # all proteins
    python scripts/data/enrich_proteins.py --top 50     # top-N by row count
    python scripts/data/enrich_proteins.py --filter-only  # just remove non-proteins
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
REQUEST_DELAY  = 0.4
OVERLAP_THRESH = 0.80   # min fraction of query tokens found in result name

OVERRIDES_PATH = os.path.join("data", "raw", "protein_name_overrides.csv")

# ── Non-protein target detection ──────────────────────────────────────────────

# Patterns that flag a target as a small molecule, cell line, or other
# non-protein entity that can never have a UniProt protein sequence.
_NON_PROTEIN_PATTERNS = [
    r'\bcocaine\b', r'\bcodeine\b', r'\bmorphine\b', r'\bheroin\b',
    r'\bamphetamine\b', r'\bampicillin\b', r'\bkanamycin\b',
    r'\btetracycline\b', r'\bdoxorubicin\b', r'\btheophylline\b',
    r'\bcaffeine\b', r'\bchloramphenicol\b', r'\bpenicillin\b',
    r'\b(ATP|ADP|AMP|GTP(?!-binding)|GDP|NAD|cAMP)\b',
    r'\bmorpholine.*GTP\b',
    r'\bsulforhodamine\b', r'\brhodamine\b', r'\bfluorescein\b',
    r'\bmalachite\s*green\b', r'\bcrystal\s*violet\b',
    r'\bochratoxin\b', r'\baflatoxin\b', r'\bzearalenone\b',
    r'\bfumonisin\b',
    r'\bsialyllactose\b',
    r'\bcell\s*line\b', r'\bPC12\b', r'\bHeLa\b', r'\bHEK293\b',
    r'\bwhole\s*cell\b',
    r'\bvitamin\s*[A-E]\d?\b', r'\briboflavin\b',
    r'\bfolic\s*acid\b',
    r'\blead\s*ion\b', r'\bmercury\s*ion\b', r'\barsenic\b',
    r'\bcholic\s*acid\b', r'\bbile\s*acid\b',
    r'\bspermine\b', r'\bspermidine\b',
    r'\bcyclic\s*adenosine\s*monophosphate\b',
    r'\badenosine\s*triphosphate\b',
    r'\bmannose-capped\s*lipoarabinomannan\b',
]

# Protein names that contain flagged keywords but ARE real proteins
_NON_PROTEIN_EXCEPTIONS = {
    "cell division cycle 42 gtp-binding protein",
    "quinoprotein glucose dehydrogenase pqqgdh",
    "flavin adenine dinucleotide-dependent glucose dehydrogenase",
}

_NON_PROTEIN_RE = re.compile(
    "|".join(_NON_PROTEIN_PATTERNS), re.IGNORECASE
)


def is_non_protein(name: str) -> bool:
    """Return True if name clearly refers to a non-protein target."""
    if not isinstance(name, str):
        return False
    if name.lower().strip() in _NON_PROTEIN_EXCEPTIONS:
        return False
    return bool(_NON_PROTEIN_RE.search(name))


# ── Noise-word stripping ──────────────────────────────────────────────────────

_NOISE_WORDS = {
    "human", "recombinant", "hiv-1", "hiv-2", "hiv1", "hiv2",
    "the", "of", "a", "an", "and", "or", "from", "in", "its",
    "full-length", "full", "length", "truncated", "domain",
    "his-tagged", "his", "tagged", "gst", "flag",
    "open", "reading", "frame", "orf",
}


def strip_noise(name: str) -> str:
    """Remove noise words and normalise spacing."""
    tokens = re.findall(r"\b[\w-]+\b", name.lower())
    cleaned = [t for t in tokens if t not in _NOISE_WORDS and len(t) > 1]
    return " ".join(cleaned)


# ── Token overlap scoring ─────────────────────────────────────────────────────

def token_overlap(query: str, result_name: str) -> float:
    """
    Fraction of cleaned query tokens found anywhere in result_name.
    Returns 0..1; >= OVERLAP_THRESH → accept match.
    """
    qt = set(re.findall(r"\b\w+\b", query.lower()))
    rt = set(re.findall(r"\b\w+\b", result_name.lower()))
    if not qt:
        return 0.0
    return len(qt & rt) / len(qt)


# ── UniProt search ────────────────────────────────────────────────────────────

def _query_uniprot(
    query: str,
    organism: str | None,
) -> tuple[str, str, str] | None:
    """
    Single UniProt REST call. Returns (accession, sequence, protein_name) or None.
    """
    q = f"({query}) AND reviewed:true"
    if organism:
        q += f" AND organism_id:{organism}"
    params = {
        "query":  q,
        "format": "json",
        "size":   "1",
        "fields": "accession,sequence,protein_name",
    }
    try:
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            hit  = results[0]
            acc  = hit["primaryAccession"]
            seq  = hit["sequence"]["value"]
            # protein_name field nests under proteinDescription → recommendedName
            pn   = ""
            try:
                pn = (hit["proteinDescription"]["recommendedName"]
                      ["fullName"]["value"])
            except (KeyError, TypeError):
                pass
            return acc, seq, pn
    except Exception as exc:
        log.debug("UniProt query failed ('%s'): %s", query, exc)
    return None


def find_sequence(
    name: str,
    overrides: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """
    Multi-strategy UniProt lookup.
    Returns (accession, sequence) or None.

    Strategy order:
      1. Manual override table (direct accession fetch)
      2. Exact quoted name, human
      3. Exact quoted name, any organism
      4. Noise-stripped name, human (token-overlap validated)
      5. Noise-stripped name, any organism (token-overlap validated)
    """
    # ── Strategy 1: override table ────────────────────────────────────────────
    if overrides:
        acc = overrides.get(name.lower().strip())
        if acc:
            result = _fetch_by_accession(acc)
            if result:
                log.debug("  Override hit: %s → %s", name, acc)
                return result

    # ── Strategy 2–3: exact quoted name ──────────────────────────────────────
    quoted = f'"{name}"'
    for org in ("9606", None):
        hit = _query_uniprot(quoted, org)
        if hit:
            acc, seq, pn = hit
            return acc, seq
        time.sleep(REQUEST_DELAY)

    # ── Strategy 4–5: noise-stripped + overlap check ──────────────────────────
    clean = strip_noise(name)
    if clean and clean != name.lower():
        for org in ("9606", None):
            hit = _query_uniprot(clean, org)
            if hit:
                acc, seq, pn = hit
                score = token_overlap(clean, pn)
                if score >= OVERLAP_THRESH:
                    log.debug("  Fuzzy match (%.0f%%): '%s' → '%s' [%s]",
                              score * 100, name, pn, acc)
                    return acc, seq
                else:
                    log.debug("  Low overlap %.0f%% for '%s' → '%s'",
                              score * 100, name, pn)
            time.sleep(REQUEST_DELAY)

    return None


def _fetch_by_accession(acc: str) -> tuple[str, str] | None:
    """Fetch sequence directly by accession ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        seq = data["sequence"]["value"]
        return acc, seq
    except Exception as exc:
        log.debug("Accession fetch failed ('%s'): %s", acc, exc)
    return None


# ── Override table ────────────────────────────────────────────────────────────

def load_overrides(path: str = OVERRIDES_PATH) -> dict[str, str]:
    """
    Load data/raw/protein_name_overrides.csv → {dataset_name_lower: uniprot_id}.
    Returns empty dict if file does not exist.
    """
    if not os.path.exists(path):
        return {}
    ov = pd.read_csv(path)
    return {row["dataset_name"].lower().strip(): row["uniprot_id"].strip()
            for _, row in ov.iterrows()
            if pd.notna(row["dataset_name"]) and pd.notna(row["uniprot_id"])}


# ── Non-protein filter ────────────────────────────────────────────────────────

def filter_non_proteins(master: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows whose target_protein is a small molecule, cell line, or other
    entity that can never be represented as a protein sequence.
    Returns (filtered_master, n_dropped).
    """
    mask = master["target_protein"].apply(
        lambda n: is_non_protein(n) if pd.notna(n) else False
    )
    n_dropped = mask.sum()
    if n_dropped:
        names = master.loc[mask, "target_protein"].value_counts()
        log.info("Removing %d non-protein rows (%d unique targets):",
                 n_dropped, len(names))
        for name, count in names.items():
            log.info("  %3d rows  %s", count, name)
    return master[~mask].reset_index(drop=True), n_dropped


# ── Main enrichment ───────────────────────────────────────────────────────────

def enrich(
    master: pd.DataFrame,
    top_n: int | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Enrich protein sequences from UniProt with fuzzy matching + override table.
    Returns (updated_master, n_proteins_found).
    """
    need = master[master["protein_sequence"].isna()]["target_protein"]
    proteins_by_count = need.value_counts()
    if top_n is not None:
        proteins_by_count = proteins_by_count.head(top_n)

    master["uniprot_id"]       = master["uniprot_id"].astype(object)
    master["protein_sequence"] = master["protein_sequence"].astype(object)

    targets = proteins_by_count.index.tolist()
    log.info("Proteins to enrich: %d (covering %d rows)",
             len(targets), proteins_by_count.sum())

    found = 0
    for i, name in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), name)
        hit = find_sequence(name, overrides=overrides)
        time.sleep(REQUEST_DELAY)

        if hit is None:
            log.warning("  NOT FOUND: %s", name)
            continue

        acc, seq = hit
        mask = master["target_protein"] == name
        master.loc[mask, "uniprot_id"]       = acc
        master.loc[mask, "protein_sequence"] = seq
        log.info("  → %s  (%d aa)  %d rows", acc, len(seq), mask.sum())
        found += 1

    return master, found


# ── Cleaned-schema enrichment ─────────────────────────────────────────────────

def enrich_cleaned(
    master: pd.DataFrame,
    top_n: int | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Enrich the cleaned schema (master_dataset_cleaned.csv).

    Only rows with target_type == "protein" and a missing protein_sequence are
    enriched. On a hit, both protein_sequence and target_id (UniProt accession)
    are written for every row sharing that target_name.

    Returns (updated_master, n_proteins_found).
    """
    need_mask = (
        (master["target_type"] == "protein") &
        master["protein_sequence"].isna()
    )
    proteins_by_count = master.loc[need_mask, "target_name"].value_counts()
    if top_n is not None:
        proteins_by_count = proteins_by_count.head(top_n)

    master["target_id"]        = master["target_id"].astype(object)
    master["protein_sequence"] = master["protein_sequence"].astype(object)

    targets = proteins_by_count.index.tolist()
    log.info("Proteins to enrich (cleaned schema): %d (covering %d rows)",
             len(targets), int(proteins_by_count.sum()))

    found = 0
    for i, name in enumerate(targets, 1):
        log.info("[%d/%d] %s", i, len(targets), name)
        hit = find_sequence(name, overrides=overrides)
        time.sleep(REQUEST_DELAY)

        if hit is None:
            log.warning("  NOT FOUND: %s", name)
            continue

        acc, seq = hit
        # only fill rows for this target that are still missing a sequence
        mask = (master["target_name"] == name) & master["protein_sequence"].isna()
        master.loc[mask, "protein_sequence"] = seq
        master.loc[mask, "target_id"]        = acc
        master.loc[mask, "target_id_source"] = "UniProt"
        log.info("  → %s  (%d aa)  %d rows", acc, len(seq), int(mask.sum()))
        found += 1

    return master, found


def _run_cleaned(args, master: pd.DataFrame) -> None:
    """Cleaned-schema enrichment driver (target_name / target_type / target_id)."""
    n_protein = (master["target_type"] == "protein").sum()
    have_seq  = master["protein_sequence"].notna().sum()
    log.info("Cleaned schema detected: %d rows (%d protein-type, %d with sequence)",
             len(master), int(n_protein), int(have_seq))

    overrides = load_overrides()
    if overrides:
        log.info("Override table: %d entries loaded from %s", len(overrides), OVERRIDES_PATH)

    master, n_found = enrich_cleaned(master, top_n=args.top, overrides=overrides)

    still_missing_rows = (
        (master["target_type"] == "protein") & master["protein_sequence"].isna()
    ).sum()
    still_missing_proteins = (
        master.loc[
            (master["target_type"] == "protein") & master["protein_sequence"].isna(),
            "target_name",
        ].nunique()
    )
    total_ready = (
        master["aptamer_sequence"].notna() &
        (master["target_type"] == "protein") &
        master["protein_sequence"].notna()
    ).sum()

    log.info("=" * 55)
    log.info("Proteins newly found        : %d", n_found)
    log.info("Protein rows still missing  : %d", int(still_missing_rows))
    log.info("Protein targets still missing: %d", int(still_missing_proteins))
    log.info("Training-ready protein rows : %d", int(total_ready))
    log.info("=" * 55)

    if args.dry_run:
        log.info("Dry-run — not writing")
        return

    master.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    default_input = os.path.join(DATA_PROCESSED, "master_dataset.csv")
    parser = argparse.ArgumentParser(description="CondAptNet protein sequence enrichment v3")
    parser.add_argument("--input",       default=default_input,
                        help="Dataset CSV to enrich (schema auto-detected)")
    parser.add_argument("--top",         type=int,  default=None)
    parser.add_argument("--output",      default=None,
                        help="Output CSV (defaults to --input, enriched in place)")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--filter-only", action="store_true",
                        help="Only remove non-protein rows, skip UniProt search "
                             "(legacy schema only)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    if not os.path.exists(args.input):
        log.error("Input not found: %s — run build_dataset.py / clean_dataset.py first",
                  args.input)
        sys.exit(1)

    master = pd.read_csv(args.input)

    # ── Auto-detect schema ────────────────────────────────────────────────────
    if "target_name" in master.columns and "aptamer_sequence" in master.columns:
        _run_cleaned(args, master)
        return

    # ── Legacy schema (master_dataset.csv) ────────────────────────────────────
    log.info("Loaded: %d rows, %d with protein_sequence",
             len(master), master["protein_sequence"].notna().sum())

    # ── Step 1: filter non-proteins ───────────────────────────────────────────
    master, n_dropped = filter_non_proteins(master)
    log.info("After filter: %d rows remain", len(master))

    if args.filter_only:
        if not args.dry_run:
            master.to_csv(args.output, index=False)
            log.info("Saved (filter only) → %s", args.output)
        return

    # ── Step 2: load override table ───────────────────────────────────────────
    overrides = load_overrides()
    if overrides:
        log.info("Override table: %d entries loaded from %s",
                 len(overrides), OVERRIDES_PATH)
    else:
        log.info("No override table found at %s", OVERRIDES_PATH)

    # ── Step 3: enrich ────────────────────────────────────────────────────────
    master, n_found = enrich(master, top_n=args.top, overrides=overrides)

    still_missing_rows    = master["protein_sequence"].isna().sum()
    still_missing_proteins = (
        master[master["protein_sequence"].isna()]["target_protein"].nunique()
    )
    total_ready = (
        master["sequence"].notna() &
        (master["needs_sequence_enrichment"] == False) &
        master["protein_sequence"].notna()
    ).sum()

    log.info("=" * 55)
    log.info("Non-protein rows dropped  : %d", n_dropped)
    log.info("Proteins newly found      : %d", n_found)
    log.info("Rows still missing seq    : %d", still_missing_rows)
    log.info("Proteins still missing    : %d", still_missing_proteins)
    log.info("Training-ready rows       : %d", total_ready)
    log.info("=" * 55)

    if args.dry_run:
        log.info("Dry-run — not writing")
        return

    master.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
