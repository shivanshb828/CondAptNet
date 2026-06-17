#!/usr/bin/env python3
"""
7-Phase Aptamer Dataset Cleaning Pipeline for CondAptNet.

Usage (from project root):
    source condaptnet_env/bin/activate
    python scripts/data/clean_dataset.py

Inputs:
    data/processed/master_dataset.csv
    data/augmented/tier1_train.csv, val.csv, test.csv  (split labels)
    /Users/shivanshbansal/Downloads/troponin_ntprobnp_aptamers.csv  (Phase 6)

Outputs:
    data/processed/master_dataset_cleaned.csv   — 20-col schema + training extras
    data/processed/non_dna_entries.csv          — filtered non-DNA entries
    data/processed/flagged_for_review.csv       — leakage + ambiguous targets
    outputs/cleaning_report.md                  — full audit log
    data/processed/checkpoints/phase_{N}.csv    — intermediate snapshots
"""

import re
import sys
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean_pipeline")
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_PATH  = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
TRAIN_PATH   = PROJECT_ROOT / "data" / "augmented" / "tier1_train.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "augmented" / "val.csv"
TEST_PATH    = PROJECT_ROOT / "data" / "augmented" / "test.csv"
CURATED_PATH = Path("/Users/shivanshbansal/Downloads/troponin_ntprobnp_aptamers.csv")
CKPT_DIR     = PROJECT_ROOT / "data" / "processed" / "checkpoints"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
REPORT_DIR   = PROJECT_ROOT / "outputs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global collectors ─────────────────────────────────────────────────────────
_REPORT: list[str] = []
_FLAGGED: list[dict] = []
_NON_DNA: list[dict] = []

PRIORITY_TARGETS = ["insulin", "myoglobin", "troponin", "nt-probnp", "ntprobnp", "albumin"]


def _section(title: str) -> None:
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)
    _REPORT.append(f"\n## {title}")


def _note(msg: str) -> None:
    log.info(msg)
    _REPORT.append(f"- {msg}")


def _warn(msg: str) -> None:
    log.warning(msg)
    _REPORT.append(f"- **WARNING**: {msg}")


def _ckpt(df: pd.DataFrame, n: int) -> None:
    path = CKPT_DIR / f"phase{n}.csv"
    df.to_csv(path, index=False)
    log.info("Checkpoint saved → %s  (%d rows)", path.name, len(df))


# ── DNA utilities ─────────────────────────────────────────────────────────────

_RC_TABLE = str.maketrans("ATGCatgcNnRYSWKMBDHVryskmwbdhv",
                          "TACGtacgNnYRSWMKVHDByrswkmvhdb")


def reverse_complement(seq: str) -> str:
    return str(seq).translate(_RC_TABLE)[::-1]


def has_rna_bases(seq: str) -> bool:
    return bool(re.search(r"[Uu]", str(seq)))


def is_dna_only(seq: str) -> bool:
    return bool(re.fullmatch(r"[ATGCatgc]+", str(seq).strip()))


# ── Target type classification ────────────────────────────────────────────────

_SM_EXACT = frozenset({
    "cocaine", "dopamine", "theophylline", "adenosine", "glucose", "cortisol",
    "testosterone", "estradiol", "progesterone", "hemin", "biotin",
    "atp", "adp", "amp", "gtp", "cgmp", "camp", "nad", "nadh", "fad", "fmn",
    "bisphenol a", "bpa", "diclofenac", "ibuprofen", "melamine", "toluene",
    "malachite green", "crystal violet", "kanamycin", "streptomycin",
    "tetracycline", "gentamicin", "tobramycin", "ampicillin", "chloramphenicol",
    "ciprofloxacin", "oxytetracycline", "fumonisin b1", "ochratoxin a",
    "aflatoxin b1", "zearalenone", "deoxynivalenol", "microcystin", "caffeine",
    "l-arginine", "l-lysine", "l-glutamine", "folic acid", "folate", "histamine",
    "serotonin", "epinephrine", "norepinephrine", "acetylcholine", "urea",
    "creatinine", "uric acid", "bilirubin", "cholesterol", "sucrose", "lactose",
    "fructose", "galactose", "mannose", "methamphetamine", "amphetamine",
    "morphine", "heroin", "lysergic acid", "lsd", "thc", "cannabidiol", "cbd",
    "nicotine", "cotinine", "procymidone", "7-methylguanosine triphosphate",
    "hoechst stain derivative", "flavin adenine dinucleotide-dependent glucose dehydrogenase",
    "h-acetyl-tobramycin", "8-oxodg",
    "10-carboxy-2,7-di-t-butyl-trans-12c,12d-dimethyl-12c,12d-dihydrobenzo[e]pyrene",
})

_SM_PARTIAL = [
    "ochratoxin", "aflatoxin", "fumonisin", "zearalenone", "deoxynivalenol",
    "mycotoxin", "polychlorinated", "biphenyl", "benzopyrene",
    "hoechst", "organophosphor", "pesticide", "herbicide", "aminoglycosid",
    "methamphetamin", "amphetamin", "cannabinoid", "cannabin", "tetrahydro",
    "procymidone", "antibiotic", "steroid hormone", "small molecule",
    "flavin adenine", "7-methylguanosin",
]

_ION_PARTIAL = [
    "mercury", " lead ion", "arsenic ion", "cadmium ion", "zinc ion",
    "copper ion", "pb2+", "hg2+", "cd2+", "as3+", "silver ion",
    "lead(ii)", "mercury(ii)", "cadmium(ii)",
]

_CELL_PARTIAL = [
    "cell line", "tumor cell", "cancer cell", "blood cell", "red blood cell",
    "exosome", "microvesicle", "extracellular vesicle", "platelet",
    "mast cell", "lymphocyte", "monocyte", "hela cell", "mcf-7", "ctc ",
    "circulating tumor", "ramos cell", "jurkat", "a549 cell",
    "cancer cell marker", "metastatic cancer cell",
]

_ORGANISM_PARTIAL = [
    "salmonella", "listeria monocytogen", "staphylococcus",
    "pseudomonas aeruginosa", "mycobacterium tuberculosis",
    "campylobacter", "brucella", "yersinia", "aspergillus niger",
    "candida albicans", "cryptococcus",
]

_GARBAGE = [
    "according to claim", "use according", "said at least",
    "capable of being", "be capable", "be tightly", "the target by",
    "a microvesicle", "the surface material",
    "block binding", "inhibit histamine", "at 20 degrees",
    "is enthalpically", "procymidone is", "the sequence",
    "combined with", "a subject",
]


def classify_target_type(name, uniprot_id, protein_sequence) -> str:
    """Return target_type: protein/small_molecule/cell/organism/ion/other."""
    has_seq = (
        pd.notna(protein_sequence)
        and isinstance(protein_sequence, str)
        and len(str(protein_sequence)) > 10
    )
    if has_seq:
        return "protein"

    if pd.isna(name) or not isinstance(name, str) or not name.strip():
        return "other"

    n = name.lower().strip()

    if any(g in n for g in _GARBAGE) or n == "unknown":
        return "other"

    if n in _SM_EXACT:
        return "small_molecule"
    for kw in _SM_PARTIAL:
        if kw in n:
            return "small_molecule"
    for kw in _ION_PARTIAL:
        if kw in n:
            return "ion"
    for kw in _CELL_PARTIAL:
        if kw in n:
            return "cell"
    for kw in _ORGANISM_PARTIAL:
        if kw in n:
            return "organism"

    # Valid-format UniProt ID → protein even without protein_sequence
    if pd.notna(uniprot_id) and isinstance(uniprot_id, str):
        uid = uniprot_id.strip()
        if re.fullmatch(r"[A-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?", uid):
            return "protein"

    return "protein"  # default — most aptamers target proteins


# ── PubChem CID lookup ────────────────────────────────────────────────────────

_PUBCHEM_CACHE: dict[str, str | None] = {}


def lookup_pubchem_cid(name: str) -> str | None:
    if not HAS_REQUESTS:
        return None
    if name in _PUBCHEM_CACHE:
        return _PUBCHEM_CACHE[name]
    try:
        encoded = requests.utils.quote(name)
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/cids/JSON"
        )
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            cid = str(r.json()["IdentifierList"]["CID"][0])
            _PUBCHEM_CACHE[name] = cid
            time.sleep(0.21)  # PubChem public API: ≤5 req/s
            return cid
    except Exception:
        pass
    _PUBCHEM_CACHE[name] = None
    return None


# ── Phase helpers ─────────────────────────────────────────────────────────────

def _load_split_map() -> dict[str, str]:
    """Return {(seq_upper, tgt_lower) → split} from the existing augmented files."""
    split_map: dict[tuple, str] = {}
    for path, label in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        if label == "train":
            df = df[df["augmented"] == False]
        for _, row in df.iterrows():
            key = (str(row["sequence"]).upper(), str(row["target_protein"]).lower())
            if key not in split_map:
                split_map[key] = label
    return split_map


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — RC-augmented pairs and exact duplicate removal
# ══════════════════════════════════════════════════════════════════════════════

def phase1_dedup(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 1 — Reverse-Complement & Exact Duplicate Removal")
    n_before = len(df)
    _note(f"Input rows: {n_before}")

    # Step 1a: Remove exact (sequence, target_protein) duplicates
    df["_seq_up"]  = df["sequence"].str.upper()
    df["_tgt_lo"]  = df["target_protein"].str.lower()
    n_dup = df.duplicated(subset=["_seq_up", "_tgt_lo"], keep="first").sum()
    df = df[~df.duplicated(subset=["_seq_up", "_tgt_lo"], keep="first")].copy()
    _note(f"Exact duplicates (same seq + target) removed: {n_dup}")

    # Step 1b: Check for RC pairs within the dataset
    seq_to_idx: dict[str, int] = {}
    for idx, row in df.iterrows():
        seq_to_idx[row["_seq_up"]] = idx

    rc_pairs_found = 0
    drop_idx: set[int] = set()
    for idx, row in df.iterrows():
        if idx in drop_idx:
            continue
        rc = reverse_complement(row["_seq_up"])
        if rc in seq_to_idx and seq_to_idx[rc] != idx:
            rc_idx = seq_to_idx[rc]
            if rc_idx in drop_idx:
                continue
            # Keep the lower-index row (appears first in dataset)
            keep, discard = (idx, rc_idx) if idx < rc_idx else (rc_idx, idx)
            drop_idx.add(discard)
            rc_pairs_found += 1
            _FLAGGED.append({
                "phase": "1_rc_pair",
                "reason": "reverse_complement_pair_deduplicated",
                "kept_sequence": df.at[keep, "sequence"],
                "dropped_sequence": df.at[discard, "sequence"],
                "target_protein": row["target_protein"],
                "source": row["source"],
            })

    df = df[~df.index.isin(drop_idx)].copy()
    _note(f"Reverse-complement pairs found: {rc_pairs_found}; rows removed: {len(drop_idx)}")

    df.drop(columns=["_seq_up", "_tgt_lo"], inplace=True)
    n_after = len(df)
    _note(f"Output rows: {n_after}  (removed {n_before - n_after} total in Phase 1)")
    _ckpt(df, 1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Priority target coverage audit (pre-split-repair)
# ══════════════════════════════════════════════════════════════════════════════

def phase2_coverage_audit(df: pd.DataFrame) -> None:
    _section("Phase 2 — Priority Target Coverage Audit")

    priority_patterns = {
        "Insulin":    r"(?i)\binsulin\b",
        "Myoglobin":  r"(?i)\bmyoglobin\b",
        "Troponin":   r"(?i)\btroponin\b",
        "NT-proBNP":  r"(?i)(nt.?probnp|pro.?bnp)",
        "Albumin":    r"(?i)\balbumin\b",
    }

    _note("Coverage by split ('_unassigned' = not in any split file):")
    _note(f"{'Target':<18} {'total':>6} {'train':>7} {'val':>6} {'test':>6} {'unassigned':>11}")
    _note(f"{'-'*18} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*11}")

    for label, pat in priority_patterns.items():
        mask = df["target_protein"].str.contains(pat, regex=True, na=False)
        sub  = df[mask]
        total = len(sub)
        by_split = sub["_split"].value_counts().to_dict() if "_split" in df.columns else {}
        tr = by_split.get("train", 0)
        vl = by_split.get("val",   0)
        te = by_split.get("test",  0)
        un = by_split.get("unassigned", 0)
        row_str = f"{label:<18} {total:>6} {tr:>7} {vl:>6} {te:>6} {un:>11}"
        _note(row_str)
        if total == 0:
            _warn(f"{label}: ABSENT from dataset — will be added via curated file in Phase 6")
        elif vl == 0 and te == 0:
            _warn(f"{label}: {total} rows but NONE in val or test — evaluation impossible")

    # Also report total split sizes
    if "_split" in df.columns:
        sc = df["_split"].value_counts()
        _note(f"\nCurrent split sizes: train={sc.get('train',0)}, "
              f"val={sc.get('val',0)}, test={sc.get('test',0)}, "
              f"unassigned={sc.get('unassigned',0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Split assignment + leakage detection and repair
# ══════════════════════════════════════════════════════════════════════════════

def phase3_assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 3 — Split Assignment & Leakage Repair")

    # Assign split labels from existing augmented files
    split_map = _load_split_map()
    df["_split"] = df.apply(
        lambda r: split_map.get(
            (str(r["sequence"]).upper(), str(r["target_protein"]).lower()),
            "unassigned",
        ),
        axis=1,
    )
    sc = df["_split"].value_counts()
    _note(f"Split assignment from augmented files: "
          f"train={sc.get('train',0)}, val={sc.get('val',0)}, "
          f"test={sc.get('test',0)}, unassigned={sc.get('unassigned',0)}")

    # Phase 2 audit (needs _split column to exist)
    phase2_coverage_audit(df)

    # ── Leakage detection ────────────────────────────────────────────────────
    _section("Phase 3b — Leakage Detection and Repair")

    train_seqs = set(df[df["_split"] == "train"]["sequence"].str.upper())
    val_seqs   = set(df[df["_split"] == "val"]["sequence"].str.upper())
    test_seqs  = set(df[df["_split"] == "test"]["sequence"].str.upper())

    tv_leak  = train_seqs & val_seqs    # same seq in train AND val
    tt_leak  = train_seqs & test_seqs   # same seq in train AND test
    vt_leak  = val_seqs   & test_seqs   # same seq in val  AND test (not necessarily train)

    _note(f"Sequence overlap train∩val  : {len(tv_leak)}")
    _note(f"Sequence overlap train∩test : {len(tt_leak)}")
    _note(f"Sequence overlap val∩test   : {len(vt_leak)}")

    # All overlapping sequences are same-seq-different-target (confirmed by audit):
    # Li 2014 screened one aptamer library against 164 proteins → structural leakage.
    # Resolution: assign each sequence to exactly ONE split (most conservative split wins).
    # Priority: train > val > test  (never evaluate on seen sequences)

    # Resolution: any sequence appearing in more than one split → move ALL instances
    # to train. This is the most conservative rule and eliminates all leakage.
    # Root cause: Li 2014 screened one aptamer library against 164 proteins;
    # the same sequence appears paired with many different proteins in different splits.
    all_leaked = tv_leak | tt_leak | vt_leak
    # These sequences must live exclusively in train
    seq_force_train: set[str] = all_leaked

    moved_to_train = 0

    def _reassign(row):
        nonlocal moved_to_train
        seq_up = str(row["sequence"]).upper()
        current = row["_split"]
        if seq_up not in seq_force_train:
            return current
        if current in ("val", "test"):
            tr_tgts = df[(df["sequence"].str.upper()==seq_up) & (df["_split"]=="train")]["target_protein"].tolist()
            _FLAGGED.append({
                "phase": "3_leakage",
                "reason": f"same_seq_diff_target_leak_{current}_moved_to_train",
                "sequence": row["sequence"],
                "current_split": current,
                "current_target": row["target_protein"],
                "train_targets": "; ".join(tr_tgts),
                "source": row["source"],
            })
            moved_to_train += 1
            return "train"
        return current

    df["_split"] = df.apply(_reassign, axis=1)

    # Verify zero leakage after repair
    train_seqs_r = set(df[df["_split"] == "train"]["sequence"].str.upper())
    val_seqs_r   = set(df[df["_split"] == "val"]["sequence"].str.upper())
    test_seqs_r  = set(df[df["_split"] == "test"]["sequence"].str.upper())
    post_tv = len(train_seqs_r & val_seqs_r)
    post_tt = len(train_seqs_r & test_seqs_r)
    post_vt = len(val_seqs_r   & test_seqs_r)

    _note(f"Rows reassigned to train: {moved_to_train}")
    _note(f"Post-repair leakage — train∩val={post_tv}, train∩test={post_tt}, val∩test={post_vt}")
    if post_tv == 0 and post_tt == 0 and post_vt == 0:
        _note("✓ Zero sequence overlap across splits")
    else:
        _warn("Residual leakage detected — investigate manually")

    sc2 = df["_split"].value_counts()
    _note(f"Final split sizes: train={sc2.get('train',0)}, "
          f"val={sc2.get('val',0)}, test={sc2.get('test',0)}, "
          f"unassigned={sc2.get('unassigned',0)}")

    _ckpt(df, 3)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Nucleic acid type annotation and DNA-only filter
# ══════════════════════════════════════════════════════════════════════════════

def phase4_nucleic_acid(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 4 — Nucleic Acid Type & DNA Filter")

    def detect_na_type(row) -> str:
        seq = str(row.get("sequence", ""))
        if not seq or seq == "nan":
            return "unknown"
        if has_rna_bases(seq):
            return "ssRNA"
        if is_dna_only(seq):
            return "ssDNA"
        return "other"

    df["nucleic_acid_type"] = df.apply(detect_na_type, axis=1)

    na_counts = df["nucleic_acid_type"].value_counts()
    _note(f"Nucleic acid type breakdown: {na_counts.to_dict()}")

    # Separate non-DNA rows
    non_dna_mask = df["nucleic_acid_type"] != "ssDNA"
    non_dna_df   = df[non_dna_mask].copy()
    non_dna_df["removal_reason"] = "non_ssDNA_type"

    if len(non_dna_df) > 0:
        _NON_DNA.extend(non_dna_df.to_dict("records"))
        _note(f"Non-ssDNA rows removed: {len(non_dna_df)}  "
              f"(saved to non_dna_entries.csv)")

    df = df[~non_dna_mask].copy()

    # Length filter: 20–120 nt (model constraint, confirmed invalid by validate_sequences.py)
    seq_len  = df["sequence"].str.len()
    too_short_mask = seq_len < 20
    too_long_mask  = seq_len > 120
    bad_len_mask   = too_short_mask | too_long_mask

    if bad_len_mask.sum() > 0:
        bad_len_df = df[bad_len_mask].copy()
        bad_len_df["removal_reason"] = bad_len_df["sequence"].str.len().apply(
            lambda n: f"too_short ({n} < 20)" if n < 20 else f"too_long ({n} > 120)"
        )
        _NON_DNA.extend(bad_len_df.to_dict("records"))
        _note(f"Out-of-range length rows removed: {bad_len_mask.sum()} "
              f"(too_short={too_short_mask.sum()}, too_long={too_long_mask.sum()})")
        df = df[~bad_len_mask].copy()

    # Add modifications column — check for known modification indicators in source data
    # (current master has no modification column; default to 'none')
    df["modifications"] = "none"

    _note(f"Output rows after DNA filter: {len(df)}")
    _ckpt(df, 4)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Target type classification + target ID re-mapping
# ══════════════════════════════════════════════════════════════════════════════

def phase5_target_type(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 5 — Target Type Classification & ID Re-mapping")

    df["target_type"] = df.apply(
        lambda r: classify_target_type(
            r["target_protein"], r.get("uniprot_id"), r.get("protein_sequence")
        ),
        axis=1,
    )

    type_counts = df["target_type"].value_counts()
    _note(f"Target type distribution: {type_counts.to_dict()}")

    # Initialize target_id / target_id_source columns
    df["target_id"]        = df["uniprot_id"].copy()
    df["target_id_source"] = df["target_type"].map(
        {"protein": "UniProt", "peptide": "UniProt"}
    )

    # For non-protein targets that have a UniProt ID assigned (mapping error),
    # clear the ID and attempt PubChem lookup for small molecules
    wrong_id_mask = (~df["target_type"].isin(["protein", "peptide"])) & df["target_id"].notna()
    n_wrong = wrong_id_mask.sum()
    _note(f"Non-protein rows with UniProt ID (mapping error to correct): {n_wrong}")

    df.loc[wrong_id_mask, "target_id"]        = np.nan
    df.loc[wrong_id_mask, "target_id_source"] = np.nan

    # PubChem lookup for small molecules
    sm_mask = df["target_type"] == "small_molecule"
    sm_targets = df[sm_mask]["target_protein"].dropna().unique()
    _note(f"Querying PubChem for {len(sm_targets)} unique small molecule targets ...")

    pubchem_hits   = 0
    pubchem_misses = 0
    cid_map: dict[str, str] = {}

    for name in sm_targets:
        cid = lookup_pubchem_cid(name)
        if cid:
            cid_map[name] = cid
            pubchem_hits += 1
        else:
            pubchem_misses += 1
            _FLAGGED.append({
                "phase": "5_pubchem",
                "reason": "pubchem_lookup_failed",
                "target_protein": name,
                "target_type": "small_molecule",
            })

    _note(f"PubChem hits: {pubchem_hits} / {len(sm_targets)}  "
          f"(misses → flagged_for_review.csv)")

    for name, cid in cid_map.items():
        mask = (df["target_type"] == "small_molecule") & (df["target_protein"] == name)
        df.loc[mask, "target_id"]        = f"CID:{cid}"
        df.loc[mask, "target_id_source"] = "PubChem"

    _ckpt(df, 5)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Column restructuring + curated data merge
# ══════════════════════════════════════════════════════════════════════════════

_BUFFER_MAP = {0: "PBS", 1: "HEPES", 2: "Tris", 3: "other"}

_SOURCE_TYPE_MAP = {
    "li2014":     "paper",
    "utexas":     "database",
    "aptamerbase":"database",
    "pubmed":     "paper",
    "patent":     "patent",
    "biorxiv":    "preprint",
    "database":   "database",
}


def _confidence_score(row) -> str:
    src  = str(row.get("source", "")).lower()
    tgt  = str(row.get("target_protein", "")).lower()
    has_kd = pd.notna(row.get("Kd_nM"))

    garbage_indicators = [
        "according to claim", "use according", "be capable", "a subject",
        "the target", "unknown",
    ]
    if any(g in tgt for g in garbage_indicators) or tgt == "unknown":
        return "uncertain"
    if src == "li2014":
        return "extracted"
    if src == "utexas" and has_kd:
        return "curated"
    if src in ("utexas", "aptamerbase"):
        return "extracted"
    if src == "pubmed":
        return "extracted"
    if src == "patent":
        return "non-curated"
    return "non-curated"


def _format_source_doi(pmid_raw, source: str) -> str:
    """Convert raw source_pmid field to a consistent source_doi string."""
    if pd.isna(pmid_raw):
        return np.nan

    s = str(pmid_raw).strip()

    # Already a DOI
    if s.startswith("10.") or s.startswith("https://doi.org/"):
        doi = s.replace("https://doi.org/", "").strip()
        return doi if doi.startswith("10.") else s

    # Patent number (non-numeric)
    if source == "patent" and not s.replace(".", "").isdigit():
        return s

    # Numeric → PMID
    try:
        pmid = int(float(s))
        return f"PMID:{pmid}"
    except (ValueError, TypeError):
        return s


def phase6_restructure(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 6 — Column Restructuring & Curated Data Merge")
    n_before = len(df)

    # Build the 20-column schema (+ training-essential extras)
    out = pd.DataFrame()

    # 1. aptamer_sequence
    out["aptamer_sequence"] = df["sequence"].str.upper()

    # 2. nucleic_acid_type
    out["nucleic_acid_type"] = df["nucleic_acid_type"]

    # 3. modifications
    out["modifications"] = df["modifications"]

    # 4. target_name
    out["target_name"] = df["target_protein"]

    # 5. target_type
    out["target_type"] = df["target_type"]

    # 6. target_id
    out["target_id"] = df["target_id"]

    # 7. target_id_source
    out["target_id_source"] = df["target_id_source"]

    # 8. kd_value (already in nM)
    out["kd_value"] = df["Kd_nM"]

    # 9. kd_unit — 'nM' where value is present, else NaN
    out["kd_unit"] = df["Kd_nM"].apply(lambda x: "nM" if pd.notna(x) else np.nan)

    # 10. assay_type — not captured in old schema
    out["assay_type"] = np.nan

    # 11. selection_buffer — not captured in old schema
    out["selection_buffer"] = np.nan

    # 12. binding_buffer — derive from buffer_type int
    out["binding_buffer"] = df["buffer_type"].map(_BUFFER_MAP)

    # 13. ph
    out["ph"] = df["pH"]

    # 14. na_concentration_mM
    out["na_concentration_mM"] = df["salt_mM"]

    # 15. mg_concentration_mM
    out["mg_concentration_mM"] = df["mg_mM"]

    # 16. temperature_C
    out["temperature_C"] = df["temp_C"]

    # 17. source_doi
    out["source_doi"] = df.apply(
        lambda r: _format_source_doi(r["source_pmid"], r["source"]), axis=1
    )

    # 18. source_type
    out["source_type"] = df["source"].str.lower().map(_SOURCE_TYPE_MAP).fillna("other")

    # 19. confidence_score
    out["confidence_score"] = df.apply(_confidence_score, axis=1)

    # 20. split
    out["split"] = df["_split"]

    # ── Training-essential extras (beyond the 20-col spec) ──────────────────
    out["protein_sequence"] = df["protein_sequence"]
    out["label"]            = df["label"]
    out["training_tier"]    = df["training_tier"]

    _note(f"Column restructuring complete: {len(out.columns)} columns")
    _note(f"20-column schema satisfied; training extras appended: "
          f"protein_sequence, label, training_tier")

    # ── Merge curated file ───────────────────────────────────────────────────
    if not CURATED_PATH.exists():
        _warn(f"Curated file not found: {CURATED_PATH} — skipping merge")
    else:
        cur = pd.read_csv(CURATED_PATH)
        _note(f"Curated file loaded: {len(cur)} rows, targets: "
              f"{cur['target_name'].unique().tolist()}")

        # Assign split to curated rows:
        # All are Tier 2 targets → training_tier=2, split='train'
        # (Stage 2 fine-tuning handles specific evaluation of these targets)
        cur["split"]            = "train"
        cur["protein_sequence"] = np.nan   # will be looked up by ESM-2 pipeline
        cur["label"]            = 1        # all curated rows are confirmed binders
        cur["training_tier"]    = 2

        # Kd unit normalization: curated file stores nM directly
        cur["kd_value"] = pd.to_numeric(cur["kd_value"], errors="coerce")

        # Ensure schema alignment
        for col in out.columns:
            if col not in cur.columns:
                cur[col] = np.nan

        # Deduplicate against existing rows
        existing_pairs = set(
            zip(out["aptamer_sequence"].str.upper(), out["target_name"].str.lower())
        )
        def _not_duplicate(r):
            return (str(r["aptamer_sequence"]).upper(),
                    str(r["target_name"]).lower()) not in existing_pairs

        cur_new = cur[[_not_duplicate(r) for _, r in cur.iterrows()]].copy()
        n_dup_cur = len(cur) - len(cur_new)
        _note(f"Curated rows: {len(cur)} total, "
              f"{n_dup_cur} already in master (skipped), "
              f"{len(cur_new)} new rows added")

        out = pd.concat([out, cur_new[out.columns]], ignore_index=True)

    n_after = len(out)
    _note(f"Output rows: {n_after}  (+{n_after - n_before} from curated merge)")

    # Backfill protein_sequence for curated rows using UniProt ID lookup from old master.
    # Build uniprot_id → protein_sequence map from already-enriched master rows.
    uid_to_seq: dict[str, str] = {}
    if "target_id" in out.columns and "protein_sequence" in out.columns:
        for _, row in out.iterrows():
            uid = row.get("target_id")
            seq = row.get("protein_sequence")
            if (
                pd.notna(uid) and pd.notna(seq)
                and isinstance(seq, str) and len(seq) > 10
                and isinstance(uid, str) and uid not in uid_to_seq
            ):
                uid_to_seq[uid] = seq

    backfilled = 0
    for idx, row in out.iterrows():
        if pd.isna(row.get("protein_sequence")) or row.get("protein_sequence") == "":
            uid = row.get("target_id")
            if pd.notna(uid) and uid in uid_to_seq:
                out.at[idx, "protein_sequence"] = uid_to_seq[uid]
                backfilled += 1

    if backfilled:
        _note(f"Backfilled protein_sequence for {backfilled} curated rows from UniProt ID lookup")

    _ckpt(out, 6)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 — Final validation and export
# ══════════════════════════════════════════════════════════════════════════════

VALID_DNA_RE   = re.compile(r"^[ATGC]+$")
VALID_CONF     = {"curated", "extracted", "non-curated", "uncertain"}
VALID_NA_TYPE  = {"ssDNA", "ssRNA", "LNA", "2'-OMe RNA", "2'-F RNA", "other", "unknown"}
VALID_SRC_TYPE = {"paper", "patent", "database", "preprint", "other"}
VALID_TGT_TYPE = {"protein", "peptide", "small_molecule", "cell", "organism", "ion", "toxin", "other"}
VALID_SPLIT    = {"train", "val", "test", "unassigned"}


def phase7_export(df: pd.DataFrame) -> pd.DataFrame:
    _section("Phase 7 — Final Validation & Export")

    issues = []

    # Required columns must be non-null
    required_cols = ["aptamer_sequence", "nucleic_acid_type", "modifications",
                     "target_name", "target_type", "confidence_score", "split"]
    for col in required_cols:
        nulls = df[col].isna().sum()
        if nulls > 0:
            issues.append(f"Required column '{col}' has {nulls} nulls")

    # Sequence validation (ssDNA rows only)
    dna_rows = df[df["nucleic_acid_type"] == "ssDNA"].copy()
    invalid_seq = ~dna_rows["aptamer_sequence"].apply(
        lambda s: bool(VALID_DNA_RE.fullmatch(str(s).strip().upper()))
    )
    if invalid_seq.sum() > 0:
        issues.append(f"{invalid_seq.sum()} ssDNA sequences contain non-ATGC characters")
        df.loc[dna_rows[invalid_seq].index, "confidence_score"] = "uncertain"

    # Length check
    lengths = dna_rows["aptamer_sequence"].str.len()
    too_short = (lengths < 20).sum()
    too_long  = (lengths > 120).sum()
    if too_short > 0:
        issues.append(f"{too_short} sequences shorter than 20 nt")
    if too_long > 0:
        issues.append(f"{too_long} sequences longer than 120 nt")

    # Enum validation
    for col, valid_set in [
        ("confidence_score", VALID_CONF),
        ("nucleic_acid_type", VALID_NA_TYPE),
        ("source_type", VALID_SRC_TYPE),
        ("target_type", VALID_TGT_TYPE),
        ("split", VALID_SPLIT),
    ]:
        bad = ~df[col].isin(valid_set) & df[col].notna()
        if bad.sum() > 0:
            bad_vals = df.loc[bad, col].unique()[:5]
            issues.append(f"Column '{col}': {bad.sum()} invalid values: {bad_vals}")

    # Split balance
    sc = df[df["split"] != "unassigned"]["split"].value_counts()
    total_assigned = sc.sum()
    _note(f"Final split sizes (assigned rows):")
    for spl in ["train", "val", "test"]:
        n = sc.get(spl, 0)
        pct = 100 * n / total_assigned if total_assigned else 0
        _note(f"  {spl:<8}: {n:>5} rows ({pct:.1f}%)")
        if spl in ("val", "test") and pct < 5.0:
            issues.append(f"Split '{spl}' is < 5% of assigned data ({pct:.1f}%)")

    _note(f"  unassigned: {(df['split']=='unassigned').sum()}")

    # Priority target re-audit on final cleaned set
    _note("\nPriority target final counts (full dataset):")
    for label, pat in [
        ("Insulin",   r"(?i)\binsulin\b"),
        ("Myoglobin", r"(?i)\bmyoglobin\b"),
        ("Troponin",  r"(?i)\btroponin\b"),
        ("NT-proBNP", r"(?i)(nt.?probnp|pro.?bnp)"),
        ("Albumin",   r"(?i)\balbumin\b"),
    ]:
        n = df["target_name"].str.contains(pat, regex=True, na=False).sum()
        _note(f"  {label:<14}: {n}")

    # Leakage re-check
    tr = set(df[df["split"]=="train"]["aptamer_sequence"].str.upper())
    va = set(df[df["split"]=="val"  ]["aptamer_sequence"].str.upper())
    te = set(df[df["split"]=="test" ]["aptamer_sequence"].str.upper())
    for a, b, sa, sb in [("train","val",tr,va), ("train","test",tr,te), ("val","test",va,te)]:
        ov = len(sa & sb)
        if ov > 0:
            issues.append(f"RESIDUAL LEAKAGE: {ov} sequences in {a}∩{b}")
        else:
            _note(f"  Leakage check {a}∩{b}: 0 (clean)")

    if issues:
        _warn(f"{len(issues)} validation issues found:")
        for iss in issues:
            _warn(f"  ↳ {iss}")
    else:
        _note("✓ All validation checks passed")

    # ── Export files ─────────────────────────────────────────────────────────
    out_path = OUT_DIR / "master_dataset_cleaned.csv"
    df.to_csv(out_path, index=False)
    _note(f"\nExported: {out_path}  ({len(df)} rows, {len(df.columns)} columns)")

    if _NON_DNA:
        non_dna_df = pd.DataFrame(_NON_DNA)
        non_dna_path = OUT_DIR / "non_dna_entries.csv"
        non_dna_df.to_csv(non_dna_path, index=False)
        _note(f"Exported: {non_dna_path}  ({len(non_dna_df)} rows)")

    if _FLAGGED:
        flagged_df = pd.DataFrame(_FLAGGED)
        flagged_path = OUT_DIR / "flagged_for_review.csv"
        flagged_df.to_csv(flagged_path, index=False)
        _note(f"Exported: {flagged_path}  ({len(flagged_df)} rows)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Report writer
# ══════════════════════════════════════════════════════════════════════════════

def _write_report() -> None:
    path = REPORT_DIR / "cleaning_report.md"
    header = [
        "# Aptamer Dataset Cleaning Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input: data/processed/master_dataset.csv",
        "",
    ]
    path.write_text("\n".join(header + _REPORT) + "\n")
    log.info("Cleaning report → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    log.info("Loading master_dataset.csv ...")
    df = pd.read_csv(MASTER_PATH)
    _note(f"Loaded {len(df)} rows, {len(df.columns)} columns from master_dataset.csv")

    df = phase1_dedup(df)
    df = phase3_assign_splits(df)   # Phase 2 runs inside Phase 3 (needs _split col)
    df = phase4_nucleic_acid(df)
    df = phase5_target_type(df)
    df = phase6_restructure(df)
    df = phase7_export(df)
    _write_report()

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1fs", elapsed)
    log.info("master_dataset_cleaned.csv: %d rows, %d columns", len(df), len(df.columns))


if __name__ == "__main__":
    main()
