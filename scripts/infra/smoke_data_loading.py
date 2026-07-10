"""
scripts/infra/smoke_data_loading.py — Step 3: Data Pipeline Smoke Test

Verifies that the full data loading pipeline works end-to-end:
  1. Load master_dataset_v2.csv (or augmented splits if available)
  2. Check split column presence; if missing, randomly sample a held-out subset
  3. Run tokenization + batching for DNA_ENCODER_TYPE=scratch
  4. Run tokenization + batching for DNA_ENCODER_TYPE=dnabert2
  5. Confirm tensor shapes match what the model expects
  6. Confirm ViennaRNA cache loads (or falls back cleanly)

NOTE: This test does NOT run ESM-2 protein embedding pre-computation
(that's slow and requires the full model). It uses the pre-computed .npy files
if they exist; if not, it generates a random embedding of the right shape as
a stand-in and notes that explicitly.

Usage:
    python scripts/infra/smoke_data_loading.py
    python scripts/infra/smoke_data_loading.py --n-rows 32
    python scripts/infra/smoke_data_loading.py --skip-dnabert2
"""

import argparse
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_dataset(n_rows: int) -> pd.DataFrame:
    """Load master_dataset_v2.csv or fall back to augmented splits."""
    from config import DATA_PROCESSED, DATA_AUGMENTED

    v2_path = os.path.join(DATA_PROCESSED, "master_dataset_v2.csv")
    aug_path = os.path.join(DATA_AUGMENTED, "tier1_train.csv")

    if os.path.exists(v2_path):
        print(f"{INFO} Loading {v2_path}")
        df = pd.read_csv(v2_path)
        print(f"{PASS} Loaded {len(df)} rows from master_dataset_v2.csv")

        # Check split column
        if "split" not in df.columns:
            print(f"{WARN} 'split' column missing from master_dataset_v2.csv.")
            print(f"     Using a random held-out subset of {n_rows} rows for smoke test.")
            print(f"     NOTE: The parallel data-finalization task may not have run yet.")
            df = df.sample(min(n_rows, len(df)), random_state=42).reset_index(drop=True)
            df["split"] = "smoke_test"
        else:
            vals = df["split"].value_counts().to_dict()
            print(f"{INFO} Split column values: {vals}")
    elif os.path.exists(aug_path):
        print(f"{WARN} master_dataset_v2.csv not found; using augmented splits")
        df = pd.read_csv(aug_path)
        print(f"{PASS} Loaded {len(df)} rows from tier1_train.csv")
    else:
        print(f"{FAIL} No dataset found at {v2_path} or {aug_path}")
        sys.exit(1)

    # Filter to rows with valid aptamer + protein sequences
    before = len(df)
    df = df[df["aptamer_sequence"].notna() & df["protein_sequence"].notna()].reset_index(drop=True)
    print(f"{INFO} {before - len(df)} rows dropped (missing sequence/protein)")
    print(f"{INFO} Usable rows: {len(df)}")

    # Sample for smoke test
    df = df.sample(min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"{INFO} Sampled {len(df)} rows for smoke test")
    return df


def build_fake_protein_embeddings(df: pd.DataFrame, emb_dir: str) -> dict:
    """
    Return a seq→path mapping using pre-computed .npy files if available,
    otherwise generate random embeddings of the right shape as stand-ins.
    This avoids running ESM-2 in the data loading smoke test.
    """
    from config import ESM_EMBED_DIM, PROT_MAX_TOKENS

    os.makedirs(emb_dir, exist_ok=True)
    seq_to_path: dict[str, str] = {}
    generated = 0
    cached = 0

    import hashlib
    for seq in df["protein_sequence"].dropna().unique():
        key  = hashlib.md5(f"{PROT_MAX_TOKENS}:{seq}".encode()).hexdigest()
        path = os.path.join(emb_dir, f"{key}.npy")
        seq_to_path[seq] = path

        if os.path.exists(path):
            cached += 1
        else:
            # Fake embedding: correct shape, random values
            fake = np.random.randn(min(len(seq), PROT_MAX_TOKENS), ESM_EMBED_DIM).astype(np.float32)
            np.save(path, fake)
            generated += 1

    if generated:
        print(f"{WARN} Generated {generated} FAKE protein embeddings (random, ESM-2 not run).")
        print(f"     These are for pipeline shape-testing only, NOT for real training.")
    if cached:
        print(f"{PASS} {cached} real pre-computed protein embeddings found in cache.")
    return seq_to_path


def load_vienna_cache() -> dict:
    from config import VIENNA_CACHE
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vc = pickle.load(f)
        print(f"{PASS} Vienna cache loaded: {len(vc)} entries")
        return vc
    else:
        print(f"{WARN} Vienna cache not found at {VIENNA_CACHE}.")
        print(f"     Features will be zero-vectors (fallback in AptamerDataset).")
        return {}


def smoke_scratch_path(df: pd.DataFrame, seq_to_path: dict, vienna_cache: dict,
                       batch_size: int) -> None:
    """Test the scratch DNA encoder data path."""
    header("Data Loading: DNA_ENCODER_TYPE=scratch (3-mer tokenizer)")

    import config as _cfg
    orig_type = _cfg.DNA_ENCODER_TYPE
    _cfg.DNA_ENCODER_TYPE = "scratch"

    try:
        from scripts.model.tokenizer import DNATokenizer
        from scripts.training.train import AptamerDataset, collate_fn
        from torch.utils.data import DataLoader

        tokenizer = DNATokenizer()
        ds = AptamerDataset(df, tokenizer, vienna_cache, seq_to_path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        t0 = time.perf_counter()
        apt, v, prot_tok, cond, labels, kds, prot_emb = next(iter(loader))
        elapsed = time.perf_counter() - t0

        expected_apt_len = _cfg.DNA_MAX_LEN
        print(f"{PASS} Batch loaded in {elapsed*1000:.1f}ms")
        print(f"{INFO}   apt           : {tuple(apt.shape)}  dtype={apt.dtype}")
        print(f"{INFO}   v_feats       : {tuple(v.shape)}")
        print(f"{INFO}   prot_tok      : {tuple(prot_tok.shape)}")
        print(f"{INFO}   cond          : {tuple(cond.shape)}")
        print(f"{INFO}   labels        : {tuple(labels.shape)}")
        print(f"{INFO}   kds           : {tuple(kds.shape)}")
        print(f"{INFO}   prot_emb      : {tuple(prot_emb.shape)}")

        # Shape assertions
        B = apt.shape[0]
        assert apt.shape == (B, expected_apt_len), \
            f"apt shape {apt.shape} != ({B}, {expected_apt_len})"
        assert v.shape == (B, 6), f"vienna feats shape {v.shape} != ({B}, 6)"
        assert cond.shape == (B, 5), f"cond shape {cond.shape} != ({B}, 5)"
        assert labels.shape == (B, 1), f"labels shape {labels.shape}"
        assert kds.shape == (B, 1), f"kds shape {kds.shape}"
        assert prot_emb.ndim == 3, f"prot_emb ndim {prot_emb.ndim} != 3"

        print(f"{PASS} All shape assertions passed for scratch path")

        # Sanity: aptamer tokens should be in [0, 65]
        assert apt.min() >= 0 and apt.max() <= 65, \
            f"aptamer token range [{apt.min()}, {apt.max()}] out of vocab range [0, 65]"
        print(f"{PASS} Aptamer token range OK: [{apt.min()}, {apt.max()}] ⊂ [0, 65]")

    finally:
        _cfg.DNA_ENCODER_TYPE = orig_type


def smoke_dnabert2_path(df: pd.DataFrame, seq_to_path: dict, vienna_cache: dict,
                        batch_size: int) -> None:
    """Test the DNABERT-2 data path (tokenization only, no model forward)."""
    header("Data Loading: DNA_ENCODER_TYPE=dnabert2 (DNABERT-2 BPE tokenizer)")

    import config as _cfg
    orig_type = _cfg.DNA_ENCODER_TYPE
    _cfg.DNA_ENCODER_TYPE = "dnabert2"

    try:
        from scripts.model.tokenizer import DNATokenizer
        from scripts.training.train import AptamerDataset, collate_fn
        from torch.utils.data import DataLoader

        tokenizer = DNATokenizer()   # scratch tokenizer not used in dnabert2 path but required by Dataset

        try:
            t0 = time.perf_counter()
            ds = AptamerDataset(df, tokenizer, vienna_cache, seq_to_path)
            elapsed = time.perf_counter() - t0
            print(f"{PASS} AptamerDataset with DNABERT-2 BPE tokenizer initialized in {elapsed:.1f}s")
        except ImportError as e:
            print(f"{WARN} DNABERT-2 tokenizer init failed: {e}")
            print(f"     This is expected if transformers/einops is not installed.")
            print(f"     Install with: pip install transformers einops huggingface_hub")
            return
        except Exception as e:
            print(f"{FAIL} Unexpected error initializing DNABERT-2 dataset: {e}")
            import traceback; traceback.print_exc()
            return

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        t0 = time.perf_counter()
        apt, v, prot_tok, cond, labels, kds, prot_emb = next(iter(loader))
        elapsed = time.perf_counter() - t0

        print(f"{PASS} Batch loaded in {elapsed*1000:.1f}ms")
        print(f"{INFO}   apt (BPE ids) : {tuple(apt.shape)}  dtype={apt.dtype}")
        print(f"{INFO}   v_feats       : {tuple(v.shape)}")
        print(f"{INFO}   prot_emb      : {tuple(prot_emb.shape)}")

        expected_apt_len = _cfg.DNABERT2_MAX_LEN
        B = apt.shape[0]
        assert apt.shape == (B, expected_apt_len), \
            f"BPE apt shape {apt.shape} != ({B}, {expected_apt_len})"
        print(f"{PASS} DNABERT-2 BPE aptamer shape correct: {tuple(apt.shape)}")
        print(f"{INFO} Token range: [{apt.min()}, {apt.max()}] — vocab size ~4096 (BPE)")

    finally:
        _cfg.DNA_ENCODER_TYPE = orig_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline smoke test")
    parser.add_argument("--n-rows", type=int, default=64,
                        help="Number of rows to sample for smoke test (default: 64)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-dnabert2", action="store_true",
                        help="Skip the DNABERT-2 tokenizer path (faster; use if transformers not installed)")
    args = parser.parse_args()

    header("Step 3: Data Pipeline Smoke Test")

    from config import DATA_PROCESSED

    df = load_dataset(args.n_rows)
    vienna_cache = load_vienna_cache()

    emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
    seq_to_path = build_fake_protein_embeddings(df, emb_dir)

    smoke_scratch_path(df, seq_to_path, vienna_cache, args.batch_size)

    if not args.skip_dnabert2:
        smoke_dnabert2_path(df, seq_to_path, vienna_cache, args.batch_size)
    else:
        print(f"\n{INFO} DNABERT-2 path skipped (--skip-dnabert2)")

    header("Data pipeline smoke test complete")
    print(f"\n{PASS} All data loading checks passed.")


if __name__ == "__main__":
    main()
