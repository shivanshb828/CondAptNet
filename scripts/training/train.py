"""
CondAptNet Stage 1 — Broad Pretraining.

Trains the full model on the augmented training set produced by
scripts/data/augment.py. ESM-2 backbone is frozen; LoRA adapters and all other
layers are trainable (set_stage1()).

Data: reads the protein-family splits written to data/augmented/ —
    tier1_train.csv (augmented: hard negatives, cross-target negatives,
                     truncations, scrambles), val.csv and test.csv (never
                     augmented). Splits were assigned by protein family during
                     cleaning (never randomly), with zero sequence leakage.
Primary metric: MCC. Also logs AUC-ROC, AUC-PR, sensitivity.

Usage:
    source condaptnet_env/bin/activate
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py --max-epochs 3
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py --batch-size 8

Prerequisite: run `python scripts/data/augment.py` first to build data/augmented/.
Checkpoints saved to: models/checkpoints/pretrain/
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, RANDOM_SEED, DATA_PROCESSED, DATA_AUGMENTED, CHECKPOINTS_DIR, VIENNA_CACHE,
    BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE_BASE, LEARNING_RATE_LORA, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_METRIC, GRAD_CLIP,
    DNA_MAX_LEN, PROT_MAX_TOKENS, DEFAULT_PH, DEFAULT_SALT_MM,
    DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
)
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer
from scripts.training.losses import CondAptNetLoss
from scripts.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

def precompute_protein_embeddings(
    protein_seqs: list[str],
    protein_encoder,
    emb_dir: str,
    device: torch.device,
    prot_max_tokens: int = PROT_MAX_TOKENS,
) -> dict[str, str]:
    """
    Run ESM-2 ONCE per unique protein sequence and cache to disk as .npy files.
    Returns mapping  protein_sequence → npy_file_path.

    Per CLAUDE.md: "ESM-2 embeddings: cache to disk as numpy, never to GPU memory."
    This makes Stage 1 training ~N× faster (N = unique proteins in the dataset).
    """
    import hashlib
    os.makedirs(emb_dir, exist_ok=True)
    bc = protein_encoder.alphabet.get_batch_converter()

    seq_to_path: dict[str, str] = {}
    unique = [s for s in set(protein_seqs) if isinstance(s, str)]
    log.info("Pre-computing protein embeddings: %d unique proteins → %s", len(unique), emb_dir)

    protein_encoder.eval()
    with torch.no_grad():
        for i, seq in enumerate(unique, 1):
            # Key includes prot_max_tokens: embeddings truncated to a different
            # token cap are NOT interchangeable, so they must not share a cache file.
            key  = hashlib.md5(f"{prot_max_tokens}:{seq}".encode()).hexdigest()
            path = os.path.join(emb_dir, f"{key}.npy")
            seq_to_path[seq] = path

            if os.path.exists(path):
                log.info("  [%d/%d] cached   (%d aa)", i, len(unique), len(seq))
                continue

            # Tokenise (truncate to prot_max_tokens)
            _, _, tok = bc([("p", seq)])
            tok = tok[:, :prot_max_tokens].to(device)

            # Get representation from last layer
            n_layers = protein_encoder.esm.num_layers
            results  = protein_encoder.esm(tok, repr_layers=[n_layers],
                                           return_contacts=False)
            emb = results["representations"][n_layers]  # [1, seq_len, 480]
            emb = emb[0].cpu().float().numpy()           # [seq_len, 480]

            np.save(path, emb)
            log.info("  [%d/%d] computed  (%d aa → %s shape)", i, len(unique),
                     len(seq), emb.shape)

    return seq_to_path


class AptamerDataset(Dataset):
    """
    Loads aptamer-protein pairs from a filtered DataFrame.

    Requires rows where both `aptamer_sequence` and `protein_sequence` are
    non-null and `split` is train/val/test (not 'unassigned').

    Protein embeddings are loaded from pre-computed .npy files (ESM-2 is NOT
    called during training). Pass `seq_to_emb_path` from precompute_protein_embeddings().

    ViennaRNA features are looked up from the pickle cache; sequences not in
    the cache get zero-vectors.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: DNATokenizer,
                 vienna_cache: dict, seq_to_emb_path: dict[str, str],
                 max_prot_len: int = PROT_MAX_TOKENS) -> None:
        self.df              = df.reset_index(drop=True)
        self.tokenizer       = tokenizer
        self.vc              = vienna_cache
        self.seq_to_emb_path = seq_to_emb_path
        self.max_prot_len    = max_prot_len

        # DNA-encoder-dependent aptamer tokenization. Default "scratch" uses the
        # 3-mer DNATokenizer (unchanged). "dnabert2" needs DNABERT-2's own BPE
        # tokenizer producing input_ids padded to DNABERT2_MAX_LEN — a parallel
        # path, NOT the 3-mer pipeline. Config/transformers are imported lazily
        # here so the scratch path pulls in no extra dependency.
        from config import DNA_ENCODER_TYPE
        self.dna_encoder_type = DNA_ENCODER_TYPE
        self._bpe_tokenizer = None
        if DNA_ENCODER_TYPE == "dnabert2":
            from config import DNABERT2_MODEL_NAME, DNABERT2_REVISION, DNABERT2_MAX_LEN
            from transformers import AutoTokenizer
            # Pin to an immutable commit SHA — trust_remote_code=True executes
            # repo code, so never resolve the moving default branch at runtime.
            self._bpe_tokenizer = AutoTokenizer.from_pretrained(
                DNABERT2_MODEL_NAME, revision=DNABERT2_REVISION, trust_remote_code=True)
            self._bpe_max_len = DNABERT2_MAX_LEN

    def _encode_aptamer(self, seq: str) -> torch.Tensor:
        """Aptamer -> token ids, per the active DNA encoder.

        scratch  : 3-mer ids padded to DNA_MAX_LEN (via DNATokenizer).
        dnabert2 : DNABERT-2 BPE input_ids padded to DNABERT2_MAX_LEN.
        Both return a fixed-length LongTensor so collate_fn can stack them.
        """
        if self._bpe_tokenizer is not None:
            enc = self._bpe_tokenizer(
                seq.upper(), return_tensors="pt", padding="max_length",
                truncation=True, max_length=self._bpe_max_len,
            )
            return enc["input_ids"].squeeze(0).long()
        encoded = self.tokenizer.encode_padded(seq, DNA_MAX_LEN)
        return (encoded.clone().detach() if isinstance(encoded, torch.Tensor)
                else torch.tensor(encoded, dtype=torch.long))

    def _vienna_feats(self, seq: str) -> torch.Tensor:
        if seq in self.vc:
            d = self.vc[seq]
            return torch.tensor([
                d.get("mfe", 0.0),
                float(d.get("stem_count", 0)),
                float(d.get("loop_count", 0)),
                d.get("bp_prob_mean", 0.0),
                d.get("bp_prob_max", 0.0),
                len(seq) / DNA_MAX_LEN,
            ], dtype=torch.float32)
        # Attempt live computation (requires ViennaRNA)
        try:
            import RNA
            fc = RNA.fold_compound(seq)
            struct, mfe = fc.mfe()
            stems = struct.count("(")
            loops = struct.count(".")
            bpp_matrix = fc.bpp()
            bp_vals = [bpp_matrix[i][j] for i in range(1, len(seq)+1)
                       for j in range(i+1, len(seq)+1) if bpp_matrix[i][j] > 0.01]
            bp_mean = float(np.mean(bp_vals)) if bp_vals else 0.0
            bp_max  = float(max(bp_vals))     if bp_vals else 0.0
            self.vc[seq] = dict(mfe=mfe, stem_count=stems, loop_count=loops,
                                bp_prob_mean=bp_mean, bp_prob_max=bp_max)
            return torch.tensor([mfe, float(stems), float(loops),
                                  bp_mean, bp_max, len(seq)/DNA_MAX_LEN],
                                 dtype=torch.float32)
        except Exception:
            return torch.zeros(6, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq  = row["aptamer_sequence"]
        prot = row["protein_sequence"]

        # aptamer tokens — 3-mer ids (scratch) or DNABERT-2 BPE input_ids (dnabert2)
        apt_tok = self._encode_aptamer(seq)

        # vienna features [6]
        v_feats = self._vienna_feats(seq)

        # protein embedding [prot_len, ESM_EMBED_DIM] — loaded from .npy cache
        prot_emb = torch.from_numpy(np.load(self.seq_to_emb_path[prot])).float()
        if prot_emb.shape[0] > self.max_prot_len:
            prot_emb = prot_emb[:self.max_prot_len]

        # condition vector [5]
        def _f(col, default):
            v = row.get(col, default)
            return float(v) if pd.notna(v) else float(default)

        condition = torch.tensor([
            _f("ph",                  DEFAULT_PH),
            _f("na_concentration_mM", DEFAULT_SALT_MM),
            _f("temperature_C",       DEFAULT_TEMP_C),
            float(DEFAULT_BUFFER),     # no buffer_type column in cleaned schema
            _f("mg_concentration_mM", DEFAULT_MG_MM),
        ], dtype=torch.float32)

        # label [1]
        label = torch.tensor([float(row["label"])], dtype=torch.float32)

        # Kd: log10(nM + 1), NaN if missing
        kd_raw = row.get("kd_value", float("nan"))
        if pd.notna(kd_raw) and float(kd_raw) >= 0:
            kd = torch.tensor([np.log10(float(kd_raw) + 1)], dtype=torch.float32)
        else:
            kd = torch.tensor([float("nan")], dtype=torch.float32)

        return apt_tok, v_feats, prot_emb, condition, label, kd


def collate_fn(batch):
    apt_list, v_list, emb_list, cond_list, lbl_list, kd_list = zip(*batch)

    # Aptamer tokens: all padded to DNA_MAX_LEN → just stack
    apt   = torch.stack(apt_list)          # [B, DNA_MAX_LEN]
    v     = torch.stack(v_list)            # [B, 6]
    cond  = torch.stack(cond_list)         # [B, 5]
    lbls  = torch.stack(lbl_list)          # [B, 1]
    kds   = torch.stack(kd_list)           # [B, 1]

    # Protein embeddings: variable prot_len → pad with zeros to max in batch
    max_prot = max(e.shape[0] for e in emb_list)
    esm_dim  = emb_list[0].shape[1]
    prot_emb = torch.stack([
        F.pad(e, (0, 0, 0, max_prot - e.shape[0]))   # pad sequence dim only
        for e in emb_list
    ])                                     # [B, max_prot, ESM_EMBED_DIM]

    # protein_tokens placeholder (unused when protein_emb is passed to model)
    prot_tok = torch.zeros(len(apt_list), 1, dtype=torch.long)

    return apt, v, prot_tok, cond, lbls, kds, prot_emb


# ── Data split ────────────────────────────────────────────────────────────────

def split_by_protein_family(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign entire protein families to train / val / test.
    Never mixes rows from the same protein across splits.
    Used by finetune.py to re-split small filtered target subsets; Stage 1
    train.py instead reads the pre-split augmented files in data/augmented/.
    """
    families = sorted(df["target_name"].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    n        = len(families)
    n_train  = max(1, int(n * train_frac))
    n_val    = max(1, int(n * val_frac))

    train_f  = set(families[:n_train])
    val_f    = set(families[n_train : n_train + n_val])
    test_f   = set(families[n_train + n_val :])

    tr  = df[df["target_name"].isin(train_f)].reset_index(drop=True)
    va  = df[df["target_name"].isin(val_f)].reset_index(drop=True)
    te  = df[df["target_name"].isin(test_f)].reset_index(drop=True)

    log.info("Split: %d train / %d val / %d test rows  "
             "(%d / %d / %d families)",
             len(tr), len(va), len(te),
             len(train_f), len(val_f), len(test_f))
    return tr, va, te


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, max_batches=None,
                use_amp=False, grad_accum=1):
    model.train()
    total_loss = bce_sum = kd_sum = 0.0
    all_labels: list = []
    all_probs:  list = []
    all_kd_true: list = []
    all_kd_pred: list = []
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    t_batch = time.time()
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else torch.amp.autocast("cpu", enabled=False))

    optimizer.zero_grad(set_to_none=True)
    for batch_i, (apt, v, prot_tok, cond, labels, kds, prot_emb) in enumerate(loader):
        if max_batches is not None and batch_i >= max_batches:
            break

        apt      = apt.to(device)
        v        = v.to(device)
        prot_emb = prot_emb.to(device)
        cond     = cond.to(device)
        labels   = labels.to(device)
        kds      = kds.to(device)

        with amp_ctx:
            out = model(apt, v, prot_tok, cond, protein_emb=prot_emb)

        # Loss always in float32 for numerical stability
        loss, bce, kd_l = criterion(
            out.binding_prob.float(),
            labels,
            out.kd_pred.float() if out.kd_pred is not None else None,
            kds,
        )
        # Gradient accumulation: scale so summed grads over `grad_accum`
        # micro-batches equal the gradient of one effective batch.
        (loss / grad_accum).backward()

        # Step on the last micro-batch of each accumulation window (and on the
        # final batch of the epoch, so a partial tail window isn't dropped).
        is_last = (batch_i + 1 == n_batches)
        if (batch_i + 1) % grad_accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        bce_sum    += bce.item()
        kd_sum     += kd_l.item() if isinstance(kd_l, torch.Tensor) else kd_l

        all_labels.extend(labels.detach().cpu().squeeze(-1).tolist())
        all_probs.extend(out.binding_prob.detach().float().cpu().squeeze(-1).tolist())
        all_kd_true.extend(kds.detach().cpu().squeeze(-1).tolist())
        all_kd_pred.extend(out.kd_pred.detach().float().cpu().squeeze(-1).tolist()
                           if out.kd_pred is not None else [float("nan")] * labels.shape[0])

        if (batch_i + 1) % 10 == 0 or (batch_i + 1) == n_batches:
            elapsed = time.time() - t_batch
            log.info("  batch %d/%d  loss=%.4f  %.1fs/batch",
                     batch_i + 1, n_batches, loss.item(), elapsed / 10)
            t_batch = time.time()

    n = max(batch_i + 1, 1)
    return (total_loss/n, bce_sum/n, kd_sum/n,
            all_labels, all_probs, all_kd_true, all_kd_pred)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, max_batches=None, use_amp=False):
    model.eval()
    total_loss = bce_sum = kd_sum = 0.0
    all_labels: list = []
    all_probs:  list = []
    all_kd_true: list = []
    all_kd_pred: list = []
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else torch.amp.autocast("cpu", enabled=False))

    for batch_i, (apt, v, prot_tok, cond, labels, kds, prot_emb) in enumerate(loader):
        if max_batches is not None and batch_i >= max_batches:
            break
        apt      = apt.to(device)
        v        = v.to(device)
        prot_emb = prot_emb.to(device)
        cond     = cond.to(device)
        labels   = labels.to(device)
        kds      = kds.to(device)

        with amp_ctx:
            out = model(apt, v, prot_tok, cond, protein_emb=prot_emb)

        loss, bce, kd_l = criterion(
            out.binding_prob.float(),
            labels,
            out.kd_pred.float() if out.kd_pred is not None else None,
            kds,
        )

        total_loss += loss.item()
        bce_sum    += bce.item()
        kd_sum     += kd_l.item() if isinstance(kd_l, torch.Tensor) else kd_l

        all_labels.extend(labels.cpu().squeeze(-1).tolist())
        all_probs.extend(out.binding_prob.float().cpu().squeeze(-1).tolist())
        all_kd_true.extend(kds.cpu().squeeze(-1).tolist())
        all_kd_pred.extend(out.kd_pred.float().cpu().squeeze(-1).tolist()
                           if out.kd_pred is not None else [float("nan")] * labels.shape[0])

    n = max(batch_i + 1, 1)
    return (total_loss/n, bce_sum/n, kd_sum/n,
            all_labels, all_probs, all_kd_true, all_kd_pred)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CondAptNet Stage 1 training")
    parser.add_argument("--max-epochs",  type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr-base",     type=float, default=LEARNING_RATE_BASE)
    parser.add_argument("--lr-lora",     type=float, default=LEARNING_RATE_LORA)
    parser.add_argument("--augmented-dir", type=str, default=DATA_AUGMENTED,
                        help="Directory holding tier1_train.csv / val.csv / test.csv "
                             "produced by scripts/data/augment.py")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(CHECKPOINTS_DIR, "pretrain"))
    parser.add_argument("--prot-max-tokens", type=int, default=PROT_MAX_TOKENS,
                        help="Truncate protein to this many tokens during ESM-2 embedding "
                             "(applied once at pre-compute time)")
    parser.add_argument("--max-prot-len", type=int, default=PROT_MAX_TOKENS,
                        help="Cap protein embedding length per sample fed to the CNN "
                             "(can be smaller than prot-max-tokens; reduces cross-attn memory)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch (for smoke-testing)")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM,
                        help="Accumulate gradients over this many micro-batches before "
                             "an optimizer step. Effective batch = batch-size * grad-accum. "
                             "Use a small --batch-size with --grad-accum>1 to fit long "
                             "proteins (e.g. --batch-size 8 --grad-accum 4 = effective 32) "
                             "without the interaction-matrix OOM at --max-prot-len 1024.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest epoch_*.pt checkpoint in "
                             "--checkpoint-dir, restoring optimizer, scheduler, "
                             "and best val MCC so early stopping continues correctly")
    parser.add_argument("--use-amp", action="store_true", default=False,
                        help="Enable BF16 automatic mixed precision (A100/Ampere GPUs). "
                             "Halves activation memory — enables batch=32 + prot_len=1024 "
                             "on A100 that would OOM without it.")
    parser.add_argument("--allow-cpu", action="store_true", default=False,
                        help="Allow training on CPU when no GPU/MPS device is found. "
                             "Disabled by default to prevent silent hours-long CPU runs "
                             "when a GPU driver fails. Only use for unit tests or CI.")
    parser.add_argument("--balanced-sampling", action="store_true", default=False,
                        help="Use WeightedRandomSampler for ~50/50 pos/neg per batch. "
                             "Positives are resampled with replacement each epoch so the "
                             "full positive set is seen over training, not a fixed slice. "
                             "Overrides shuffle=True on the train DataLoader.")
    args = parser.parse_args()

    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Runtime device check overrides config.py (catches late CUDA init on Colab)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Hard refusal: training on CPU wastes hours silently and is never
        # intentional on this codebase (GCP L4 or Apple MPS expected).
        # A GPU driver failure should be caught here, not discovered via
        # a 2-hour CPU run. Override with --allow-cpu if you genuinely
        # need CPU (e.g. unit tests or CI without a GPU).
        if not getattr(args, "allow_cpu", False):
            log.error(
                "No CUDA or MPS device found — torch.cuda.is_available()=False, "
                "torch.backends.mps.is_available()=False. "
                "Training on CPU is refused to prevent silent multi-hour CPU runs. "
                "Check your GPU driver (run: nvidia-smi). "
                "To override intentionally: pass --allow-cpu."
            )
            sys.exit(1)
        device = torch.device("cpu")
        log.warning("CPU training permitted via --allow-cpu — this will be very slow")
    log.info("Device confirmed at runtime: %s", device)

    use_amp = args.use_amp and device.type == "cuda"
    if use_amp:
        log.info("BF16 AMP enabled — activations in bfloat16, loss in float32")
    elif args.use_amp:
        log.warning("--use-amp requested but device is %s (not CUDA) — AMP disabled", device)

    torch.manual_seed(RANDOM_SEED)

    # ── Load data: augmented protein-family splits from augment.py ────────────
    # train = augmented (hard negatives, cross-target negatives, truncations,
    # scrambles); val/test = never augmented. Splits are leakage-free by family.
    train_csv = os.path.join(args.augmented_dir, "tier1_train.csv")
    val_csv   = os.path.join(args.augmented_dir, "val.csv")
    test_csv  = os.path.join(args.augmented_dir, "test.csv")

    if not os.path.exists(train_csv):
        log.error("Augmented training file not found: %s\n"
                  "Run `python scripts/data/augment.py` first.", train_csv)
        sys.exit(1)

    log.info("Loading augmented splits from %s", args.augmented_dir)

    def _load_ready(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        d = pd.read_csv(path)
        ready = d["aptamer_sequence"].notna() & d["protein_sequence"].notna()
        return d[ready].reset_index(drop=True)

    train_df = _load_ready(train_csv)
    val_df   = _load_ready(val_csv)
    test_df  = _load_ready(test_csv)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    log.info("Split (augmented): %d train / %d val / %d test rows",
             len(train_df), len(val_df), len(test_df))

    if len(train_df) == 0:
        log.error("Train split is empty in %s.", train_csv)
        sys.exit(1)

    # ── Load auxiliary tools ──────────────────────────────────────────────────
    tokenizer = DNATokenizer()

    vienna_cache: dict = {}
    if os.path.exists(VIENNA_CACHE):
        with open(VIENNA_CACHE, "rb") as f:
            vienna_cache = pickle.load(f)
        log.info("Vienna cache: %d entries", len(vienna_cache))
    else:
        log.warning("Vienna cache not found — features will be computed on-the-fly")

    # ── Vienna cache coverage guard ───────────────────────────────────────────
    # Abort if > 1% of sequences across any split are missing from the cache.
    # Missing sequences get zero ViennaRNA features, which the model can learn
    # as a spurious "non-binder" signal. Run scripts/data/vienna_features.py
    # first to fix: python scripts/data/vienna_features.py --input data/augmented/tier1_train.csv
    _CACHE_THRESHOLD = 0.01
    _splits_to_check = [(train_df, "train"), (val_df, "val"), (test_df, "test")]
    _guard_failed = False
    for _split_df, _split_name in _splits_to_check:
        if len(_split_df) == 0:
            continue
        _missing = [s for s in _split_df["aptamer_sequence"].dropna()
                    if str(s).strip().upper() not in vienna_cache]
        _rate = len(_missing) / len(_split_df)
        if _rate > _CACHE_THRESHOLD:
            log.error(
                "Vienna cache gap in %s split: %d / %d sequences missing (%.1f%%) — "
                "threshold is %.0f%%. Fix: run `python scripts/data/vienna_features.py "
                "--input data/augmented/tier1_train.csv` (incremental, skips cached). "
                "First 5 missing: %s",
                _split_name, len(_missing), len(_split_df), 100 * _rate,
                100 * _CACHE_THRESHOLD, [s[:30] for s in _missing[:5]],
            )
            _guard_failed = True
    if _guard_failed:
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────────────
    log.info("Building CondAptNet...")
    model = CondAptNet(predict_kd=True)
    model.set_stage1()
    model = model.to(device)
    # Move protein_encoder back to CPU for embedding pre-computation
    model.protein_encoder = model.protein_encoder.to("cpu")

    log.info("Total params:     %d", model.total_params())
    log.info("Trainable params: %d (%.2f%%)",
             model.trainable_params(),
             100 * model.trainable_params() / model.total_params())

    # ── Pre-compute protein embeddings (ESM-2, run ONCE per unique protein) ───
    emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
    all_prot_seqs = df["protein_sequence"].dropna().unique().tolist()
    seq_to_emb = precompute_protein_embeddings(
        all_prot_seqs,
        model.protein_encoder,
        emb_dir,
        device=torch.device("cpu"),   # pre-compute on CPU to avoid MPS OOM
        prot_max_tokens=args.prot_max_tokens,
    )
    # After pre-computation, move protein_encoder to main device
    model.protein_encoder = model.protein_encoder.to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    log.info("Building datasets (max_prot_len=%d)...", args.max_prot_len)
    train_ds = AptamerDataset(train_df, tokenizer, vienna_cache, seq_to_emb,
                               max_prot_len=args.max_prot_len)

    val_ds = None
    if len(val_df) > 0:
        val_ds = AptamerDataset(val_df, tokenizer, vienna_cache, seq_to_emb,
                                max_prot_len=args.max_prot_len)

    # MPS requires num_workers=0; CUDA benefits from prefetch workers + pinned memory
    _num_workers = 0 if device.type in ("mps", "cpu") else 2
    _pin_memory  = device.type == "cuda"

    # Balanced sampling: ~50/50 pos/neg per batch via WeightedRandomSampler.
    # Each sample's weight = 1/class_count — so minority and majority classes
    # contribute equally in expectation. Sampling with replacement means the
    # full positive set is reachable over an epoch, not a fixed 50% slice.
    if args.balanced_sampling:
        labels_arr = train_df["label"].values.astype(int)
        class_counts = np.bincount(labels_arr)           # [neg_count, pos_count]
        sample_weights = np.where(labels_arr == 1,
                                  1.0 / class_counts[1],
                                  1.0 / class_counts[0])
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        # Verify balance on a quick sample before training
        _check = np.random.default_rng(0).choice(
            len(train_ds), size=min(256, len(train_ds)),
            p=sample_weights / sample_weights.sum(), replace=True)
        _bal = labels_arr[_check].mean()
        log.info("Balanced sampler active: pos_rate in sampled check=%.3f (target ~0.50) "
                 "| class counts: neg=%d pos=%d", _bal, class_counts[0], class_counts[1])
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,       # sampler is mutually exclusive with shuffle
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin_memory,
            persistent_workers=_num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin_memory,
            persistent_workers=_num_workers > 0,
        )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   collate_fn=collate_fn, num_workers=_num_workers,
                   pin_memory=_pin_memory, persistent_workers=_num_workers > 0)
        if val_ds else None
    )

    # ── Optimizer: separate LR for LoRA vs rest ───────────────────────────────
    lora_params  = [p for n, p in model.named_parameters()
                    if "lora_" in n and p.requires_grad]
    base_params  = [p for n, p in model.named_parameters()
                    if "lora_" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr_base},
        {"params": lora_params, "lr": args.lr_lora},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    criterion = CondAptNetLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    # val_metric / best_val_metric track EARLY_STOPPING_METRIC (config.py).
    # Default -1.0 works for both MCC (range [-1,1]) and AUC-ROC (range [0,1]).
    log.info("Early-stopping metric: %s  patience=%d", EARLY_STOPPING_METRIC, EARLY_STOPPING_PATIENCE)
    best_val_metric = -1.0
    patience_count  = 0
    start_epoch     = 1
    best_ckpt_path  = os.path.join(args.checkpoint_dir, "best.pt")

    if args.resume:
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(args.checkpoint_dir, "epoch_*.pt")))
        if ckpts:
            resume_path = ckpts[-1]
            log.info("Resuming from %s", resume_path)
            # weights_only=True: refuse to unpickle arbitrary objects from a
            # checkpoint (pickle-deserialization RCE surface). Our checkpoints
            # hold only tensors / plain dicts / numbers / strings, so this is safe.
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            # strict=False tolerates new keys added by lazy-init modules (e.g.
            # condition_encoder._film_heads) that weren't exercised when the
            # checkpoint was saved.  We explicitly verify critical keys were loaded.
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            benign_missing = [k for k in missing if "_film_heads" in k]
            real_missing   = [k for k in missing if k not in benign_missing]
            if real_missing:
                raise RuntimeError(f"Critical weights missing from checkpoint: {real_missing}")
            if unexpected:
                log.warning("Checkpoint has unexpected keys (ignored): %s", unexpected)
            if benign_missing:
                log.info("Lazily-init keys not in checkpoint (random init, OK): %s",
                         benign_missing)
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            # Backward-compat: old checkpoints used "best_val_mcc"/"val_mcc".
            # When switching metric, start fresh at -1.0 (correct for both MCC
            # and AUC-ROC) rather than carrying over a value from a different metric.
            old_key = "best_val_metric"
            if old_key not in ckpt:
                log.warning(
                    "Checkpoint has no 'best_val_metric' key (older format or metric switch). "
                    "Resetting best_val_metric=-1.0 for metric=%s. "
                    "Patience counter preserved: %d.",
                    EARLY_STOPPING_METRIC, ckpt.get("patience_count", 0),
                )
            best_val_metric = ckpt.get("best_val_metric", -1.0)
            patience_count  = ckpt.get("patience_count", 0)
            start_epoch     = ckpt.get("epoch", 0) + 1
            log.info("Resumed at epoch %d  best_val_%s=%.4f  patience=%d",
                     start_epoch - 1, EARLY_STOPPING_METRIC, best_val_metric, patience_count)
        else:
            log.warning("--resume set but no epoch_*.pt found in %s — starting fresh",
                        args.checkpoint_dir)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log.info("GPU memory: %.1fGB free / %.1fGB total before training",
                 free / 1e9, total / 1e9)

    log.info("=" * 65)
    log.info("Training for up to %d epochs on device=%s", args.max_epochs, device)
    log.info("Micro-batch=%d  grad-accum=%d  → effective batch=%d  | max_prot_len=%d  AMP=%s",
             args.batch_size, args.grad_accum, args.batch_size * args.grad_accum,
             args.max_prot_len, use_amp)
    log.info("=" * 65)

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()

        (tr_loss, tr_bce, tr_kd,
         tr_labels, tr_probs,
         tr_kd_t, tr_kd_p) = train_epoch(model, train_loader, optimizer,
                                          criterion, device,
                                          max_batches=args.max_batches,
                                          use_amp=use_amp,
                                          grad_accum=args.grad_accum)

        tr_m = compute_metrics(tr_labels, tr_probs,
                               kd_true=tr_kd_t, kd_pred=tr_kd_p)

        elapsed = time.time() - t0

        if val_loader is not None:
            (va_loss, va_bce, va_kd,
             va_labels, va_probs,
             va_kd_t, va_kd_p) = eval_epoch(model, val_loader, criterion, device,
                                             max_batches=args.max_batches,
                                             use_amp=use_amp)
            va_m = compute_metrics(va_labels, va_probs,
                                   kd_true=va_kd_t, kd_pred=va_kd_p)
            # val_metric is the configurable early-stopping signal.
            # MCC: return -1.0 sentinel when degenerate (0/0) so patience
            #      increments correctly without NaN propagating.
            # AUC: never degenerate when both classes are present in val.
            if EARLY_STOPPING_METRIC == "mcc":
                val_metric = va_m["mcc"] if not va_m["mcc_degenerate"] else -1.0
            else:
                val_metric = va_m[EARLY_STOPPING_METRIC]

            va_mcc_str = ("UNDEF(0/0-all-pos/neg)"
                          if va_m["mcc_degenerate"] else f"{va_m['mcc']:.3f}")
            tr_mcc_str = ("UNDEF(0/0)"
                          if tr_m["mcc_degenerate"] else f"{tr_m['mcc']:.3f}")
            log.info(
                "Epoch %3d/%d | "
                "loss=%.4f (bce=%.4f kd=%.4f) | "
                "train AUC=%.3f MCC=%s | "
                "val loss=%.4f AUC=%.3f MCC=%s [stop_metric(%s)=%.4f] | "
                "%.0fs",
                epoch, args.max_epochs,
                tr_loss, tr_bce, tr_kd,
                tr_m["auroc"], tr_mcc_str,
                va_loss, va_m["auroc"], va_mcc_str,
                EARLY_STOPPING_METRIC, val_metric,
                elapsed,
            )
        else:
            if EARLY_STOPPING_METRIC == "mcc":
                val_metric = tr_m["mcc"] if not tr_m["mcc_degenerate"] else -1.0
            else:
                val_metric = tr_m[EARLY_STOPPING_METRIC]
            tr_mcc_str = ("UNDEF(0/0)"
                          if tr_m["mcc_degenerate"] else f"{tr_m['mcc']:.3f}")
            log.info(
                "Epoch %3d/%d | "
                "loss=%.4f (bce=%.4f kd=%.4f) | "
                "train AUC=%.3f MCC=%s [stop_metric(%s)=%.4f] | "
                "(no val split) | %.0fs",
                epoch, args.max_epochs,
                tr_loss, tr_bce, tr_kd,
                tr_m["auroc"], tr_mcc_str,
                EARLY_STOPPING_METRIC, val_metric,
                elapsed,
            )

        # Save every epoch
        ckpt = {
            "epoch":            epoch,
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "val_metric":       val_metric,
            "best_val_metric":  best_val_metric,
            "stop_metric":      EARLY_STOPPING_METRIC,
            "patience_count":   patience_count,
            "train_loss":       tr_loss,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        # Save best
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_count = 0
            ckpt["best_val_metric"]  = best_val_metric
            ckpt["patience_count"]   = patience_count
            torch.save(ckpt, best_ckpt_path)
            log.info("  → New best %s=%.4f  saved to %s",
                     EARLY_STOPPING_METRIC, best_val_metric, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOPPING_PATIENCE:
                log.info("Early stopping at epoch %d (patience=%d, metric=%s)",
                         epoch, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_METRIC)
                break

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Training complete. Best val %s=%.4f", EARLY_STOPPING_METRIC, best_val_metric)
    log.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
