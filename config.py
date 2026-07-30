"""
Central configuration for CondAptNet.
All hyperparameters live here — never hardcode values in model files.
"""

import torch
import os

# ── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"[CondAptNet] Device: {DEVICE}")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW         = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED   = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_AUGMENTED   = os.path.join(PROJECT_ROOT, "data", "augmented")
VIENNA_CACHE     = os.path.join(DATA_PROCESSED, "vienna_cache.pkl")
PROTEIN_EMB_DIR  = os.path.join(DATA_PROCESSED, "protein_embeddings")
CHECKPOINTS_DIR  = os.path.join(PROJECT_ROOT, "models", "checkpoints")

# ── DNA Encoder ──────────────────────────────────────────────────────────────
DNA_KMER_SIZE   = 3
DNA_VOCAB_SIZE  = 64 + 2   # 64 k-mers + [PAD] + [UNK]
DNA_PAD_ID      = 0
DNA_UNK_ID      = 1
DNA_EMBED_DIM   = 128
DNA_NUM_LAYERS  = 6
DNA_NUM_HEADS   = 8
DNA_FF_DIM      = 512
DNA_DROPOUT     = 0.1
DNA_MAX_LEN     = 120      # max aptamer nucleotides

# ── DNA Encoder selector (A/B ablation) ──────────────────────────────────────
# "scratch"  = the from-scratch 6-layer Transformer above (default, unchanged).
# "dnabert2" = pretrained DNABERT-2 foundation model + LoRA (dna_encoder_pretrained.py).
# NOTE: this is a switchable ablation, NOT a replacement — the scratch encoder
# remains the default and is bit-for-bit unchanged when this stays "scratch".
DNA_ENCODER_TYPE    = "scratch"
DNABERT2_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
# Pin to an exact commit SHA — never resolve the HF repo's moving default branch
# at runtime. This matters because loading DNABERT-2 uses trust_remote_code=True,
# which executes arbitrary Python fetched from the repo (bert_layers.py etc.);
# an unpinned ref would run whatever upstream pushes next. This SHA was the
# repo's main-branch HEAD as of 2026-07-04 and is the exact revision this
# codebase was built and smoke-tested against (bundles bert_layers.py +
# pytorch_model.bin + tokenizer). Bumping it must be a deliberate, re-verified
# change, not an accident.
DNABERT2_REVISION   = "7bce263b15377fc15361f52cfab88f8b586abda0"
DNABERT2_EMBED_DIM  = 768
# Session-2 spike measured BPE token counts on the real cleaned dataset:
# 120 nt -> 27 tokens, dataset-wide max 28 (BPE != nt count). 32 gives headroom.
DNABERT2_MAX_LEN      = 32
DNABERT2_LORA_RANK    = 8
DNABERT2_LORA_ALPHA   = 16
DNABERT2_LORA_DROPOUT = 0.05

# ── Protein Encoder (ESM-2) ───────────────────────────────────────────────────
ESM_MODEL_NAME  = "esm2_t12_35M_UR50D"  # 35M params, 480-dim output
ESM_EMBED_DIM   = 480
LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# ── Condition Encoder (FiLM) ─────────────────────────────────────────────────
# Input: 5 raw scalars [pH, salt_mM, temp_C, buffer_type, mg_mM]
# mg_mM is separate from salt_mM — divalent Mg2+ critically affects aptamer folding
CONDITION_INPUT_DIM = 5
CONDITION_HIDDEN    = 64
CONDITION_DIM       = 128   # output dim fed into FiLM scale/shift projection
BUFFER_TYPES    = {"PBS": 0, "HEPES": 1, "Tris": 2, "other": 3}

# Physiological defaults for Continuity's in-body sensing context
# Used as imputation when condition fields are missing in training data
DEFAULT_PH       = 7.4    # blood/interstitial fluid pH
DEFAULT_SALT_MM  = 150.0  # ~150 mM Na+ (physiological)
DEFAULT_TEMP_C   = 37.0   # body temperature
DEFAULT_BUFFER   = 0      # PBS (closest to physiological)
DEFAULT_MG_MM    = 2.0    # 2 mM Mg2+ (physiological divalent cation)

# ── Cross-Attention (Fusion) ──────────────────────────────────────────────────
FUSION_DIM         = 256   # both encoder dims projected to this before attention
CROSS_ATTN_HEADS   = 8
CROSS_ATTN_DROPOUT = 0.1

# ── CNN Interaction Head ─────────────────────────────────────────────────────
CNN_CHANNELS    = [64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_NUM_BLOCKS  = 17
# Channel-wise (Dropout2d) regularization inside each ConvBlock — guards the
# deep 17-block, 256-channel stack against overfitting on our ~3.3k real rows.
CNN_DROPOUT     = 0.1

# ── Output Heads ─────────────────────────────────────────────────────────────
KD_OUTPUT_DIM   = 1   # log-scale nM
BINDING_OUTPUT_DIM = 1  # sigmoid probability

# ── Qualitative Binding Label ─────────────────────────────────────────────────
# Integer encoding: 0 = low, 1 = medium, 2 = high
BIND_LABEL_LOW    = 0
BIND_LABEL_MEDIUM = 1
BIND_LABEL_HIGH   = 2
BIND_LABEL_NAMES  = {0: "low", 1: "medium", 2: "high"}

# Primary: Kd thresholds in raw nM (converted to log10(nM+1) internally)
BIND_LABEL_KD_HIGH_NM   = 10.0    # < 10 nM    → high
BIND_LABEL_KD_MEDIUM_NM = 1000.0  # < 1000 nM  → medium  (≥ 1000 nM → low)

# Fallback: binding_prob thresholds when kd_pred is unavailable
BIND_LABEL_PROB_HIGH   = 0.66   # > 0.66 → high
BIND_LABEL_PROB_MEDIUM = 0.33   # > 0.33 → medium  (≤ 0.33 → low)

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE              = 32
LEARNING_RATE_BASE      = 1e-4
LEARNING_RATE_LORA      = 1e-5
WEIGHT_DECAY            = 1e-4
MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 10
# Metric used for early stopping and best-checkpoint selection.
# "auroc"  — recommended while model is threshold-collapsed (all-positive predictions);
#            non-degenerate by construction, tracks real ranking improvement.
# "mcc"    — threshold=0.5 MCC; appropriate once model produces mixed predictions,
#            but returns a degenerate -1.0 sentinel when all-positive, silently
#            consuming patience even while AUC improves.
EARLY_STOPPING_METRIC   = "auroc"
GRAD_CLIP               = 1.0

# Class imbalance — positives weighted 3× in BCE loss
POSITIVE_CLASS_WEIGHT   = 3.0
# Loss component weights — total = BCE_WEIGHT*bce + KD_WEIGHT*mse_kd
BCE_WEIGHT              = 1.0
KD_WEIGHT               = 0.5
# Kd head predicts in log10(nM+1) space; KD_LOG_MAX is the clamp ceiling
KD_LOG_MAX              = 7.0   # corresponds to ~10 µM upper bound
# Max protein tokens fed to ESM-2 (BOS + residues + EOS)
PROT_MAX_TOKENS         = 1024

# ── Data Validation Bounds ───────────────────────────────────────────────────
SEQ_MIN_LEN     = 20
SEQ_MAX_LEN     = 120
GC_MIN          = 0.20
GC_MAX          = 0.80
MAX_HOMOPOLYMER = 8
VALID_BASES     = set("ATGC")

# ── Evaluation ───────────────────────────────────────────────────────────────
# Primary metric is MCC; list determines logging order
EVAL_METRICS    = ["mcc", "auroc", "auprc", "sensitivity", "specificity", "pearson_r_kd"]

# ── Target Lists ─────────────────────────────────────────────────────────────
# Tier 2: validation benchmarks only — NOT deployment targets
VALIDATION_TARGETS  = ["insulin", "myoglobin", "NT-proBNP",
                        "troponin_I", "troponin_T", "albumin"]
# Tier 3: update when Continuity confirms real device targets
DEPLOYMENT_TARGETS  = []

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED     = 42
