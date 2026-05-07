"""
Central configuration for CondAptNet.
All hyperparameters live here — never hardcode values in model files.
"""

import torch
import os

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

# ── Protein Encoder (ESM-2) ───────────────────────────────────────────────────
ESM_MODEL_NAME  = "esm2_t12_35M_UR50D"  # 35M params, 480-dim output
ESM_EMBED_DIM   = 480
LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# ── Condition Encoder (FiLM) ─────────────────────────────────────────────────
CONDITION_DIM   = 128
# Condition vector layout: [pH, salt_mM, temp_C, buffer_type_onehot × 4]
CONDITION_INPUT_DIM = 7   # pH, salt, temp, + 4-class buffer one-hot
BUFFER_TYPES    = {"PBS": 0, "HEPES": 1, "Tris": 2, "other": 3}

# ── Cross-Attention (Fusion) ──────────────────────────────────────────────────
FUSION_DIM      = 128
CROSS_ATTN_HEADS  = 8
CROSS_ATTN_DROPOUT = 0.1

# ── CNN Interaction Head ─────────────────────────────────────────────────────
CNN_CHANNELS    = [64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_NUM_BLOCKS  = 17

# ── Output Heads ─────────────────────────────────────────────────────────────
KD_OUTPUT_DIM   = 1   # log-scale nM
BINDING_OUTPUT_DIM = 1  # sigmoid probability

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE              = 32
LEARNING_RATE_BASE      = 1e-4
LEARNING_RATE_LORA      = 1e-5
WEIGHT_DECAY            = 1e-4
MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP               = 1.0

# Class imbalance — positives weighted 3× in BCE loss
POSITIVE_CLASS_WEIGHT   = 3.0

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

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED     = 42
