# CondAptNet

**Conditional Aptamer-Protein Interaction Network**

A novel deep learning architecture for predicting DNA aptamer binding to arbitrary protein targets, built for [Continuity](https://continuity.bio)'s real-time physiological biosensing platform.

---

## Overview

CondAptNet is a **general-purpose** DNA aptamer-protein interaction prediction model. Given any protein's amino acid sequence and a DNA aptamer sequence, it predicts whether the aptamer will bind that protein and how strongly.

It is designed in three tiers:

```
TIER 1 — GENERAL MODEL
Trained on hundreds of diverse protein families.
Generalizes to any protein. This is the core product.

TIER 2 — VALIDATION BENCHMARK
Fine-tuned on insulin, myoglobin, NT-proBNP, troponin I/T, albumin.
These are well-studied proteins with published aptamers and known Kd values.
Used to verify the model works before trusting it on real device targets.
These are NOT the actual deployment targets for Continuity's device.

TIER 3 — DEPLOYMENT TARGETS (TBD)
The real Continuity biomarker set, not yet confirmed.
Plug-and-play: update config.py and run finetune.py when targets are known.
No architectural changes required.
```

---

## Why a New Model

| Model | Year | Key Limitation |
|---|---|---|
| Apta-MCTS | 2021 | Shallow Random Forest, no generalization |
| AptaTrans | 2023 | Converts DNA→RNA (lossy), 164 proteins from 2012 |
| AptaBERT | 2023 | Proprietary data, not reproducible |
| AptaBLE | 2026 | RNA-focused, no physiological condition encoding |

**CondAptNet's novel contributions:**
- **Native DNA encoding** — no T→U substitution used by all prior models
- **ESM-2 protein encoder** — Meta's LLM pretrained on 250M protein sequences
- **Physiological condition injection** — pH, salt, temperature, buffer via FiLM
- **Dual output** — binding probability AND Kd affinity regression
- **Broadest training distribution** — hundreds of diverse protein families
- **Plug-and-play fine-tuning** — swap deployment targets without architecture changes

---

## Architecture

```
DNA Aptamer Sequence     Protein Sequence       Condition Vector
[A, T, G, C — native]   [amino acids]          [pH, salt, temp, buffer]
        │                      │                        │
        ▼                      ▼                        ▼
  DNA Encoder            Protein Encoder          Condition Encoder
  6-layer Transformer    ESM-2 (35M params)       Linear MLP
  3-mer tokenization     Fine-tuned via LoRA      128-dim output
  + ViennaRNA features   480-dim per residue
        │                      │                        │
        └──────────┬───────────┘                        │
                   ▼                                    │
       Symmetric Bidirectional                          │
         Cross-Attention          ◄─────────────────────┘
       (aptamer ↔ protein)        FiLM condition injection
                   │
                   ▼
          Interaction Matrix
          17-block CNN
          channels: 64 → 128 → 256
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Binding probability         │  → P(aptamer binds) ∈ [0,1]
    │  Kd regression               │  → predicted affinity (nM)
    └──────────────────────────────┘
```

### Component Rationale

**DNA Encoder:** Transformer with native 3-mer tokenization. No T→U conversion. Augmented with ViennaRNA secondary structure features (MFE, stem count, loop count, base pair probabilities).

**Protein Encoder (ESM-2):** Pretrained on 250 million protein sequences. Deeply understands any protein — including ones with zero aptamer training data. This is what makes the general model possible. Fine-tuned with LoRA (rank=8) to run on Apple M-series chips.

**Symmetric Bidirectional Cross-Attention:** Both molecules attend to each other simultaneously. Validated by AptaBLE (2026) as superior to unidirectional approaches.

**FiLM Condition Injection:** pH, salt, temperature, and buffer modulate cross-attention feature maps via learned scale and shift parameters. First aptamer model to encode physiological context.

**17-block CNN:** Extracts hierarchical features from the 2D aptamer-protein interaction map. From AptaTrans (2023).

**Dual Output Head:** Binary binding classification + Kd regression for affinity-ranked candidate generation.

---

## Training Pipeline

### Stage 1 — Broad Pretraining
- Data: UTexas Aptamer DB + PubMed SELEX literature 2000–2025, hundreds of protein families
- ESM-2 frozen, all other layers trained
- Split: by protein family (never randomly)
- Output: general aptamer interaction model

### Stage 2 — Validation Fine-Tuning
- Data: insulin, myoglobin, NT-proBNP, troponin I/T, albumin
- Purpose: benchmark model against published Kd values, verify generalization
- Identical script to Stage 3 — just a different protein set in config

### Stage 3 — Deployment Fine-Tuning (TBD)
- Data: actual Continuity device targets (not yet confirmed)
- Update `DEPLOYMENT_TARGETS` in config.py and run `finetune.py`
- No other changes required

### Stage 4 — Active Learning (ongoing)
- Lab validation results → retrain Stage 2/3 continuously

---

## Data Sources

| Source | Type | Size |
|---|---|---|
| UTexas Aptamer Database (Zenodo: doi.org/10.5281/zenodo.8264921) | Primary curated DB — bulk download | ~896 ssDNA rows (filtered from 1,495) |
| PubMed SELEX 2000–2025 | Literature extraction | ~500–1000 pairs |
| Li et al. 2014 benchmark (doi:10.1371/journal.pone.0086729) | Standard benchmark — labels + targets | 2,320 entries, 164 proteins (sequences need enrichment) |
| GitHub AptamerBase dump (github.com/micheldumontier/aptamerbase) | Pre-2016 entries, supplementary | Supplementary (original site shut down 2016) |
| Therapeutic literature | Clinical-stage aptamers | ~50–100 pairs |

### Augmentation
- Reverse complement of all positive sequences (doubles data — AptaTrans validated)
- Systematic truncations (length variants)
- Cross-target negatives (hard negatives — binder for A = non-binder for B)
- Scrambled sequences (composition preserved, order destroyed → non-binders)

---

## Project Structure

```
continuitybioML/
├── CLAUDE.md                    # Full context for Claude Code
├── README.md                    # This file
├── config.py                    # All hyperparameters
├── data/
│   ├── raw/                     # Source data, never modified
│   ├── processed/               # Cleaned, unified format
│   └── augmented/               # Training splits by tier
├── models/
│   ├── encoders/
│   │   ├── dna_encoder.py
│   │   ├── protein_encoder.py
│   │   └── condition_encoder.py
│   ├── attention/
│   │   └── cross_attention.py
│   ├── interaction/
│   │   └── cnn_head.py
│   ├── output/
│   │   └── dual_head.py
│   ├── condaptnet.py
│   └── checkpoints/
│       ├── pretrain/
│       ├── validation/
│       └── deployment/          # Empty until Tier 3 targets confirmed
├── scripts/
│   ├── data/                    # Collection, validation, augmentation
│   ├── model/                   # Tokenizer
│   ├── training/                # Train, finetune, losses
│   └── evaluation/              # Metrics, evaluate
└── outputs/
    ├── candidates/
    └── motifs/
```

---

## Environment

- Python 3.11 (virtual environment: `condaptnet_env`)
- PyTorch 2.11.0
- Apple MPS — M-series GPU acceleration confirmed working
- ESM-2: `esm2_t12_35M_UR50D` (35M params, 480-dim)
- ViennaRNA — secondary structure prediction

### Setup

```bash
git clone https://github.com/shivanshbansal/continuitybioML
cd continuitybioML
python3.11 -m venv condaptnet_env
source condaptnet_env/bin/activate
pip install torch torchvision torchaudio
pip install fair-esm pandas numpy scikit-learn biopython ViennaRNA requests tqdm
python scripts/verify_env.py
```

### Usage

```bash
source condaptnet_env/bin/activate

# Collect data
python scripts/data/collect_aptamers.py

# Precompute structure features
python scripts/data/vienna_features.py

# Stage 1: broad pretraining
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py

# Stage 2 or 3: fine-tuning
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py

# Evaluate
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt
```

---

## Evaluation Metrics

| Metric | Why |
|---|---|
| MCC | Best single metric for imbalanced binary classification |
| AUC-ROC | Discriminative ability across thresholds |
| AUC-PR | More informative than ROC when positives are rare |
| Sensitivity | False negatives = missed candidates |
| Pearson r (Kd) | Regression head quality |

Accuracy is not reported as a primary metric — data is inherently imbalanced and accuracy is misleading.

---

## Reference Papers

| Paper | Key Contribution |
|---|---|
| Li et al., PLOS ONE (2014) | Standard benchmark — 725 pairs, 164 proteins |
| Lee et al., PLOS ONE (2021) — Apta-MCTS | MCTS generation; RF baseline |
| Shin et al., BMC Bioinformatics (2023) — AptaTrans | Interaction matrix + CNN; pretraining; augmentation |
| Morsch et al., bioRxiv (2023) — AptaBERT | BERT-style aptamer pretraining |
| Atom Bioworks, bioRxiv (2026) — AptaBLE | Symmetric bidirectional cross-attention; current SOTA |

---

## Build Status

- [x] Environment setup (Python 3.11, PyTorch, ESM-2, ViennaRNA, MPS)
- [x] Project structure
- [x] config.py — all hyperparams + physiological defaults (37°C, 150mM Na, 2mM Mg)
- [x] collect_aptamers.py
- [x] validate_sequences.py
- [x] vienna_features.py
- [x] tokenizer.py
- [x] dna_encoder.py — 6-layer Transformer [B, L, 128], MPS verified
- [x] protein_encoder.py — ESM-2 + LoRA, 0.55% trainable [B, L, 480], MPS verified
- [x] condition_encoder.py — FiLM MLP [B, 128], MPS verified
- [x] cross_attention.py — symmetric bidirectional [B, L, 256], MPS verified
- [x] cnn_head.py — 17-block CNN [B, 256], MPS verified
- [x] dual_head.py — binding + Kd heads, MPS verified
- [x] build_dataset.py — UTexas (896 rows) + Li2014 (2320 rows) → master_dataset.csv
- [ ] condaptnet.py (full assembly)
- [ ] losses.py
- [ ] train.py
- [ ] evaluate.py
- [ ] finetune.py

---

## License

Private — built for Continuity (continuity.bio). All rights reserved.

---

*Built by Shivansh Bansal — Continuity ML Pipeline*