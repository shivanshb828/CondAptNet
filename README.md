# CondAptNet

**Conditional Aptamer-Protein Interaction Network**

A novel deep learning architecture for predicting DNA aptamer binding to arbitrary protein targets, built for [Continuity](https://continuity.bio)'s real-time physiological biosensing platform.

---

## Overview

CondAptNet is a general-purpose DNA aptamer-protein interaction prediction model that generalizes across arbitrary protein targets and is fine-tunable for specific biosensing applications. It is the first aptamer interaction model to combine native DNA encoding, a large pretrained protein language model (ESM-2), symmetric bidirectional cross-attention, and explicit physiological condition conditioning into a single unified architecture.

Traditional aptamer discovery uses SELEX — a wet lab process costing weeks and thousands of dollars per target protein. CondAptNet computationally predicts which DNA sequences will bind to a given protein, dramatically accelerating candidate discovery before synthesis begins.

---

## Why a New Model

Every existing aptamer prediction model has critical limitations:

| Model | Year | Key Limitation |
|---|---|---|
| Apta-MCTS | 2021 | Shallow Random Forest, no generalization beyond training proteins |
| AptaTrans | 2023 | Converts DNA→RNA (lossy), trained on only 164 proteins from 2012 |
| AptaBERT | 2023 | Proprietary training data, not reproducible |
| AptaBLE | 2026 | Best existing model, still RNA-focused, no condition encoding |

**CondAptNet's novel contributions:**

- **Native DNA encoding** — no T→U substitution used by all prior models
- **ESM-2 protein encoder** — Meta's language model pretrained on 250M protein sequences; no prior aptamer model uses this
- **Physiological condition injection** — pH, salt, temperature, and buffer encoded via FiLM conditioning; first aptamer model to do this
- **Dual output head** — binary binding classification AND continuous Kd regression
- **Broadest training distribution** — hundreds of diverse protein families, not just 164 proteins from 2012
- **Two-stage training** — broad pretraining then target-specific fine-tuning

---

## Architecture

```
DNA Aptamer Sequence          Protein Sequence           Condition Vector
[A,T,G,C — native DNA]        [amino acids]              [pH, salt, temp, buffer]
        │                           │                            │
        ▼                           ▼                            ▼
  DNA Encoder                Protein Encoder             Condition Encoder
  6-layer Transformer        ESM-2 (35M params)          Linear MLP
  3-mer tokenization         Fine-tuned via LoRA         128-dim output
  + ViennaRNA features       480-dim embeddings
        │                           │                            │
        └───────────────┬───────────┘                            │
                        ▼                                        │
            Symmetric Bidirectional                              │
              Cross-Attention          ◄────────────────────────┘
            (aptamer ↔ protein)        FiLM condition injection
                        │
                        ▼
               Interaction Matrix
               17-block CNN
               channels: 64 → 128 → 256
                        │
                        ▼
                 Dual Output Head
          ┌──────────────────────────┐
          │  Binding probability     │  → P(aptamer binds protein)
          │  Kd regression           │  → predicted affinity (nM)
          └──────────────────────────┘
```

### Component Rationale

**DNA Encoder:** Transformer with native 3-mer tokenization. Preserves DNA chemistry without the T→U conversion used by all prior models. Augmented with ViennaRNA secondary structure features (MFE, stem count, loop count).

**Protein Encoder (ESM-2):** Meta AI's protein language model pretrained on 250 million protein sequences. Provides rich representations of effectively any protein — including targets with zero aptamer training data. Fine-tuned with LoRA (rank=8) to remain computationally tractable on consumer hardware.

**Symmetric Bidirectional Cross-Attention:** Both molecules attend to each other simultaneously, capturing mutual binding geometry. Validated by AptaBLE (2026) as superior to unidirectional approaches.

**FiLM Condition Injection:** Experimental conditions (pH, salt, temperature, buffer type) modulate cross-attention feature maps via scale and shift parameters. Critical for Continuity's physiological sensing context where conditions vary and directly affect binding behavior.

**17-block CNN:** Extracts local and hierarchical interaction features from the 2D aptamer-protein interface map. Architecture from AptaTrans (2023), validated to outperform simpler pooling approaches.

**Dual Output Head:** Binding classification head (sigmoid) for binder/non-binder prediction. Kd regression head (ReLU, log-scale) for affinity estimation when training labels include dissociation constants.

---

## Target Proteins (Continuity Application)

The model is general-purpose but fine-tuned for Continuity's biosensor targets:

| Protein | Role | Biosensor Purpose |
|---|---|---|
| Insulin | Metabolic hormone | Blood glucose regulation monitoring |
| Myoglobin | Muscle protein | Muscle stress detection |
| NT-proBNP | Cardiac peptide | Cardiovascular stress monitoring |
| Troponin I | Cardiac protein | Heart attack biomarker |
| Troponin T | Cardiac protein | Heart attack biomarker |
| Albumin | Abundant serum protein | **Anti-target** — filter non-specific binders |

---

## Training Strategy

### Two-Stage Pipeline

**Stage 1 — Broad Pretraining**
Train on all collected aptamer-protein pairs spanning hundreds of diverse protein families (AptamerBase + PubMed SELEX literature 2000–2025). ESM-2 frozen, all other components trained. Goal: learn general aptamer-protein interaction priors.

**Stage 2 — Target-Specific Fine-Tuning**
Unfreeze ESM-2 LoRA layers. Fine-tune on Continuity's 5 target proteins + albumin anti-target. Lower learning rates. Smaller batches if data-limited.

**Stage 3 — Active Learning Loop**
Experimental validation results from Continuity's lab feed back into Stage 2 retraining. Every confirmed binder/non-binder improves predictions over time.

### Data Split
Always split by protein family, never randomly. Random splitting leaks information — a model that has seen 8 of 10 insulin aptamers in training will appear to generalize to insulin but hasn't actually learned to generalize. Family-held-out splits test true generalization.

### Evaluation Metrics (priority order)
1. MCC (Matthews Correlation Coefficient) — best single metric for imbalanced binary classification
2. AUC-ROC — discriminative ability
3. AUC-PR — precision-recall for imbalanced data
4. Sensitivity — false negatives = missed candidates
5. Pearson r on Kd — regression head quality

---

## Data Sources

| Source | Type | Estimated Size |
|---|---|---|
| AptamerBase | Curated DNA/RNA aptamer pairs | ~800 pairs |
| PubMed SELEX 2000–2025 | Literature extraction | ~500–1000 new pairs |
| Li et al. 2014 benchmark | Standard benchmark | 725 pairs, 164 proteins |
| Therapeutic aptamer literature | Clinical-stage aptamers | ~50–100 pairs |
| Target-specific searches | Insulin, troponin, etc. | ~50–200 pairs |

### Data Augmentation
- **Reverse complement** — flip every positive sequence (validated by AptaTrans, doubles data)
- **Truncations** — systematically shorten known aptamers to generate length variants
- **Cross-target negatives** — known binders for protein A labeled as non-binders for protein B
- **Scrambled sequences** — shuffle nucleotides, preserve composition, label as non-binders

---

## Project Structure

```
continuitybioML/
├── CLAUDE.md                    # Full project context for Claude Code
├── README.md                    # This file
├── config.py                    # All hyperparameters (single source of truth)
│
├── data/
│   ├── raw/                     # Downloaded data, never modified
│   ├── processed/               # Cleaned, validated, unified format
│   └── augmented/               # Post-augmentation training splits
│
├── models/
│   ├── encoders/
│   │   ├── dna_encoder.py       # Native DNA transformer
│   │   ├── protein_encoder.py   # ESM-2 + LoRA wrapper
│   │   └── condition_encoder.py # FiLM condition MLP
│   ├── attention/
│   │   └── cross_attention.py   # Symmetric bidirectional cross-attention
│   ├── interaction/
│   │   └── cnn_head.py          # 17-block CNN interaction head
│   ├── output/
│   │   └── dual_head.py         # Binding + Kd output heads
│   └── condaptnet.py            # Full model assembly
│
├── scripts/
│   ├── data/                    # Collection, validation, augmentation
│   ├── model/                   # Tokenizer
│   ├── training/                # Train, fine-tune, loss functions
│   └── evaluation/              # Metrics and evaluation suite
│
└── outputs/
    ├── candidates/              # Generated aptamer candidates
    └── motifs/                  # MEME motif analysis
```

---

## Environment

- Python 3.11 (virtual environment: `condaptnet_env`)
- PyTorch 2.11.0
- Apple MPS (M-series GPU acceleration)
- ESM-2: `esm2_t12_35M_UR50D` (35M parameters, 480-dim embeddings)
- ViennaRNA for secondary structure prediction

### Setup

```bash
# Clone repository
git clone https://github.com/shivanshbansal/continuitybioML
cd continuitybioML

# Create virtual environment with Python 3.11
python3.11 -m venv condaptnet_env
source condaptnet_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install fair-esm pandas numpy scikit-learn biopython ViennaRNA requests tqdm

# Verify environment
python scripts/verify_env.py
```

### Training

```bash
# Activate environment first
source condaptnet_env/bin/activate

# Collect data (run once, takes ~20 min)
python scripts/data/collect_aptamers.py

# Precompute ViennaRNA features
python scripts/data/vienna_features.py

# Stage 1: Broad pretraining
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py

# Stage 2: Fine-tuning on Continuity targets
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py

# Evaluate
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt
```

---

## Reference Papers

| Paper | Contribution |
|---|---|
| Li et al., PLOS ONE (2014) | Standard benchmark dataset — 725 pairs, 164 proteins |
| Lee et al., PLOS ONE (2021) — Apta-MCTS | MCTS generation algorithm; baseline Random Forest classifier |
| Shin et al., BMC Bioinformatics (2023) — AptaTrans | Interaction matrix + CNN architecture; pretraining strategy; reverse-complement augmentation |
| Morsch et al., bioRxiv (2023) — AptaBERT | BERT-style aptamer pretraining; 96% ROC-AUC shows transformers work well |
| Atom Bioworks, bioRxiv (2026) — AptaBLE | Symmetric bidirectional cross-attention; current SOTA |

---

## Current Status

- [x] Environment setup and verified (Python 3.11, PyTorch, ESM-2, ViennaRNA, MPS)
- [x] Project structure initialized
- [x] Configuration file (`config.py`)
- [x] PubMed data collection script (`collect_aptamers.py`)
- [ ] Sequence validation pipeline
- [ ] ViennaRNA feature precomputation
- [ ] DNA tokenizer
- [ ] DNA encoder
- [ ] ESM-2 + LoRA protein encoder
- [ ] Condition encoder
- [ ] Symmetric cross-attention module
- [ ] CNN interaction head
- [ ] Dual output head
- [ ] Full model assembly
- [ ] Training loop
- [ ] Evaluation suite
- [ ] Stage 2 fine-tuning

---

## License

Private — built for Continuity (continuity.bio). All rights reserved.

---

*Built by Shivansh Bansal — Continuity ML Pipeline*