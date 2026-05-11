# CondAptNet

**Conditional Aptamer-Protein Interaction Network**

A novel deep learning architecture for predicting DNA aptamer binding to arbitrary protein targets, built for [Continuity](https://continuity.bio)'s real-time physiological biosensing platform.

---

## Overview

CondAptNet is a **general-purpose** DNA aptamer-protein interaction prediction model. Given any protein's amino acid sequence and a DNA aptamer sequence, it predicts whether the aptamer will bind that protein and how strongly.

It is designed in three tiers:

```
TIER 1 — GENERAL MODEL
Trained on 2,364 aptamer-protein pairs across 252 protein families.
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
- **Broadest training distribution** — 252 diverse protein families
- **Plug-and-play fine-tuning** — swap deployment targets without architecture changes

---

## Architecture

```
DNA Aptamer Sequence     Protein Sequence       Condition Vector
[A, T, G, C — native]   [amino acids]          [pH, salt, temp, buffer, Mg]
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
          17-block CNN (GroupNorm, GELU, residual)
          channels: 64 → 128 → 256
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Binding probability         │  → P(aptamer binds) ∈ [0,1]
    │  Kd regression               │  → predicted affinity (nM, log scale)
    └──────────────────────────────┘
```

### Component Rationale

**DNA Encoder:** Transformer with native 3-mer tokenization. No T→U conversion. Augmented with ViennaRNA secondary structure features (MFE, stem count, loop count, base pair probabilities).

**Protein Encoder (ESM-2):** Pretrained on 250 million protein sequences. Deeply understands any protein — including ones with zero aptamer training data. Fine-tuned with LoRA (rank=8, α=16) to run on Apple M-series and T4 GPUs. ESM-2 embeddings are pre-cached to disk; the frozen backbone runs only once per protein.

**Symmetric Bidirectional Cross-Attention:** Both molecules attend to each other simultaneously. Validated by AptaBLE (2026) as superior to unidirectional approaches.

**FiLM Condition Injection:** pH, salt, temperature, Mg²⁺, and buffer modulate cross-attention feature maps via learned scale and shift parameters. First aptamer model to encode physiological context. Physiological defaults: pH 7.4, 150 mM Na⁺, 37°C, 2 mM Mg²⁺, PBS.

**17-block CNN (GroupNorm):** Extracts hierarchical features from the 2D aptamer-protein interaction map. GroupNorm replaces BatchNorm2d for full MPS/CUDA native execution. From AptaTrans (2023).

**Dual Output Head:** Binary binding classification + Kd regression for affinity-ranked candidate generation. Kd head is skippable at inference when no affinity label is available.

---

## Training Pipeline

### Stage 1 — Broad Pretraining
- Data: 6,914 training rows (augmented) across 252 protein families
- ESM-2 frozen (only LoRA adapters + all other layers train)
- Split: by protein family (never randomly) — 70 / 15 / 15
- Loss: BCE (binding) + MSE (Kd where available)
- Gradient checkpointing on CNN head for T4 memory efficiency
- Output: general aptamer interaction model

### Stage 2 — Validation Fine-Tuning
- Data: insulin, myoglobin, NT-proBNP, troponin I/T, albumin
- Purpose: benchmark against published Kd values, verify generalization
- ESM-2 LoRA unfrozen at lower learning rate

### Stage 3 — Deployment Fine-Tuning (TBD)
- Data: actual Continuity device targets (not yet confirmed)
- Update `DEPLOYMENT_TARGETS` in config.py and run `finetune.py`
- No other changes required

### Stage 4 — Active Learning (ongoing)
- Lab validation results → new labeled data → retrain Stage 2/3

---

## Data

### Sources

| Source | Type | Size |
|---|---|---|
| UTexas Aptamer Database ([Zenodo](https://doi.org/10.5281/zenodo.8264921)) | Primary curated DB | ~896 ssDNA rows |
| PubMed SELEX 2000–2025 | Literature extraction | ~500–1000 pairs |
| Li et al. 2014 ([PLOS ONE](https://doi.org/10.1371/journal.pone.0086729)) | Standard benchmark | 2,320 entries, 164 proteins |
| Therapeutic literature | Clinical-stage aptamers | ~50–100 pairs |

### Dataset Stats (current)

| Split | Rows |
|---|---|
| master_dataset.csv (training-ready) | 2,364 rows, 252 proteins |
| tier1_train.csv (post-augmentation) | 6,914 rows |
| val.csv | 297 rows |
| test.csv | 282 rows |
| ViennaRNA structure cache | 6,330 sequences |

### Augmentation

- **Reverse complement** — doubles positive data (AptaTrans validated)
- **Systematic truncations** — 2–3 nt from each end
- **Cross-target negatives** — binder for protein A is a hard negative for protein B
- **Scrambled sequences** — composition preserved, order destroyed → non-binder label

---

## Training on Google Colab (T4)

A ready-to-run notebook is provided at `notebooks/train_colab.ipynb`.

**6-cell flow:**
1. GPU check (asserts T4 with ≥12 GB VRAM)
2. Clone repo and install dependencies
3. Download data from Google Drive (master_dataset.csv, vienna_cache.pkl, protein_embeddings/)
4. Resume detection — finds latest `epoch_*.pt` checkpoint automatically
5. Train with T4-tuned settings (batch_size=16, max_prot_len=128, PYTORCH_ALLOC_CONF=expandable_segments:True)
6. Evaluate best checkpoint on val and test splits

Training supports `--resume` to recover after Colab disconnects.

---

## Project Structure

```
continuitybioML/
├── CLAUDE.md                    # Full context for Claude Code
├── README.md                    # This file
├── config.py                    # All hyperparameters
├── data/
│   ├── raw/
│   │   └── protein_name_overrides.csv   # Manual UniProt accession overrides (tracked)
│   ├── processed/               # master_dataset.csv, vienna_cache.pkl, protein_embeddings/
│   └── augmented/               # tier1_train.csv, val.csv, test.csv
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
│   ├── condaptnet.py            # Full model assembly
│   └── checkpoints/
│       ├── pretrain/
│       ├── validation/
│       └── deployment/          # Empty until Tier 3 targets confirmed
├── notebooks/
│   └── train_colab.ipynb        # T4 GPU training notebook
├── scripts/
│   ├── data/
│   │   ├── collect_aptamers.py  # PubMed SELEX collection
│   │   ├── build_dataset.py     # Unify sources → master_dataset.csv
│   │   ├── enrich_proteins.py   # UniProt sequence lookup + non-protein filtering
│   │   ├── augment.py           # Rev-comp, truncations, negatives, scrambles
│   │   ├── vienna_features.py   # ViennaRNA structure features (incremental cache)
│   │   └── validate_sequences.py
│   ├── model/
│   │   └── tokenizer.py         # DNA 3-mer tokenizer
│   ├── training/
│   │   ├── train.py             # Stage 1 (--resume, --max-batches, CUDA/MPS/CPU)
│   │   ├── finetune.py          # Stage 2 and 3
│   │   └── losses.py            # Combined BCE + MSE loss
│   └── evaluation/
│       ├── evaluate.py
│       └── metrics.py
└── outputs/
    ├── candidates/
    └── motifs/
```

---

## Environment

- Python 3.11 (virtual environment: `condaptnet_env`)
- PyTorch 2.11.0
- Apple MPS — M-series GPU acceleration confirmed working
- CUDA — T4 Google Colab confirmed working
- ESM-2: `esm2_t12_35M_UR50D` (35M params, 480-dim)
- ViennaRNA — secondary structure prediction

### Setup (local)

```bash
git clone https://github.com/shivanshb828/CondAptNet.git
cd CondAptNet
python3.11 -m venv condaptnet_env
source condaptnet_env/bin/activate
pip install torch torchvision torchaudio
pip install fair-esm pandas numpy scikit-learn biopython ViennaRNA requests tqdm
python scripts/verify_env.py
```

### Usage

```bash
source condaptnet_env/bin/activate

# Collect and process data
python scripts/data/collect_aptamers.py
python scripts/data/build_dataset.py
python scripts/data/enrich_proteins.py
python scripts/data/vienna_features.py
python scripts/data/augment.py

# Stage 1: broad pretraining (local MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py

# Resume after interruption
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py --resume

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

- [x] Environment setup (Python 3.11, PyTorch, ESM-2, ViennaRNA, MPS + CUDA)
- [x] config.py — all hyperparams + physiological defaults (37°C, 150mM Na, 2mM Mg)
- [x] collect_aptamers.py — PubMed SELEX collection
- [x] build_dataset.py — UTexas + Li2014 → master_dataset.csv
- [x] enrich_proteins.py — UniProt lookup, fuzzy matching, non-protein filtering, manual overrides
- [x] validate_sequences.py — length, GC%, homopolymer, alphabet QC
- [x] vienna_features.py — ViennaRNA features, incremental pickle cache (6,330 sequences)
- [x] augment.py — rev-comp, truncations, cross-target negatives, scrambles → 6,914 train rows
- [x] tokenizer.py — DNA 3-mer tokenizer (66-token vocab)
- [x] dna_encoder.py — 6-layer Transformer [B, L, 128], MPS verified
- [x] protein_encoder.py — ESM-2 + LoRA (0.55% trainable) [B, L, 480], MPS verified
- [x] condition_encoder.py — FiLM MLP [B, 128], MPS verified
- [x] cross_attention.py — symmetric bidirectional [B, L, 256], MPS verified
- [x] cnn_head.py — 17-block CNN with GroupNorm [B, 256], MPS + CUDA verified
- [x] dual_head.py — binding sigmoid + Kd ReLU, MPS verified
- [x] condaptnet.py — full assembly + gradient checkpointing, end-to-end verified
- [x] losses.py — combined BCE + MSE loss
- [x] train.py — Stage 1 loop (--resume, --max-batches, CUDA/MPS/CPU auto-detect)
- [x] evaluate.py — MCC, AUC-ROC, AUC-PR, sensitivity, Pearson r
- [x] notebooks/train_colab.ipynb — T4 Colab training notebook
- [ ] finetune.py — Stage 2/3 fine-tuning loop
- [ ] Tier 3 deployment targets (pending Continuity confirmation)

---

## License

Private — built for Continuity (continuity.bio). All rights reserved.

---

*Built by Shivansh Bansal — Continuity ML Pipeline*
