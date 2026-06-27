# CLAUDE.md — CondAptNet Project Guide
## For Claude Code: Read this entire file before doing anything

---

## 1. PROJECT OVERVIEW

### What This Is
**CondAptNet** (Conditional Aptamer-Protein Interaction Network) is a novel deep learning model for predicting DNA aptamer binding to proteins. It is being built for **Continuity** (continuity.bio), a biosensing company developing a real-time physiological monitoring platform that uses DNA aptamers as molecular binders to detect biomarkers.

### The Core Problem Being Solved
Traditional aptamer discovery uses SELEX (Systematic Evolution of Ligands by Exponential Enrichment) — a wet lab process that takes weeks to months and costs thousands of dollars per target protein. CondAptNet uses machine learning to computationally predict which DNA sequences will bind to a given protein, dramatically accelerating candidate discovery before any lab work begins.

### CRITICAL: Three-Tier Target Structure
This is the most important thing to understand about the project scope. There are THREE distinct tiers of protein targets. Do not confuse them.

```
TIER 1 — GENERAL MODEL (primary scientific contribution)
  The model is trained on hundreds of diverse protein families.
  It must generalize to ANY protein given only its amino acid sequence.
  This is the core product. Do not optimize for specific proteins here.

TIER 2 — VALIDATION TARGETS (test that the model works — NOT device targets)
  Insulin, myoglobin, NT-proBNP, troponin I/T, albumin (anti-target).
  These are chosen because:
    - Published aptamers with known Kd values exist in literature
    - Proteins are commercially available for cheap in vitro testing
    - Predictions can be experimentally verified in the lab
  These are NOT the actual device targets.
  These are NOT what Continuity's biosensor will monitor in the real product.
  They exist purely to benchmark model performance against known ground truth.

TIER 3 — DEPLOYMENT TARGETS (actual device, TBD)
  The real Continuity biomarker set is not yet confirmed.
  The pipeline has a plug-and-play fine-tuning slot ready for these.
  When targets are confirmed, update DEPLOYMENT_TARGETS in config.py
  and run finetune.py. No architectural changes required.
```

**Never confuse Tier 2 with Tier 3.** Insulin and troponin are validation tools,
not end goals. Every architectural decision must prioritize Tier 1 generalization.
The Tier 2 fine-tuning layer is lightweight and replaceable by Tier 3 when ready.

---

## 2. WHY A NEW MODEL (Not an Existing One)

Every existing aptamer prediction model has critical limitations:

| Existing Model | Key Limitation |
|---|---|
| Apta-MCTS (2021) | Random Forest, shallow features, no generalization |
| AptaTrans (2023) | Converts DNA→RNA (lossy), trained on only 164 proteins from 2012 |
| AptaBERT (2023) | Proprietary training data, not reproducible |
| AptaBLE (2026) | Best existing model, still RNA-focused, no condition encoding |

**CondAptNet's novel contributions:**
1. First model with a native DNA encoder — no T→U substitution
2. ESM-2 protein encoder (Meta, 250M protein sequences) — no prior aptamer model does this
3. Physiological condition injection (pH, salt, temperature, buffer) via FiLM conditioning
4. Dual output head — binary binding classification AND continuous Kd regression
5. General model trained on broadest protein diversity to date
6. Three-tier training — broad pretraining → validation benchmark → deployment (plug-and-play)

---

## 3. SCIENTIFIC BACKGROUND

### What is an Aptamer?
A DNA aptamer is a short single-stranded DNA sequence (typically 20–100 nucleotides)
that folds into a 3D structure and binds to a specific target protein with high affinity
and specificity. Like a synthetic antibody made of DNA. Selected through SELEX from
libraries of ~10^15 random sequences.

### Key Terminology
- SELEX: wet lab process for finding aptamers
- Kd: dissociation constant — measures binding affinity; lower = stronger; good aptamers ~nM
- k-mer: subsequence of length k; DNA 3-mers are all 64 possible 3-nucleotide combinations
- ViennaRNA: software predicting DNA/RNA secondary structure (stem-loops, hairpins)
- MFE: Minimum Free Energy — most stable predicted secondary structure
- ESM-2: Meta's protein language model pretrained on 250M protein sequences
- LoRA: Low-Rank Adaptation — efficient fine-tuning, updates <1% of parameters
- FiLM: Feature-wise Linear Modulation — injects scalar conditions into neural networks
- ZDOCK: molecular docking tool for validating aptamer-protein binding

### Validation Targets (Tier 2 — benchmark only)
| Protein | Why Chosen for Validation |
|---|---|
| Insulin | Published aptamers, well-characterized Kd, cheap in vitro testing |
| Myoglobin | Published aptamers, well-studied binding |
| NT-proBNP | Published aptamers, physiologically relevant |
| Troponin I | Rich cardiac biomarker literature |
| Troponin T | Can compare I vs T prediction quality |
| Albumin | Anti-target — filter non-specific binders |

### Deployment Targets (Tier 3 — TBD)
Not yet confirmed by Continuity. Update DEPLOYMENT_TARGETS in config.py when known.

---

## 4. ARCHITECTURE

### Plain English Summary
CondAptNet takes three inputs — a DNA aptamer sequence, a protein sequence, and
experimental conditions — and outputs: (1) probability that the aptamer binds the
protein, and (2) predicted binding affinity (Kd) when possible.

Five stages:
1. DNA Encoder — reads and represents the aptamer sequence
2. Protein Encoder — reads and represents the protein (using ESM-2)
3. Condition Encoder — encodes pH, salt, temperature, buffer
4. Cross-Attention — makes aptamer and protein reason about each other jointly
5. CNN + Output Head — finds interaction patterns, produces final predictions

### Full Architecture Diagram
```
INPUT
├── DNA aptamer sequence (A, T, G, C — 20–100 nt, native DNA)
├── Protein sequence (amino acids)
└── Condition vector [pH, salt_mM, temp_C, buffer_type]

         │                        │                    │
         ▼                        ▼                    ▼
┌─────────────────┐    ┌──────────────────┐   ┌──────────────────┐
│   DNA ENCODER   │    │ PROTEIN ENCODER  │   │ CONDITION ENCODER│
│                 │    │                  │   │                  │
│ 3-mer tokenize  │    │ ESM-2 pretrained │   │ Linear(4,64)     │
│ (native DNA,    │    │ esm2_t12_35M     │   │ ReLU             │
│  no T→U)        │    │ 35M params       │   │ Linear(64,128)   │
│                 │    │                  │   │                  │
│ + ViennaRNA     │    │ Fine-tuned via   │   │ Output: 128-dim  │
│   structure     │    │ LoRA (rank=8)    │   └────────┬─────────┘
│   features      │    │                  │            │
│                 │    │ Output: 480-dim  │            │
│ Transformer     │    │ per amino acid   │            │
│ 6 layers        │    │                  │            │
│ 8 heads         │    └────────┬─────────┘            │
│ dim=128         │             │                      │
│                 │             │                      │
│ Output: 128-dim │             │                      │
│ per nucleotide  │             │                      │
└────────┬────────┘             │                      │
         │                      │                      │
         └──────────┬───────────┘                      │
                    ▼                                  │
      ┌─────────────────────────────┐                  │
      │  SYMMETRIC BIDIRECTIONAL    │◄─────────────────┘
      │     CROSS-ATTENTION         │  FiLM conditioning
      │                             │  condition modulates
      │  Aptamer attends to protein │  attention weights
      │  AND protein attends to     │
      │  aptamer simultaneously     │
      │                             │
      │  8 heads, dropout=0.1       │
      └──────────────┬──────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   INTERACTION MATRIX  │
         │   + 17-block CNN      │
         │                       │
         │  channels: 64→128→256 │
         │  kernel: 3×3          │
         │  BatchNorm + GELU     │
         │  residual connections │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    DUAL OUTPUT HEAD   │
         │                       │
         │  Binding: sigmoid     │  → P(bind) ∈ [0,1]
         │  Affinity: ReLU       │  → Kd (nM, log scale)
         └───────────────────────┘
```

### Component Rationale

**Native DNA Encoder:** Every prior model converts DNA→RNA because datasets were
RNA-dominated. DNA has different structural properties (thymine ≠ uracil). Native
encoding eliminates a systematic error baked into all existing models.

**ESM-2 Protein Encoder:** 250M protein sequences pretrained. Deeply understands ANY
protein including ones with zero aptamer training data (NT-proBNP, troponin, future
deployment targets). This is what makes general prediction possible. No prior aptamer
model uses a large protein language model.

**LoRA:** Updates only small adapter matrices in ESM-2 attention layers (<1% of
parameters). Preserves pretrained knowledge while allowing task-specific adaptation.
Runs on Apple M-series chips without full fine-tuning memory requirements.

**Symmetric Bidirectional Cross-Attention:** Validated by AptaBLE (2026) as superior
to unidirectional approaches. Captures mutual binding geometry — both molecules inform
each other simultaneously rather than one being subordinate.

**FiLM Condition Injection:** pH, salt, temperature, buffer modulate cross-attention
feature maps via learned scale and shift parameters. First aptamer model to do this.
Critical for Continuity's physiological sensing context.

**17-block CNN:** Extracts local and hierarchical features from 2D aptamer-protein
interaction map. Architecture from AptaTrans (2023), validated effective.

**Dual Output Head:** Binding classification (sigmoid) + Kd regression (ReLU, log
scale). Enables affinity-ranked candidate generation, not just binary classification.

---

## 5. TRAINING STRATEGY

### Three-Stage Pipeline

**Stage 1 — Broad Pretraining (primary contribution)**
- ALL collected aptamer-protein pairs, hundreds of diverse protein families
- ESM-2 frozen, all other layers trained
- Loss: BCE (binding) + MSE (Kd where available)
- Split: by protein family, never randomly
- Checkpoint: models/checkpoints/pretrain/

**Stage 2 — Validation Fine-Tuning (benchmark only)**
- Validation targets: insulin, myoglobin, NT-proBNP, troponin I/T, albumin
- ESM-2 LoRA unfrozen, lower learning rates
- Purpose: verify generalization, benchmark against published Kd values
- Checkpoint: models/checkpoints/validation/

**Stage 3 — Deployment Fine-Tuning (TBD, plug-and-play)**
- Same script as Stage 2, different protein set
- Update DEPLOYMENT_TARGETS in config.py when confirmed
- Checkpoint: models/checkpoints/deployment/

**Stage 4 — Active Learning (ongoing)**
- Lab results → new labeled data → retrain Stage 2/3

### Critical Rules
- ALWAYS split by protein family, never randomly
- Primary metric: MCC (not accuracy — data is imbalanced)
- Also report: AUC-ROC, AUC-PR, sensitivity, Pearson r (Kd)
- Class weights: weight positive examples 3x in loss function

---

## 6. DATA PIPELINE

### Sources (priority order)
1. UTexas Aptamer Database (Zenodo: doi.org/10.5281/zenodo.8264921) — 1,495 entries, ssDNA filter → ~896 usable rows; bulk download at data/raw/utexas_aptamer_db/
2. PubMed SELEX 2000–2025 — ~500–1000 new pairs (collect_aptamers.py)
3. Li et al. 2014 File S1 (doi:10.1371/journal.pone.0086729) — 2,320 label+target entries, 164 proteins; sequences need PubMed lookup (needs_sequence_enrichment=True)
4. GitHub AptamerBase dump (github.com/micheldumontier/aptamerbase) — pre-2016 entries, supplementary; original aptamer.uni.lu shut down in 2016
5. Therapeutic aptamer literature — ~50–100 clinical pairs
6. Validation-specific searches — for Stage 2 only

NOTE: Physiological defaults for missing condition fields (Continuity device context):
  temp_C=37.0, salt_mM=150.0 (Na+), mg_mM=2.0 (Mg2+), pH=7.4, buffer_type=0 (PBS)

### Required Fields
```
sequence                    : string  — DNA only (A, T, G, C); None if needs enrichment
target_protein              : string  — protein name
uniprot_id                  : string  — for ESM-2 lookup; None until enrichment pass
protein_sequence            : string  — full amino acid sequence; None until enrichment pass
Kd_nM                       : float   — null if not reported
pH                          : float   — physiological default 7.4 if missing
salt_mM                     : float   — physiological default 150.0 mM Na+ if missing
temp_C                      : float   — physiological default 37.0°C if missing
buffer_type                 : int     — 0=PBS, 1=HEPES, 2=Tris, 3=other
mg_mM                       : float   — physiological default 2.0 mM Mg2+ if missing
label                       : int     — 1=binder, 0=non-binder
source_pmid                 : string  — PubMed ID
training_tier               : int     — 1=general, 2=validation, 3=deployment
augmented                   : bool
aug_method                  : string  — null if not augmented
needs_sequence_enrichment   : bool    — True for Li2014 rows pending sequence lookup
source                      : string  — 'utexas', 'li2014', 'pubmed', etc.
```

### Augmentation
- Reverse complement: hard negatives (label=0) — RC folds into a different 3D structure, is NOT a confirmed binder
- Truncations: 2–5 nt systematic removal from ends
- Cross-target negatives: binder for protein A = non-binder for protein B
- Scrambled sequences: shuffle nucleotides → label as non-binder

### Validation Rules
- Length: 20–120 nt
- Characters: A, T, G, C only
- GC content: 20–80%
- No homopolymer run > 8 nt
- No exact duplicates for same target

---

## 7. FILE STRUCTURE

```
continuitybioML/
├── CLAUDE.md
├── README.md
├── config.py
├── condaptnet_env/              ← never edit
├── data/
│   ├── raw/
│   │   ├── pubmed_results.csv
│   │   ├── aptamerbase/
│   │   └── li2014_benchmark/
│   ├── processed/
│   │   ├── master_dataset.csv   ← training_tier column distinguishes tiers
│   │   ├── negatives.csv
│   │   ├── vienna_cache.pkl
│   │   └── protein_embeddings/
│   └── augmented/
│       ├── tier1_train.csv
│       ├── tier2_train.csv
│       ├── val.csv
│       └── test.csv             ← held out by protein family
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
│       └── deployment/          ← empty until Tier 3 targets confirmed
├── scripts/
│   ├── data/
│   │   ├── collect_aptamers.py
│   │   ├── build_dataset.py
│   │   ├── augment.py
│   │   ├── vienna_features.py
│   │   └── validate_sequences.py
│   ├── model/
│   │   └── tokenizer.py
│   ├── training/
│   │   ├── train.py             ← Stage 1
│   │   ├── finetune.py          ← Stage 2 and 3 (same script)
│   │   └── losses.py
│   └── evaluation/
│       ├── evaluate.py
│       └── metrics.py
└── outputs/
    ├── candidates/
    └── motifs/
```

---

## 8. CONFIGURATION (config.py)

All hyperparameters live here. Never hardcode in model files.

```python
DEVICE = "mps"  # Apple Silicon

# DNA Encoder
DNA_KMER_SIZE = 3
DNA_EMBED_DIM = 128
DNA_NUM_LAYERS = 6
DNA_NUM_HEADS = 8
DNA_DROPOUT = 0.1
DNA_MAX_LENGTH = 120

# Protein Encoder
ESM_MODEL_NAME = "esm2_t12_35M_UR50D"
ESM_EMBED_DIM = 480
LORA_RANK = 8
LORA_ALPHA = 16

# Cross-attention
CROSS_ATTN_HEADS = 8
FUSION_DIM = 256

# Condition
CONDITION_DIM = 4
CONDITION_HIDDEN = 64

# CNN
CNN_CHANNELS = [64, 128, 256]
CNN_KERNEL_SIZE = 3

# Training
BATCH_SIZE = 32
LEARNING_RATE_BASE = 1e-4
LEARNING_RATE_LORA = 1e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Validation targets (Tier 2 — benchmark only, NOT deployment targets)
VALIDATION_TARGETS = [
    "insulin", "myoglobin", "NT-proBNP",
    "troponin_I", "troponin_T", "albumin"
]

# Deployment targets (Tier 3 — update when Continuity confirms)
DEPLOYMENT_TARGETS = []
```

---

## 9. BUILD ORDER

```
Phase 1: DATA PIPELINE
  1.1  collect_aptamers.py      PubMed automated collection
  1.2  validate_sequences.py    sequence QC
  1.3  vienna_features.py       precompute + cache structure features
  1.4  build_dataset.py         unify sources, assign training_tier
  1.5  augment.py               reverse complement, truncations, negatives

Phase 2: TOKENIZER
  2.1  tokenizer.py             DNA 3-mer tokenizer, test on examples

Phase 3: ENCODERS (build and test each independently)
  3.1  dna_encoder.py           6-layer transformer, verify shapes
  3.2  protein_encoder.py       ESM-2 + LoRA, test on insulin sequence
  3.3  condition_encoder.py     condition MLP

Phase 4: FUSION
  4.1  cross_attention.py       symmetric bidirectional, verify shapes
  4.2  cnn_head.py              17-block CNN

Phase 5: OUTPUT + ASSEMBLY
  5.1  dual_head.py             binding + Kd
  5.2  condaptnet.py            full model, end-to-end forward pass test

Phase 6: TRAINING
  6.1  losses.py                combined BCE + MSE loss
  6.2  train.py                 Stage 1 with MPS support
  6.3  evaluate.py              MCC, AUC, PR

Phase 7: FINE-TUNING
  7.1  finetune.py              Stage 2 (validation) + Stage 3 (deployment)
```

---

## 10. ENVIRONMENT

```bash
# Always activate first
cd continuitybioML
source condaptnet_env/bin/activate

# Verified working
# Python 3.11.3, PyTorch 2.11.0, MPS=True, ESM-2=OK, ViennaRNA=OK

# MPS training
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py
```

MPS rules:
- Always use float32 (float64 unreliable on MPS)
- Use `torch.device("mps")` explicitly
- Add PYTORCH_ENABLE_MPS_FALLBACK=1 if NotImplementedError occurs

---

## 11. CODING STANDARDS

- Every model file: docstring with input/output shapes
- Every model file: `if __name__ == "__main__":` test with dummy data and shape assertions
- All hyperparameters: import from config.py, never hardcode
- ESM-2 embeddings: cache to disk as numpy, never to GPU memory
- Save checkpoints every epoch, not just best

---

## 12. WHAT NOT TO DO

- Never convert T→U — native DNA only
- Never randomly split train/val/test — always by protein family
- Never confuse Tier 2 validation targets with Tier 3 deployment targets
- Never hardcode hyperparameters in model files
- Never train without early stopping
- Never report accuracy as primary metric — use MCC
- Never cache ESM-2 embeddings to GPU
- Never use float64 on MPS
- Never optimize architecture for validation targets — keep Stage 1 fully general

---

## 13. REFERENCE PAPERS

| Paper | Contribution to CondAptNet |
|---|---|
| Li et al. PLOS ONE 2014 | Standard benchmark — 725 pairs, 164 proteins |
| Apta-MCTS PLOS ONE 2021 | MCTS generation; RF baseline |
| AptaTrans BMC Bioinf 2023 | Interaction matrix + CNN; pretraining; augmentation |
| AptaBERT bioRxiv 2023 | Transformer pretraining on aptamers |
| AptaBLE bioRxiv 2026 | Symmetric bidirectional cross-attention; current SOTA |

---

## 14. CURRENT STATUS

### Completed
- [x] Python 3.11 virtual environment (condaptnet_env)
- [x] All dependencies installed and verified (PyTorch 2.11, ESM-2, ViennaRNA, MPS)
- [x] Project folder structure created
- [x] config.py — all hyperparams, physiological defaults (37°C, 150mM Na, 2mM Mg), tier lists
- [x] collect_aptamers.py — PubMed Entrez collection
- [x] validate_sequences.py — sequence QC (length, GC%, homopolymer, alphabet)
- [x] vienna_features.py — ViennaRNA feature extraction + pickle cache
- [x] tokenizer.py — DNA 3-mer tokenizer (66-token vocab, all tests pass)
- [x] dna_encoder.py — 6-layer Pre-LN Transformer, MPS-verified, output [B, L, 128]
- [x] protein_encoder.py — ESM-2 (35M) + LoRA rank-8, 0.55% trainable, output [B, L, 480]
- [x] condition_encoder.py — FiLM MLP, 5-scalar input (pH, salt, temp, buffer, Mg), output [B, 128]
- [x] cross_attention.py — symmetric bidirectional, FiLM-modulated, output [B, L, 256]
- [x] cnn_head.py — 17-block residual CNN, global avg pool, output [B, 256]
- [x] dual_head.py — binding sigmoid + Kd ReLU, skippable Kd head
- [x] build_dataset.py — parses UTexas (896 ssDNA rows) + Li2014 (2320 rows); master_dataset.csv produced
- [x] Data sources updated: UTexas Aptamer DB replaces defunct AptamerBase
- [x] condaptnet.py — full model assembly, set_stage1/2 helpers, end-to-end forward pass verified
- [x] losses.py — combined BCE + MSE (CondAptNetLoss)
- [x] train.py — Stage 1 training loop (MPS/CUDA/CPU, protein-family splits, early stopping)
- [x] evaluate.py — MCC, AUC-ROC, AUC-PR, Pearson r(Kd)
- [x] Scraper pipeline — 4 parsers + 10 source adapters + merge.py + main.py (178 tests passing)
- [x] finetune.py — Stage 2 (validation) + Stage 3 (deployment) fine-tuning; 24 tests passing
- [x] dual_head.py — added binding_label output (0=low/1=medium/2=high, Kd-first with prob fallback)
- [x] condaptnet.py — CondAptNetOutput updated to include binding_label
- [x] Data pipeline complete:
  - master_dataset.csv: 4643 rows (UTexas 789 + Li2014 2320 + patent 792 + aptamerbase 712 + pubmed 30)
  - UniProt enrichment: 3 passes → 3707 training-ready rows
  - vienna_cache.pkl: 6489 sequences cached
  - augmented/: tier1_train=14014 / val=502 / test=392 rows (273/58/60 protein families)
- [x] 7-phase data cleaning pipeline (clean_dataset.py):
  - Phase 1: removed 77 exact dupes + 5 RC pairs
  - Phase 2: priority target audit (troponin/myoglobin/insulin thin pre-curated-merge)
  - Phase 3: fixed 596-row Li2014 structural leakage → zero sequence overlap across splits
  - Phase 4: added nucleic_acid_type; removed 67 non-DNA + 10 out-of-range-length rows
  - Phase 5: target_type classification (protein=4370, organism/cell/other/sm=114)
  - Phase 6: restructured to 20-column schema; merged 13 curated troponin/NT-proBNP/myoglobin/insulin rows (training_tier=2); protein_sequence backfilled for 11/13
  - Phase 7: all validations pass (0 leakage; 2 warnings: val=4.4%/test=3.7% from leakage repair)
  - master_dataset_cleaned.csv: 4497 rows, 23 columns (20-col spec + protein_sequence, label, training_tier)
  - Also: non_dna_entries.csv (77), flagged_for_review.csv (609), outputs/cleaning_report.md
- [x] train.py + finetune.py — _film_heads checkpoint bug fixed; strict=False resume guard added
- [x] condition_encoder.py — pre-registers FUSION_DIM FiLM head so checkpoint keys are stable
- [x] Pre-training cleanup pass (all pipeline plumbing verified end-to-end):
  - enrich_proteins.py — v3: auto-detects legacy vs cleaned schema; `--input/--output` flags. Re-ran on master_dataset_cleaned.csv: all 17 Tier-2 rows already enriched; 228 residual "protein" targets are garbled scraper noise (organisms/cells/spores/truncated names) that fail UniProt name search — left unenriched, dropped downstream. 3700 protein rows have sequences.
  - train.py — Stage 1 now reads the augmented protein-family splits in data/augmented/ (tier1_train.csv/val.csv/test.csv) instead of the raw cleaned CSV; the augmentation pipeline is now actually used and the 858 unassigned rows are folded into train. `--augmented-dir` arg.
  - evaluate.py — rewritten: was crashing (stale master_dataset.csv schema, re-randomized split, wrong AptamerDataset signature, 6-tuple vs 7-tuple, no protein_emb). Now mirrors train.py exactly (augmented splits, pre-computed ESM-2 embeddings, strict=False ckpt load).
  - augment.py — added a leakage guard dropping any folded-in/synthesized train row whose aptamer_sequence appears in val/test. Regenerated splits: tier1_train=15904 / val=159 / test=136, verified zero sequence overlap.
  - condition_encoder.py — docstring corrected (5-dim input incl. mg_mM, was stale 4-dim).

### Next Up (in order)
1. Run Stage 1 training: `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py` (reads data/augmented/; run `python scripts/data/augment.py` first if splits are stale)
2. Evaluate on test set: `python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt`
3. Run Stage 2 fine-tuning: `python scripts/training/finetune.py --stage validation`

### Open Decisions
- esm2_t12_35M confirmed (480-dim, LoRA rank-8 working on MPS)
- Kd head: active now, skippable at inference for rows without Kd labels
- Tier 3 deployment targets: pending Continuity confirmation (DEPLOYMENT_TARGETS=[] in config.py)

---

## 15. QUICK COMMANDS

```bash
source condaptnet_env/bin/activate
python scripts/verify_env.py
python scripts/data/collect_aptamers.py
python models/encoders/dna_encoder.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt
```

---

## 16. CONTACT

- Developer: Shivansh Bansal
- Company: Continuity (continuity.bio)
- Academic: Incoming UCLA undergraduate, computational biology
- Background: ISEF 2025 2nd place computational biology

*Update Current Status as components are completed.*