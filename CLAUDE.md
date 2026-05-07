# CLAUDE.md — CondAptNet Project Guide
## For Claude Code: Read this entire file before doing anything

---

## 1. PROJECT OVERVIEW

### What This Is
**CondAptNet** (Conditional Aptamer-Protein Interaction Network) is a novel deep learning model for predicting DNA aptamer binding to proteins. It is being built for **Continuity** (continuity.bio), a biosensing company developing a real-time physiological monitoring platform that uses DNA aptamers as molecular binders to detect biomarkers.

### The Core Problem Being Solved
Traditional aptamer discovery uses SELEX (Systematic Evolution of Ligands by Exponential Enrichment) — a wet lab process that takes weeks to months and costs thousands of dollars per target protein. CondAptNet uses machine learning to computationally predict which DNA sequences will bind to a given protein, dramatically accelerating candidate discovery before any lab work begins.

### Why a New Model (Not an Existing One)
Every existing aptamer prediction model has critical limitations for this use case:

| Existing Model | Key Limitation |
|---|---|
| Apta-MCTS (2021) | Random Forest, shallow features, no generalization |
| AptaTrans (2023) | Converts DNA→RNA (lossy), trained on only 164 proteins from 2012 |
| AptaBERT (2023) | Proprietary training data, not reproducible |
| AptaBLE (2026) | Best existing model, but still RNA-focused, no condition encoding |

**CondAptNet's novel contributions:**
1. First model with a **native DNA encoder** — no T→U substitution
2. Uses **ESM-2** (Meta, 250M protein sequences) as protein encoder — no prior aptamer model does this
3. **Physiological condition injection** (pH, salt, temperature, buffer) via FiLM conditioning
4. **Dual output head** — binary binding classification AND continuous Kd regression
5. **Trained on broadest protein diversity to date** — general model, not 5-protein model
6. **Two-stage training** — broad pretraining then target-specific fine-tuning

---

## 2. SCIENTIFIC BACKGROUND

### What is an Aptamer?
A DNA aptamer is a short single-stranded DNA sequence (typically 20–100 nucleotides) that folds into a 3D structure and binds to a specific target protein with high affinity and specificity. Think of it as a synthetic antibody made of DNA. Aptamers are selected through SELEX from libraries of ~10^16 random sequences.

### Key Terminology
- **SELEX**: Systematic Evolution of Ligands by Exponential Enrichment — the wet lab process for finding aptamers
- **Kd**: Dissociation constant — measures binding affinity. Lower Kd = stronger binding. Good aptamers have Kd in nM range
- **API**: Aptamer-Protein Interaction — the binding event we're predicting
- **k-mer**: Subsequence of length k. For DNA, 3-mers are AAA, AAT, AAG, AAC... etc (64 total)
- **ViennaRNA**: Software that predicts DNA/RNA secondary structure (stem-loops, hairpins, etc.)
- **MFE**: Minimum Free Energy — the most stable predicted secondary structure
- **ESM-2**: Meta's protein language model pretrained on 250M protein sequences
- **LoRA**: Low-Rank Adaptation — efficient fine-tuning that updates only small adapter matrices, not full model weights
- **FiLM**: Feature-wise Linear Modulation — technique for injecting conditioning vectors into neural networks
- **ZDOCK**: Molecular docking tool used to validate aptamer-protein binding computationally

### The 5 Primary Target Proteins (Continuity-specific)
These are the proteins Continuity's biosensor will monitor. The model must be especially accurate for these, but is trained to generalize broadly:

| Protein | Role | Why It Matters |
|---|---|---|
| Insulin | Metabolic hormone | Blood glucose regulation |
| Myoglobin | Muscle protein | Muscle stress biomarker |
| NT-proBNP | Cardiac peptide | Cardiovascular stress marker |
| Troponin I | Cardiac protein | Heart attack marker |
| Troponin T | Cardiac protein | Heart attack marker |
| Albumin | Serum protein | **Anti-target** — high abundance, want aptamers that DON'T bind this |

### Critical Data Reality
- The standard benchmark used by ALL prior models (Li et al. 2014) contains only 725 aptamer-protein pairs from 164 proteins accessed in 2012
- NT-proBNP and troponin almost certainly have ZERO representation in that dataset
- This is why we need broad data collection from 2012–2025 literature
- Our model's generalization comes from ESM-2's protein knowledge, not aptamer training data alone

---

## 3. ARCHITECTURE — CondAptNet

### Full Architecture Diagram
```
INPUT
├── DNA aptamer sequence (string of A, T, G, C — 20–100 nt)
├── Protein sequence (string of amino acids)
└── Condition vector [pH, salt_mM, temp_C, buffer_type]

         │                        │                    │
         ▼                        ▼                    ▼
┌─────────────────┐    ┌──────────────────┐   ┌──────────────────┐
│   DNA ENCODER   │    │ PROTEIN ENCODER  │   │ CONDITION ENCODER│
│                 │    │                  │   │                  │
│ 3-mer tokenize  │    │ ESM-2 pretrained │   │ 4 scalars →      │
│ (native DNA,    │    │ (esm2_t12_35M)   │   │ Linear(4,64) →   │
│  no T→U)        │    │                  │   │ ReLU →           │
│                 │    │ Fine-tuned with   │   │ Linear(64,128)   │
│ + ViennaRNA     │    │ LoRA (rank=8)    │   │                  │
│   structure     │    │                  │   │ Output: 128-dim  │
│   features      │    │ Output: 480-dim  │   │ condition vector │
│                 │    │ embeddings per   │   └────────┬─────────┘
│ Transformer     │    │ amino acid       │            │
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
      │     CROSS-ATTENTION         │  FiLM conditioning:
      │                             │  condition vector modulates
      │  Aptamer tokens attend to   │  attention weights
      │  protein tokens AND         │
      │  protein tokens attend to   │
      │  aptamer tokens             │
      │  simultaneously             │
      │                             │
      │  8 attention heads          │
      │  dropout=0.1                │
      └──────────────┬──────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   INTERACTION MATRIX  │
         │                       │
         │  Outer product of     │
         │  aptamer × protein    │
         │  embeddings           │
         │                       │
         │  17-block CNN         │
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
         │  Head 1: Binding      │
         │  Linear → Sigmoid     │
         │  Output: P(binding)   │
         │  [0,1] probability    │
         │                       │
         │  Head 2: Affinity     │
         │  Linear → ReLU        │
         │  Output: predicted Kd │
         │  (nM, log scale)      │
         │  Only when Kd labeled │
         └───────────────────────┘
```

### Why Each Component

**Native DNA Encoder:** Every prior model converts DNA to RNA (T→U substitution) because they trained on RNA-dominated datasets. DNA and RNA have different structural properties — thymine (T) forms different interactions than uracil (U). Training on native DNA eliminates this systematic error.

**ESM-2 Protein Encoder:** ESM-2 was pretrained on 250 million protein sequences by Meta AI. This means it already has rich representations of essentially any protein — including NT-proBNP and troponin that have zero aptamer training data. No prior aptamer model uses ESM-2 or any large protein language model.

**LoRA Fine-tuning:** Full fine-tuning of ESM-2 is computationally prohibitive. LoRA adds small trainable adapter matrices (rank=8) to ESM-2's attention layers, updating <1% of parameters while preserving pretrained knowledge. This runs on Apple M-series chips.

**Symmetric Bidirectional Cross-Attention:** Validated by AptaBLE (2026) as superior to AptaTrans's unidirectional approach. Both molecules attend to each other simultaneously, capturing mutual binding geometry rather than treating one as a fixed context for the other.

**FiLM Condition Injection:** The condition vector (pH, salt, temp, buffer) is projected and used to compute scale (γ) and shift (β) parameters that modulate the cross-attention feature maps. This is the first aptamer model to encode experimental conditions — critical for Continuity's physiological sensing context.

**17-block CNN:** Directly from AptaTrans, validated to extract hierarchical features from the 2D interaction map. Channels 64→128→256 with residual connections and GELU activation.

**Dual Output Head:** Prior models only output binary binding. The Kd regression head enables affinity-ranked candidate generation — essential for prioritizing which sequences to synthesize and test experimentally.

---

## 4. TRAINING STRATEGY

### Two-Stage Training

**Stage 1 — Broad Pretraining (General Model)**
- Train on ALL collected aptamer-protein pairs spanning hundreds of diverse proteins
- Goal: learn general aptamer-protein interaction priors
- Freeze ESM-2, train everything else
- Loss: Binary cross-entropy on binding labels + MSE on Kd where available
- Evaluate: held-out protein families (never random split)

**Stage 2 — Target-Specific Fine-Tuning (Continuity)**
- Unfreeze ESM-2 LoRA layers
- Fine-tune on insulin, myoglobin, NT-proBNP, troponin I/T, albumin data
- Lower learning rate (1e-5 for LoRA, 1e-4 for other layers)
- Smaller batch size if limited data

**Stage 3 — Active Learning Loop (Ongoing)**
- Experimental results from Continuity's lab → new labeled data → retrain Stage 2
- Every confirmed binder/non-binder is gold — add immediately

### Data Split Strategy
**CRITICAL: Never split randomly.** Always split by protein family.

Wrong way: randomly assign 80% of all pairs to train, 20% to test
Right way: assign entire protein families to train/val/test

Random splitting leaks information — if you have 10 insulin aptamers and 8 go to train and 2 to test, the model has effectively seen insulin. Family-held-out splits test true generalization.

### Class Imbalance Handling
Positive pairs (real binders) will always be fewer than negatives. Use:
- Class weights in loss function (weight positives by 3x)
- Oversampling of positives during training
- Report MCC and AUC-PR, NOT accuracy (accuracy is misleading with imbalanced data)

### Evaluation Metrics (in priority order)
1. **MCC** (Matthews Correlation Coefficient) — best single metric for imbalanced binary classification
2. **AUC-ROC** — discriminative ability across thresholds
3. **AUC-PR** (Precision-Recall) — more informative than ROC when positives are rare
4. **Sensitivity (recall)** — false negatives = missed candidates = expensive in drug discovery
5. **Specificity** — false positives = wasted synthesis/testing cost
6. **Pearson r on Kd** — for regression head evaluation

---

## 5. DATA PIPELINE

### Data Sources (in priority order)
1. **AptamerBase** (aptamer.uni.lu) — manual download, most curated
2. **PubMed SELEX literature 2012–2025** — automated fetch + manual extraction
3. **Li et al. 2014 supplementary** (File S1 from doi:10.1371/journal.pone.0086729) — the standard benchmark, 725 pairs
4. **Therapeutic aptamer literature** — clinical-stage aptamers, high quality
5. **Target-specific searches** — insulin, troponin, myoglobin, NT-proBNP, albumin

### Required Fields Per Record
```
sequence          : string  — DNA aptamer sequence (A, T, G, C only)
target_protein    : string  — protein name
uniprot_id        : string  — UniProt accession (for ESM-2 lookup)
protein_sequence  : string  — full amino acid sequence
Kd_nM             : float   — dissociation constant in nM (null if not reported)
pH                : float   — buffer pH (null if not reported)
salt_mM           : float   — salt concentration in mM (null if not reported)
temp_C            : float   — temperature in Celsius (null if not reported)
buffer_type       : int     — 0=PBS, 1=HEPES, 2=Tris, 3=other
label             : int     — 1=binder, 0=non-binder
source_pmid       : string  — PubMed ID
augmented         : bool    — True if generated by augmentation
aug_method        : string  — 'reverse_complement', 'truncation', 'mutation', null
```

### Data Augmentation (apply after collection, before training)
1. **Reverse complement** — flip every positive aptamer sequence and label as binder. Doubles dataset. Validated by AptaTrans.
2. **Truncations** — systematically remove 2-5 nt from each end of known aptamers. Creates length variants.
3. **Cross-target negatives** — known binders for protein A labeled as non-binders for protein B (hard negatives, better than random)
4. **Scrambled sequences** — shuffle nucleotides of known binders (destroys sequence order, keeps composition). Label as non-binders.

### Sequence Validation Rules
Before adding any sequence to the dataset:
- Length: 20–120 nucleotides
- Characters: only A, T, G, C (no ambiguous bases)
- GC content: 20–80% (extreme values are problematic)
- No homopolymer runs longer than 8 nt (e.g., AAAAAAAAA is suspicious)
- Not an exact duplicate of existing sequence for same target

### ViennaRNA Feature Extraction
For every aptamer sequence, precompute and cache:
- MFE (minimum free energy) — scalar
- Dot-bracket secondary structure string
- Stem count — number of helical regions
- Loop count — number of loop regions
- Base pair probability matrix (condensed to statistics)

Cache these to `data/processed/vienna_cache.pkl` — ViennaRNA is slow and you call it thousands of times.

---

## 6. FILE STRUCTURE

```
continuitybioML/
│
├── CLAUDE.md                    ← THIS FILE
├── config.py                    ← Central configuration (all hyperparams)
├── README.md
│
├── condaptnet_env/              ← Python 3.11 virtual environment (never edit)
│
├── data/
│   ├── raw/                     ← Downloaded/scraped data, never modified
│   │   ├── pubmed_results.csv   ← PubMed search results
│   │   ├── aptamerbase/         ← AptamerBase downloads
│   │   └── li2014_benchmark/    ← Li et al. 2014 File S1
│   ├── processed/               ← Cleaned, validated, unified format
│   │   ├── master_dataset.csv   ← All positive pairs, cleaned
│   │   ├── negatives.csv        ← All negative pairs
│   │   ├── vienna_cache.pkl     ← Precomputed ViennaRNA features
│   │   └── protein_embeddings/  ← Cached ESM-2 embeddings per protein
│   └── augmented/               ← Post-augmentation training data
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── encoders/
│   │   ├── dna_encoder.py       ← Native DNA transformer encoder
│   │   ├── protein_encoder.py   ← ESM-2 + LoRA wrapper
│   │   └── condition_encoder.py ← FiLM condition MLP
│   ├── attention/
│   │   └── cross_attention.py   ← Symmetric bidirectional cross-attention
│   ├── interaction/
│   │   └── cnn_head.py          ← 17-block CNN interaction head
│   ├── output/
│   │   └── dual_head.py         ← Binding + Kd output heads
│   ├── condaptnet.py            ← Full model assembly
│   └── checkpoints/             ← Saved model weights
│       ├── pretrain/
│       └── finetune/
│
├── scripts/
│   ├── data/
│   │   ├── collect_aptamers.py  ← PubMed automated collection
│   │   ├── build_dataset.py     ← Unify all sources into master_dataset.csv
│   │   ├── augment.py           ← Data augmentation pipeline
│   │   ├── vienna_features.py   ← Precompute ViennaRNA features
│   │   └── validate_sequences.py← Quality control for sequences
│   ├── model/
│   │   └── tokenizer.py         ← DNA 3-mer tokenizer
│   ├── training/
│   │   ├── train.py             ← Main training loop
│   │   ├── finetune.py          ← Stage 2 fine-tuning script
│   │   └── losses.py            ← Combined binding + Kd loss
│   ├── evaluation/
│   │   ├── evaluate.py          ← Full evaluation suite
│   │   └── metrics.py           ← MCC, AUC, PR calculation
│   └── verify_env.py            ← Environment check (already created)
│
├── notebooks/                   ← Jupyter for exploration only, not production
│
└── outputs/
    ├── candidates/              ← Generated aptamer candidates per target
    └── motifs/                  ← MEME motif analysis outputs
```

---

## 7. CONFIGURATION (config.py)

All hyperparameters live in `config.py`. Never hardcode values in model files. Always import from config. Key settings:

```python
# Device (auto-detected)
DEVICE = "mps"  # Apple Silicon GPU — your machine

# DNA Encoder
DNA_KMER_SIZE = 3
DNA_EMBED_DIM = 128
DNA_NUM_LAYERS = 6
DNA_NUM_HEADS = 8

# Protein Encoder
ESM_MODEL_NAME = "esm2_t12_35M_UR50D"  # 35M params, 480-dim output
ESM_EMBED_DIM = 480
LORA_RANK = 8

# Training
BATCH_SIZE = 32
LEARNING_RATE_BASE = 1e-4
LEARNING_RATE_LORA = 1e-5
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
```

---

## 8. BUILD ORDER (DO NOT SKIP STEPS)

Build in this exact order. Each component is testable before the next:

```
Phase 1: DATA PIPELINE
  Step 1.1  collect_aptamers.py     — PubMed collection (run and let collect)
  Step 1.2  validate_sequences.py   — sequence QC rules
  Step 1.3  vienna_features.py      — precompute structure features
  Step 1.4  build_dataset.py        — unify all sources
  Step 1.5  augment.py              — reverse complements, truncations, negatives

Phase 2: TOKENIZER
  Step 2.1  tokenizer.py            — DNA 3-mer tokenizer, test on example sequences

Phase 3: ENCODERS (build and test each independently)
  Step 3.1  dna_encoder.py          — 6-layer transformer, test input→output shapes
  Step 3.2  protein_encoder.py      — ESM-2 + LoRA, test on insulin sequence
  Step 3.3  condition_encoder.py    — MLP for condition vector

Phase 4: FUSION
  Step 4.1  cross_attention.py      — symmetric bidirectional, test shapes
  Step 4.2  cnn_head.py             — 17-block CNN, test on interaction matrix

Phase 5: OUTPUT + ASSEMBLY
  Step 5.1  dual_head.py            — binding + Kd heads
  Step 5.2  condaptnet.py           — full model assembly, end-to-end forward pass test

Phase 6: TRAINING
  Step 6.1  losses.py               — combined loss function
  Step 6.2  train.py                — full training loop with MPS support
  Step 6.3  evaluate.py             — MCC, AUC, PR metrics

Phase 7: FINE-TUNING
  Step 7.1  finetune.py             — Stage 2 on Continuity targets
```

---

## 9. ENVIRONMENT & DEPENDENCIES

### Virtual Environment
**ALWAYS activate before running anything:**
```bash
cd continuitybioML
source condaptnet_env/bin/activate
```

The prompt should show `(condaptnet_env)` when active.

### Verified Working (as of setup)
- Python 3.11.3
- PyTorch 2.11.0
- CUDA: False (Mac, expected)
- MPS: True (Apple Silicon GPU — USE THIS for training)
- ESM-2: Available (fair-esm installed)
- ViennaRNA: Available
- Pandas 3.0.2

### Installing New Packages
Always install inside the virtual environment:
```bash
source condaptnet_env/bin/activate
pip install <package>
```

### MPS (Apple Silicon) Usage
All tensor operations and model training must use MPS device:
```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
tensor = tensor.to(device)
```

**MPS gotchas:**
- Some PyTorch ops aren't MPS-supported yet. If you get `NotImplementedError`, add `PYTORCH_ENABLE_MPS_FALLBACK=1` to the run command
- MPS doesn't support float64 well — use float32 everywhere
- Batch size may need to be smaller than CUDA equivalents

---

## 10. CODING STANDARDS

### Every Model File Must Have
1. Docstring explaining what the module does and its input/output shapes
2. A `if __name__ == "__main__":` test block that runs a forward pass with dummy data
3. Imports from `config.py` for all hyperparameters (no hardcoded values)
4. Shape assertions in forward pass during development

### Shape Convention (always document these)
```python
# DNA encoder input:  [batch_size, seq_len] (token ids)
# DNA encoder output: [batch_size, seq_len, DNA_EMBED_DIM]
# Protein encoder:    [batch_size, protein_len, ESM_EMBED_DIM]
# Cross-attention:    [batch_size, apt_len, protein_len, FUSION_DIM]
# CNN input:          [batch_size, channels, apt_len, protein_len]
# Final output:       [batch_size, 1] binding prob, [batch_size, 1] Kd
```

### Error Handling
- Validate sequence inputs (correct alphabet, length range) before forward pass
- Log warnings for sequences outside expected ranges, don't silently drop them
- Save checkpoints every epoch, not just at the end

### Testing Pattern
Every module file ends with:
```python
if __name__ == "__main__":
    import torch
    from config import DEVICE, BATCH_SIZE

    # Create dummy inputs
    batch_size = 4
    apt_len = 50
    protein_len = 200

    dummy_aptamer = torch.randint(0, 64, (batch_size, apt_len)).to(DEVICE)
    # ... etc

    # Run forward pass
    model = YourModule().to(DEVICE)
    output = model(dummy_aptamer)

    # Assert shapes
    assert output.shape == (batch_size, ...), f"Wrong shape: {output.shape}"
    print(f"Test passed. Output shape: {output.shape}")
```

---

## 11. REFERENCE PAPERS (key decisions sourced from these)

| Paper | Year | Key Contribution to CondAptNet |
|---|---|---|
| Li et al. PLOS ONE | 2014 | Source of standard benchmark dataset (725 pairs, 164 proteins). We extend this |
| Apta-MCTS PLOS ONE | 2021 | MCTS generation algorithm (Phase 5 of pipeline). RF classifier as baseline |
| AptaTrans BMC Bioinf | 2023 | Interaction matrix + 17-block CNN architecture. Pretraining strategy. Reverse-complement augmentation |
| AptaBERT bioRxiv | 2023 | BERT-style pretraining on aptamers. ROC-AUC 96% shows transformers work well here |
| AptaBLE bioRxiv | 2026 | Symmetric bidirectional cross-attention. Currently SOTA. We improve with ESM-2 + DNA encoder + conditions |

### Critical Facts from Literature
- Standard training dataset: 580 positive + 1740 negative pairs (Li et al. 2014)
- AptaTrans ROC-AUC: 0.921 on benchmark (our target to beat)
- AptaBLE outperforms AlphaFold3: 71% vs 62% on 26 diverse pairs
- All prior models convert DNA to RNA — we do NOT do this
- AptaTrans data augmentation: reverse complement doubles training data (validated)
- Optimal sequence length per Apta-MCTS: 70–90 nt gave best docking scores

---

## 12. WHAT NOT TO DO

- **Never convert T→U** in aptamer sequences. We use native DNA.
- **Never randomly split** train/val/test. Always split by protein family.
- **Never hardcode hyperparameters** in model files. Use config.py.
- **Never train without early stopping.** Small dataset = fast overfitting.
- **Never report accuracy as primary metric** on imbalanced data. Use MCC.
- **Never cache ESM-2 embeddings to GPU memory.** Cache to disk as numpy arrays.
- **Never run training without activating condaptnet_env first.**
- **Never add sequences without running validate_sequences.py first.**
- **Never use float64 tensors on MPS.** Always float32.
- **Never skip the `if __name__ == "__main__"` test block** in model files.

---

## 13. CURRENT STATUS

### Completed
- [x] Python 3.11 virtual environment (condaptnet_env)
- [x] All dependencies installed and verified (PyTorch, ESM-2, ViennaRNA, Pandas)
- [x] MPS (Apple Silicon GPU) confirmed working
- [x] Project folder structure created (continuitybioML/)
- [x] config.py written with all hyperparameters
- [x] collect_aptamers.py written (PubMed data collection)
- [x] verify_env.py confirmed working

### In Progress
- [ ] Running collect_aptamers.py to gather PubMed literature

### Next Up (in order)
1. Download Li et al. 2014 File S1 (the standard benchmark dataset)
2. Manually download AptamerBase entries for all 5 target proteins
3. Build validate_sequences.py
4. Build vienna_features.py
5. Build the DNA tokenizer (tokenizer.py)
6. Build the DNA encoder (dna_encoder.py)

### Decisions Still To Make
- Whether to use esm2_t6_8M (fast) or esm2_t12_35M (better) based on RAM constraints
- Whether Kd regression head is trainable immediately or deferred until Kd-labeled data is sufficient
- Exact FiLM conditioning implementation (additive vs multiplicative modulation)

---

## 14. QUICK COMMANDS REFERENCE

```bash
# Activate environment (always do this first)
source condaptnet_env/bin/activate

# Run data collection (takes ~20 min)
python scripts/data/collect_aptamers.py

# Run environment check
python scripts/verify_env.py

# Test any model module
python models/encoders/dna_encoder.py

# Run training (once data and model are ready)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py

# Run evaluation
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt
```

---

## 15. CONTACT & CONTEXT

- **Developer:** Shivansh Bansal
- **Company:** Continuity (continuity.bio) — biosensing platform
- **Device context:** Real-time physiological biomarker monitoring using DNA aptamers
- **Academic context:** Incoming UCLA undergraduate, computational biology
- **Background:** ISEF 2025 2nd place computational biology (AlphaFold3, Schrödinger, SVM, HEK293 validation)
- **Key collaborator context:** This project is for Continuity's internal ML pipeline

---

*Last updated: Project initialization. Update the Current Status section as components are completed.*