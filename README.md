# CondAptNet

**Conditional Aptamer-Protein Interaction Network**

A novel deep learning architecture for predicting DNA aptamer binding to arbitrary protein targets, built for [Continuity](https://continuity.bio)'s real-time physiological biosensing platform.

---

## Overview

CondAptNet is a **general-purpose** DNA aptamer-protein interaction prediction model. Given any protein's amino acid sequence and a DNA aptamer sequence, it predicts whether the aptamer will bind that protein and how strongly.

It is designed in three tiers:

```
TIER 1 — GENERAL MODEL
Trained on aptamer-protein pairs across hundreds of diverse protein families.
Generalizes to any protein given only its amino acid sequence.
This is the core scientific contribution.

TIER 2 — VALIDATION BENCHMARK
Fine-tuned on insulin, myoglobin, NT-proBNP, troponin I/T, albumin.
Well-studied proteins with published aptamers and known Kd values.
Used to verify model performance before trusting it on real device targets.
These are NOT the deployment targets for Continuity's device.

TIER 3 — DEPLOYMENT TARGETS (TBD)
The real Continuity biomarker set, not yet confirmed.
Plug-and-play: update DEPLOYMENT_TARGETS in config.py and run finetune.py.
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

**DNA Encoder:** Transformer with native 3-mer tokenization. No T→U conversion. Augmented with ViennaRNA secondary structure features (MFE, stem count, loop count, base pair probabilities). **Switchable** via `DNA_ENCODER_TYPE` in config.py — see [DNA Encoder Ablation](#dna-encoder-ablation-scratch-vs-dnabert-2) below for the pretrained DNABERT-2 alternative.

**Protein Encoder (ESM-2):** Pretrained on 250 million protein sequences. Deeply understands any protein — including ones with zero aptamer training data. Fine-tuned with LoRA (rank=8, α=16). ESM-2 embeddings are pre-cached to disk; the frozen backbone runs only once per unique protein.

**Symmetric Bidirectional Cross-Attention:** Both molecules attend to each other simultaneously. Validated by AptaBLE (2026) as superior to unidirectional approaches.

**FiLM Condition Injection:** pH, salt, temperature, Mg²⁺, and buffer modulate cross-attention feature maps via learned scale and shift parameters. First aptamer model to encode physiological context. Physiological defaults: pH 7.4, 150 mM Na⁺, 37°C, 2 mM Mg²⁺, PBS.

**17-block CNN (GroupNorm):** Extracts hierarchical features from the 2D aptamer-protein interaction map. GroupNorm replaces BatchNorm2d for full MPS/CUDA native execution. Each `ConvBlock` applies channel-wise `Dropout2d` (`CNN_DROPOUT=0.1`) after each GELU — the standard regularizer for spatially correlated conv activations — guarding the deep 256-channel stack against overfitting on the few thousand real training rows.

**Dual Output Head:** Binary binding classification + Kd regression for affinity-ranked candidate generation. Kd head is skippable at inference when no affinity label is available.

### DNA Encoder Ablation (scratch vs. DNABERT-2)

Our protein arm gets transfer learning (ESM-2, pretrained on 250M sequences) while the DNA arm learns from scratch on a small dataset. To test whether a pretrained DNA foundation model closes that asymmetry, the DNA encoder is **switchable** behind a single config flag — an A/B ablation, not a replacement:

| `DNA_ENCODER_TYPE` | Encoder | Output dim | Tokenization | Trainable |
|---|---|---|---|---|
| `"scratch"` (default) | 6-layer Transformer, trained from scratch | 128 | 3-mer (66-token vocab) + ViennaRNA bias | full |
| `"dnabert2"` | [DNABERT-2 117M](https://github.com/MAGICS-LAB/DNABERT_2) (Zhou et al., ICLR 2024) + LoRA | 768 | BPE (multi-species genomic) | LoRA only (0.25%) |

- **Default is unchanged.** With `DNA_ENCODER_TYPE="scratch"` the from-scratch encoder is bit-for-bit identical; DNABERT-2 pulls in no dependency and downloads nothing.
- **LoRA target:** DNABERT-2 uses a *fused* `Wqkv` projection (`nn.Linear(768→2304)`), not the separate q/v projections ESM-2 exposes, so LoRA (rank 8) adapts Q/K/V jointly. Loading uses a direct-construction pattern (`models/encoders/dna_encoder_pretrained.py`) because the plain `AutoModel.from_pretrained` path is broken on transformers 5.x.
- **Domain-shift caveat:** DNABERT-2 was pretrained on genomic DNA, not short synthetic ssDNA aptamers (20–120 nt). Whether it beats the from-scratch encoder is an open empirical question — the point of making it benchmarkable. The comparative training run is a follow-up.
- **BPE ≠ nucleotides:** a 120 nt aptamer is ~27 BPE tokens (dataset max 28), so `DNABERT2_MAX_LEN=32`.

Enable with `DNA_ENCODER_TYPE = "dnabert2"` in config.py (also needs `pip install transformers einops`).

---

## Training Pipeline

### Stage 1 — Broad Pretraining
- Data: all curated + harvested aptamer-protein pairs across diverse protein families
- ESM-2 frozen (only LoRA adapters + all other layers train)
- Split: by protein family (never randomly) — 70 / 15 / 15
- Loss: BCE (binding) + MSE (Kd where available)
- Gradient checkpointing on CNN head for T4 memory efficiency
- Output: general aptamer interaction model

### Stage 2 — Validation Fine-Tuning
- Data: insulin, myoglobin, NT-proBNP, troponin I/T, albumin
- Purpose: benchmark against published Kd values, verify generalization
- ESM-2 LoRA unfrozen at 10× lower learning rate

### Stage 3 — Deployment Fine-Tuning (TBD)
- Data: actual Continuity device targets (not yet confirmed)
- Update `DEPLOYMENT_TARGETS` in config.py and run `finetune.py --stage deployment`
- No other changes required

### Stage 4 — Active Learning (ongoing)
- Lab validation results → new labeled data → retrain Stage 2/3

---

## Data

### Sources

| Source | Type | Notes |
|---|---|---|
| UTexas Aptamer Database ([Zenodo](https://doi.org/10.5281/zenodo.8264921)) | Curated DB | ~896 ssDNA rows after DNA-only filter |
| Li et al. 2014 ([PLOS ONE](https://doi.org/10.1371/journal.pone.0086729)) | Benchmark | 2,320 entries, 164 proteins; sequences need enrichment pass |
| PubMed / PMC | Literature | Full-text XML + supplementary files via Entrez |
| Semantic Scholar | Literature | Open academic index |
| OpenAlex | Literature | Open access full-text index |
| bioRxiv / medRxiv | Preprints | Content API |
| PatentsView (US) | Patents | REST API, no auth required |
| EPO OPS | Patents | Requires OAuth2 credentials |
| WIPO PatentScope | Patents | Web scraping |
| Lens.org | Patents | Requires Bearer token |
| Google Patents | Patents | Conservative XHR scraping |

### Scraper Architecture

All literature and patent sources are harvested by a unified scraper pipeline (`scripts/data/scraper/`):

- **4 parsers** — PDF, Excel/CSV, XML (PMC NXML), plain text/HTML
- **10 source adapters** — each rate-limited, with per-adapter failure isolation
- **Supplementary file fetcher** — PMC supplementary Excel/CSV/PDF files auto-downloaded and parsed (this is where most aptamer sequence tables live)
- **merge.py** — deduplicates against master_dataset.csv by exact (sequence, target) key; writes only new unique rows to `scraped_dataset.csv`; never overwrites master
- **Append-only provenance log** — byte offset, source URL, file hash, extraction timestamp for every record

Scraped records follow a 20-column schema distinct from master_dataset.csv and are merged after manual or automated review.

### Dataset (current)

| File | Rows | Notes |
|---|---|---|
| master_dataset.csv | 3,821 total | Built from UTexas + Li2014 |
| master_dataset.csv (training-ready) | 2,364 | Has protein_sequence + DNA sequence |
| scraped_dataset.csv | grows with scraper runs | New unique rows only |
| tier1_train.csv | 6,914 (post-augmentation) | After augment.py |
| val.csv | ~297 | Split by protein family |
| test.csv | ~282 | Held out by protein family |
| vienna_cache.pkl | ~6,330 sequences | Pre-computed structure features |

~1,457 master rows have `needs_sequence_enrichment=True` (Li2014 rows awaiting PubMed sequence lookup).

### Augmentation

- **Reverse complement** — hard negatives (label=0): RC folds into a different 3D structure, so it is NOT a confirmed binder
- **Systematic truncations** — 2–3 nt from each end
- **Cross-target negatives** — binder for protein A is a hard negative for protein B
- **Scrambled sequences** — composition preserved, order destroyed → non-binder label

### Sequence Safety Rules

- **LLMs never generate sequences.** All sequences are regex-extracted and validated against source documents.
- **Original Kd unit always stored** before nM conversion.
- **DNA only** — RNA sequences (containing U) are filtered out at ingestion.

---

## Project Structure

```
continuitybioML/
├── CLAUDE.md                    # Full context for Claude Code
├── README.md                    # This file
├── config.py                    # All hyperparameters
├── .env                         # API credentials (not tracked)
├── data/
│   ├── raw/
│   │   ├── utexas_aptamer_db/   # aptamer_database.xlsx (source)
│   │   ├── li2014_benchmark/    # pone.0086729.s001.xlsx, s003.xlsx
│   │   ├── scraped_dataset.csv  # Output from scraper (never master)
│   │   ├── scraper_provenance.jsonl
│   │   ├── scraper_coverage_report.txt
│   │   └── protein_name_overrides.csv
│   ├── processed/
│   │   ├── master_dataset.csv
│   │   ├── vienna_cache.pkl
│   │   └── protein_embeddings/  # Pre-cached ESM-2 .npy files
│   └── augmented/
│       ├── tier1_train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── encoders/
│   │   ├── dna_encoder.py
│   │   ├── dna_encoder_pretrained.py  # Optional DNABERT-2 + LoRA (A/B ablation)
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
│   │   ├── collect_aptamers.py
│   │   ├── build_dataset.py
│   │   ├── enrich_proteins.py
│   │   ├── augment.py
│   │   ├── vienna_features.py
│   │   ├── validate_sequences.py
│   │   └── scraper/             # Automated data harvesting pipeline
│   │       ├── main.py          # CLI: run all/selected sources
│   │       ├── merge.py         # Dedup + append to scraped_dataset.csv
│   │       ├── config.py        # Rate limits, API keys, paths
│   │       ├── schema.py        # 20-column schema + validation
│   │       ├── parsers/
│   │       │   ├── pdf_parser.py
│   │       │   ├── excel_parser.py
│   │       │   ├── xml_parser.py    # PMC NXML
│   │       │   └── text_parser.py   # HTML + plain text
│   │       ├── adapters/
│   │       │   ├── base.py          # Rate-limited session, provenance logging
│   │       │   ├── pubmed_pmc.py    # PubMed + PMC full text + supplementary files
│   │       │   ├── semantic_scholar.py
│   │       │   ├── openalex.py
│   │       │   ├── biorxiv.py
│   │       │   ├── patents_us.py
│   │       │   ├── patents_epo.py
│   │       │   ├── patents_wipo.py
│   │       │   ├── lens.py
│   │       │   ├── google_patents.py
│   │       │   ├── databases.py     # Local UTexas / Li2014
│   │       │   └── supp_fetcher.py  # PMC supplementary file downloader
│   │       └── utils/
│   │           ├── provenance.py
│   │           ├── rate_limiter.py
│   │           └── deduplication.py
│   ├── model/
│   │   └── tokenizer.py
│   ├── training/
│   │   ├── train.py             # Stage 1 (--resume, --max-batches, CUDA/MPS/CPU)
│   │   ├── finetune.py          # Stage 2 and 3 (--stage validation|deployment)
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── spikes/
│       └── inspect_dnabert2.py     # Throwaway DNABERT-2 internals probe (Session 2)
└── outputs/
    ├── candidates/
    └── motifs/
```

---

## Environment Setup

### Local (Apple Silicon)

```bash
git clone <repo>
cd continuitybioML
python3.11 -m venv condaptnet_env
source condaptnet_env/bin/activate
pip install torch torchvision torchaudio
pip install fair-esm pandas numpy scikit-learn biopython ViennaRNA requests tqdm openpyxl pdfplumber lxml beautifulsoup4
python scripts/verify_env.py
```

### Credentials (.env)

Copy `.env` and fill in what you have. Everything except EPO and Lens works without credentials:

```bash
# Optional — speeds PubMed from 3 req/s to 10 req/s
NCBI_API_KEY=

# Optional — higher Semantic Scholar rate limit
SEMANTIC_SCHOLAR_API_KEY=

# Required for Lens patent adapter (free at access.lens.org)
LENS_API_TOKEN=

# Required for EPO patent adapter (register at developers.epo.org)
EPO_CLIENT_KEY=
EPO_CLIENT_SECRET=
```

`ENTREZ_EMAIL` is already set in the scraper config.

---

## Usage

### Data harvesting (scraper)

```bash
source condaptnet_env/bin/activate

# Dry run — validate without writing to disk
python -m scripts.data.scraper.main --sources pubmed,semantic_scholar --dry-run

# Real run — literature sources only (no credentials needed)
python -m scripts.data.scraper.main \
    --sources pubmed,semantic_scholar,openalex,biorxiv \
    --max-per-source 500

# Full run including patents (needs Lens/EPO credentials)
python -m scripts.data.scraper.main --max-per-source 1000
```

Output: `data/raw/scraped_dataset.csv` (new unique rows only — master_dataset.csv is never modified).

### Data processing

```bash
# Build master_dataset.csv from raw sources
python scripts/data/build_dataset.py

# Enrich protein sequences via UniProt
python scripts/data/enrich_proteins.py

# Precompute ViennaRNA structure features
python scripts/data/vienna_features.py

# Generate augmented training splits
python scripts/data/augment.py
```

### Training

```bash
# Stage 1: broad pretraining (local MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py

# Resume after interruption
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py --resume

# Smoke test (2 batches per epoch)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/train.py --max-batches 2

# Stage 2: validation fine-tuning
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py --stage validation

# Stage 3: deployment fine-tuning (after updating DEPLOYMENT_TARGETS in config.py)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py --stage deployment

# Include scraped data in fine-tuning
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/training/finetune.py \
    --stage validation \
    --extra-data data/raw/scraped_dataset.csv

# Evaluate
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/pretrain/best.pt
```

### Google Colab (T4)

Open `notebooks/train_colab.ipynb`. It auto-detects the latest checkpoint and resumes. T4-tuned defaults: `batch_size=16`, `max_prot_len=128`, `PYTORCH_ALLOC_CONF=expandable_segments:True`.

---

## Evaluation Metrics

| Metric | Why |
|---|---|
| MCC | Best single metric for imbalanced binary classification (primary) |
| AUC-ROC | Discriminative ability across thresholds |
| AUC-PR | More informative than ROC when positives are rare |
| Sensitivity | False negatives = missed candidates |
| Pearson r (Kd) | Regression head quality |

Accuracy is not reported as a primary metric — class imbalance makes it misleading.

---

## Reference Papers

| Paper | Key Contribution |
|---|---|
| Li et al., PLOS ONE (2014) | Standard benchmark — 725 pairs, 164 proteins |
| Lee et al., PLOS ONE (2021) — Apta-MCTS | MCTS generation; RF baseline |
| Shin et al., BMC Bioinformatics (2023) — AptaTrans | Interaction matrix + CNN; pretraining; augmentation strategy |
| Morsch et al., bioRxiv (2023) — AptaBERT | BERT-style aptamer pretraining |
| Atom Bioworks, bioRxiv (2026) — AptaBLE | Symmetric bidirectional cross-attention; current SOTA |

---

## Build Status

### Model Pipeline
- [x] Environment setup (Python 3.11, PyTorch, ESM-2, ViennaRNA, MPS + CUDA)
- [x] config.py — all hyperparams + physiological defaults (37°C, 150mM Na, 2mM Mg)
- [x] tokenizer.py — DNA 3-mer tokenizer (66-token vocab)
- [x] dna_encoder.py — 6-layer Transformer [B, L, 128], MPS verified
- [x] dna_encoder_pretrained.py — optional DNABERT-2 117M + LoRA on fused Wqkv [B, L, 768], MPS verified (gated on `DNA_ENCODER_TYPE="dnabert2"`)
- [x] protein_encoder.py — ESM-2 + LoRA (0.55% trainable) [B, L, 480], MPS verified
- [x] condition_encoder.py — FiLM MLP [B, 128], MPS verified
- [x] cross_attention.py — symmetric bidirectional [B, L, 256], MPS verified; DNA embed dim now configurable (128 scratch / 768 dnabert2)
- [x] cnn_head.py — 17-block CNN with GroupNorm + channel-wise Dropout2d [B, 256], MPS + CUDA verified
- [x] dual_head.py — binding sigmoid + Kd ReLU
- [x] condaptnet.py — full assembly + gradient checkpointing, end-to-end verified
- [x] losses.py — combined BCE + MSE loss
- [x] train.py — Stage 1 loop (--resume, --max-batches, CUDA/MPS/CPU auto-detect)
- [x] finetune.py — Stage 2/3 fine-tuning (--stage validation|deployment, --extra-data)
- [x] evaluate.py — MCC, AUC-ROC, AUC-PR, sensitivity, Pearson r
- [x] notebooks/train_colab.ipynb — T4 Colab training notebook

### Data Pipeline
- [x] collect_aptamers.py — PubMed SELEX collection
- [x] build_dataset.py — UTexas + Li2014 → master_dataset.csv (3,821 rows)
- [x] enrich_proteins.py — UniProt lookup, fuzzy matching, non-protein filtering
- [x] validate_sequences.py — length, GC%, homopolymer, alphabet QC
- [x] vienna_features.py — ViennaRNA features, incremental pickle cache
- [x] augment.py — rev-comp, truncations, cross-target negatives, scrambles

### Scraper Pipeline
- [x] 4 document parsers — PDF, Excel/CSV/TSV, PMC XML, HTML/text
- [x] supp_fetcher.py — PMC supplementary file downloader (Excel/CSV/PDF)
- [x] 10 source adapters — PubMed+PMC, Semantic Scholar, OpenAlex, bioRxiv, PatentsView, EPO, WIPO, Lens, Google Patents, local databases
- [x] merge.py — deduplication against master + append-safe output
- [x] main.py — CLI orchestrator with per-adapter failure isolation
- [x] Provenance logging — byte-level JSONL audit trail

### Pending
- [ ] Sequence enrichment pass — PubMed lookup for Li2014 rows (needs_sequence_enrichment=True)
- [ ] UniProt enrichment — protein_sequence for rows missing it
- [ ] Stage 1 training run
- [ ] Stage 2 validation fine-tuning
- [ ] Tier 3 deployment targets (pending Continuity confirmation)

---

## License

Private — built for Continuity (continuity.bio). All rights reserved.

---

*Built by Shivansh Bansal — Continuity ML Pipeline*
