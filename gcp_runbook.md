# GCP Runbook — CondAptNet Training

**Purpose:** Step-by-step guide to provision a GCP GPU instance, run the
smoke tests that verify the environment, and launch a full Stage 1 training run.

**Branch this runbook was written for:** `infra/gcp-smoke-test`
**Last updated:** 2026-07-08

---

## Contents

1. [Instance Sizing Decision](#1-instance-sizing-decision)
2. [Prerequisites](#2-prerequisites)
3. [Provision the VM](#3-provision-the-vm)
4. [Install Dependencies on the VM](#4-install-dependencies-on-the-vm)
5. [Copy the Repo to the VM](#5-copy-the-repo-to-the-vm)
6. [Run Smoke Tests](#6-run-smoke-tests)
7. [Launch Full Training Run](#7-launch-full-training-run)
8. [Monitoring and Cost Control](#8-monitoring-and-cost-control)
9. [Known Issues and Gotchas](#9-known-issues-and-gotchas)
10. [Smoke Test Expected Results](#10-smoke-test-expected-results)

---

## 1. Instance Sizing Decision

### Model parameter counts

| Component | Params (scratch path) | Params (dnabert2 path) |
|---|---|---|
| DNA Encoder (scratch 6-layer Transformer) | ~3.0M | — |
| DNABERT-2 117M + LoRA | — | ~117.5M |
| ESM-2 (35M) + LoRA (rank=8) | ~35.2M | ~35.2M |
| Cross-attention + FiLM | ~1.8M | ~1.8M |
| 17-block CNN (64→128→256) | ~5.1M | ~5.1M |
| Dual output head | ~0.3M | ~0.3M |
| **Total** | **~45.4M** | **~160M** |
| Trainable (Stage 1) | ~10.5M (23%) | ~15M (9%) |

### GPU memory estimate (float32, batch=32, prot_len=512)

| Config | Params | Activations | AdamW state | Peak estimate |
|---|---|---|---|---|
| scratch, batch=32 | ~730MB | ~1.2GB | ~1.5GB | **~3.5GB** |
| scratch, batch=32, prot=1024 | ~730MB | ~4.8GB | ~1.5GB | **~7GB** |
| dnabert2, batch=32 | ~2.6GB | ~1.8GB | ~5.2GB | **~10GB** |
| dnabert2, batch=32, prot=1024 | ~2.6GB | ~5.5GB | ~5.2GB | **~14GB** |

**CNN interaction matrix is the OOM risk.** It's `[B, 1, apt_len, prot_len]`
before the first conv — at batch=32, prot_len=1024, apt_len=50:
`32 * 50 * 1024 * 4B ≈ 6.5MB` per channel. With 64 channels: ~415MB just for
the interaction matrix. This is manageable on T4 (16GB) but tight. Use
`--max-prot-len 512` to cap protein length and reduce this if OOM occurs.

### Recommended instances

| Use case | GCP instance | GPU | VRAM | ~Cost/hr |
|---|---|---|---|---|
| Smoke tests + dev | `n1-standard-8` + T4 | NVIDIA T4 | 16 GB | $0.35 |
| Full A/B training run | `a2-highgpu-1g` | NVIDIA A100 | 40 GB | $3.67 |
| Budget full run | `n1-standard-8` + T4 | NVIDIA T4 | 16 GB | $0.35 |

**AWS g5.xlarge equivalent** (mentioned in prior planning):
- g5.xlarge = 4 vCPU / 16GB RAM / A10G 24GB
- GCP nearest: `n1-standard-8` + T4 (cheaper; T4 has 16GB vs A10G's 24GB)
- For headroom parity with A10G, use A100 or reduce `--max-prot-len` to 512

---

## 2. Prerequisites

On your **local machine**:
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- A GCP project with billing enabled
- GPU quota for T4 (default) or A100 (may need quota request) in your zone
- The CondAptNet repo cloned locally

Check GPU quota:
```bash
gcloud compute regions describe us-central1 --format="json" \
  | jq '.quotas[] | select(.metric == "NVIDIA_T4_GPUS")'
```

Request quota increase if needed:
```
GCP Console → IAM & Admin → Quotas → search "NVIDIA T4 GPUs" → Request increase
```

---

## 3. Provision the VM

### Option A: Use the setup script (recommended)

```bash
cd ~/condaptnet   # or wherever the repo is

# Smoke test VM (T4):
PROJECT_ID=your-project-id \
REPO_URL=https://github.com/YOUR_ORG/condaptnet.git \
bash scripts/infra/gcp_setup.sh --instance-type T4 --zone us-central1-a

# Full training VM (A100):
PROJECT_ID=your-project-id \
REPO_URL=https://github.com/YOUR_ORG/condaptnet.git \
bash scripts/infra/gcp_setup.sh --instance-type A100 --zone us-central1-a
```

### Option B: Manual gcloud commands

```bash
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export VM_NAME="condaptnet-smoke"

# T4 instance:
gcloud compute instances create $VM_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True \
  --scopes=cloud-platform

# Wait ~60s for boot, then verify:
gcloud compute ssh $VM_NAME --zone=$ZONE -- nvidia-smi
```

### Connect to the VM

```bash
gcloud compute ssh condaptnet-smoke --zone=us-central1-a
```

### Stop/start the VM (to save costs when not training)

```bash
# Stop (billing stops for compute; disk storage continues):
gcloud compute instances stop condaptnet-smoke --zone=us-central1-a

# Start again:
gcloud compute instances start condaptnet-smoke --zone=us-central1-a
```

---

## 4. Install Dependencies on the VM

Run these commands **on the GCP VM** (after SSH):

```bash
# Verify CUDA is available
nvidia-smi
nvcc --version

# System packages
sudo apt-get update -qq
sudo apt-get install -y git python3.11 python3.11-venv python3.11-dev \
    build-essential wget curl

# ViennaRNA (required for structure features)
# Option A: pip wheel (may not be available for all Python versions)
pip install viennarna

# Option B: build from source (if wheel fails)
cd /tmp
wget -q https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz
tar xzf ViennaRNA-2.6.4.tar.gz
cd ViennaRNA-2.6.4
./configure --without-perl --without-ruby --without-doc --prefix=/usr/local --with-python3
make -j$(nproc) && sudo make install
cd ~
python3 -c "import RNA; print('ViennaRNA OK:', RNA.__version__)"
```

---

## 5. Copy the Repo to the VM

### Option A: Git clone (if repo is on GitHub)

```bash
# On the VM:
git clone https://github.com/YOUR_ORG/condaptnet.git ~/condaptnet
cd ~/condaptnet
git checkout infra/gcp-smoke-test
```

### Option B: rsync from local machine (if private / no remote yet)

```bash
# On your LOCAL machine:
gcloud compute scp --recurse ./ condaptnet-smoke:~/condaptnet \
  --zone=us-central1-a \
  --exclude=condaptnet_env --exclude=.git --exclude=__pycache__
```

### Create the Python environment

```bash
# On the VM:
cd ~/condaptnet
python3.11 -m venv condaptnet_env
source condaptnet_env/bin/activate

# PyTorch with CUDA 12.1 support (DO NOT use plain pip install torch — CPU only)
pip install torch==2.2.2 torchvision \
  --index-url https://download.pytorch.org/whl/cu121

# Core ML dependencies
pip install \
  fair-esm==2.0.0 \
  transformers==4.40.2 \
  einops \
  huggingface_hub \
  peft \
  pandas numpy scipy scikit-learn biopython

# Verify CUDA torch build:
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
# Expected output:
# PyTorch: 2.2.2+cu121
# CUDA available: True
# GPU: Tesla T4  (or A100)
```

---

## 6. Run Smoke Tests

```bash
# On the VM, with env activated:
cd ~/condaptnet
source condaptnet_env/bin/activate

# Run all 5 smoke tests (both encoder configs):
bash scripts/infra/run_all_smoke_tests.sh --require-cuda --n-steps 50

# Or run individual steps:
python scripts/infra/smoke_cuda_check.py --require-cuda     # Step 2
python scripts/infra/smoke_data_loading.py                  # Step 3
python scripts/infra/smoke_checkpoint.py --n-steps 5        # Step 4
python scripts/infra/smoke_training.py --n-steps 50 --encoder scratch  # Step 5a
python scripts/infra/smoke_training.py --n-steps 50 --encoder dnabert2 # Step 5b

# Skip DNABERT-2 for faster run (if only validating infrastructure):
bash scripts/infra/run_all_smoke_tests.sh --require-cuda --skip-dnabert2
```

See [Section 10](#10-smoke-test-expected-results) for expected output from each step.

---

## 7. Launch Full Training Run

Only do this after all smoke tests pass.

### Stage 1: Broad Pretraining

```bash
cd ~/condaptnet
source condaptnet_env/bin/activate

# Ensure augmented splits are current (run once; skip if splits already exist):
python scripts/data/augment.py

# Stage 1 — scratch encoder (default):
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/training/train.py \
  --use-amp \
  --batch-size 32 \
  --grad-accum 1 \
  --max-prot-len 512 \
  --max-epochs 100

# Stage 1 — DNABERT-2 encoder (for A/B comparison):
# First, set DNA_ENCODER_TYPE in config.py:
#   DNA_ENCODER_TYPE = "dnabert2"
# Then:
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/training/train.py \
  --use-amp \
  --batch-size 16 \        # smaller batch: DNABERT-2 uses more memory
  --grad-accum 2 \         # effective batch = 32
  --max-prot-len 512 \
  --max-epochs 100
```

> **Note on `--max-prot-len`:** Set to 512 on T4 (16GB) to avoid OOM from the
> CNN interaction matrix. Use 1024 on A100 (40GB). The CNN matrix is
> `[B, 1, apt_len, prot_len]` — halving prot_len cuts this tensor 4×.

### Resume from checkpoint

```bash
python scripts/training/train.py --resume --checkpoint-dir models/checkpoints/pretrain/
```

### Stage 2: Validation Fine-tuning (after Stage 1 is complete)

```bash
python scripts/training/finetune.py \
  --stage validation \
  --pretrain-checkpoint models/checkpoints/pretrain/best.pt
```

---

## 8. Monitoring and Cost Control

### Watch GPU utilization while training

```bash
# In a second SSH session:
watch -n 5 nvidia-smi

# Or log to file (background):
nvidia-smi dmon -s u -d 10 > nvidia_monitor.log &
```

### Check training progress

```bash
tail -f training.log   # if you redirect stdout

# Or look at checkpoint timestamps:
ls -lt models/checkpoints/pretrain/
```

### Estimated costs

These are estimates based on Step 5 timing (see actual measured times in
smoke test output — use those numbers, not these):

| Config | ~sec/step (T4) | ~sec/step (A100) | Full run (100ep, T4) | Cost (T4) |
|---|---|---|---|---|
| scratch, batch=32 | ~0.8s | ~0.25s | ~35h | ~$12 |
| dnabert2, batch=16 | ~2.5s | ~0.7s | ~55h | ~$19 |

**These are rough pre-smoke-test estimates.** Replace with actual measured
timings from Step 5 of the smoke test before budgeting the full run.

### Stop VM to save money when not training

```bash
# From local machine:
gcloud compute instances stop condaptnet-smoke --zone=us-central1-a

# When ready to resume:
gcloud compute instances start condaptnet-smoke --zone=us-central1-a
gcloud compute ssh condaptnet-smoke --zone=us-central1-a
cd ~/condaptnet && source condaptnet_env/bin/activate
python scripts/training/train.py --resume
```

### Delete VM when done (saves disk storage too)

```bash
gcloud compute instances delete condaptnet-smoke --zone=us-central1-a
```

---

## 9. Known Issues and Gotchas

### Issue 1: CUDA routing — the Mac vs GCP difference

**Background:** This codebase was developed on Apple Silicon (MPS). On Mac,
`PYTORCH_ENABLE_MPS_FALLBACK=1` is needed because some ops fall back from MPS
to CPU. On Linux/GCP/CUDA this env var is **harmless** (MPS is Apple-only,
so there's nothing to fall back) but **misleading** — seeing it in a GCP shell
session means the shell env was copied from a Mac setup.

**What to check:** Run `smoke_cuda_check.py --require-cuda`. It will:
1. Confirm `torch.cuda.is_available() == True`
2. Explicitly verify model parameters land on `cuda:0` (not CPU)
3. Confirm gradients are on GPU after backward pass
4. Warn if `PYTORCH_ENABLE_MPS_FALLBACK` is set (harmless, but flag it)

**History:** The VS Code extension environment was once observed to not route
CUDA correctly (ops silently ran on CPU). The Colab browser environment did
route correctly. `smoke_cuda_check.py` detects this class of problem.

### Issue 2: weights_only=True on checkpoint load

`torch.load(..., weights_only=True)` blocks pickle-deserialization of arbitrary
Python objects (guards against RCE from malicious checkpoint files). This was
added in PR #6 (`security/pin-dnabert2-revision-and-weights-only`) and is
already in `train.py` (line 617) and `finetune.py` (lines 332, 415).

**Compatibility:** `weights_only=True` allows: tensors, plain dicts, ints,
floats, strings, and lists/tuples of those types. It blocks: custom Python
classes, NumPy arrays embedded directly, and any pickled object.

Our checkpoint format stores `{epoch: int, model: state_dict, optimizer:
state_dict, scheduler: state_dict, val_mcc: float, ...}` — all plain
Python types + tensor state dicts. This is compatible. `smoke_checkpoint.py`
verifies this explicitly.

**If you ever see:** `_pickle.UnpicklingError: Weights only load failed` after
adding something new to a checkpoint, the new value is a non-tensor Python
object. Either convert it to a plain type (int/float/str/list) before saving,
or keep it in a separate sidecar file.

### Issue 3: DNABERT-2 trust_remote_code=True on HF hub

DNABERT-2 requires `trust_remote_code=True` to load its custom `bert_layers.py`
from Hugging Face. The code is pinned to a specific commit SHA
(`DNABERT2_REVISION = "7bce263b..."` in `config.py`) so that the fetched
code can't change under us. Do NOT change this SHA without:
1. Reading the diff between old and new SHA on GitHub
2. Re-running `scripts/spikes/inspect_dnabert2.py` to confirm the LoRA target
   (`Wqkv`) still exists under the same module path

### Issue 4: ViennaRNA on the VM

`pip install viennarna` may fail if no pre-built wheel exists for the Python
version on the VM. In that case, build from source (see Section 4). If ViennaRNA
is unavailable, the training will still run — `AptamerDataset._vienna_feats()`
falls back to zero-vectors for sequences not in the cache. This degrades model
quality but doesn't crash.

### Issue 5: ESM-2 protein embedding pre-computation is slow

`train.py` calls `precompute_protein_embeddings()` before training starts.
This runs ESM-2 ONCE per unique protein in the dataset (~3700 unique proteins)
and caches the results to `data/processed/protein_embeddings/`.

On first run this takes ~45–90 minutes (T4). On subsequent runs (cached), it's
a few seconds. The embeddings are ~3.5GB on disk total.

**GCP-specific:** If you're copying the dataset from your local machine, copy
`data/processed/protein_embeddings/` too to avoid re-running ESM-2:
```bash
# From local machine:
gcloud compute scp --recurse data/processed/protein_embeddings/ \
  condaptnet-smoke:~/condaptnet/data/processed/ --zone=us-central1-a
```

### Issue 6: OOM on CNN interaction matrix

The biggest OOM risk is `CNNHead` which processes a `[B, 1, apt_len, prot_len]`
tensor. With `B=32, apt_len=50, prot_len=1024` this is manageable on A100 but
tight on T4.

**Mitigations (in order of preference):**
1. `--max-prot-len 512` (reduces CNN input by 4×, minimal quality impact)
2. `--batch-size 8 --grad-accum 4` (same effective batch, 4× less activation)
3. `--use-amp` (BF16 halves activation memory on A100; T4 does NOT support BF16)

### Issue 7: `_film_heads` missing from checkpoint (benign)

`ConditionEncoder._film_heads` is a `nn.ModuleDict` that's lazily populated
the first time a new `fusion_dim` is seen. If a checkpoint was saved before
`_film_heads` was ever called (e.g., very early in training), those keys will
be absent. `train.py` uses `strict=False` and explicitly classifies these as
`benign_missing` — they get random initialization, which is correct behavior
(they'd have been random if they had been in the checkpoint too, since they
hadn't been trained yet).

---

## 10. Smoke Test Expected Results

### Step 2: CUDA Routing Check

```
[PASS] PYTORCH_ENABLE_MPS_FALLBACK not set (expected on Linux GCP)
[PASS] CUDA device 0   : Tesla T4
[INFO]   VRAM          : 16.0 GB total, 15.7 GB free
[PASS] All model parameters confirmed on cuda
[PASS] Forward pass output on cuda:0
[PASS] All gradients confirmed on cuda:0 — no silent CPU fallback
[PASS] CondAptNet forward pass succeeded in ~250ms
```

**Failure modes:**
- `CUDA available: False` → CUDA driver not installed; run `sudo nvidia-smi` to check
- Tensor on wrong device → PyTorch installed without CUDA support; re-install
  with `--index-url https://download.pytorch.org/whl/cu121`

### Step 3: Data Pipeline

```
[PASS] Loaded 4499 rows from master_dataset_v2.csv
[INFO] Split column values: {'val': 159, 'test': 136, 'train': ~4200}
[PASS] Batch loaded in ~15ms
[PASS] All shape assertions passed for scratch path
[PASS] DNABERT-2 BPE aptamer shape correct: torch.Size([8, 32])
```

**If split column is missing:** The test samples randomly and notes it. This
is expected if `master_dataset_v2.csv` is from before the split assignment step.

### Step 4: Checkpoint

```
[PASS] 5 steps completed in ~8s
[PASS] torch.load(..., weights_only=True) succeeded in 0.3s
[PASS] Checkpoint has all required keys
[PASS] Model state_dict restored (strict=False, N lazy-init keys OK)
[PASS] Optimizer state_dict restored
[PASS] Scheduler state_dict restored
[PASS] All model parameters match between original and restored model
[PASS] Original and restored model produce identical binding_prob output
```

### Step 5: Training Smoke (timing table)

Replace the placeholders below with actual measured numbers from your run:

```
Encoder      Steps  First loss   Last loss  sec/step   Peak MB  Loss OK
─────────────────────────────────────────────────────────────────────────
scratch         50     0.XXXX      0.XXXX     X.XXXs    XXXX.X  [PASS]
dnabert2        50     0.XXXX      0.XXXX     X.XXXs    XXXX.X  [PASS]
```

**What to look for:**
- `sec/step` is the primary cost-estimation number. Multiply by 15904/batch_size
  steps/epoch × 100 epochs ÷ 3600 = hours for full run.
- Loss should decrease over 50 steps. If not, it may indicate NaN gradients,
  wrong device placement, or the optimizer not getting gradients (check AMP
  settings and grad clip).
- Peak MB should be < 14000 on T4 (16GB) with `--max-prot-len 512`. If you see
  OOM, reduce `--max-prot-len` or `--batch-size`.

---

*End of runbook. File issues or corrections on the GitHub repo.*
