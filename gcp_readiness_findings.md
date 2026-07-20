# GCP Readiness Findings — CondAptNet

**Generated:** 2026-07-15  
**Target VM:** `continuity-l4-dev` — NVIDIA L4 (Ada Lovelace, 24 GB VRAM), zone `us-west1-a`  
**Scope:** Static codebase analysis only (see § VM Access Blocked below for runtime gaps)

---

## VM ACCESS STATUS

**gcloud CLI is not installed on the local machine** that runs this Conductor workspace.  
Tasks 1, 2 (installed-package side), 3, and 5 (VM-side directory checks) could not be
completed from here. The commands in Task 1 of the prompt must be run interactively on
the VM by someone with gcloud or direct SSH access. The remaining tasks (4, 6, 7) are
fully answered below from static codebase analysis.

**Concrete gap:** Before training, someone must SSH to `continuity-l4-dev` and manually
run the full Task-1 checklist (whoami / nvidia-smi / pip list / etc.) plus confirm
the home-directory layout (~/checkpoints, ~/datasets, ~/models, ~/logs). Everything
below tells you exactly what to look for when you do that.

---

## 1. ENVIRONMENT VERIFICATION (runtime — NOT RUN)

Commands to run on the VM to satisfy Task 1:

```bash
whoami && hostname && pwd
nvidia-smi
df -h /
free -h
nproc
command -v conda || true
python3 --version
git --version
git lfs version
pip list 2>/dev/null | grep -E "torch|esm|transformers|viennarna|biopython|scikit|scipy|numpy|pandas"
echo $CUDA_HOME
nvcc --version
cat /etc/os-release
```

Expected values when healthy:
- `nvidia-smi`: L4 (24478 MiB VRAM), CUDA driver ≥ 525.x, CUDA runtime shown
- `python3 --version`: 3.11.x (the venv was built with 3.11)
- `torch.__version__`: should end in `+cu121` or `+cu124` (CUDA build, NOT cpu)

**If any of these hit an escalation condition (nvidia-smi fails, conda unavailable, etc.),
stop immediately per the escalation rule in the task brief.**

---

## 2. PACKAGE GAP REPORT

### Packages confirmed required by the codebase (actual imports, not runbook guess-list)

| Package | Import | Role | Required for training? |
|---|---|---|---|
| `torch` | `import torch` | PyTorch core | **YES — must be CUDA build** |
| `numpy` | `import numpy as np` | Array ops | YES |
| `pandas` | `import pandas as pd` | DataFrames | YES |
| `scipy` | `from scipy.stats import pearsonr` | Pearson r metric | YES |
| `scikit-learn` | `from sklearn.metrics import ...` | MCC/AUC metrics | YES |
| `fair-esm` | `import esm` | ESM-2 protein encoder | **YES — core dependency** |
| `transformers` | `from transformers import AutoTokenizer` | DNABERT-2 encoder (optional path) | Only if `DNA_ENCODER_TYPE="dnabert2"` |
| `viennarna` | `import RNA` | Structure features (live fallback only; cache covers training) | NO if vienna_cache.pkl is present |
| `biopython` | `from Bio import Entrez` | PubMed scraping only | NO for training |
| `requests` | `import requests` | PubMed scraping only | NO for training |

### Packages in the runbook install list but NOT imported anywhere in production code

| Package | Runbook claims it's needed | Actual status |
|---|---|---|
| `peft` | Yes (`pip install peft`) | **Not imported anywhere in scripts/ or models/**. LoRA is implemented from scratch in `models/encoders/protein_encoder.py`. Not needed for training. |
| `einops` | Yes (`pip install einops`) | **Not imported anywhere in scripts/ or models/**. Not needed. |
| `huggingface_hub` | Yes | Not directly imported, but installed as a transitive dependency of `transformers`. OK to include. |

### Critical version constraint

The installed PyTorch build MUST be CUDA-linked, not the default CPU wheel:

```bash
# BAD (what plain `pip install torch` gives you):
pip install torch                  # → torch-2.x.x+cpu

# GOOD (what's needed for the L4):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# or cu124 depending on the VM's installed CUDA runtime
```

**How to verify on the VM:**
```python
import torch
print(torch.__version__)          # must include "+cu121" or "+cu124", NOT "+cpu"
print(torch.cuda.is_available())  # must be True
print(torch.cuda.get_device_name(0))  # must show "NVIDIA L4"
```

### No requirements.txt or environment.yml exists in the repo

All install instructions live only in `gcp_runbook.md § 4`. Anyone setting up a fresh
VM must follow that document manually. Consider adding a `requirements.txt` as a follow-up.

---

## 3. GPU / CUDA COMPATIBILITY FOR L4

### L4 hardware profile
- Architecture: **Ada Lovelace** (NOT Ampere, NOT Turing)
- Compute capability: **8.9**
- VRAM: **24 GB**
- BF16 support: **YES** (Ada Lovelace fully supports bfloat16)
- Minimum driver for CUDA 12.x: ≥ 525.x

### Code-side assumption mismatch

The docstring for `--use-amp` in `scripts/training/train.py:458` says:
> "Enable BF16 automatic mixed precision (A100/Ampere GPUs)."

This is **inaccurate for the L4**. The L4 is Ada Lovelace and **does** support BF16.
The flag will work correctly on the L4 — the docstring is just misleading. Not a blocker.

Similarly, `gcp_runbook.md § 9 Known Issues` states:
> "T4 does NOT support BF16"

This is accurate (T4 = Turing), and implies the `--use-amp` path was only for A100.
On the L4, `--use-amp` is **safe and recommended**.

### CUDA runtime vs driver compatibility

The runbook installs PyTorch with `--index-url https://download.pytorch.org/whl/cu121`,
targeting CUDA 12.1. The L4 VM's NVIDIA driver must be ≥ 525.x to support CUDA 12.1.
A typical GCP Deep Learning VM image (`common-cu121`) ships with a compatible driver.

**Must verify on VM:** `nvidia-smi` output should show `CUDA Version: 12.x` in the
top-right corner. If it shows 11.x, the PyTorch cu121 build will not see the GPU.

---

## 4. HARDCODED PATHS — FULL LIST

### Critical (will break on GCP VM)

| File | Line | Path | Impact |
|---|---|---|---|
| `scripts/data/clean_dataset.py` | 55 | `/Users/shivanshbansal/Downloads/troponin_ntprobnp_aptamers.csv` | Breaks Phase 6 of `clean_dataset.py` if it needs to be re-run on the VM. `clean_dataset.py` is a one-time data prep script (already run); the output `master_dataset_cleaned.csv` is already committed. **Not a training blocker** unless the cleaning pipeline must be rerun. |

### Informational (Colab-only, not used by training scripts)

| File | Context |
|---|---|
| `notebooks/train_colab.ipynb` | Hard-codes `PROJECT_DIR = "/content/CondAptNet"`, Colab-specific paths and `!git` shell cells. This notebook is a Colab helper, not used by `train.py` or any smoke test. Not a blocker on GCP. |

### Confirmed safe (not hardcoded)

`config.py` uses `PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))` throughout.
All data paths (`DATA_RAW`, `DATA_PROCESSED`, `DATA_AUGMENTED`, `CHECKPOINTS_DIR`,
`VIENNA_CACHE`, `PROTEIN_EMB_DIR`) are relative to `PROJECT_ROOT` and will resolve
correctly wherever the repo is checked out on the VM.

---

## 5. DATA AND CHECKPOINT LOCATIONS

### Files that exist on the LOCAL machine (need transfer to VM if VM is fresh)

| File/Dir | Local size | Location |
|---|---|---|
| `data/processed/master_dataset_v2.csv` | 2.9 MB | Split column present, 0 blanks (verified) |
| `data/augmented/tier1_train.csv` + `val.csv` + `test.csv` | 12 MB total | Required by `train.py --augmented-dir` |
| `data/processed/vienna_cache.pkl` | 1.9 MB | Covers ~6489 sequences; avoids live ViennaRNA calls |
| `data/processed/protein_embeddings/` | 64 MB | ESM-2 pre-computed .npy files; avoids 45–90 min re-compute |

**Transfer command (from local machine, once gcloud is available):**
```bash
gcloud compute scp --recurse data/ continuity-l4-dev:~/condaptnet/data/ \
  --zone=us-west1-a \
  --exclude="data/raw" --exclude="data/archive"
```

### Home-directory layout expected on VM (per onboarding runbook)

The task brief mentions `~/checkpoints`, `~/datasets`, `~/models`, `~/logs` should exist.
**These are NOT inside the repo** — they are VM-level directories. Existence must be
confirmed on the VM. The repo uses `models/checkpoints/` (relative), not `~/checkpoints`.
If the onboarding created these at `~/`, they are separate from what `train.py` reads.

Confirm on VM:
```bash
ls ~/checkpoints ~/datasets ~/models ~/logs 2>&1
```

---

## 6. CONFIG VALUES FOR THE L4

### Batch size and protein length

The current `config.py` defaults: `BATCH_SIZE = 32`, `PROT_MAX_TOKENS = 1024`.

| Config | T4 (16 GB) | **L4 (24 GB)** | A100 (40 GB) |
|---|---|---|---|
| `--batch-size` | 16 | **32** | 32 |
| `--max-prot-len` | 512 | **1024** | 1024 |
| `--use-amp` | ❌ (T4=Turing, no BF16) | **✅ (Ada Lovelace, BF16 OK)** | ✅ |
| `--grad-accum` | 2 (to hit eff. 32) | **1** | 1 |

**Recommended launch command for L4:**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/training/train.py \
  --use-amp \
  --batch-size 32 \
  --grad-accum 1 \
  --max-prot-len 1024 \
  --max-epochs 100
```

**Note:** `PYTORCH_ENABLE_MPS_FALLBACK=1` must NOT be set on GCP. `smoke_cuda_check.py`
will warn if it is. Do not copy Mac dev shell exports to GCP launch commands.

### Colab notebook VRAM branching is wrong for L4

`notebooks/train_colab.ipynb` selects settings based on VRAM thresholds:
- ≥ 35 GB → A100 branch (batch=32, prot_len=1024, AMP=True)
- ≥ 14 GB → T4/V100 branch (batch=16, prot_len=512, AMP=False)

L4 at 24 GB falls into the T4 branch, which would select suboptimal settings. The
notebook is not used on GCP (`train.py` is used instead), so this is not a training
blocker — but if anyone runs the notebook on the L4, they should manually override to
the A100-equivalent settings.

### `gcp_setup.sh` and `gcp_runbook.md` reference T4/A100 only

Neither the setup script nor the runbook mentions the L4. The `gcp_setup.sh` only
provisions `nvidia-tesla-t4` or `nvidia-tesla-a100` accelerators. Since the VM
already exists (`continuity-l4-dev`), the provisioning sections are moot, but
the timing estimates and batch-size guidance in the runbook are calibrated for T4
or A100, not L4. Use the L4 column in the table above instead.

---

## 7. SECURITY AND DATA INTEGRITY CHECKS

| Check | Status | Evidence |
|---|---|---|
| `DNABERT2_REVISION` pinned to SHA | ✅ PASS | `config.py` line 45: `DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"` |
| All `torch.load` calls use `weights_only=True` | ✅ PASS | 6 call sites verified: `train.py:617`, `finetune.py:332`, `finetune.py:415`, `smoke_checkpoint.py:203`, `evaluate.py:134`, `dna_encoder_pretrained.py:132` — all use `weights_only=True` |
| `master_dataset_v2.csv` split column fully populated | ✅ PASS | Locally verified: 4499 rows, `split` column present, 0 blanks. Values: train=3512, val=476, test=511 |

---

## VERDICT

**NOT READY TO PROCEED TO TRAINING — the following must be resolved first:**

### Blockers (must be done before `train.py`)

1. **VM runtime verification not done** — `gcloud` is unavailable from this workspace.
   Someone must manually SSH to `continuity-l4-dev` and run the Task-1 command checklist
   (nvidia-smi, pip list, python3 --version, nvcc --version, etc.). This is a prerequisite.

2. **Data files may not be on the VM** — `master_dataset_v2.csv`, augmented splits,
   `vienna_cache.pkl`, and ideally `protein_embeddings/` must be present on the VM before
   training starts. Transfer them from local machine (§ 5 above).

3. **PyTorch CUDA build must be verified** — the installed torch must be a CUDA build
   (`+cu121` or `+cu124`), not CPU-only. Cannot confirm without VM access.

4. **Smoke tests must pass on the VM** — run `smoke_cuda_check.py --require-cuda`,
   `smoke_data_loading.py`, `smoke_checkpoint.py`, and `smoke_training.py` per the runbook.
   Do not launch full training until all smoke tests pass.

### Non-blockers (flag for follow-up, not required before first run)

- `scripts/data/clean_dataset.py:55` hardcodes `/Users/shivanshbansal/Downloads/...` —
  only matters if the cleaning pipeline must be re-run on GCP. The cleaned output
  (`master_dataset_v2.csv`) is already present locally.
- `gcp_runbook.md` and `gcp_setup.sh` are calibrated for T4/A100, not L4. Update
  the recommended launch command documentation after the first run is verified.
- The `--use-amp` docstring says "A100/Ampere" — update to include L4/Ada Lovelace.
- No `requirements.txt` — all installs are prose-only in the runbook. Add one as follow-up.
- `peft` and `einops` are in the runbook install list but not imported in production code.
  They can be omitted from the VM's venv install (saves ~50 MB) but are harmless to include.
