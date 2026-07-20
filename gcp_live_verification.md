# GCP Live Verification — continuity-l4-dev

**Generated:** 2026-07-15  
**Target VM:** `continuity-l4-dev` — NVIDIA L4, zone `us-west1-a`  
**Task scope:** Runtime SSH verification of Tasks 1–5 from the task brief

---

## SSH CONNECTION STATUS — BLOCKED

**The live verification could not be completed.** SSH to `continuity-l4-dev` fails
with a hard DNS error:

```
ssh shivanshbansal@continuity-l4-dev
→ ssh: Could not resolve hostname continuity-l4-dev: nodename nor servname provided,
  or not known
```

**Root cause:** `continuity-l4-dev` is a GCP compute instance name, not a
DNS-resolvable public hostname. Connecting to it requires one of:
1. `gcloud compute ssh continuity-l4-dev --zone=us-west1-a` — requires the gcloud CLI
2. Direct `ssh <external-ip>` with the VM's actual external IP address and an
   authorized SSH key

**What was checked and found missing:**

| Prerequisite | Status |
|---|---|
| `gcloud` CLI in PATH | Not found — checked `which gcloud`, full PATH, Homebrew, miniconda, conda, bash/zsh login shells |
| `~/.config/gcloud/` directory | Not present — gcloud has never been installed or authenticated on this machine |
| `~/.ssh/google_compute_engine` key | Not found — no GCP-generated SSH key pair |
| `~/.ssh/config` host entry for `continuity-l4-dev` or a GCP IP | Not found |
| `~/.ssh/known_hosts` GCP entry | Not found (only `hoffman2.idre.ucla.edu` present) |

**Per the escalation rule in the task brief:** SSH access fails to open. Reporting
here rather than attempting workarounds, per explicit instruction.

---

## HOW TO UNBLOCK

One of these must be in place before the live verification can run:

**Option A — Install gcloud CLI (recommended, matches runbook):**
```bash
# On local Mac:
brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project <your-gcp-project-id>
gcloud compute ssh continuity-l4-dev --zone=us-west1-a
```

**Option B — Direct SSH with static IP (if the VM has a static external IP):**
```bash
# Swathi or whoever set up the VM can provide:
# 1. The VM's external IP (from GCP Console → Compute Engine → VM instances)
# 2. Confirm the authorized SSH key (add ~/.ssh/id_ed25519.pub or similar to the VM's metadata)
# Then:
ssh shivanshbansal@<external-ip>
```

---

## COMMANDS TO RUN ONCE ACCESS IS ESTABLISHED

Run these in order. Report full output for each. Stop and escalate if any hit an
escalation condition (nvidia-smi fails, permission denied, etc.).

### Task 1 — Environment verification

```bash
whoami
hostname
pwd
nvidia-smi
df -h /
free -h
nproc
command -v conda || true
python3 --version
git --version
git lfs version
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version 2>/dev/null || echo "nvcc not found"
cat /etc/os-release
```

### Task 1 continued — PyTorch CUDA build check

```bash
python3 -c "
import torch
print('torch version :', torch.__version__)
print('CUDA version  :', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device name   :', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('Compute cap   :', f'{props.major}.{props.minor}')
    free, total = torch.cuda.mem_get_info()
    print('VRAM          :', f'{total/1e9:.1f} GB total, {free/1e9:.1f} GB free')
"
```

Expected output when healthy:
```
torch version : 2.2.2+cu121  (or +cu124; must NOT be +cpu)
CUDA version  : 12.1 (or 12.4)
CUDA available: True
Device name   : NVIDIA L4
Compute cap   : 8.9
VRAM          : 24.6 GB total, ~24.x GB free
```

**If `torch.__version__` ends in `+cpu`** → PyTorch was installed without CUDA support.
This is silent — `torch.cuda.is_available()` returns False, training silently runs on
CPU. Reinstall with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
But do NOT install in this task — just report the mismatch.

### Task 2 — Package gap check

```bash
python3 -c "
packages = [
    'torch', 'numpy', 'pandas', 'scipy', 'sklearn',
    'esm', 'transformers', 'RNA', 'Bio', 'requests',
]
for p in packages:
    try:
        mod = __import__(p)
        version = getattr(mod, '__version__', 'no __version__')
        print(f'[PRESENT] {p} == {version}')
    except ImportError:
        print(f'[MISSING] {p}')
"
```

Also run: `pip list | grep -Ei "torch|esm|transformers|viennarna|biopython|scikit|scipy|numpy|pandas|requests"`

### Task 3 — Home directory layout

```bash
ls ~/projects/ 2>/dev/null || echo "MISSING: ~/projects"
ls ~/datasets/ 2>/dev/null || echo "MISSING: ~/datasets"
ls ~/checkpoints/ 2>/dev/null || echo "MISSING: ~/checkpoints"
ls ~/models/ 2>/dev/null || echo "MISSING: ~/models"
ls ~/logs/ 2>/dev/null || echo "MISSING: ~/logs"

# Repo location and git status:
ls ~/projects/condaptnet/ 2>/dev/null || ls ~/condaptnet/ 2>/dev/null || echo "MISSING: repo not found"
cd ~/projects/condaptnet 2>/dev/null || cd ~/condaptnet 2>/dev/null
git status
git branch
git log --oneline -5
```

### Task 4 — Disk space

```bash
df -h /
# Should show >> 80 MB free; the data transfer needs ~80 MB:
# data/augmented/ (12 MB) + master_dataset_v2.csv (2.9 MB)
# + vienna_cache.pkl (1.9 MB) + protein_embeddings/ (64 MB)
```

---

## WHAT CAN STILL BE ANSWERED FROM STATIC ANALYSIS

All static findings from `gcp_readiness_findings.md` remain valid. Summary of the
items NOT blocked by VM access:

| Check | Verdict | Source |
|---|---|---|
| `DNABERT2_REVISION` pinned to SHA | ✅ PASS | `config.py` line 45 |
| All `torch.load` use `weights_only=True` | ✅ PASS | 6 call sites verified |
| `master_dataset_v2.csv` split column — 0 blanks | ✅ PASS | Verified locally |
| Hardcoded macOS path in `clean_dataset.py:55` | ⚠️ NON-BLOCKER | Script already ran; output committed |
| Package list — what the code actually imports | ✅ Complete | See `gcp_readiness_findings.md § 2` |
| L4 config recommendation (`--use-amp`, prot_len=1024) | ✅ Complete | See `gcp_readiness_findings.md § 6` |

---

## UPDATED READINESS VERDICT

**NOT READY — blocked on VM access.**

Before any training or smoke tests can happen:

1. **[BLOCKED] SSH access to `continuity-l4-dev` must be established.** Either install
   gcloud (Option A above) or get the VM's external IP + SSH key (Option B above).
   This is Swathi's domain — one of the escalation items.

2. **[PENDING] All Task-1 runtime checks must be run and pass** once access is
   established: nvidia-smi confirms L4 + driver, PyTorch confirms CUDA build (+cu121),
   pip list confirms all required packages are present.

3. **[PENDING] Data files must be transferred** to the VM once disk space is confirmed.

4. **[PENDING] Smoke tests must pass** before full training launch.

The static analysis items (security checks, config values, path audit) are all clear
and do not need to be re-run once VM access is obtained.
