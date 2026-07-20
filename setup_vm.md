# VM Environment Setup — continuity-l4-dev

**GCP Project:** continuity-ai-502519  
**Zone:** us-west1-a  
**Machine type:** g2-standard-4 (1× NVIDIA L4)  
**Access method:** `gcloud compute ssh` via IAP tunnel only

> **Reproducibility note:** This environment is designed to be fully reproducible in
> backup zones per Swathi's plan — no system-level dependencies, everything lives in
> the user's home directory.

---

## Hard constraints (apply to every future task on this VM)

- Never create another VM, never change machine type, GPU, or disk config.
- Never use Spot/preemptible, never touch quotas/IAM/service accounts.
- Never enable additional cloud APIs, never create Vertex AI jobs.
- Never run `sudo`, never change SSH config, never touch NVIDIA drivers.
- All packages and environments live in `$HOME` only — no system-wide installs.
- No system Anaconda/Miniconda — Miniforge3 in `$HOME/miniforge3` is the only conda.
- No CUDA driver or toolkit changes — the pre-installed drivers are used as-is.

---

## 1. Miniforge install

Downloaded and installed Miniforge3 in batch (non-interactive) mode to the user home
directory — no `sudo` required:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
```

The `-b` flag runs in batch mode (no prompts, no `conda init` auto-modification of
`.bashrc`). The install target is `$HOME/miniforge3`.

---

## 2. Add conda to .bashrc

Manually sourced conda in `.bashrc` so it is available in all interactive and
non-interactive shells:

```bash
echo 'source $HOME/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

Verified line present in `.bashrc` (live check 2026-07-20):

```
source $HOME/miniforge3/etc/profile.d/conda.sh
```

---

## 3. Create the conda environment

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -n condaptnet python=3.11 -y
conda activate condaptnet
```

---

## 4. Install PyTorch with CUDA 13.2 support

Exact command run — matches the pre-installed CUDA 13.2 driver on the L4:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Verified result (live check 2026-07-20):

```
torch==2.13.0+cu132
torchvision==0.28.0+cu132
```

GPU verification:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
# 2.13.0+cu132 True NVIDIA L4
```

---

## 5. Install project dependencies

```bash
conda activate condaptnet
pip install fair-esm biopython pandas scikit-learn scipy ViennaRNA \
    beautifulsoup4 requests lxml pdfplumber openpyxl transformers \
    huggingface_hub pytest
```

---

## 6. Full verified dependency list

Captured via `pip list --format=freeze` on the live VM (2026-07-20). This is the
authoritative record of the installed environment:

```
annotated-doc==0.0.4
anyio==4.14.2
beautifulsoup4==4.15.0
biopython==1.87
bs4==0.0.2
certifi==2026.6.17
cffi==2.1.0
charset-normalizer==3.4.9
click==8.4.2
cryptography==49.0.0
cuda-bindings==13.0.3
cuda-pathfinder==1.2.2
cuda-toolkit==13.2.1
et_xmlfile==2.0.0
fair-esm==2.0.0
filelock==3.29.0
fsspec==2026.4.0
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.24.0
idna==3.18
iniconfig==2.3.0
Jinja2==3.1.6
joblib==1.5.3
lxml==6.1.1
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.24.0
networkx==3.6.1
numpy==2.4.4
nvidia-cublas==13.4.0.1
nvidia-cuda-cupti==13.2.75
nvidia-cuda-nvrtc==13.2.78
nvidia-cuda-runtime==13.2.75
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.2.0.46
nvidia-cufile==1.17.1.22
nvidia-curand==10.4.2.55
nvidia-cusolver==12.2.0.1
nvidia-cusparse==12.7.10.1
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.2.78
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.2.75
openpyxl==3.1.5
packaging==26.2
pandas==3.0.3
pdfminer.six==20260107
pdfplumber==0.11.10
pillow==12.2.0
pip==26.1.2
pluggy==1.6.0
pycparser==3.0
Pygments==2.20.0
pypdfium2==5.12.1
pytest==9.1.1
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.7.19
reportlab==5.0.0
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.17.1
setuptools==83.0.0
shellingham==1.5.4
six==1.17.0
soupsieve==2.9
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.13.0+cu132
torchvision==0.28.0+cu132
tqdm==4.69.0
transformers==5.14.1
triton==3.7.1
typer==0.27.0
typing_extensions==4.15.0
urllib3==2.7.0
ViennaRNA==2.7.2
wheel==0.47.0
```

---

## 7. Connecting to the VM

```bash
# Check status
gcloud compute instances describe continuity-l4-dev \
  --project=continuity-ai-502519 --zone=us-west1-a \
  --format="value(status)"

# Start (if TERMINATED)
gcloud compute instances start continuity-l4-dev \
  --project=continuity-ai-502519 --zone=us-west1-a

# SSH via IAP tunnel
gcloud compute ssh continuity-l4-dev \
  --project=continuity-ai-502519 --zone=us-west1-a --tunnel-through-iap

# Stop when done (always stop — never leave running unattended)
gcloud compute instances stop continuity-l4-dev \
  --project=continuity-ai-502519 --zone=us-west1-a
```

---

## 8. Quick sanity check on login

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate condaptnet
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA L4
```
