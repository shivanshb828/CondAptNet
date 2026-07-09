#!/usr/bin/env bash
# scripts/infra/gcp_setup.sh
#
# Provisions a GCP VM (n1-standard-8 + T4 or a2-highgpu-1g) and installs
# everything CondAptNet needs to run training. Run this ONCE from your local
# machine that has `gcloud` authenticated.
#
# Usage:
#   bash scripts/infra/gcp_setup.sh [--instance-type T4|A100] [--zone ZONE]
#   bash scripts/infra/gcp_setup.sh --instance-type A100 --zone us-central1-a
#
# Environment variables (optional overrides):
#   PROJECT_ID   GCP project ID (defaults to `gcloud config get-value project`)
#   ZONE         GCP zone (default: us-central1-a)
#   INSTANCE_TYPE T4 or A100 (default: T4)
#   VM_NAME      Name of the VM to create (default: condaptnet-smoke)
#   DISK_GB      Boot disk size in GB (default: 200)
#
# After this script completes, connect with:
#   gcloud compute ssh $VM_NAME --zone $ZONE -- -L 8888:localhost:8888
# Then run the smoke tests:
#   bash scripts/infra/run_all_smoke_tests.sh

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_TYPE="${INSTANCE_TYPE:-T4}"
VM_NAME="${VM_NAME:-condaptnet-smoke}"
DISK_GB="${DISK_GB:-200}"
GIT_BRANCH="${GIT_BRANCH:-infra/gcp-smoke-test}"
REPO_URL="${REPO_URL:-}"   # Fill in or pass via env: e.g. https://github.com/ORG/REPO.git

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --zone)          ZONE="$2"; shift 2;;
        --vm-name)       VM_NAME="$2"; shift 2;;
        --disk-gb)       DISK_GB="$2"; shift 2;;
        --repo-url)      REPO_URL="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Machine spec by GPU type ──────────────────────────────────────────────────
# Sizing rationale:
#   Model params at inference:
#     scratch path:  ~45.5M (DNA 3M + ESM-2 35M + LoRA 0.5M + cross-attn/CNN 7M)
#     dnabert2 path: ~160M  (DNABERT-2 117M + ESM-2 35M + LoRA ~0.5M + rest)
#   Memory estimate (float32, batch=32, prot_len=512):
#     Activation map (interaction matrix) for scratch:  ~32 * 50 * 512 * 256 * 4B ≈ 850MB
#     Plus optimizer state (AdamW) ≈ 3x param count    for scratch: ~550MB
#     Total scratch path peak GPU ≈ 4–6 GB → T4 (16GB) fits comfortably.
#     DNABERT-2 path: +117M extra params ≈ +1.4GB → still fits T4; use A100
#     for full A/B run to get headroom and speed.
#   AWS g5.xlarge equivalent (4 vCPU / 16 GB RAM / A10G 24GB VRAM):
#     GCP nearest: n1-standard-4 + T4 (16GB).  Cheaper; T4 is 65% the compute
#     of A10G but the model fits easily, so it's the right smoke-test choice.
#     For the real A/B full training run, upgrade to a2-highgpu-1g (A100-40GB):
#     faster (3-4× A10G for fp32 matmul), 40GB VRAM leaves margin for dnabert2
#     path at batch=32 + prot_len=1024.

case "$INSTANCE_TYPE" in
    T4)
        MACHINE_TYPE="n1-standard-8"
        ACCELERATOR="type=nvidia-tesla-t4,count=1"
        IMAGE_FAMILY="common-cu121"
        IMAGE_PROJECT="deeplearning-platform-release"
        echo "→ Provisioning: $MACHINE_TYPE + T4 (16GB) — smoke test / development"
        ;;
    A100)
        MACHINE_TYPE="a2-highgpu-1g"
        ACCELERATOR="type=nvidia-tesla-a100,count=1"
        IMAGE_FAMILY="common-cu121"
        IMAGE_PROJECT="deeplearning-platform-release"
        echo "→ Provisioning: $MACHINE_TYPE + A100 (40GB) — full A/B training run"
        ;;
    *)
        echo "ERROR: Unknown instance type '$INSTANCE_TYPE'. Choose T4 or A100."
        exit 1
        ;;
esac

echo ""
echo "GCP project : $PROJECT_ID"
echo "Zone        : $ZONE"
echo "VM name     : $VM_NAME"
echo "Disk        : ${DISK_GB}GB"
echo "Branch      : $GIT_BRANCH"
echo ""

# ── Create VM ─────────────────────────────────────────────────────────────────
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "VM $VM_NAME already exists — skipping creation."
else
    echo "Creating VM..."
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="$ACCELERATOR" \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${DISK_GB}GB" \
        --boot-disk-type=pd-ssd \
        --metadata=install-nvidia-driver=True \
        --scopes=cloud-platform
    echo "VM created. Waiting 60s for boot..."
    sleep 60
fi

# ── Install dependencies on the VM ────────────────────────────────────────────
echo "Installing dependencies on $VM_NAME..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s << 'REMOTE_SETUP'
set -euo pipefail

echo "=== System update ==="
sudo apt-get update -qq
sudo apt-get install -y git python3.11 python3.11-venv python3.11-dev \
    build-essential wget curl libhdf5-dev pkg-config

# ViennaRNA — must be built from source (no conda here; we use venv)
if ! python3 -c "import RNA" 2>/dev/null; then
    echo "=== Building ViennaRNA ==="
    cd /tmp
    wget -q https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz
    tar xzf ViennaRNA-2.6.4.tar.gz
    cd ViennaRNA-2.6.4
    ./configure --without-perl --without-ruby --without-doc --prefix=/usr/local \
        --with-python3
    make -j"$(nproc)"
    sudo make install
    cd ~
fi

echo "ViennaRNA: $(python3 -c 'import RNA; print(RNA.__version__)' 2>/dev/null || echo 'NOT FOUND')"

REMOTE_SETUP

# ── Clone or update repo ───────────────────────────────────────────────────────
if [ -n "$REPO_URL" ]; then
    echo "Cloning repo..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s << REMOTE_REPO
set -euo pipefail
if [ ! -d ~/condaptnet ]; then
    git clone "$REPO_URL" ~/condaptnet
fi
cd ~/condaptnet
git fetch origin
git checkout "$GIT_BRANCH"
git pull origin "$GIT_BRANCH"
REMOTE_REPO
else
    echo ""
    echo "NOTE: --repo-url not set. Copy the repo manually:"
    echo "  gcloud compute scp --recurse ./ ${VM_NAME}:~/condaptnet --zone=${ZONE}"
    echo ""
fi

# ── Python environment ─────────────────────────────────────────────────────────
echo "Setting up Python environment on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s << 'REMOTE_PYENV'
set -euo pipefail
cd ~/condaptnet

# CUDA version check — must see 12.x
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2- || echo "NOT FOUND")
echo "CUDA version: $CUDA_VER"
nvidia-smi || echo "WARNING: nvidia-smi failed — driver may not be ready"

# Create venv if needed
if [ ! -d condaptnet_env ]; then
    python3.11 -m venv condaptnet_env
fi
source condaptnet_env/bin/activate
pip install --upgrade pip wheel -q

# PyTorch with CUDA 12.1 support
# NOTE: use the cu121 index — the default PyPI torch is CPU-only
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Project dependencies
pip install \
    fair-esm==2.0.0 \
    transformers==4.40.2 \
    einops \
    huggingface_hub \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    biopython \
    peft \
    -q

# ViennaRNA Python bindings (built in previous step)
pip install viennarna 2>/dev/null || echo "ViennaRNA Python wheel not found — using system build"

python -c "
import torch
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
REMOTE_PYENV

echo ""
echo "Setup complete. Connect with:"
echo "  gcloud compute ssh $VM_NAME --zone $ZONE"
echo ""
echo "Then run smoke tests:"
echo "  cd ~/condaptnet"
echo "  source condaptnet_env/bin/activate"
echo "  bash scripts/infra/run_all_smoke_tests.sh"
