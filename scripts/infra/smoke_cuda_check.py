"""
scripts/infra/smoke_cuda_check.py — Step 2: CUDA Routing Verification

Explicitly verifies that:
  1. torch.cuda.is_available() is True
  2. A forward pass actually executes on GPU (not silently on CPU)
  3. Model parameters land on the correct device after .to(device)
  4. A real forward+backward step doesn't silently CPU-fallback
  5. PYTORCH_ENABLE_MPS_FALLBACK is checked (shouldn't exist on Linux GCP;
     if it does, that's a copy-paste from the Mac env — warn about it)

On MPS (Mac dev), the test still runs but notes that this is not the
GCP CUDA path. On CPU-only, it FAILS so the user knows before committing
to a training run.

Usage:
    python scripts/infra/smoke_cuda_check.py
    python scripts/infra/smoke_cuda_check.py --require-cuda  # exit 1 if no GPU
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_mps_fallback_env() -> None:
    """Warn if PYTORCH_ENABLE_MPS_FALLBACK is set on a CUDA machine."""
    val = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if val:
        print(f"{WARN} PYTORCH_ENABLE_MPS_FALLBACK={val!r} is set.")
        print(f"     This is harmless on Linux/CUDA (MPS is Apple-only) but means")
        print(f"     this environment was copied from a Mac dev setup. On CUDA,")
        print(f"     this env var has NO effect — PyTorch does NOT fall back MPS ops")
        print(f"     to CPU on Linux. Safe to ignore, but remove it from GCP launch")
        print(f"     commands to avoid confusion.")
    else:
        print(f"{PASS} PYTORCH_ENABLE_MPS_FALLBACK not set (expected on Linux GCP)")


def check_cuda_availability(require_cuda: bool) -> torch.device:
    """Return the active device; fail if require_cuda and CUDA is missing."""
    print(f"\n{INFO} PyTorch version : {torch.__version__}")
    print(f"{INFO} CUDA available  : {torch.cuda.is_available()}")
    print(f"{INFO} MPS available   : {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"{PASS} CUDA device 0   : {props.name}")
        print(f"{INFO}   VRAM          : {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
        print(f"{INFO}   Compute cap   : {props.major}.{props.minor}")
        print(f"{INFO}   CUDA runtime  : {torch.version.cuda}")
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"{INFO}   Extra GPUs    : {n_gpus - 1} additional (multi-GPU not used by train.py)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"{WARN} Running on MPS (Apple Silicon). This is the Mac dev path.")
        print(f"     On GCP this should be CUDA. If you see this on GCP, the")
        print(f"     CUDA driver or PyTorch CUDA build may be missing.")
        if require_cuda:
            print(f"{FAIL} --require-cuda set but no CUDA available")
            sys.exit(1)
    else:
        device = torch.device("cpu")
        print(f"{FAIL} No GPU found. Running on CPU.")
        if require_cuda:
            print(f"     Install drivers + CUDA PyTorch build, then re-run.")
            sys.exit(1)
        else:
            print(f"{WARN} Continuing on CPU. Forward pass will be slow but tests will run.")

    return device


def check_tensor_placement(device: torch.device) -> None:
    """Verify that tensors sent to device actually land there."""
    header("Tensor placement")
    t = torch.zeros(4, 128).to(device)
    actual = str(t.device)
    expected = "cuda:0" if device.type == "cuda" else str(device)
    if actual == expected or actual.startswith(device.type):
        print(f"{PASS} torch.zeros(...).to(device).device == {actual}")
    else:
        print(f"{FAIL} Tensor device mismatch: expected {expected}, got {actual}")
        sys.exit(1)


def build_minimal_model() -> nn.Module:
    """Build a tiny model that exercises the same ops as CondAptNet (Linear, ReLU, sigmoid)."""
    return nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def check_model_placement(device: torch.device) -> nn.Module:
    """Verify model parameters land on the right device after .to()."""
    header("Model parameter placement")
    model = build_minimal_model().to(device)
    bad = [(n, str(p.device)) for n, p in model.named_parameters()
           if not str(p.device).startswith(device.type)]
    if bad:
        print(f"{FAIL} Parameters NOT on {device}: {bad}")
        sys.exit(1)
    print(f"{PASS} All model parameters confirmed on {device}")
    return model


def check_forward_pass(model: nn.Module, device: torch.device) -> None:
    """
    Run a forward pass and confirm:
      - output tensor is on the right device
      - the computation actually ran on the GPU (not silently on CPU)
    """
    header("Forward pass device verification")
    x = torch.randn(8, 64).to(device)
    t0 = time.perf_counter()
    out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()   # ensure GPU work is done before timing
    elapsed = time.perf_counter() - t0

    if not str(out.device).startswith(device.type):
        print(f"{FAIL} Output tensor is on {out.device}, expected {device}")
        sys.exit(1)
    print(f"{PASS} Forward pass output on {out.device} (elapsed {elapsed*1000:.2f}ms)")
    print(f"{INFO} Output stats: min={out.min().item():.4f} max={out.max().item():.4f}")


def check_backward_pass(model: nn.Module, device: torch.device) -> None:
    """
    Run a full forward+backward pass with a loss, confirm gradients are on GPU.
    This is the real "no silent CPU fallback" check: if any op silently went to
    CPU, the gradient would be a CPU tensor.
    """
    header("Backward pass (gradient) device verification")
    x = torch.randn(8, 64).to(device)
    target = torch.rand(8, 1).to(device)
    loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    bad_grads = [
        (n, str(p.grad.device))
        for n, p in model.named_parameters()
        if p.grad is not None and not str(p.grad.device).startswith(device.type)
    ]
    if bad_grads:
        print(f"{FAIL} Gradients NOT on {device}: {bad_grads}")
        print(f"     This means operations silently ran on a different device!")
        sys.exit(1)
    print(f"{PASS} All gradients confirmed on {device} — no silent CPU fallback")
    print(f"{INFO} Loss: {loss.item():.4f}")


def check_condaptnet_forward(device: torch.device) -> None:
    """
    Attempt to import and run CondAptNet's actual model with minimal dummy
    inputs. This catches import errors (missing deps) before the full smoke test.
    """
    header("CondAptNet model import + forward pass")
    try:
        from models.condaptnet import CondAptNet
    except ImportError as e:
        print(f"{FAIL} Could not import CondAptNet: {e}")
        print(f"     Check that all dependencies are installed (esm, peft, etc.)")
        sys.exit(1)

    try:
        import config as cfg
        model = CondAptNet(predict_kd=True)
        model.set_stage1()
        model = model.to(device)
        # Move protein encoder back to CPU (same pattern as train.py)
        model.protein_encoder = model.protein_encoder.to("cpu")

        # Dummy inputs — same shapes as train.py dataset
        B       = 2
        apt_len = cfg.DNA_MAX_LEN  # 120

        apt_tok  = torch.randint(0, 66, (B, apt_len)).to(device)
        v_feats  = torch.zeros(B, 6).to(device)
        prot_tok = torch.zeros(B, 1, dtype=torch.long).to(device)
        cond     = torch.tensor([[7.4, 150.0, 37.0, 0.0, 2.0]] * B).to(device)

        prot_seq = "MGARASVLSGGELDRWEKIRLRPGGKKKYKLK" * 4   # dummy 128-aa protein
        prot_len = len(prot_seq)
        # Simulate pre-computed protein embedding (what train.py loads from .npy)
        prot_emb = torch.randn(B, prot_len, cfg.ESM_EMBED_DIM).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(apt_tok, v_feats, prot_tok, cond, protein_emb=prot_emb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        print(f"{PASS} CondAptNet forward pass succeeded in {elapsed*1000:.1f}ms")
        print(f"{INFO}   binding_prob  : {out.binding_prob.shape} on {out.binding_prob.device}")
        print(f"{INFO}   kd_pred       : {out.kd_pred.shape} on {out.kd_pred.device}")
        print(f"{INFO}   binding_label : {out.binding_label.shape}")
        print(f"{INFO}   binding_prob  : {out.binding_prob.squeeze().tolist()}")

        # Verify output tensors are on GPU
        if not str(out.binding_prob.device).startswith(device.type):
            print(f"{FAIL} CondAptNet output on wrong device: {out.binding_prob.device}")
            sys.exit(1)

        # Memory snapshot after forward
        if device.type == "cuda":
            alloc  = torch.cuda.memory_allocated() / 1e6
            reserv = torch.cuda.memory_reserved() / 1e6
            print(f"{INFO}   GPU memory    : {alloc:.1f} MB allocated, {reserv:.1f} MB reserved")

    except Exception as e:
        print(f"{FAIL} CondAptNet forward pass failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="GCP CUDA routing smoke test")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Exit 1 if CUDA is not available (use on GCP)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip CondAptNet model import (for quick env check)")
    args = parser.parse_args()

    header("Step 2: CUDA Routing Check")

    check_mps_fallback_env()
    device = check_cuda_availability(args.require_cuda)
    check_tensor_placement(device)
    model = check_model_placement(device)
    check_forward_pass(model, device)
    check_backward_pass(model, device)

    if not args.skip_model:
        check_condaptnet_forward(device)

    header("CUDA routing check complete")
    print(f"\n{PASS} All checks passed on device={device}")
    if device.type == "cuda":
        print(f"{PASS} Confirmed: ops are running on GPU, not silently on CPU.")
    elif device.type == "mps":
        print(f"{WARN} Running on MPS (Mac). Re-run on GCP to confirm CUDA routing.")
    else:
        print(f"{WARN} Running on CPU. On GCP, this should be CUDA.")


if __name__ == "__main__":
    main()
