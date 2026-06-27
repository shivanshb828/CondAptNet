"""
Dual Output Head for CondAptNet.

Two branches operating on the CNN's fixed-size feature vector:
    1. Binding head  → scalar probability ∈ [0, 1]  (Sigmoid)
    2. Kd head       → scalar in log10(nM+1) space ≥ 0  (ReLU)

The Kd head is skippable at inference when Kd labels are unavailable,
controlled by predict_kd=True/False in forward().

A parameter-free qualitative label (0=low, 1=medium, 2=high) is always
produced alongside the continuous outputs:
    - Primary:  Kd thresholds in log10(nM+1) space (BIND_LABEL_KD_*)
    - Fallback: binding_prob thresholds when kd_pred is None (BIND_LABEL_PROB_*)

Input shape:
    features     : [batch, CNN_CHANNELS[-1]]   (256-dim from CNNHead)

Output:
    DualHeadOutput named tuple:
        binding_prob  : [batch, 1]   float32   ∈ [0, 1]
        kd_pred       : [batch, 1]   float32   ≥ 0  (log10(nM+1)), or None
        binding_label : [batch, 1]   int64     0=low | 1=medium | 2=high

Usage:
    python models/output/dual_head.py
"""

import math
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, CNN_CHANNELS,
    BIND_LABEL_LOW, BIND_LABEL_MEDIUM, BIND_LABEL_HIGH,
    BIND_LABEL_KD_HIGH_NM, BIND_LABEL_KD_MEDIUM_NM,
    BIND_LABEL_PROB_HIGH, BIND_LABEL_PROB_MEDIUM,
)

# Precompute log10(nM+1) thresholds once at import time
_KD_HIGH_LOG   = math.log10(BIND_LABEL_KD_HIGH_NM   + 1)  # log10(11)  ≈ 1.041
_KD_MEDIUM_LOG = math.log10(BIND_LABEL_KD_MEDIUM_NM + 1)  # log10(1001) ≈ 3.000


class DualHeadOutput(NamedTuple):
    binding_prob  : torch.Tensor           # [batch, 1], sigmoid ∈ [0,1]
    kd_pred       : Optional[torch.Tensor] # [batch, 1], ReLU ≥ 0 (log10(nM+1)), or None
    binding_label : torch.Tensor           # [batch, 1], int64: 0=low|1=medium|2=high


class DualHead(nn.Module):
    """
    Two-branch prediction head over the CNN feature vector.

    binding_head : Linear(256,128) → GELU → Linear(128,1) → Sigmoid
    kd_head      : Linear(256,128) → GELU → Linear(128,1) → ReLU
    """

    def __init__(self, in_features: int = CNN_CHANNELS[-1]) -> None:
        super().__init__()

        self.binding_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.kd_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    @staticmethod
    def _qualify_binding(
        binding_prob: torch.Tensor,
        kd_pred: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Converts continuous outputs to a qualitative label without any learned params.

        Primary (kd_pred available): lower Kd = stronger binding
            < _KD_HIGH_LOG   (< 10 nM)   → 2  high
            < _KD_MEDIUM_LOG (< 1000 nM) → 1  medium
            else                          → 0  low

        Fallback (kd_pred is None): binding_prob thresholds
            > BIND_LABEL_PROB_HIGH   (> 0.66) → 2  high
            > BIND_LABEL_PROB_MEDIUM (> 0.33) → 1  medium
            else                               → 0  low

        Returns: [batch, 1] int64 tensor on same device as inputs.
        """
        if kd_pred is not None:
            dev = kd_pred.device
            low = torch.full_like(kd_pred, BIND_LABEL_LOW, dtype=torch.long)
            med = torch.full_like(kd_pred, BIND_LABEL_MEDIUM, dtype=torch.long)
            hig = torch.full_like(kd_pred, BIND_LABEL_HIGH, dtype=torch.long)
            label = torch.where(kd_pred < _KD_MEDIUM_LOG, med, low)
            label = torch.where(kd_pred < _KD_HIGH_LOG,   hig, label)
        else:
            low = torch.full_like(binding_prob, BIND_LABEL_LOW, dtype=torch.long)
            med = torch.full_like(binding_prob, BIND_LABEL_MEDIUM, dtype=torch.long)
            hig = torch.full_like(binding_prob, BIND_LABEL_HIGH, dtype=torch.long)
            label = torch.where(binding_prob > BIND_LABEL_PROB_MEDIUM, med, low)
            label = torch.where(binding_prob > BIND_LABEL_PROB_HIGH,   hig, label)
        return label

    def forward(
        self,
        features: torch.Tensor,
        predict_kd: bool = True,
    ) -> DualHeadOutput:
        """
        Args:
            features   : [batch, in_features]  float32
            predict_kd : if False, Kd head is skipped and kd_pred=None

        Returns:
            DualHeadOutput(binding_prob, kd_pred, binding_label)
        """
        assert features.is_floating_point(), f"features must be float, got {features.dtype}"
        assert features.dim() == 2, f"Expected [batch, features], got {features.shape}"

        binding_prob = self.binding_head(features)           # [batch, 1]
        kd_pred      = self.kd_head(features) if predict_kd else None
        binding_label = self._qualify_binding(binding_prob, kd_pred)  # [batch, 1] int64

        return DualHeadOutput(
            binding_prob=binding_prob,
            kd_pred=kd_pred,
            binding_label=binding_label,
        )


if __name__ == "__main__":
    from config import BIND_LABEL_NAMES

    torch.manual_seed(42)
    batch_size   = 4
    in_features  = CNN_CHANNELS[-1]   # 256

    features = torch.randn(batch_size, in_features, dtype=torch.float32).to(DEVICE)

    model = DualHead().to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Both heads active
        out = model(features, predict_kd=True)
        assert out.binding_prob.shape  == (batch_size, 1), f"binding_prob wrong: {out.binding_prob.shape}"
        assert out.kd_pred.shape       == (batch_size, 1), f"kd_pred wrong: {out.kd_pred.shape}"
        assert out.binding_label.shape == (batch_size, 1), f"binding_label wrong: {out.binding_label.shape}"
        assert out.binding_prob.min()  >= 0.0 and out.binding_prob.max() <= 1.0, "binding_prob outside [0,1]"
        assert out.kd_pred.min()       >= 0.0, "kd_pred is negative (ReLU failed)"
        assert out.binding_label.dtype == torch.long, "binding_label must be int64"
        assert out.binding_label.min() >= 0 and out.binding_label.max() <= 2, "binding_label out of {0,1,2}"

        print(f"binding_prob  shape: {out.binding_prob.shape}   range [{out.binding_prob.min():.3f}, {out.binding_prob.max():.3f}]")
        print(f"kd_pred       shape: {out.kd_pred.shape}        range [{out.kd_pred.min():.3f}, {out.kd_pred.max():.3f}]")
        labels_str = [BIND_LABEL_NAMES[v.item()] for v in out.binding_label.squeeze(1)]
        print(f"binding_label shape: {out.binding_label.shape}  values {out.binding_label.squeeze().tolist()} → {labels_str}")

        # Kd head skipped → fallback to binding_prob thresholds
        out_no_kd = model(features, predict_kd=False)
        assert out_no_kd.binding_prob.shape  == (batch_size, 1)
        assert out_no_kd.kd_pred             is None, "kd_pred should be None when predict_kd=False"
        assert out_no_kd.binding_label.shape == (batch_size, 1), "binding_label missing when kd_pred=None"
        assert out_no_kd.binding_label.dtype == torch.long
        print(f"predict_kd=False → kd_pred is None, binding_label (prob fallback): {out_no_kd.binding_label.squeeze().tolist()}  ✓")

        # Deterministic label check: force kd_pred to known value
        # kd = 0.5 (< _KD_HIGH_LOG ≈ 1.041) → HIGH (2)
        known_kd   = torch.tensor([[0.5]], dtype=torch.float32).to(DEVICE)
        known_prob = torch.tensor([[0.5]], dtype=torch.float32).to(DEVICE)
        lbl = DualHead._qualify_binding(known_prob, known_kd)
        assert lbl.item() == 2, f"Expected HIGH (2) for kd=0.5, got {lbl.item()}"
        # kd = 2.0 (≥ _KD_HIGH_LOG, < _KD_MEDIUM_LOG ≈ 3.0) → MEDIUM (1)
        lbl = DualHead._qualify_binding(known_prob, torch.tensor([[2.0]]).to(DEVICE))
        assert lbl.item() == 1, f"Expected MEDIUM (1) for kd=2.0, got {lbl.item()}"
        # kd = 4.0 (≥ _KD_MEDIUM_LOG) → LOW (0)
        lbl = DualHead._qualify_binding(known_prob, torch.tensor([[4.0]]).to(DEVICE))
        assert lbl.item() == 0, f"Expected LOW (0) for kd=4.0, got {lbl.item()}"
        # fallback: prob=0.8 → HIGH (2)
        lbl = DualHead._qualify_binding(torch.tensor([[0.8]]).to(DEVICE), None)
        assert lbl.item() == 2, f"Expected HIGH (2) for prob=0.8, got {lbl.item()}"
        # fallback: prob=0.5 → MEDIUM (1)
        lbl = DualHead._qualify_binding(torch.tensor([[0.5]]).to(DEVICE), None)
        assert lbl.item() == 1, f"Expected MEDIUM (1) for prob=0.5, got {lbl.item()}"
        # fallback: prob=0.2 → LOW (0)
        lbl = DualHead._qualify_binding(torch.tensor([[0.2]]).to(DEVICE), None)
        assert lbl.item() == 0, f"Expected LOW (0) for prob=0.2, got {lbl.item()}"
        print("Deterministic label checks passed  ✓")

    print(f"\nDevice: {out.binding_prob.device} | dtype: {out.binding_prob.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
    print("\nDualHead test passed.")
