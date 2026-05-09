"""
Dual Output Head for CondAptNet.

Two branches operating on the CNN's fixed-size feature vector:
    1. Binding head  → scalar probability ∈ [0, 1]  (Sigmoid)
    2. Kd head       → scalar in log-nM space ≥ 0    (ReLU)

The Kd head is skippable at inference when Kd labels are unavailable,
controlled by predict_kd=True/False in forward().

Input shape:
    features     : [batch, CNN_CHANNELS[-1]]   (256-dim from CNNHead)

Output:
    DualHeadOutput named tuple:
        binding_prob : [batch, 1]   float32  ∈ [0, 1]
        kd_pred      : [batch, 1]   float32  ≥ 0  (log-nM), or None

Usage:
    python models/output/dual_head.py
"""

import sys
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEVICE, CNN_CHANNELS


class DualHeadOutput(NamedTuple):
    binding_prob: torch.Tensor          # [batch, 1], sigmoid ∈ [0,1]
    kd_pred:      Optional[torch.Tensor]  # [batch, 1], ReLU ≥ 0, or None


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
            DualHeadOutput(binding_prob, kd_pred)
        """
        assert features.dtype == torch.float32, "features must be float32"
        assert features.dim() == 2, f"Expected [batch, features], got {features.shape}"

        binding_prob = self.binding_head(features)    # [batch, 1]

        kd_pred = self.kd_head(features) if predict_kd else None

        return DualHeadOutput(binding_prob=binding_prob, kd_pred=kd_pred)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size   = 4
    in_features  = CNN_CHANNELS[-1]   # 256

    features = torch.randn(batch_size, in_features, dtype=torch.float32).to(DEVICE)

    model = DualHead().to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Both heads active
        out = model(features, predict_kd=True)
        assert out.binding_prob.shape == (batch_size, 1), \
            f"binding_prob wrong: {out.binding_prob.shape}"
        assert out.kd_pred.shape == (batch_size, 1), \
            f"kd_pred wrong: {out.kd_pred.shape}"
        assert out.binding_prob.min() >= 0.0 and out.binding_prob.max() <= 1.0, \
            "binding_prob outside [0,1]"
        assert out.kd_pred.min() >= 0.0, "kd_pred is negative (ReLU failed)"

        print(f"binding_prob shape: {out.binding_prob.shape}  range [{out.binding_prob.min():.3f}, {out.binding_prob.max():.3f}]")
        print(f"kd_pred shape:      {out.kd_pred.shape}      range [{out.kd_pred.min():.3f}, {out.kd_pred.max():.3f}]")

        # Kd head skipped
        out_no_kd = model(features, predict_kd=False)
        assert out_no_kd.binding_prob.shape == (batch_size, 1)
        assert out_no_kd.kd_pred is None, "kd_pred should be None when predict_kd=False"
        print(f"predict_kd=False → kd_pred is None  ✓")

    print(f"\nDevice: {out.binding_prob.device} | dtype: {out.binding_prob.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
    print("\nDualHead test passed.")
