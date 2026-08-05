"""
CondAptNet loss function.

Combined binary cross-entropy (binding) + masked MSE (Kd regression).

    total = BCE_WEIGHT * bce  +  KD_WEIGHT * mse_kd

BCE uses manual minority-class weighting (POSITIVE_CLASS_WEIGHT = 3.0,
applied to NEGATIVES — the minority class in training) to compensate for
class imbalance. Train split is 57% positive / 43% negative, so negatives
are the minority class. Kd MSE is computed only over rows where a
ground-truth Kd value exists (non-NaN), so batches with no Kd labels
contribute zero Kd loss rather than erroring out.

Kd values are expected in log10(nM + 1) space:
    - training target: log10(Kd_nM + 1)   (computed in dataset)
    - model output: kd_pred from DualHead  (ReLU ≥ 0, same space)

Usage:
    criterion = CondAptNetLoss()
    loss, bce_loss, kd_loss = criterion(binding_prob, labels, kd_pred, kd_targets)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import POSITIVE_CLASS_WEIGHT, BCE_WEIGHT, KD_WEIGHT


class CondAptNetLoss(nn.Module):
    """
    Args:
        pos_weight  : multiplier on positive-class BCE terms (default 3.0)
        bce_weight  : scale factor on the BCE component in the combined loss
        kd_weight   : scale factor on the Kd MSE component
    """

    def __init__(
        self,
        pos_weight: float = POSITIVE_CLASS_WEIGHT,
        bce_weight: float = BCE_WEIGHT,
        kd_weight:  float = KD_WEIGHT,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.kd_weight  = kd_weight

    def forward(
        self,
        binding_prob:  torch.Tensor,            # [B, 1]  sigmoid output ∈ (0,1)
        labels:        torch.Tensor,            # [B, 1]  float {0.0, 1.0}
        kd_pred:       torch.Tensor | None,     # [B, 1]  ReLU output ≥ 0, or None
        kd_targets:    torch.Tensor | None,     # [B, 1]  log10(nM+1), NaN for missing
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss : scalar — back-prop target
            bce_loss   : scalar — for logging
            kd_loss    : scalar — for logging (0.0 if no Kd labels in batch)
        """
        # ── Weighted BCE ──────────────────────────────────────────────────────
        # weight tensor: pos_weight for NEGATIVE rows (minority class, 43% of
        # train), 1.0 for positives. Train is 57% positive / 43% negative, so
        # negatives are the minority and need the upweight. Prior setting had
        # pos_weight applied to the majority class (positives), which amplified
        # the existing positive bias and caused TN=0 across epochs 1-3.
        pw = torch.where(labels.bool(),
                         torch.ones_like(labels),
                         torch.full_like(labels, self.pos_weight))
        bce_loss = F.binary_cross_entropy(binding_prob, labels, weight=pw)

        # ── Masked MSE on Kd ─────────────────────────────────────────────────
        kd_loss = torch.tensor(0.0, device=binding_prob.device)
        if kd_pred is not None and kd_targets is not None:
            mask = ~torch.isnan(kd_targets)
            if mask.any():
                kd_loss = F.mse_loss(kd_pred[mask], kd_targets[mask])

        total_loss = self.bce_weight * bce_loss + self.kd_weight * kd_loss
        return total_loss, bce_loss, kd_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    B = 8

    binding_prob = torch.sigmoid(torch.randn(B, 1))
    labels       = torch.tensor([[1],[0],[1],[1],[0],[0],[1],[0]], dtype=torch.float32)
    kd_pred      = torch.relu(torch.randn(B, 1))
    kd_targets   = torch.tensor([[2.1],[float("nan")],[1.5],[3.0],
                                  [float("nan")],[float("nan")],[2.8],[float("nan")]],
                                 dtype=torch.float32)

    criterion = CondAptNetLoss()
    total, bce, kd = criterion(binding_prob, labels, kd_pred, kd_targets)

    print(f"total_loss : {total.item():.4f}")
    print(f"bce_loss   : {bce.item():.4f}")
    print(f"kd_loss    : {kd.item():.4f}")
    assert total.item() > 0
    assert bce.item() > 0
    assert kd.item() > 0

    # kd_pred=None → no Kd loss
    total_no_kd, _, kd_zero = criterion(binding_prob, labels, None, None)
    assert kd_zero.item() == 0.0, "kd_loss should be 0 when kd_pred is None"

    # all-NaN kd_targets → no Kd loss
    all_nan = torch.full((B, 1), float("nan"))
    total_nan_kd, _, kd_zero2 = criterion(binding_prob, labels, kd_pred, all_nan)
    assert kd_zero2.item() == 0.0, "kd_loss should be 0 when all targets are NaN"

    print("\nCondAptNetLoss tests passed.")
