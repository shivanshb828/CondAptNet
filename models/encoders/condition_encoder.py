"""
Condition Encoder for CondAptNet — FiLM conditioning MLP.

Encodes experimental conditions [pH, salt_mM, temp_C, buffer_type, mg_mM] into a
128-dim vector, then projects that vector into FiLM scale (γ) and shift (β)
parameters used to modulate cross-attention feature maps. mg_mM (divalent Mg2+)
is a separate input because it critically affects aptamer folding.

Architecture (CONDITION_INPUT_DIM=5 → CONDITION_HIDDEN=64 → CONDITION_DIM=128):
    Linear(5, 64) → GELU → Linear(64, 128) → GELU → 128-dim condition vector
    get_film_params(target_dim) → Linear(128, target_dim*2) → γ [target_dim],
                                                                β [target_dim]

Input shapes:
    condition : [batch, 5]   float32  — [pH, salt_mM, temp_C, buffer_type, mg_mM]

Output shapes:
    encode()         → [batch, CONDITION_DIM]          (128-dim)
    get_film_params() → gamma [batch, target_dim],
                        beta  [batch, target_dim]

Usage:
    python models/encoders/condition_encoder.py
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEVICE, CONDITION_INPUT_DIM, CONDITION_HIDDEN, CONDITION_DIM, FUSION_DIM


class ConditionEncoder(nn.Module):
    """
    MLP that encodes a 4-element condition vector into 128-dim FiLM parameters.

    get_film_params(target_dim) returns (γ, β) each of shape [batch, target_dim].
    FiLM modulation: x' = γ * x + β  applied element-wise to feature maps.
    """

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(CONDITION_INPUT_DIM, CONDITION_HIDDEN),
            nn.GELU(),
            nn.Linear(CONDITION_HIDDEN, CONDITION_DIM),
            nn.GELU(),
        )

        # FiLM projection heads keyed by str(target_dim).
        # Pre-register FUSION_DIM so checkpoint keys are stable across runs
        # (lazy init creates the key on first forward pass, which breaks
        #  load_state_dict if the head wasn't used before saving).
        self._film_heads: dict[int, nn.Linear] = nn.ModuleDict({
            str(FUSION_DIM): nn.Linear(CONDITION_DIM, FUSION_DIM * 2),
        })

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition : [batch, 5]  float32  — [pH, salt_mM, temp_C, buffer_type, mg_mM]

        Returns:
            [batch, CONDITION_DIM]  float32
        """
        assert condition.shape[-1] == CONDITION_INPUT_DIM, (
            f"Expected condition dim {CONDITION_INPUT_DIM}, got {condition.shape[-1]}"
        )
        assert condition.dtype == torch.float32, "condition must be float32"
        return self.mlp(condition)

    def get_film_params(
        self,
        condition: torch.Tensor,
        target_dim: int,
        condition_vec: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project condition into FiLM scale γ and shift β for a given feature dim.

        Args:
            condition     : [batch, 5]            raw condition input
            target_dim    : int                   dim of the feature map to modulate
            condition_vec : [batch, CONDITION_DIM] pre-computed (skips encode if given)

        Returns:
            gamma : [batch, target_dim]
            beta  : [batch, target_dim]
        """
        if condition_vec is None:
            condition_vec = self.forward(condition)  # [batch, CONDITION_DIM]

        key = str(target_dim)
        if key not in self._film_heads:
            head = nn.Linear(CONDITION_DIM, target_dim * 2).to(condition_vec.device)
            self._film_heads[key] = head

        film_out = self._film_heads[key](condition_vec)   # [batch, target_dim*2]
        gamma, beta = film_out.chunk(2, dim=-1)           # each [batch, target_dim]
        return gamma, beta


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4

    # Condition vector: [pH, salt_mM, temp_C, buffer_type, mg_mM]
    condition = torch.tensor([
        [7.4, 150.0, 37.0, 0.0, 2.0],  # physiological defaults
        [7.0, 150.0, 25.0, 1.0, 1.5],
        [7.4,  50.0, 37.0, 2.0, 2.0],
        [6.8, 100.0, 30.0, 3.0, 3.0],
    ], dtype=torch.float32).to(DEVICE)

    model = ConditionEncoder().to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Test encode
        cond_vec = model(condition)
        assert cond_vec.shape == (batch_size, CONDITION_DIM), \
            f"Expected ({batch_size}, {CONDITION_DIM}), got {cond_vec.shape}"
        print(f"Encode output shape:  {cond_vec.shape}")

        # Test FiLM params at FUSION_DIM=256
        from config import FUSION_DIM
        gamma, beta = model.get_film_params(condition, target_dim=FUSION_DIM)
        assert gamma.shape == (batch_size, FUSION_DIM), \
            f"Expected ({batch_size}, {FUSION_DIM}), got {gamma.shape}"
        assert beta.shape == (batch_size, FUSION_DIM), \
            f"Expected ({batch_size}, {FUSION_DIM}), got {beta.shape}"
        print(f"FiLM gamma shape:     {gamma.shape}")
        print(f"FiLM beta  shape:     {beta.shape}")

        # Test FiLM params at DNA_EMBED_DIM=128
        from config import DNA_EMBED_DIM
        gamma2, beta2 = model.get_film_params(condition, target_dim=DNA_EMBED_DIM)
        assert gamma2.shape == (batch_size, DNA_EMBED_DIM)
        print(f"FiLM gamma (128) shape: {gamma2.shape}")

    print(f"\nDevice: {cond_vec.device} | dtype: {cond_vec.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
    print("\nConditionEncoder test passed.")
