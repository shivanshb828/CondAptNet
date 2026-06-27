"""
Symmetric Bidirectional Cross-Attention for CondAptNet.

Both aptamer and protein attend to each other simultaneously. Condition FiLM
parameters modulate the projected features before the attention computation,
so experimental conditions (pH, salt, temp, buffer) influence how the two
sequences reason about each other.

Architecture per direction:
    project to FUSION_DIM → apply FiLM (γ·x + β) → MultiheadAttention

Both directions share the same condition FiLM params but have independent
projection weights and attention layers.

Input shapes:
    aptamer_emb  : [batch, apt_len,  DNA_EMBED_DIM]   (128)
    protein_emb  : [batch, prot_len, ESM_EMBED_DIM]   (480)
    condition    : [batch, 5]                          raw scalars

Output shapes:
    apt_out      : [batch, apt_len,  FUSION_DIM]       (256)
    prot_out     : [batch, prot_len, FUSION_DIM]       (256)

Usage:
    python models/attention/cross_attention.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE,
    DNA_EMBED_DIM, ESM_EMBED_DIM,
    FUSION_DIM, CROSS_ATTN_HEADS, CROSS_ATTN_DROPOUT,
    CONDITION_INPUT_DIM,
)
from models.encoders.condition_encoder import ConditionEncoder


class SymmetricCrossAttention(nn.Module):
    """
    Symmetric bidirectional cross-attention with FiLM condition injection.

    Aptamer attends to protein (apt→prot), and protein attends to aptamer
    (prot→apt) in parallel, each producing FUSION_DIM-dim output tokens.
    """

    def __init__(self) -> None:
        super().__init__()

        # Project both encoder outputs to FUSION_DIM before attention
        self.apt_proj  = nn.Linear(DNA_EMBED_DIM, FUSION_DIM)
        self.prot_proj = nn.Linear(ESM_EMBED_DIM,  FUSION_DIM)

        # Aptamer attends to protein
        self.apt_to_prot_attn = nn.MultiheadAttention(
            embed_dim=FUSION_DIM,
            num_heads=CROSS_ATTN_HEADS,
            dropout=CROSS_ATTN_DROPOUT,
            batch_first=True,
        )

        # Protein attends to aptamer
        self.prot_to_apt_attn = nn.MultiheadAttention(
            embed_dim=FUSION_DIM,
            num_heads=CROSS_ATTN_HEADS,
            dropout=CROSS_ATTN_DROPOUT,
            batch_first=True,
        )

        self.apt_norm  = nn.LayerNorm(FUSION_DIM)
        self.prot_norm = nn.LayerNorm(FUSION_DIM)

        self.condition_encoder = ConditionEncoder()

    def forward(
        self,
        aptamer_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            aptamer_emb : [batch, apt_len,  DNA_EMBED_DIM]
            protein_emb : [batch, prot_len, ESM_EMBED_DIM]
            condition   : [batch, 5]  float32

        Returns:
            apt_out     : [batch, apt_len,  FUSION_DIM]
            prot_out    : [batch, prot_len, FUSION_DIM]
        """
        # Project both to FUSION_DIM
        apt  = self.apt_proj(aptamer_emb)    # [batch, apt_len,  FUSION_DIM]
        prot = self.prot_proj(protein_emb)   # [batch, prot_len, FUSION_DIM]

        # FiLM conditioning — same γ/β applied to both sequences
        cond_vec = self.condition_encoder(condition)          # [batch, 128]
        gamma, beta = self.condition_encoder.get_film_params(
            condition, target_dim=FUSION_DIM, condition_vec=cond_vec
        )                                                     # each [batch, FUSION_DIM]

        # Broadcast FiLM over sequence length: [batch, 1, FUSION_DIM]
        gamma = gamma.unsqueeze(1)
        beta  = beta.unsqueeze(1)
        apt  = gamma * apt  + beta
        prot = gamma * prot + beta

        # Aptamer queries protein (apt→prot cross-attention)
        apt_attended, _  = self.apt_to_prot_attn(
            query=apt, key=prot, value=prot
        )

        # Protein queries aptamer (prot→apt cross-attention)
        prot_attended, _ = self.prot_to_apt_attn(
            query=prot, key=apt, value=apt
        )

        # Residual + LayerNorm
        apt_out  = self.apt_norm(apt  + apt_attended)
        prot_out = self.prot_norm(prot + prot_attended)

        return apt_out, prot_out


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    apt_len    = 50
    prot_len   = 200

    aptamer_emb = torch.randn(batch_size, apt_len,  DNA_EMBED_DIM,
                              dtype=torch.float32).to(DEVICE)
    protein_emb = torch.randn(batch_size, prot_len, ESM_EMBED_DIM,
                              dtype=torch.float32).to(DEVICE)
    # Condition vector: [pH, salt_mM, temp_C, buffer_type, mg_mM]
    condition   = torch.tensor([
        [7.4, 150.0, 37.0, 0.0, 2.0],
        [7.0, 150.0, 25.0, 1.0, 1.5],
        [7.4,  50.0, 37.0, 2.0, 2.0],
        [6.8, 100.0, 30.0, 3.0, 3.0],
    ], dtype=torch.float32).to(DEVICE)

    model = SymmetricCrossAttention().to(DEVICE)
    model.eval()

    with torch.no_grad():
        apt_out, prot_out = model(aptamer_emb, protein_emb, condition)

    assert apt_out.shape  == (batch_size, apt_len,  FUSION_DIM), \
        f"apt_out wrong: {apt_out.shape}"
    assert prot_out.shape == (batch_size, prot_len, FUSION_DIM), \
        f"prot_out wrong: {prot_out.shape}"

    print(f"apt_out shape:  {apt_out.shape}")
    print(f"prot_out shape: {prot_out.shape}")
    print(f"Device: {apt_out.device} | dtype: {apt_out.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
    print("\nSymmetricCrossAttention test passed.")
