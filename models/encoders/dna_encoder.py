"""
Native DNA Transformer Encoder for CondAptNet.

Encodes DNA aptamer sequences as dense embeddings without T→U conversion.
Combines:
  1. k-mer token embeddings  (learnable)
  2. Sinusoidal positional encoding
  3. ViennaRNA structure features injected as a per-sequence scalar projection
  4. 6-layer Transformer encoder

Input shapes:
    token_ids     : [batch, seq_len]           LongTensor of 3-mer IDs
    vienna_feats  : [batch, 6]                 float32 — mfe, stem_count,
                                               loop_count, bp_prob_mean,
                                               bp_prob_max, seq_len_norm
Output shape:
    embeddings    : [batch, seq_len, DNA_EMBED_DIM]

Usage:
    python models/encoders/dna_encoder.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, DNA_VOCAB_SIZE, DNA_EMBED_DIM, DNA_NUM_LAYERS,
    DNA_NUM_HEADS, DNA_FF_DIM, DNA_DROPOUT, DNA_MAX_LEN, DNA_PAD_ID
)

VIENNA_FEAT_DIM = 6   # mfe, stem_count, loop_count, bp_prob_mean, bp_prob_max, seq_len_norm


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal PE added to token embeddings."""

    def __init__(self, embed_dim: int, max_len: int = DNA_MAX_LEN, dropout: float = DNA_DROPOUT) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float)
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ViennaProjection(nn.Module):
    """
    Projects scalar ViennaRNA features into the embedding space so they can
    be added as a sequence-level bias to every token in the aptamer.
    """

    def __init__(self, in_dim: int = VIENNA_FEAT_DIM, out_dim: int = DNA_EMBED_DIM) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, vienna_feats: torch.Tensor) -> torch.Tensor:
        # vienna_feats: [batch, VIENNA_FEAT_DIM]
        # Returns:      [batch, 1, DNA_EMBED_DIM]  (broadcast over seq_len)
        return self.proj(vienna_feats).unsqueeze(1)


class DNAEncoder(nn.Module):
    """
    6-layer Transformer encoder operating on native DNA k-mer tokens.

    Args:
        use_vienna: if True, expects vienna_feats to be passed in forward().
                    Set False during inference when structure features aren't available.
    """

    def __init__(self, use_vienna: bool = True) -> None:
        super().__init__()
        self.use_vienna = use_vienna

        self.token_embedding = nn.Embedding(
            num_embeddings=DNA_VOCAB_SIZE,
            embedding_dim=DNA_EMBED_DIM,
            padding_idx=DNA_PAD_ID,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(DNA_EMBED_DIM)

        if use_vienna:
            self.vienna_proj = ViennaProjection()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DNA_EMBED_DIM,
            nhead=DNA_NUM_HEADS,
            dim_feedforward=DNA_FF_DIM,
            dropout=DNA_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=DNA_NUM_LAYERS,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(DNA_EMBED_DIM)

    def _make_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """True where token is PAD (transformer ignores these positions)."""
        return token_ids == DNA_PAD_ID

    def forward(
        self,
        token_ids: torch.Tensor,
        vienna_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids    : [batch, seq_len]      LongTensor
            vienna_feats : [batch, VIENNA_FEAT_DIM] float32 (required if use_vienna=True)

        Returns:
            embeddings   : [batch, seq_len, DNA_EMBED_DIM]
        """
        assert token_ids.dtype == torch.long, "token_ids must be LongTensor"
        assert token_ids.dim() == 2, f"Expected [batch, seq_len], got {token_ids.shape}"

        x = self.token_embedding(token_ids)   # [batch, seq_len, embed_dim]
        x = self.pos_encoding(x)

        if self.use_vienna and vienna_feats is not None:
            assert vienna_feats.dtype == torch.float32, "vienna_feats must be float32"
            v = self.vienna_proj(vienna_feats)  # [batch, 1, embed_dim]
            x = x + v

        padding_mask = self._make_padding_mask(token_ids)  # [batch, seq_len]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.out_norm(x)

        assert x.shape == (*token_ids.shape, DNA_EMBED_DIM), f"Shape error: {x.shape}"
        return x


if __name__ == "__main__":
    import torch
    from config import DEVICE, DNA_EMBED_DIM

    batch_size = 4
    seq_len    = 50   # number of 3-mer tokens (= aptamer_len - 2)

    token_ids    = torch.randint(2, DNA_VOCAB_SIZE, (batch_size, seq_len), dtype=torch.long).to(DEVICE)
    vienna_feats = torch.randn(batch_size, VIENNA_FEAT_DIM, dtype=torch.float32).to(DEVICE)

    # Sprinkle some PADs to verify masking
    token_ids[0, 40:] = DNA_PAD_ID

    model = DNAEncoder(use_vienna=True).to(DEVICE)
    model.eval()

    with torch.no_grad():
        out = model(token_ids, vienna_feats)

    expected_shape = (batch_size, seq_len, DNA_EMBED_DIM)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"DNAEncoder test passed. Output shape: {out.shape}")
    print(f"Device: {out.device}  | dtype: {out.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")
