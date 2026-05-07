"""
Protein Encoder for CondAptNet — ESM-2 + LoRA fine-tuning.

Wraps Meta's ESM-2 (esm2_t12_35M_UR50D, 480-dim output) with LoRA adapters
injected into every attention layer. Full ESM-2 weights are frozen; only the
LoRA matrices (rank=8) are trained, updating <1% of parameters.

Input shapes:
    tokens    : [batch, protein_len]   LongTensor  (ESM-2 alphabet token IDs)
    attn_mask : [batch, protein_len]   LongTensor  1=real token, 0=pad

Output shape:
    embeddings : [batch, protein_len, ESM_EMBED_DIM]   float32
                 (ESM-2 per-residue representations, LoRA-adapted)

Usage (standalone test):
    python models/encoders/protein_encoder.py
"""

import sys
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE, ESM_MODEL_NAME, ESM_EMBED_DIM,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)

try:
    import esm as esm_lib
    _ESM_AVAILABLE = True
except ImportError:
    _ESM_AVAILABLE = False


# ── LoRA implementation ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adaptation.
    The original weight is frozen; only A and B matrices are trained.

    W' = W + (alpha/rank) * B @ A
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
        dropout: float = LORA_DROPOUT,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_features  = linear.in_features
        out_features = linear.out_features

        self.lora_A  = nn.Linear(in_features, rank, bias=False)
        self.lora_B  = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialise: A ~ N(0, 1/sqrt(rank)), B = 0  →  delta W = 0 at init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base    = self.linear(x)
        delta   = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base + delta


def inject_lora(model: nn.Module, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> int:
    """
    Replace every `query` and `value` projection in ESM-2 attention layers
    with a LoRALinear wrapper. Returns the count of replaced layers.
    """
    replaced = 0
    for name, module in model.named_modules():
        # ESM-2 attention projections are named 'q_proj', 'k_proj', 'v_proj'
        # inside `esm.model.esm2.ESM2` TransformerLayer → SelfAttention
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank=rank, alpha=alpha)
            replaced += 1
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank=rank, alpha=alpha)
            replaced += 1
    return replaced


# ── Protein Encoder ───────────────────────────────────────────────────────────

class ProteinEncoder(nn.Module):
    """
    ESM-2 protein language model with LoRA adapters for efficient fine-tuning.

    Stage 1 (pretraining): call freeze_esm() to train only LoRA matrices.
    Stage 2 (fine-tuning): call unfreeze_lora() — LoRA matrices already
                           trainable; nothing else changes.
    """

    def __init__(self) -> None:
        super().__init__()

        if not _ESM_AVAILABLE:
            raise ImportError("fair-esm not installed. Run: pip install fair-esm")

        # Load pretrained ESM-2
        model, alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
        self.esm        = model
        self.alphabet   = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        # Inject LoRA into Q and V projections of every attention layer
        n_replaced = inject_lora(self.esm, rank=LORA_RANK, alpha=LORA_ALPHA)
        if n_replaced == 0:
            # Fallback: ESM-2 sometimes uses 'self_attn' submodule names
            n_replaced = self._inject_lora_fallback()

        # Freeze all ESM-2 parameters except LoRA matrices
        self.freeze_esm()

    def _inject_lora_fallback(self) -> int:
        """
        Fallback injection for ESM-2 versions that store attention as
        self_attn with in_proj_weight (merged QKV) or separate dense layers.
        """
        replaced = 0
        for module in self.esm.modules():
            if isinstance(module, nn.MultiheadAttention):
                if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                    # Can't easily split merged QKV — wrap out_proj instead
                    if isinstance(module.out_proj, nn.Linear):
                        module.out_proj = LoRALinear(module.out_proj)
                        replaced += 1
        return replaced

    def freeze_esm(self) -> None:
        """Freeze all ESM-2 weights (LoRA matrices stay trainable)."""
        for name, p in self.esm.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

    def unfreeze_lora(self) -> None:
        """Enable gradient updates on LoRA matrices only (Stage 2)."""
        for name, p in self.esm.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_sequences(self, protein_sequences: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper: convert list of (label, aa_sequence) pairs to ESM token tensors.
        Returns (tokens, attention_mask) on self.device.
        """
        _, _, tokens = self.batch_converter(protein_sequences)
        # ESM-2 tokens: 0=padding, 1=BOS, 2=EOS; non-padding = 1 in mask
        attn_mask = (tokens != self.alphabet.padding_idx).long()
        device = next(self.parameters()).device
        return tokens.to(device), attn_mask.to(device)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens        : [batch, protein_len]  LongTensor (ESM-2 token IDs)
            attention_mask: [batch, protein_len]  LongTensor (1=real, 0=pad)
                            If None, no masking is applied.

        Returns:
            embeddings    : [batch, protein_len, ESM_EMBED_DIM]  float32
        """
        assert tokens.dtype == torch.long, "tokens must be LongTensor"

        # ESM-2 forward — returns dict with 'representations'
        repr_layers = [self.esm.num_layers]
        results = self.esm(tokens, repr_layers=repr_layers, return_contacts=False)
        # Shape: [batch, protein_len, ESM_EMBED_DIM]
        embeddings = results["representations"][self.esm.num_layers]

        assert embeddings.dtype == torch.float32, "ESM-2 output should be float32"
        assert embeddings.shape[-1] == ESM_EMBED_DIM, (
            f"ESM embed dim mismatch: got {embeddings.shape[-1]}, expected {ESM_EMBED_DIM}"
        )
        return embeddings


if __name__ == "__main__":
    print(f"ESM available: {_ESM_AVAILABLE}")
    if not _ESM_AVAILABLE:
        print("Install fair-esm: pip install fair-esm")
        sys.exit(1)

    print(f"Loading {ESM_MODEL_NAME}...")
    encoder = ProteinEncoder().to(DEVICE)
    encoder.eval()

    total   = encoder.total_params()
    trainable = encoder.trainable_params()
    print(f"Total params:     {total:,}")
    print(f"Trainable (LoRA): {trainable:,}  ({100*trainable/total:.2f}%)")

    # Test on insulin sequence (human, 110 aa preproinsulin)
    insulin_seq = (
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    )
    seqs = [("insulin", insulin_seq), ("insulin_copy", insulin_seq)]
    tokens, attn_mask = encoder.encode_sequences(seqs)
    print(f"Token shape: {tokens.shape}")

    with torch.no_grad():
        out = encoder(tokens, attn_mask)

    batch, seq_len, dim = out.shape
    assert dim == ESM_EMBED_DIM, f"Wrong embed dim: {dim}"
    print(f"ProteinEncoder test passed. Output shape: {out.shape}")
    print(f"Device: {out.device} | dtype: {out.dtype}")
