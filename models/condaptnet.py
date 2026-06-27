"""
CondAptNet — Full Model Assembly.

Wires all components into a single end-to-end model:
    tokenizer → DNA encoder → protein encoder → condition encoder
    → symmetric cross-attention (FiLM) → CNN head → dual output head

Forward pass inputs:
    aptamer_tokens  : [batch, apt_token_len]   LongTensor  (3-mer IDs from tokenizer)
    vienna_feats    : [batch, 6]               float32     (ViennaRNA features)
    protein_tokens  : [batch, prot_len]        LongTensor  (ESM-2 alphabet IDs)
    condition       : [batch, 5]               float32     [pH, salt_mM, temp_C, buffer, mg_mM]

Forward pass outputs (CondAptNetOutput named tuple):
    binding_prob    : [batch, 1]   float32  ∈ [0,1]
    kd_pred         : [batch, 1]   float32  ≥ 0 (log10(nM+1)), or None
    binding_label   : [batch, 1]   int64    0=low | 1=medium | 2=high

Shapes at each stage (batch=4, apt_len=50 tokens, prot_len=200 aa):
    DNA encoder   → [4, 50, 128]
    Protein enc.  → [4, 202, 480]  (202 = 200 + BOS + EOS from ESM-2)
    Condition enc.→ [4, 128]
    Cross-attn    → apt [4, 50, 256], prot [4, 202, 256]
    CNN head      → [4, 256]
    Dual head     → binding [4, 1], kd [4, 1], label [4, 1]

Usage:
    python models/condaptnet.py
"""

import sys
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DEVICE,
    DNA_EMBED_DIM, ESM_EMBED_DIM, FUSION_DIM,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
)
from models.encoders.dna_encoder      import DNAEncoder, VIENNA_FEAT_DIM
from models.encoders.protein_encoder  import ProteinEncoder
from models.encoders.condition_encoder import ConditionEncoder
from models.attention.cross_attention  import SymmetricCrossAttention
from models.interaction.cnn_head       import CNNHead
from models.output.dual_head           import DualHead, DualHeadOutput


class CondAptNetOutput(NamedTuple):
    binding_prob  : torch.Tensor           # [batch, 1]  float32 ∈ [0,1]
    kd_pred       : Optional[torch.Tensor] # [batch, 1]  float32 ≥ 0 (log10(nM+1)), or None
    binding_label : torch.Tensor           # [batch, 1]  int64   0=low|1=medium|2=high


class CondAptNet(nn.Module):
    """
    Full CondAptNet model.

    Stage 1 (pretraining):  protein_encoder.freeze_esm() — only LoRA trains
    Stage 2/3 (fine-tuning): protein_encoder.unfreeze_lora() — LoRA adapts further
    """

    def __init__(self, predict_kd: bool = True) -> None:
        super().__init__()
        self.predict_kd = predict_kd

        self.dna_encoder       = DNAEncoder(use_vienna=True)
        self.protein_encoder   = ProteinEncoder()
        self.condition_encoder = ConditionEncoder()
        self.cross_attention   = SymmetricCrossAttention()
        self.cnn_head          = CNNHead()
        self.dual_head         = DualHead()

    # ── Training-stage helpers ────────────────────────────────────────────────

    def set_stage1(self) -> None:
        """Stage 1: freeze ESM-2 backbone, train everything else."""
        self.protein_encoder.freeze_esm()

    def set_stage2(self) -> None:
        """Stage 2/3: unfreeze ESM-2 LoRA for target-specific adaptation."""
        self.protein_encoder.unfreeze_lora()

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        aptamer_tokens  : torch.Tensor,
        vienna_feats    : torch.Tensor,
        protein_tokens  : torch.Tensor,
        condition       : torch.Tensor,
        predict_kd      : Optional[bool] = None,
        protein_emb     : Optional[torch.Tensor] = None,
    ) -> CondAptNetOutput:
        """
        Args:
            aptamer_tokens : [batch, apt_token_len]  LongTensor
            vienna_feats   : [batch, 6]              float32
            protein_tokens : [batch, prot_len]       LongTensor (ESM-2 alphabet)
                             Ignored when protein_emb is provided.
            condition      : [batch, 5]              float32
            protein_emb    : [batch, prot_len, ESM_EMBED_DIM]  float32  (optional)
                             Pre-computed ESM-2 embeddings. When supplied, the
                             protein_encoder forward pass is skipped entirely —
                             use this during Stage 1 training to avoid re-running
                             frozen ESM-2 on every batch (cache with numpy).

        Returns:
            CondAptNetOutput(binding_prob, kd_pred)
        """
        use_kd = predict_kd if predict_kd is not None else self.predict_kd

        # ── Encoders ──────────────────────────────────────────────────────────
        apt_emb  = self.dna_encoder(aptamer_tokens, vienna_feats)
        # apt_emb: [batch, apt_token_len, DNA_EMBED_DIM=128]

        if protein_emb is not None:
            prot_emb = protein_emb
        else:
            prot_emb = self.protein_encoder(protein_tokens)
        # prot_emb: [batch, prot_len, ESM_EMBED_DIM=480]

        # ── Symmetric cross-attention with FiLM conditioning ─────────────────
        apt_fused, prot_fused = self.cross_attention(apt_emb, prot_emb, condition)
        # apt_fused:  [batch, apt_token_len, FUSION_DIM=256]
        # prot_fused: [batch, prot_len,      FUSION_DIM=256]

        # ── Interaction matrix → CNN ──────────────────────────────────────────
        # CNNHead uses per-block gradient checkpointing internally.
        features = self.cnn_head(apt_fused, prot_fused)
        # features: [batch, 256]

        # ── Dual output ───────────────────────────────────────────────────────
        out = self.dual_head(features, predict_kd=use_kd)

        return CondAptNetOutput(
            binding_prob=out.binding_prob,
            kd_pred=out.kd_pred,
            binding_label=out.binding_label,
        )


if __name__ == "__main__":
    import torch
    from config import DEVICE, DNA_VOCAB_SIZE, DNA_PAD_ID

    torch.manual_seed(42)
    batch_size = 4
    apt_tokens = 50   # 3-mer tokens (~52 nt aptamer)
    prot_len   = 202  # ESM-2 adds BOS/EOS, so raw 200 aa → 202 tokens

    print("Building CondAptNet...")
    model = CondAptNet(predict_kd=True)
    model.set_stage1()   # freeze ESM-2 backbone for Stage 1
    model = model.to(DEVICE)
    model.eval()

    total     = model.total_params()
    trainable = model.trainable_params()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
    print()

    # ── Build dummy inputs ────────────────────────────────────────────────────
    aptamer_tokens = torch.randint(2, DNA_VOCAB_SIZE, (batch_size, apt_tokens),
                                   dtype=torch.long).to(DEVICE)
    aptamer_tokens[0, 45:] = DNA_PAD_ID   # simulate padding

    # [mfe, stem_count, loop_count, bp_prob_mean, bp_prob_max, seq_len_norm]
    vienna_feats = torch.tensor([
        [-2.5, 1.0, 1.0, 0.15, 0.70, 50/120],
        [-4.1, 1.0, 1.0, 0.22, 0.85, 49/120],
        [-1.2, 0.0, 0.0, 0.01, 0.05, 48/120],
        [-3.8, 1.0, 2.0, 0.30, 0.90, 51/120],
    ], dtype=torch.float32).to(DEVICE)
    # vienna_feats: [batch, 6]

    # ESM-2 tokens: use alphabet.padding_idx=1 for pad, real tokens 4+
    protein_tokens = torch.randint(4, 30, (batch_size, prot_len),
                                   dtype=torch.long).to(DEVICE)

    condition = torch.tensor([
        [7.4, 150.0, 37.0, 0.0, 2.0],   # physiological defaults
        [7.0, 150.0, 25.0, 1.0, 1.5],
        [7.4,  50.0, 37.0, 2.0, 2.0],
        [6.8, 100.0, 30.0, 3.0, 3.0],
    ], dtype=torch.float32).to(DEVICE)

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("Running forward pass (batch=4, apt_tokens=50, prot_len=202)...")
    with torch.no_grad():
        out = model(aptamer_tokens, vienna_feats, protein_tokens, condition)

    from config import BIND_LABEL_NAMES
    print()
    print(f"binding_prob  shape: {out.binding_prob.shape}   values: {out.binding_prob.squeeze().tolist()}")
    print(f"kd_pred       shape: {out.kd_pred.shape}        values: {out.kd_pred.squeeze().tolist()}")
    labels_str = [BIND_LABEL_NAMES[v.item()] for v in out.binding_label.squeeze(1)]
    print(f"binding_label shape: {out.binding_label.shape}  values: {out.binding_label.squeeze().tolist()} → {labels_str}")
    print()

    assert out.binding_prob.shape  == (batch_size, 1)
    assert out.kd_pred.shape       == (batch_size, 1)
    assert out.binding_label.shape == (batch_size, 1)
    assert out.binding_prob.min()  >= 0.0 and out.binding_prob.max() <= 1.0
    assert out.kd_pred.min()       >= 0.0
    assert out.binding_label.dtype == torch.long
    assert out.binding_label.min() >= 0 and out.binding_label.max() <= 2

    # ── Test predict_kd=False ─────────────────────────────────────────────────
    with torch.no_grad():
        out_no_kd = model(aptamer_tokens, vienna_feats, protein_tokens,
                          condition, predict_kd=False)
    assert out_no_kd.kd_pred             is None, "kd_pred should be None when predict_kd=False"
    assert out_no_kd.binding_label.shape == (batch_size, 1), "binding_label missing when kd_pred=None"
    assert out_no_kd.binding_label.dtype == torch.long
    print("predict_kd=False → kd_pred is None, binding_label (prob fallback) present  ✓")

    # ── Test set_stage2 (unfreeze LoRA) ──────────────────────────────────────
    model.set_stage2()
    trainable2 = model.trainable_params()
    assert trainable2 >= trainable, "Stage 2 should have >= trainable params"
    print(f"Stage 2 trainable: {trainable2:,}  ({100*trainable2/total:.2f}%)  ✓")

    print()
    print(f"Device: {out.binding_prob.device} | dtype: {out.binding_prob.dtype}")
    print("\nCondAptNet end-to-end test passed.")
