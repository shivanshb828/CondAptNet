"""
Interaction Matrix + 17-block CNN Head for CondAptNet.

Constructs a 2D aptamer-protein interaction map via outer product, then
extracts hierarchical features using 17 convolutional blocks with residual
connections. Global average pooling collapses variable sequence lengths into
a fixed-size feature vector for the output heads.

Architecture:
    outer_product([batch,apt,D], [batch,prot,D])
        → [batch, D, apt_len, prot_len]   interaction matrix (2D image)
    1×1 conv to CNN_CHANNELS[0]
    17 ConvBlocks (GroupNorm → GELU → Conv3×3 → GroupNorm → GELU → Conv3×3)
        channels: 64 (blocks 1–6) → 128 (blocks 7–12) → 256 (blocks 13–17)
        residual connections where in_channels == out_channels
        1×1 projection conv at channel transitions
    Global average pooling → [batch, 256]

Input shapes:
    apt_fused  : [batch, apt_len,  FUSION_DIM]   (from cross-attention)
    prot_fused : [batch, prot_len, FUSION_DIM]   (from cross-attention)

Output shape:
    features   : [batch, CNN_CHANNELS[-1]]        (256-dim fixed vector)

Usage:
    python models/interaction/cnn_head.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as _grad_ckpt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEVICE, FUSION_DIM, CNN_CHANNELS, CNN_KERNEL_SIZE, CNN_NUM_BLOCKS


class ConvBlock(nn.Module):
    """
    One residual convolutional block: two Conv2d layers with GroupNorm+GELU.
    A 1×1 projection shortcut is added when in_channels != out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = CNN_KERNEL_SIZE) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


def _build_channel_schedule(num_blocks: int, channels: list[int]) -> list[tuple[int, int]]:
    """
    Distribute num_blocks across channel stages evenly.
    Returns list of (in_ch, out_ch) tuples, one per block.
    """
    n_stages = len(channels)
    # Assign floor(num_blocks / n_stages) blocks to each stage,
    # remainder goes to the last stage.
    per_stage = num_blocks // n_stages
    remainder = num_blocks % n_stages

    schedule: list[tuple[int, int]] = []
    for s, ch in enumerate(channels):
        in_ch  = channels[s - 1] if s > 0 else channels[0]
        count  = per_stage + (1 if s == n_stages - 1 else 0) * remainder
        # First block in stage transitions from previous channel
        for b in range(count):
            if b == 0 and s > 0:
                schedule.append((channels[s - 1], ch))
            else:
                schedule.append((ch, ch))

    # Pad or trim to exactly num_blocks
    while len(schedule) < num_blocks:
        schedule.append((channels[-1], channels[-1]))
    return schedule[:num_blocks]


class CNNHead(nn.Module):
    """
    Outer-product interaction matrix fed through 17 residual Conv blocks.
    Produces a fixed-size 256-dim vector regardless of sequence lengths.
    """

    def __init__(self) -> None:
        super().__init__()

        # Project interaction matrix from FUSION_DIM channels to first CNN channel
        self.entry_conv = nn.Conv2d(FUSION_DIM, CNN_CHANNELS[0], kernel_size=1, bias=False)

        # Build 17 blocks
        schedule = _build_channel_schedule(CNN_NUM_BLOCKS, CNN_CHANNELS)
        self.blocks = nn.ModuleList([
            ConvBlock(in_ch, out_ch) for in_ch, out_ch in schedule
        ])

        self.final_norm = nn.GroupNorm(8, CNN_CHANNELS[-1])
        self.gap = nn.AdaptiveAvgPool2d(1)   # global average pooling → [batch, C, 1, 1]

    def _interact_and_project(
        self, apt_fused: torch.Tensor, prot_fused: torch.Tensor
    ) -> torch.Tensor:
        """
        Build outer-product interaction matrix and run the 1×1 entry conv.
        Extracted as a method so it can be gradient-checkpointed: saves only the
        small apt_fused/prot_fused inputs instead of the large [B,D,apt,prot] matrix.
        """
        apt_len  = apt_fused.size(1)
        prot_len = prot_fused.size(1)
        apt_exp  = apt_fused.unsqueeze(2).expand(-1, -1, prot_len, -1)
        prot_exp = prot_fused.unsqueeze(1).expand(-1, apt_len, -1, -1)
        interaction = apt_exp * prot_exp
        interaction = interaction.permute(0, 3, 1, 2).contiguous()  # [B, D, apt, prot]
        return self.entry_conv(interaction)                           # [B, 64, apt, prot]

    def forward(self, apt_fused: torch.Tensor, prot_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            apt_fused  : [batch, apt_len,  FUSION_DIM]
            prot_fused : [batch, prot_len, FUSION_DIM]

        Returns:
            features   : [batch, CNN_CHANNELS[-1]]
        """
        batch = apt_fused.size(0)

        # Checkpoint the interaction matrix construction so the large
        # [batch, FUSION_DIM, apt_len, prot_len] tensor is never stored between
        # forward and backward — only the small apt_fused/prot_fused are saved.
        x = _grad_ckpt.checkpoint(
            self._interact_and_project, apt_fused, prot_fused, use_reentrant=False
        )   # [batch, 64, apt_len, prot_len]

        # Per-block gradient checkpointing: stores each block's input only,
        # never all 17 blocks' intermediate activations simultaneously.
        for block in self.blocks:
            x = _grad_ckpt.checkpoint(block, x, use_reentrant=False)

        x = self.final_norm(x)
        x = self.gap(x)    # [batch, 256, 1, 1]
        x = x.flatten(1)   # [batch, 256]

        assert x.shape == (batch, CNN_CHANNELS[-1]), f"Unexpected shape: {x.shape}"
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    apt_len    = 50
    prot_len   = 200

    apt_fused  = torch.randn(batch_size, apt_len,  FUSION_DIM, dtype=torch.float32).to(DEVICE)
    prot_fused = torch.randn(batch_size, prot_len, FUSION_DIM, dtype=torch.float32).to(DEVICE)

    model = CNNHead().to(DEVICE)
    model.eval()

    with torch.no_grad():
        features = model(apt_fused, prot_fused)

    expected = (batch_size, CNN_CHANNELS[-1])
    assert features.shape == expected, f"Expected {expected}, got {features.shape}"

    print(f"Output shape:  {features.shape}")
    print(f"Device: {features.device} | dtype: {features.dtype}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")

    # Verify variable-length inputs produce same output shape
    apt2  = torch.randn(batch_size, 30,  FUSION_DIM, dtype=torch.float32).to(DEVICE)
    prot2 = torch.randn(batch_size, 350, FUSION_DIM, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        features2 = model(apt2, prot2)
    assert features2.shape == expected, f"Variable-length shape wrong: {features2.shape}"
    print(f"Variable-length test shape: {features2.shape}  (same ✓)")

    print("\nCNNHead test passed.")
