"""
Pretrained DNA Encoder for CondAptNet — DNABERT-2 (117M) + LoRA.

A/B ablation alternative to the from-scratch `DNAEncoder`. Selected only when
`config.DNA_ENCODER_TYPE == "dnabert2"`; the scratch encoder remains the
default and is untouched. Motivation: our protein arm gets transfer learning
(ESM-2), but the DNA arm learns DNA "language" from scratch on a small dataset.
DNABERT-2 (Zhou et al., ICLR 2024) is a BPE-tokenized multi-species genomic
foundation model — this lets us benchmark whether pretrained DNA features beat
the from-scratch encoder on aptamer-protein binding.

Domain-shift caveat (deliberate, benchmarkable): DNABERT-2 was pretrained on
genomic DNA (regulatory regions, reference genomes), NOT short synthetic ssDNA
aptamers (20–120 nt). Whether it wins is an open empirical question — hence the
flag, not a replacement.

Interface (mirrors DNAEncoder so condaptnet.py needs no forward-signature change):
    forward(token_ids, vienna_feats=None) -> [batch, seq_len, DNABERT2_EMBED_DIM=768]
  - `token_ids` here are DNABERT-2 BPE input_ids (produced by the parallel
    tokenization path in train.py's Dataset), NOT the 3-mer IDs the scratch
    encoder consumes.
  - `vienna_feats` is accepted for signature compatibility but IGNORED —
    DNABERT-2 has no ViennaRNA-bias injection path. (Structure-feature fusion
    for this branch is left as future work; out of scope for the ablation.)

Loading pattern (confirmed by scripts/spikes/inspect_dnabert2.py against the
real model — the plain `AutoModel.from_pretrained` path does NOT work on
transformers 5.x). See `_load_dnabert2` below for the three failure modes and
the working direct-construction workaround.

LoRA target (confirmed by the spike): DNABERT-2's attention uses a FUSED QKV
linear — `encoder.layer.{i}.attention.self.Wqkv = nn.Linear(768 -> 2304)` — not
the separate q_proj/v_proj that ESM-2 exposes. We therefore wrap `Wqkv` with the
shared `LoRALinear` (imported from protein_encoder), which adapts the FULL fused
2304-dim output jointly (Q, K and V together). This differs from the ESM-2
injector, which adapts Q and V separately; documented inline at the injection site.

Usage (standalone test):
    python models/encoders/dna_encoder_pretrained.py
"""

import importlib
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE,
    DNABERT2_MODEL_NAME, DNABERT2_REVISION, DNABERT2_EMBED_DIM, DNABERT2_MAX_LEN,
    DNABERT2_LORA_RANK, DNABERT2_LORA_ALPHA, DNABERT2_LORA_DROPOUT,
)
# Reuse the exact LoRA implementation the protein encoder uses — do not duplicate.
from models.encoders.protein_encoder import LoRALinear


def _ensure_triton_stub() -> None:
    """
    DNABERT-2's remote `bert_layers.py` has a top-level `import triton`, and
    transformers' `check_imports` does a static preflight scan that hard-fails
    if triton can't be resolved — even though the model code wraps the triton
    flash-attention path in try/except and falls back to plain torch.

    We refuse to install real triton (no reliable macOS/MPS build; not needed).
    Inject a minimal stub with a real __spec__ so `find_spec("triton")` succeeds;
    at runtime the model's `import triton.language` raises on the missing
    submodule -> caught -> torch attention fallback.
    """
    if importlib.util.find_spec("triton") is not None:
        return
    stub = types.ModuleType("triton")
    stub.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
    stub.__version__ = "0.0.0-stub"
    sys.modules.setdefault("triton", stub)


def _load_dnabert2(model_name: str, revision: str = DNABERT2_REVISION) -> tuple[nn.Module, object]:
    """
    Load DNABERT-2's base BertModel + tokenizer on transformers 5.x.

    `revision` pins EVERY fetch (remote code, config, tokenizer, weights) to one
    immutable commit SHA. This is a security control, not a convenience: because
    trust_remote_code=True runs arbitrary Python from the repo, an unpinned ref
    would execute whatever upstream pushes to the default branch next.

    The plain `AutoModel.from_pretrained(model_name, trust_remote_code=True)`
    path fails three ways on transformers 5.9.0 (all confirmed by the Session-2
    spike):
      1. `check_imports` hard-requires `triton`      -> handled by _ensure_triton_stub().
      2. custom BertConfig has no `pad_token_id` attr -> patched from the tokenizer.
      3. from_pretrained runs model __init__ inside a meta-device context, and
         DNABERT-2's `rebuild_alibi_tensor` mixes meta/cpu tensors ->
         "Tensor on device meta is not on the expected device cpu!".
         The documented BertConfig workaround (repo issue #38) does NOT fix (3).

    Working pattern: fetch the remote model CLASS, construct it directly (plain
    cpu init, no meta context), then load the checkpoint manually with the
    `bert.` prefix stripped and the `cls.` MLM head dropped.

    Returns (model, tokenizer). Only `pooler.*` is missing after load (unused —
    we read per-token last_hidden_state, never the pooled output).
    """
    _ensure_triton_stub()
    try:
        from transformers import AutoTokenizer, AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "DNABERT-2 encoder needs `transformers`, `einops`, `huggingface_hub`. "
            "Install with: pip install transformers einops huggingface_hub"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(
        model_name, revision=revision, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tokenizer.pad_token_id

    # Pinning this call matters most: it fetches and executes bert_layers.py.
    model_cls = get_class_from_dynamic_module(
        "bert_layers.BertModel", model_name, revision=revision, trust_remote_code=True
    )
    model = model_cls(cfg)  # direct construction — sidesteps 5.x meta-init

    weights_path = hf_hub_download(model_name, "pytorch_model.bin", revision=revision)
    raw_sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    remapped = {}
    for k, v in raw_sd.items():
        if k.startswith("bert."):      # base-encoder weights live under `bert.`
            remapped[k[len("bert."):]] = v
        elif k.startswith("cls."):     # MLM head — not part of the base encoder
            continue
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # Only the (unused) pooler should be missing; anything else is a real problem.
    real_missing = [m for m in missing if not m.startswith("pooler.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Unexpected DNABERT-2 checkpoint mismatch: "
            f"missing={real_missing} unexpected={list(unexpected)}"
        )
    return model, tokenizer


class PretrainedDNAEncoder(nn.Module):
    """
    DNABERT-2 (117M) genomic foundation model + LoRA adapters on the fused Wqkv
    attention projections. Mirrors DNAEncoder's forward contract so it drops
    into condaptnet.py's DNA-encoder slot.

    Training-stage convention (mirrors ProteinEncoder):
        Stage 1 (pretraining):  freeze_dnabert() — only LoRA + downstream train.
        Stage 2/3 (fine-tuning): unfreeze_lora() — LoRA adapts further.
    """

    def __init__(
        self,
        model_name: str = DNABERT2_MODEL_NAME,
        lora_rank: int = DNABERT2_LORA_RANK,
        lora_alpha: float = DNABERT2_LORA_ALPHA,
        lora_dropout: float = DNABERT2_LORA_DROPOUT,
    ) -> None:
        super().__init__()

        self.dnabert, self.tokenizer = _load_dnabert2(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_len = DNABERT2_MAX_LEN

        n = self._inject_lora_wqkv(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        if n == 0:
            raise RuntimeError("LoRA injection found no Wqkv modules in DNABERT-2 "
                               "— the module layout may have changed; re-run the spike.")
        self.n_lora = n
        self.freeze_dnabert()

    # ── LoRA injection ────────────────────────────────────────────────────────

    def _inject_lora_wqkv(self, rank: int, alpha: float, dropout: float) -> int:
        """
        Wrap each fused-QKV `Wqkv` (nn.Linear(768 -> 2304)) with LoRALinear.

        Unlike ESM-2 — where inject_lora() wraps SEPARATE q_proj and v_proj so
        LoRA adapts Q and V independently — DNABERT-2 fuses Q, K, V into one
        Wqkv linear. Wrapping Wqkv with LoRALinear applies a single low-rank
        delta to the FULL 2304-dim fused output, i.e. Q, K and V are adapted
        jointly by the same A/B matrices. This is the intended fused-QKV variant
        (LoRALinear is projection-agnostic; it low-rank-adapts whatever nn.Linear
        it wraps), so we reuse it rather than duplicating the class.
        """
        replaced = 0
        for module in self.dnabert.modules():
            if hasattr(module, "Wqkv") and isinstance(module.Wqkv, nn.Linear):
                module.Wqkv = LoRALinear(module.Wqkv, rank=rank, alpha=alpha,
                                         dropout=dropout)
                replaced += 1
        return replaced

    # ── Training-stage helpers (mirror ProteinEncoder naming) ─────────────────

    def freeze_dnabert(self) -> None:
        """Freeze all DNABERT-2 base weights; LoRA matrices stay trainable."""
        for name, p in self.dnabert.named_parameters():
            p.requires_grad = "lora_" in name

    def unfreeze_lora(self) -> None:
        """Enable gradient updates on LoRA matrices only (Stage 2/3)."""
        for name, p in self.dnabert.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Tokenization helper (used by the training Dataset's dnabert2 path) ─────

    def tokenize(self, sequences: list[str]) -> torch.Tensor:
        """
        BPE-tokenize raw nucleotide strings to padded input_ids [batch, max_len].
        Provided so the data path has one canonical tokenization entry point.
        """
        enc = self.tokenizer(
            [s.upper() for s in sequences],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_len,
        )
        return enc["input_ids"]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids: torch.Tensor,
        vienna_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids    : [batch, seq_len]  LongTensor of DNABERT-2 BPE input_ids
            vienna_feats : accepted for interface parity with DNAEncoder; IGNORED.

        Returns:
            embeddings   : [batch, seq_len, DNABERT2_EMBED_DIM=768]
        """
        assert token_ids.dtype == torch.long, "token_ids must be LongTensor (BPE input_ids)"
        assert token_ids.dim() == 2, f"Expected [batch, seq_len], got {token_ids.shape}"

        # Reconstruct the attention mask from padding (no mask threaded through
        # condaptnet.forward — the scratch encoder derives padding the same way).
        attention_mask = (token_ids != self.pad_token_id).long()

        out = self.dnabert(input_ids=token_ids, attention_mask=attention_mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state

        assert hidden.shape[-1] == DNABERT2_EMBED_DIM, (
            f"DNABERT-2 embed dim mismatch: got {hidden.shape[-1]}, "
            f"expected {DNABERT2_EMBED_DIM}"
        )
        return hidden


if __name__ == "__main__":
    import pandas as pd

    torch.manual_seed(42)
    print(f"Loading {DNABERT2_MODEL_NAME} (this downloads ~450MB on first run)...")
    encoder = PretrainedDNAEncoder().to(DEVICE)
    encoder.eval()

    total = encoder.total_params()
    trainable = encoder.trainable_params()
    print(f"LoRA-wrapped Wqkv modules: {encoder.n_lora}")
    print(f"Total params:     {total:,}")
    print(f"Trainable (LoRA): {trainable:,}  ({100*trainable/total:.3f}%)")

    # Pull a few real aptamer sequences from the cleaned dataset
    csv = Path(__file__).resolve().parents[2] / "data/processed/master_dataset_cleaned.csv"
    seqs = ["ACGTACGTACGTACGTACGTACGT", "ATGCATGCATGCATGCATGCATGCATGCATGC"]
    if csv.exists():
        s = pd.read_csv(csv)["aptamer_sequence"].dropna().astype(str)
        s = s[s.str.fullmatch(r"[ACGTacgt]+")].str.upper()
        seqs = s.head(4).tolist()

    input_ids = encoder.tokenize(seqs).to(DEVICE)
    print(f"Tokenized input_ids shape: {tuple(input_ids.shape)}  (max_len={encoder.max_len})")

    with torch.no_grad():
        out = encoder(input_ids)

    assert out.shape[0] == len(seqs)
    assert out.shape[-1] == DNABERT2_EMBED_DIM
    assert torch.isfinite(out).all(), "non-finite DNABERT-2 output"
    print(f"PretrainedDNAEncoder test passed. Output shape: {tuple(out.shape)}")
    print(f"Device: {out.device} | dtype: {out.dtype}")
