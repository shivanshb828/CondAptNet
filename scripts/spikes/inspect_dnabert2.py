"""
THROWAWAY SPIKE — DNABERT-2 internals inspection (Session 2).

NOT wired into the training/eval pipeline. Its only job is to surface the facts
we need before writing the real `PretrainedDNAEncoder` LoRA wrapper (Session 3):

  1. Which attention projection module(s) DNABERT-2 actually exposes
     (fused `Wqkv` vs. separate `q_proj`/`v_proj`) — LoRA must target these.
  2. How DNABERT-2's BPE tokenizer behaves on our aptamer length range
     (20–120 nt): BPE token count != nucleotide count, so we need the real
     max token count to set DNABERT2_MAX_LEN sanely.
  3. Whether the plain `AutoModel.from_pretrained(...)` path works or the
     documented `BertConfig` workaround (DNABERT_2 issue #38) is required.

Run:
    python scripts/spikes/inspect_dnabert2.py

Notes:
  - We deliberately do NOT install `triton`. DNABERT-2's optional Triton
    flash-attention path has known install issues on non-standard envs
    (per the model's GitHub tracker); the standard attention path works
    without it.
  - Uses the real `aptamer_sequence` column from
    data/processed/master_dataset_cleaned.csv (the schema calls it
    `aptamer_sequence`, not `sequence`).
"""

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
# Pin every fetch to an immutable commit SHA — trust_remote_code=True executes
# repo code, so we never resolve the moving default branch. Mirrors
# config.DNABERT2_REVISION (kept as a literal so this throwaway spike stays
# self-contained and importable even without the package on sys.path).
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
CLEANED_CSV = ROOT / "data" / "processed" / "master_dataset_cleaned.csv"


def _ensure(pkg: str, import_name: str | None = None) -> None:
    """pip-install `pkg` into the current env if `import_name` isn't importable."""
    name = import_name or pkg
    try:
        importlib.import_module(name)
        return
    except ImportError:
        print(f"[spike] installing missing dependency: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def _install_triton_stub() -> Path:
    """
    DNABERT-2's remote `bert_layers.py` has a top-level `import triton`, and
    transformers' `check_imports` does a *static* preflight scan that hard-fails
    if `triton` can't be resolved — even though the model code itself wraps the
    triton flash-attention path in try/except and falls back to plain torch.

    We refuse to install real triton (forbidden by the task; no reliable
    macOS/MPS build anyway). Instead we drop a minimal stub package on sys.path:
    `find_spec("triton")` then succeeds (preflight passes), and at runtime the
    model's `import triton.language as tl` raises ImportError on the missing
    submodule -> caught -> torch attention fallback. Returns the stub dir.
    """
    import importlib
    stub_dir = ROOT / "scripts" / "spikes" / "_triton_stub"
    (stub_dir / "triton").mkdir(parents=True, exist_ok=True)
    init = stub_dir / "triton" / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Stub triton package — see inspect_dnabert2._install_triton_stub."""\n'
            "__version__ = '0.0.0-stub'\n"
        )
    sys.path.insert(0, str(stub_dir))
    importlib.invalidate_caches()
    return stub_dir


def main() -> None:
    # ── Dependencies (NO triton) ─────────────────────────────────────────────
    _ensure("transformers")
    _ensure("einops")
    stub_dir = _install_triton_stub()
    print(f"[spike] installed triton stub at {stub_dir} (forces torch attn path)")

    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download

    try:
        from config import DEVICE
    except Exception:
        DEVICE = "cpu"
    print(f"[spike] config DEVICE = {DEVICE}")
    print(f"[spike] transformers = {importlib.import_module('transformers').__version__}")
    print(f"[spike] torch        = {torch.__version__}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\n[spike] loading tokenizer for {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)

    # ── Load model ────────────────────────────────────────────────────────────
    # The plain `AutoModel.from_pretrained(..., trust_remote_code=True)` path does
    # NOT work with transformers 5.x. It fails in three stages (all recorded
    # here so Session 3 knows exactly what it's avoiding):
    #   (1) check_imports hard-requires `triton` (handled by the stub above),
    #   (2) the custom BertConfig has no `pad_token_id` attr under 5.x,
    #   (3) `from_pretrained` runs model __init__ inside a meta-device context,
    #       and DNABERT-2's `rebuild_alibi_tensor` mixes meta/cpu tensors ->
    #       "Tensor on device meta is not on the expected device cpu!".
    # (3) is NOT fixable by the documented BertConfig workaround (issue #38) —
    # that ValueError workaround addresses a different, older failure mode.
    #
    # The robust pattern that DOES work on transformers 5.9.0: fetch the remote
    # model CLASS, construct it directly (plain cpu init, no meta context), then
    # load the checkpoint manually with the `bert.` prefix stripped.
    plain_path_error = None
    print("[spike] probing the plain AutoModel.from_pretrained path (expected to fail on 5.x) ...")
    try:
        AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
        plain_path_error = None
        print("[spike]   -> plain path SUCCEEDED (unexpected on 5.x — good news)")
    except Exception as e:
        plain_path_error = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"[spike]   -> plain path FAILED as expected: {plain_path_error}")

    print("[spike] loading via direct-construction pattern ...")
    cfg = AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
    pad_patched = getattr(cfg, "pad_token_id", None) is None
    if pad_patched:
        cfg.pad_token_id = tokenizer.pad_token_id
        print(f"[spike]   -> patched missing cfg.pad_token_id = {cfg.pad_token_id}")
    model_cls = get_class_from_dynamic_module("bert_layers.BertModel", MODEL_NAME,
                                              revision=MODEL_REVISION, trust_remote_code=True)
    model = model_cls(cfg)  # plain construction, avoids 5.x meta-init
    weights_path = hf_hub_download(MODEL_NAME, "pytorch_model.bin", revision=MODEL_REVISION)
    raw_sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    remapped = {}
    for k, v in raw_sd.items():
        if k.startswith("bert."):      # base encoder weights live under `bert.`
            remapped[k[len("bert."):]] = v
        elif k.startswith("cls."):     # MLM head — not part of the base encoder
            continue
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[spike]   -> weights loaded | missing={list(missing)} unexpected={list(unexpected)}")
    print("[spike]   -> (missing pooler.* is expected/harmless: we use per-token "
          "last_hidden_state, not the pooled output)")

    model.eval()

    # ── Pull 3 real aptamer sequences spanning the length range ───────────────
    import pandas as pd
    df = pd.read_csv(CLEANED_CSV)
    seqs_all = df["aptamer_sequence"].dropna().astype(str)
    seqs_all = seqs_all[seqs_all.str.fullmatch(r"[ACGTacgt]+")]  # clean DNA only
    lens = seqs_all.str.len()
    # shortest, median-ish, longest — to exercise the tokenizer across the range
    idx_short = lens.idxmin()
    idx_long = lens.idxmax()
    idx_mid = (lens - lens.median()).abs().idxmin()
    sample_idx = list(dict.fromkeys([idx_short, idx_mid, idx_long]))
    sample_seqs = [seqs_all.loc[i].upper() for i in sample_idx]
    print("\n[spike] sample real aptamer sequences (nt length):")
    for s in sample_seqs:
        print(f"    len={len(s):>3}  {s[:60]}{'...' if len(s) > 60 else ''}")

    # ── Forward pass on CPU and DEVICE; check finiteness + shape ──────────────
    def forward_on(device_str: str):
        dev = torch.device(device_str)
        m = model.to(dev)
        enc = tokenizer(sample_seqs, return_tensors="pt", padding=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = m(**enc)
        # DNABERT-2 AutoModel returns a tuple; [0] is last_hidden_state
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        finite = torch.isfinite(hidden).all().item()
        print(f"[spike] forward on {device_str:>4}: hidden shape={tuple(hidden.shape)}  finite={finite}")
        return hidden

    print("\n[spike] running forward passes ...")
    forward_on("cpu")
    if DEVICE != "cpu":
        try:
            forward_on(DEVICE)
        except Exception as e:
            print(f"[spike] forward on {DEVICE} FAILED: {type(e).__name__}: {e}")
    model.to("cpu")

    # ── Tokenizer behavior across the full 20–120 nt range ────────────────────
    print("\n[spike] BPE token counts across aptamer length range:")
    probe_lens = [20, 40, 60, 80, 100, 120]
    for L in probe_lens:
        # pick a real sequence at least L long, truncate to L
        cand = seqs_all[lens >= L]
        if len(cand) == 0:
            continue
        s = cand.iloc[0][:L].upper()
        n_tok = len(tokenizer(s)["input_ids"])
        print(f"    {L:>3} nt  ->  {n_tok:>3} BPE tokens (incl. special)")

    # exact max token count for a real 120nt (or longest available) sequence
    longest = seqs_all.loc[idx_long].upper()
    max_tok = len(tokenizer(longest)["input_ids"])
    # also measure the true max over the WHOLE dataset (batched, no special-token subtraction)
    all_upper = seqs_all.str.upper().tolist()
    tok_counts = [len(tokenizer(s)["input_ids"]) for s in all_upper]
    dataset_max_tok = max(tok_counts)
    dataset_p99_tok = int(pd.Series(tok_counts).quantile(0.99))
    print(f"\n[spike] longest sequence: {len(longest)} nt -> {max_tok} BPE tokens")
    print(f"[spike] dataset-wide max BPE tokens (any seq): {dataset_max_tok}")
    print(f"[spike] dataset-wide p99 BPE tokens: {dataset_p99_tok}")

    # ── Enumerate attention-relevant modules ──────────────────────────────────
    print("\n[spike] modules matching attn/Wqkv/query/value/proj:")
    keywords = ("attn", "wqkv", "query", "value", "proj")
    found_names = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in keywords):
            shape = ""
            if hasattr(module, "weight") and hasattr(module.weight, "shape"):
                shape = f"  weight={tuple(module.weight.shape)}"
            print(f"    {name:<55} {type(module).__name__}{shape}")
            if hasattr(module, "weight"):
                found_names.append((name, type(module).__name__, tuple(module.weight.shape)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPIKE SUMMARY")
    print("=" * 70)
    print(f"(a) attention projection modules with weights ({len(found_names)}):")
    # de-dup by leaf module name (last path component) to show the pattern
    seen = set()
    for name, typ, shp in found_names:
        leaf = name.split(".")[-1]
        key = (leaf, typ, shp)
        if key in seen:
            continue
        seen.add(key)
        print(f"      .{leaf:<12} {typ:<12} weight={shp}")
    print(f"(b) longest-seq BPE tokens: {max_tok}  | dataset max: {dataset_max_tok}"
          f"  | p99: {dataset_p99_tok}")
    print("(c) plain AutoModel.from_pretrained path:")
    print(f"      failed on transformers 5.x: {plain_path_error or 'no (it worked)'}")
    print(f"      -> used direct-construction pattern; pad_token_id patch needed: {pad_patched}")
    print(f"      -> triton stub required (no triton install): True")
    print("=" * 70)


if __name__ == "__main__":
    main()
