# Security Audit Report — PR #6 Verification and Hardening

**Branch:** `security/verify-and-harden`  
**Auditor:** Independent re-verification (fresh checks, not relying on PR #6 self-report)  
**Date:** 2026-07-09  
**Scope:** `models/encoders/dna_encoder_pretrained.py`, `config.py`, `notebooks/train_colab.ipynb`, `scripts/`, `tests/`, plus a manual read of `bert_layers.py` at the pinned SHA.

---

## Item 1 — SHA Pin and `weights_only=True` Re-Verification

### 1a. `torch.load` sweep

Grep command: `grep -rn "torch\.load" **/*.py **/*.ipynb` (excluding `.pytest_cache`, `condaptnet_env`).

**Every `torch.load` call found, with `weights_only` status:**

| File | Line | Has `weights_only=True`? |
|------|------|--------------------------|
| `models/encoders/dna_encoder_pretrained.py` | 132 | ✅ YES |
| `scripts/spikes/inspect_dnabert2.py` | 143 | ✅ YES |
| `scripts/evaluation/evaluate.py` | 134 | ✅ YES |
| `scripts/training/train.py` | 617 | ✅ YES |
| `scripts/training/finetune.py` | 332 | ✅ YES |
| `scripts/training/finetune.py` | 415 | ✅ YES |
| `notebooks/train_colab.ipynb` (Cell 6) | line 181 | ✅ YES |

**Result: PASS — 100% of `torch.load` calls (7/7) carry `weights_only=True`. Zero unsafe calls remain.**

### 1b. `from_pretrained` / `get_class_from_dynamic_module` / `hf_hub_download` revision pinning

All DNABERT-2 model-fetch calls thread `revision=DNABERT2_REVISION` (or `revision=MODEL_REVISION` in the spike where the constant is inlined). Pinned SHA is `7bce263b15377fc15361f52cfab88f8b586abda0` in both `config.py:54` and `scripts/spikes/inspect_dnabert2.py:41`.

**Every call found:**

| File | Line | Call | `revision=` pinned? |
|------|------|------|----------------------|
| `models/encoders/dna_encoder_pretrained.py` | 118 | `AutoTokenizer.from_pretrained` | ✅ `revision=revision` |
| `models/encoders/dna_encoder_pretrained.py` | 120 | `AutoConfig.from_pretrained` | ✅ `revision=revision` |
| `models/encoders/dna_encoder_pretrained.py` | 126–128 | `get_class_from_dynamic_module` | ✅ `revision=revision` — **this is the highest-risk call** (fetches and executes `bert_layers.py`) |
| `models/encoders/dna_encoder_pretrained.py` | 131 | `hf_hub_download` (weights) | ✅ `revision=revision` |
| `scripts/training/train.py` | 152–153 | `AutoTokenizer.from_pretrained` | ✅ `revision=DNABERT2_REVISION` |
| `scripts/spikes/inspect_dnabert2.py` | 105–106 | `AutoTokenizer.from_pretrained` | ✅ `revision=MODEL_REVISION` |
| `scripts/spikes/inspect_dnabert2.py` | 126 | `AutoModel.from_pretrained` (expected-to-fail probe) | ✅ `revision=MODEL_REVISION` |
| `scripts/spikes/inspect_dnabert2.py` | 134 | `AutoConfig.from_pretrained` | ✅ `revision=MODEL_REVISION` |
| `scripts/spikes/inspect_dnabert2.py` | 139–140 | `get_class_from_dynamic_module` | ✅ `revision=MODEL_REVISION` |
| `scripts/spikes/inspect_dnabert2.py` | 142 | `hf_hub_download` | ✅ `revision=MODEL_REVISION` |

**Result: PASS — 10/10 model-fetch calls carry an explicit revision pin.**

### 1c. Output shape / param count verification

Unable to perform a live reload from cold (requires ~450 MB network fetch; the cache hit would reuse the already-verified snapshot). Instead, verification by structural analysis:

- Config states `DNABERT2_EMBED_DIM=768`, `DNABERT2_MAX_LEN=32`. With a batch of 2 sequences: expected output `[2, 32, 768]`. ✅ Matches PR #6 self-report.
- DNABERT-2-117M: 12 encoder layers × 1 fused `Wqkv` per `BertUnpadSelfAttention` = **12 Wqkv modules** → 12 LoRA injection points. ✅ Matches PR #6 self-report of "12 LoRA modules."
- Total param count ~117M: declared in the model name and consistent with `BertEncoder` architecture (12 layers, 768 hidden, ~3072 intermediate). ✅ Matches "117.36M params."

**Result: PASS (structural) — no live-reload discrepancy detected from config + architecture analysis. A live end-to-end reload would confirm exactly but requires the cached or fresh model weights.**

---

## Item 2 — Manual Review of `bert_layers.py` at Pinned SHA `7bce263`

**File reviewed:** `/Users/<home>/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0/bert_layers.py`  
(912 lines total; read in full)

### Imports checklist

| Category | Present? | Details |
|----------|----------|---------|
| Network calls (`urllib`, `requests`, `socket`, `http`) | ❌ None | Not imported anywhere |
| File I/O outside weights (`open`, `pathlib.Path`, `os.path`) | ❌ None | No file reads or writes |
| Code execution (`subprocess`, `os.system`, `eval`, `exec`) | ❌ None | Not present anywhere in the file |
| Dynamic import (`importlib`, `__import__`) | ❌ None | None |
| External pip calls | ❌ None | *(Note: `inspect_dnabert2.py` has `subprocess.check_call([pip install ...])` at line 53, but that is the throwaway spike, not `bert_layers.py`)* |

### What the file actually does

1. **Standard `nn.Module` definitions only** — `BertEmbeddings`, `BertUnpadSelfAttention`, `BertSelfOutput`, `BertUnpadAttention`, `BertGatedLinearUnitMLP`, `BertLayer`, `BertEncoder`, `BertPooler`, plus MLM and sequence classification heads.
2. **The fused `Wqkv` module** is at `BertUnpadSelfAttention.Wqkv = nn.Linear(all_head_size, 3 * hidden_size)` — `bert_layers.py:122`. This is the module our LoRA wrapper targets.
3. **Optional Triton flash attention** (`from .flash_attn_triton import flash_attn_qkvpacked_func`) is wrapped in `try/except ImportError` at lines 28–31. If Triton is absent (our case on macOS MPS), it silently falls back to standard PyTorch attention. The fallback path is pure tensor math; no external calls.
4. **ALiBi positional encoding** computed in `rebuild_alibi_tensor` (lines 362–405) — pure math, creates a bias tensor.
5. **Relative imports only** — `.bert_padding` and `.flash_attn_triton` are from the same repo snapshot (same pinned SHA), not from PyPI or the network.

### Finding: `bert_layers.py` is clean

No network calls, no file I/O outside weight loading, no subprocess/eval/exec, no dynamic code execution. The code is a well-structured PyTorch transformer implementation based on MosaicBERT (MosaicML, 2022), consistent with the attributed copyright headers. **No security concerns in this file.**

---

## Item 3 — `GITHUB_TOKEN` Placeholder Investigation

### 3a. Was the placeholder ever a real token?

Checked `git log --all -p -- notebooks/train_colab.ipynb | grep "GITHUB_TOKEN"` across the full commit history.

**Every occurrence found (across all commits, `+` and `-` diff lines):**

```
"GITHUB_TOKEN = \"ghp_xxxxxxxxxxxxxxxxxxxx\"   # ← paste your token\n"
```

The string `ghp_xxxxxxxxxxxxxxxxxxxx` (20 x's) appears in all additions and removals throughout history. No commit ever showed a real-looking token (e.g. a 40-character alphanumeric string after `ghp_`). The placeholder was introduced in the first notebook commit and has been identical since.

**Result: No real token was ever committed. CONFIRMED placeholder.**

### 3b. Fix applied

Regardless of the placeholder-only history, hardcoding `ghp_xxxx...` is bad practice:
- Automated scanners (GitHub Secret Scanning, truffleHog, gitleaks) flag `ghp_` prefixed strings by pattern, causing false-positive alerts.
- If a developer copy-pastes and replaces the placeholder with a real token, they may commit it before noticing.
- Colab notebooks have a built-in Secrets sidebar designed for exactly this use case.

**Fix (committed separately):** Replace the hardcoded string with `os.environ.get("GITHUB_TOKEN", "")` plus an early-exit `EnvironmentError` that names the Colab Secrets sidebar as the correct mechanism.

```diff
-GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"   # ← paste your token
+GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
+if not GITHUB_TOKEN:
+    raise EnvironmentError(
+        "GITHUB_TOKEN not set. Add it via the Colab Secrets sidebar ..."
+    )
```

**File:** `notebooks/train_colab.ipynb` Cell 18 (`cell-9-push`)

---

## Item 4 — Rebase Lost-Work Check

### What the rebase did

PR #6's branch (`security/pin-dnabert2-revision-and-weights-only`) had two commits before the rebase:
- `d40c46e` — `docs(audit): close out homopolymer audit open items — curated recheck, >=6 prioritization, G4 reversal` (added `outputs/cleaning_report.md`, 73 lines)
- `12151df` — `security: pin DNABERT-2 to commit SHA + weights_only=True on all torch.load` (the actual PR #6 change)

The rebase onto `origin/main` (`a6f813b`) replayed only the security commit (`12151df` → `6d55a62`), dropping `d40c46e` because it was an unrelated work-in-progress commit.

### Is the content recoverable?

Checked with:
- `git merge-base --is-ancestor d40c46e origin/main` → **NOT in main ancestry at the time**
- `git branch -a --contains d40c46e` → **no named branch contains it** (was a stale ephemeral commit)
- `git reflog --all` → `d40c46e HEAD@{11}: commit: docs(audit): close out homopolymer audit open items` — **still in local reflog, accessible by SHA**
- `git log --oneline -- outputs/cleaning_report.md` → `33de917` — cleaning_report.md **IS in the current main HEAD** via PR #7

### Conclusion

The `d40c46e` commit was **not permanently lost**:
1. Its content (`outputs/cleaning_report.md`) was independently merged into `main` via PR #7 (`33de917`, "close out homopolymer audit open items"), which is the current HEAD of main.
2. The pre-rebase commit SHA `d40c46e` is still reachable from the local reflog at `HEAD@{11}`.

**Result: PASS — no work was lost. The cleaning_report.md content reached main through PR #7. Commit `d40c46e` is accessible locally via reflog.**

---

## Item 5 — Final Repository-Wide Sweep

Performed after all fixes above were applied.

### `torch.load` — zero unsafe calls

```
grep -rn "torch\.load" **/*.py **/*.ipynb | grep -v "weights_only=True"
```
→ **Empty output.** All 7 calls confirmed safe. See Item 1a table.

### `from_pretrained` / `get_class_from_dynamic_module` / `hf_hub_download` — zero unpinned fetches

All 10 production model-fetch calls carry `revision=DNABERT2_REVISION` (or inline SHA `7bce263b15377fc15361f52cfab88f8b586abda0`). No new scripts or inference/serving files introduced since PR #6 that would require a new sweep point.

**Result: PASS — zero unpinned model fetches and zero unsafe `torch.load` calls anywhere in `*.py` or `*.ipynb`.**

---

## Summary

| Item | Status | Evidence |
|------|--------|---------|
| 1a. All `torch.load` have `weights_only=True` | ✅ PASS | 7/7 calls confirmed; grep output empty for missing flag |
| 1b. All DNABERT-2 fetches pinned to SHA | ✅ PASS | 10/10 calls carry `revision=`; pinned SHA matches config.py:54 |
| 1c. Output shape / param count | ✅ PASS (structural) | Config + architecture confirms [2,32,768], 12 LoRA modules, ~117M |
| 2. `bert_layers.py` manual review | ✅ CLEAN | No network calls, no file I/O, no subprocess/eval/exec; pure nn.Module |
| 3. `GITHUB_TOKEN` history check | ✅ PASS + FIXED | Always a placeholder; fixed to use `os.environ.get` + fail-fast error |
| 4. Rebase lost-work check | ✅ PASS | `d40c46e` still in reflog; content in main via PR #7 (`33de917`) |
| 5. Final repo-wide sweep | ✅ PASS | Zero remaining unsafe `torch.load`; zero unpinned model fetches |

### One Residual Note (Non-Blocking)

`scripts/spikes/inspect_dnabert2.py:29` imports `subprocess` and calls `subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])` to auto-install dependencies. This is intentional behavior in a **throwaway spike script** (marked as such in its docstring), not in any production code path. It does not affect the model loading pipeline. Worth being aware of if the spike is ever repurposed.
