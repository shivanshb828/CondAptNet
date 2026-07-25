# AGENTS.md

CondAptNet — Conditional Aptamer-Protein Interaction Network. A CLI/script-based
PyTorch ML pipeline (no web app, API, or long-running server). See `README.md` and
`CLAUDE.md` for full architecture, data pipeline, and command reference.

## Cursor Cloud specific instructions

### Environment
- Python lives in the `condaptnet_env/` virtualenv (gitignored). Activate it before
  running anything: `source condaptnet_env/bin/activate`. The startup update script
  creates it and installs all dependencies (there is no `requirements.txt`).
- There is **no GPU** in the cloud VM. `config.py` auto-detects and runs on `cpu`.
  Do **not** pass `PYTORCH_ENABLE_MPS_FALLBACK=1` (that is Apple-only) and CUDA flags
  are no-ops here; they are harmless but unnecessary.
- ESM-2 weights (`esm2_t12_35M_UR50D`, ~150MB) download from the internet on the
  first model run and are cached under `~/.cache/torch/hub`. First train/eval run is
  slower because of this download.

### Running things
- Standard commands are documented in `README.md` (Usage) and `CLAUDE.md` (Quick
  Commands). Entry points: `scripts/training/train.py` (Stage 1),
  `scripts/training/finetune.py` (Stage 2/3), `scripts/evaluation/evaluate.py`.
- Each model module has an inline `__main__` shape self-test, e.g.
  `python models/condaptnet.py`, `python scripts/model/tokenizer.py`.
- On CPU, always smoke-test training/eval with small caps instead of a full run.
  Example that completes in well under a minute:
  `python scripts/training/train.py --max-epochs 1 --max-batches 2 --batch-size 4 --max-prot-len 128 --prot-max-tokens 200`
- Training reads the committed protein-family splits in `data/augmented/`
  (`tier1_train.csv`, `val.csv`, `test.csv`). No checkpoints, `vienna_cache.pkl`, or
  precomputed `protein_embeddings/` are shipped — ESM-2 embeddings are computed on the
  fly per run (there is no persisted cache), so every train/eval run recomputes them.
- `train.py` writes checkpoints to `models/checkpoints/pretrain/` (gitignored). Point
  `evaluate.py --checkpoint` at `models/checkpoints/pretrain/best.pt`.

### Tests
- Run the suite with `python -m pytest tests/ -q` (mostly the scraper + finetune
  units; no pytest config file, no CI).
- Known pre-existing failure (not an environment problem):
  `tests/test_finetune.py::TestFinetuneCLI::test_missing_pretrain_checkpoint_warns_not_crashes`
  expects `data/processed/master_dataset_cleaned.csv`, which is not committed (the repo
  ships `data/processed/master_dataset_v2.csv` instead). All other tests pass.
