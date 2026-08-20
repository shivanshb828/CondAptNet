"""
Overfit-a-tiny-batch diagnostic.

Trains from RANDOM INIT on 16-20 carefully selected rows for 500 steps
with no regularization and high LR. The single question:

  Can the model drive loss to near-zero and produce genuinely different
  output probabilities for different inputs?

YES → architecture can learn input-dependent features; full-data collapse
      is an epochs/sampling problem, not a structural one.
NO  → something structural is blocking input-dependent gradients (dead
      path, zeroed input, cross-attention collapse, etc.) — more epochs
      on full data will never fix it.
"""

import sys
import os
import pickle
import hashlib
import glob
import random
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config import (
    VIENNA_CACHE, DATA_AUGMENTED, DATA_PROCESSED,
    DEFAULT_PH, DEFAULT_SALT_MM, DEFAULT_TEMP_C, DEFAULT_BUFFER, DEFAULT_MG_MM,
    PROT_MAX_TOKENS, DNA_MAX_LEN,
)
from models.condaptnet import CondAptNet
from scripts.model.tokenizer import DNATokenizer

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ── Device ─────────────────────────────────────────────────────────────────────
# Force CPU: the 17-block CNN interaction matrix [1, L_apt, L_prot, C] OOMs on
# MPS even for a single sample, because the full activation map must stay alive
# through all 17 blocks for backward (gradient checkpointing disabled above).
# CPU holds ~32 GB and is safe. Speed is adequate for 400 steps on 8 samples.
device = torch.device("cpu")
print(f"Device: {device} (forced — MPS OOMs on interaction matrix backward)")

# ── Load support data ──────────────────────────────────────────────────────────
with open(VIENNA_CACHE, "rb") as f:
    vcache = pickle.load(f)

emb_dir = os.path.join(DATA_PROCESSED, "protein_embeddings")
hash_to_path = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in glob.glob(os.path.join(emb_dir, "*.npy"))
}

def seq_hash(s):
    return hashlib.md5(f"{PROT_MAX_TOKENS}:{s}".encode()).hexdigest()

tok = DNATokenizer()

# ── Select tiny balanced batch ─────────────────────────────────────────────────
val = pd.read_csv(os.path.join(DATA_AUGMENTED, "val.csv"))

def row_is_usable(row):
    seq  = str(row["aptamer_sequence"]).upper()
    prot = str(row["protein_sequence"])
    if pd.isna(row["protein_sequence"]):
        return False
    # Must have real Vienna features (non-zero)
    if seq not in vcache:
        return False
    d = vcache[seq]
    if d.get("mfe", 0.0) == 0.0 and d.get("stem_count", 0) == 0:
        return False
    # Must have local protein embedding
    h = seq_hash(prot)
    if h not in hash_to_path:
        return False
    return True

usable = val[val.apply(row_is_usable, axis=1)].copy()
pos = usable[usable["label"] == 1]
neg = usable[usable["label"] == 0]
print(f"Usable rows — pos: {len(pos)}, neg: {len(neg)}")

# Pick 4 positives and 4 negatives — small enough to run per-sample forward
# passes without OOM. Diverse targets maximise the diagnostic value.
random.seed(42)
n_each = min(4, len(pos), len(neg))
chosen_pos = pos.sample(n=n_each, random_state=42)
chosen_neg = neg.sample(n=n_each, random_state=42)
tiny = pd.concat([chosen_pos, chosen_neg]).reset_index(drop=True)
print(f"\nTiny batch: {len(tiny)} rows  ({n_each} pos + {n_each} neg)")
print("Targets:", tiny["target_name"].tolist())

# ── Build input tensors ────────────────────────────────────────────────────────
def make_tensors(row):
    seq  = str(row["aptamer_sequence"]).upper()
    prot = str(row["protein_sequence"])

    ids = tok.encode_padded(seq, DNA_MAX_LEN)
    apt = (ids.clone().detach() if isinstance(ids, torch.Tensor)
           else torch.tensor(ids, dtype=torch.long))

    d = vcache[seq]
    v = torch.tensor([
        d.get("mfe", 0.0),
        float(d.get("stem_count", 0)),
        float(d.get("loop_count", 0)),
        d.get("bp_prob_mean", 0.0),
        d.get("bp_prob_max", 0.0),
        len(seq) / DNA_MAX_LEN,
    ], dtype=torch.float32)

    prot_emb = torch.from_numpy(np.load(hash_to_path[seq_hash(prot)])).float()
    if prot_emb.shape[0] > PROT_MAX_TOKENS:
        prot_emb = prot_emb[:PROT_MAX_TOKENS]

    def _f(col, default):
        val_ = row.get(col, default)
        return float(val_) if pd.notna(val_) else float(default)

    cond = torch.tensor([
        _f("ph", DEFAULT_PH),
        _f("na_concentration_mM", DEFAULT_SALT_MM),
        _f("temperature_C", DEFAULT_TEMP_C),
        float(DEFAULT_BUFFER),
        _f("mg_concentration_mM", DEFAULT_MG_MM),
    ], dtype=torch.float32)

    label = torch.tensor([float(row["label"])], dtype=torch.float32)
    prot_tok = torch.zeros(1, dtype=torch.long)  # unused when protein_emb passed

    return apt, v, prot_emb, cond, prot_tok, label

print("\n── Input verification ──")
for i, (_, row) in enumerate(tiny.iterrows()):
    apt, v, prot_emb, cond, _, label = make_tensors(row)
    seq = str(row["aptamer_sequence"]).upper()
    d = vcache[seq]
    print(f"  row {i:2d} label={int(label.item())}  target={row['target_name'][:30]:<30}"
          f"  vienna=[mfe={d['mfe']:.2f} stems={d.get('stem_count',0)}]"
          f"  prot_emb={prot_emb.shape}  cond={cond.tolist()}")

# Pre-build all tensors
samples = [make_tensors(row) for _, row in tiny.iterrows()]
labels_all = torch.tensor([float(row["label"]) for _, row in tiny.iterrows()],
                           dtype=torch.float32)

# ── Model — random init, no dropout, no gradient checkpointing ───────────────
# Gradient checkpointing recomputes 17 CNN blocks on every backward pass,
# making each training step ~10x slower than needed for this diagnostic.
# Patching it to a plain call so the test finishes in minutes, not hours.
_real_ckpt_fn = lambda fn, *args, **kwargs: fn(*args,
    **{k: v for k, v in kwargs.items() if k != 'use_reentrant'})

with mock.patch('torch.utils.checkpoint.checkpoint', side_effect=_real_ckpt_fn):
    model = CondAptNet()
# Also patch at the module level used by cnn_head at runtime
import torch.utils.checkpoint as _tuc
_tuc.checkpoint = _real_ckpt_fn
model = model.to(device)
model.train()

# Override dropout to 0 so regularization can't interfere
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# High LR — we WANT to overfit
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
criterion = nn.BCELoss()  # plain unweighted BCE

# ── Training loop ──────────────────────────────────────────────────────────────
N_STEPS = 400
LOG_EVERY = 40

print(f"\n── Overfit training: {N_STEPS} steps, LR=1e-3, no weight decay, dropout=0 ──")
print(f"  {len(samples)} samples, individual forward passes per step (avoids OOM)")
print(f"{'Step':>6}  {'Loss':>8}  {'Min prob':>9}  {'Max prob':>9}  "
      f"{'Mean pos':>9}  {'Mean neg':>9}  {'Std all':>8}")

for step in range(1, N_STEPS + 1):
    model.train()
    optimizer.zero_grad()
    all_probs = []

    for apt, v, prot_emb, cond, prot_tok, label in samples:
        out = model(
            apt.unsqueeze(0).to(device),
            v.unsqueeze(0).to(device),
            prot_tok.unsqueeze(0).to(device),
            cond.unsqueeze(0).to(device),
            protein_emb=prot_emb.unsqueeze(0).to(device),
        )
        all_probs.append(out.binding_prob)

    probs_t  = torch.cat(all_probs, dim=0).squeeze(-1)     # [N]
    labels_d = labels_all.to(device)
    loss = criterion(probs_t, labels_d)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % LOG_EVERY == 0 or step == 1:
        with torch.no_grad():
            p = probs_t.detach().cpu().numpy()
            pos_mask = labels_all.numpy() == 1
            neg_mask = ~pos_mask
            print(f"{step:>6}  {loss.item():>8.4f}  {p.min():>9.4f}  {p.max():>9.4f}  "
                  f"{p[pos_mask].mean():>9.4f}  {p[neg_mask].mean():>9.4f}  "
                  f"{p.std():>8.4f}")

# ── Final verdict ──────────────────────────────────────────────────────────────
print("\n── Final output probabilities ──")
model.eval()
final_probs = []
with torch.no_grad():
    for apt, v, prot_emb, cond, prot_tok, label in samples:
        out = model(apt.unsqueeze(0).to(device), v.unsqueeze(0).to(device),
                    prot_tok.unsqueeze(0).to(device), cond.unsqueeze(0).to(device),
                    protein_emb=prot_emb.unsqueeze(0).to(device))
        final_probs.append((out.binding_prob.item(), int(label.item())))

final_probs.sort(key=lambda x: x[0], reverse=True)
for prob, lbl in final_probs:
    bar = "+" if lbl == 1 else "-"
    print(f"  [{bar}]  {prob:.4f}  {'████' * int(prob * 20)}")

p_arr = np.array([x[0] for x in final_probs])
l_arr = np.array([x[1] for x in final_probs])
pos_mean = p_arr[l_arr == 1].mean()
neg_mean = p_arr[l_arr == 0].mean()
separation = pos_mean - neg_mean
final_loss = criterion(
    torch.tensor(p_arr, dtype=torch.float32),
    torch.tensor(l_arr, dtype=torch.float32)
).item()

print(f"\nFinal loss:      {final_loss:.4f}")
print(f"Mean prob pos:   {pos_mean:.4f}")
print(f"Mean prob neg:   {neg_mean:.4f}")
print(f"Separation:      {separation:+.4f}")
print(f"Std all probs:   {p_arr.std():.4f}")

print("\n── VERDICT ──")
if final_loss < 0.15 and p_arr.std() > 0.15 and separation > 0.20:
    print("YES — model memorized the tiny batch.")
    print("Outputs are genuinely differentiated. Architecture CAN learn input-dependent")
    print("features. Full-data collapse is an epochs/sampling problem.")
    print("→ Next step: balanced sampling + more training epochs.")
elif final_loss < 0.35 and p_arr.std() > 0.08:
    print("PARTIAL — model is learning but hasn't fully memorized.")
    print("Some differentiation visible. May need more steps or higher LR.")
    print("Increase N_STEPS to 1500 and re-check before concluding.")
else:
    print("NO — model failed to memorize 16 examples.")
    print(f"Loss={final_loss:.4f}, std={p_arr.std():.4f}, separation={separation:+.4f}")
    print("Something structural is blocking input-dependent learning.")
    print("→ Architectural debugging required before more training.")
