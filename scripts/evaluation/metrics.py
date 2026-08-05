"""
Evaluation metrics for CondAptNet.

Primary metric: MCC (Matthews Correlation Coefficient).
MCC is preferred over accuracy for imbalanced binary classification.

All functions accept numpy arrays or Python lists.

Functions:
    compute_metrics(y_true, y_prob, kd_true=None, kd_pred=None) → dict
    threshold_metrics(y_true, y_prob, threshold=0.5) → dict
    print_metrics(metrics_dict) → None

Usage:
    from scripts.evaluation.metrics import compute_metrics
    m = compute_metrics(labels, probs, kd_true=kd_vals, kd_pred=kd_preds)
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EVAL_METRICS


def compute_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
    kd_true=None,
    kd_pred=None,
) -> dict:
    """
    Compute the full CondAptNet evaluation suite.

    Args:
        y_true    : (N,) binary labels {0, 1}
        y_prob    : (N,) predicted probabilities ∈ [0, 1]
        threshold : decision threshold for binary metrics (default 0.5)
        kd_true   : (M,) ground-truth Kd values (log10 nM+1), may contain NaN
        kd_pred   : (M,) predicted Kd values

    Returns:
        dict with keys: mcc, auroc, auprc, sensitivity, specificity,
                        pearson_r_kd, pearson_p_kd, n_kd_pairs
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(float)

    # ── Threshold-dependent ───────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Detect degenerate MCC: sklearn returns 0.0 *silently* when the denominator
    # is zero (any of the four marginals is zero). This is mathematically undefined
    # (0/0), not a genuine "no correlation" result — it looks identical to a real
    # MCC=0.0 in logs and is impossible to distinguish without this flag.
    # Common when the model predicts all-positive (TN=FN=0) or all-negative (TP=FP=0).
    mcc_degenerate = int((tn + fn) == 0 or (tn + fp) == 0 or
                         (tp + fp) == 0 or (tp + fn) == 0)
    mcc = matthews_corrcoef(y_true, y_pred) if not mcc_degenerate else float("nan")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Threshold-independent ─────────────────────────────────────────────────
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = float("nan")
        auprc = float("nan")

    # ── Kd regression (only where both are available and non-NaN) ────────────
    pearson_r = float("nan")
    pearson_p = float("nan")
    n_kd      = 0
    if kd_true is not None and kd_pred is not None:
        kt = np.asarray(kd_true, dtype=float)
        kp = np.asarray(kd_pred, dtype=float)
        valid = ~(np.isnan(kt) | np.isnan(kp))
        n_kd = int(valid.sum())
        if n_kd >= 3:
            r, p = pearsonr(kt[valid], kp[valid])
            pearson_r = float(r)
            pearson_p = float(p)

    return {
        "mcc":            float(mcc) if not mcc_degenerate else float("nan"),
        "mcc_degenerate": bool(mcc_degenerate),
        "auroc":          float(auroc),
        "auprc":          float(auprc),
        "sensitivity":    float(sensitivity),
        "specificity":    float(specificity),
        "pearson_r_kd":   pearson_r,
        "pearson_p_kd":   pearson_p,
        "n_kd_pairs":     n_kd,
    }


def print_metrics(m: dict, prefix: str = "") -> None:
    """Pretty-print the metrics dict. AUC-ROC leads; MCC flagged if degenerate."""
    prefix = f"{prefix} | " if prefix else ""
    kd_str = (f"  Kd Pearson r={m['pearson_r_kd']:.3f} (n={m['n_kd_pairs']})"
               if m.get("n_kd_pairs", 0) >= 3 else "")
    if m.get("mcc_degenerate"):
        mcc_str = "MCC=UNDEF(0/0 — all-pos or all-neg predictions)"
    else:
        mcc_str = f"MCC={m['mcc']:.3f}"
    print(
        f"{prefix}"
        f"AUC-ROC={m['auroc']:.3f}  "
        f"AUC-PR={m['auprc']:.3f}  "
        f"{mcc_str}  "
        f"Sens={m['sensitivity']:.3f}  "
        f"Spec={m['specificity']:.3f}"
        f"{kd_str}"
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 200
    y_true = rng.integers(0, 2, N)
    y_prob = np.clip(y_true * 0.7 + rng.normal(0, 0.2, N), 0.0, 1.0)
    kd_true = rng.uniform(0, 4, N)
    kd_pred = kd_true + rng.normal(0, 0.5, N)
    kd_true[rng.integers(0, N, 50)] = float("nan")

    m = compute_metrics(y_true, y_prob, kd_true=kd_true, kd_pred=kd_pred)
    print_metrics(m, prefix="Test")

    assert 0.0 <= m["mcc"] <= 1.0,       f"MCC out of range: {m['mcc']}"
    assert 0.5 <= m["auroc"] <= 1.0,     f"AUROC too low: {m['auroc']}"
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert 0.0 <= m["specificity"] <= 1.0
    assert m["n_kd_pairs"] > 0

    # All NaN kd → n_kd_pairs == 0
    m2 = compute_metrics(y_true, y_prob,
                         kd_true=np.full(N, float("nan")),
                         kd_pred=np.zeros(N))
    assert m2["n_kd_pairs"] == 0

    print("\nmetrics.py tests passed.")
