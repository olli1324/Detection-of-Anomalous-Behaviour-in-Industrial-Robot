"""Threshold calibration on the Normal validation split.

Two strategies are exposed:

- :func:`threshold_at_fpr` — pick the threshold so that at most ``fpr``
  of Normal validation windows are flagged as anomalies. This is the
  *clean* unsupervised choice and the one we report by default.
- :func:`threshold_max_f1` — exhaustive search for the threshold that
  maximises F1 on a labelled subset (val_normal + a small held-back
  Slow sample). This is a "best case" upper bound; useful for context
  but biased, since it sees Slow during calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Threshold:
    value: float
    method: str
    detail: str = ""


def threshold_at_fpr(normal_scores: np.ndarray, *, fpr: float = 0.05) -> Threshold:
    """Threshold that yields a target false-positive rate on Normal val.

    Fully unsupervised: only sees Normal data.
    """
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    q = 1 - fpr
    val = float(np.quantile(normal_scores, q))
    return Threshold(value=val, method="fpr", detail=f"fpr_target={fpr}")


def threshold_max_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_steps: int = 200,
) -> Threshold:
    """Best-F1 threshold on a labelled (Normal + Slow) score array.

    NOTE: This violates the unsupervised assumption (it sees Slow labels).
    Use only as a "best case" reference, never as the primary metric.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} vs labels {labels.shape}")
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return Threshold(value=lo, method="max_f1", detail="degenerate")
    grid = np.linspace(lo, hi, n_steps)
    best_f1 = -1.0
    best_thr = lo
    for t in grid:
        pred = (scores >= t).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return Threshold(
        value=best_thr,
        method="max_f1",
        detail=f"best_f1={best_f1:.4f}",
    )
