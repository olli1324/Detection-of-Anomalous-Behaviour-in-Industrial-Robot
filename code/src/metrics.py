"""Anomaly-detection metrics for time-series models.

We deliberately compute everything **per window**. Per-window metrics
- match the granularity of the anomaly score the model produces;
- avoid the well-known "point-adjust" pitfall described in
  Kim et al., "Towards a Rigorous Evaluation of Time-series Anomaly
  Detection" (NeurIPS 2022); that trick treats any flagged window in
  a labelled segment as a full true positive, which inflates F1.

For completeness we also report per-action metrics so we can see whether
the model performs uniformly across actions (or, for example, struggles
on the action 9 missing-from-Slow case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class WindowMetrics:
    roc_auc: float
    pr_auc: float
    f1_at_thr: float
    precision_at_thr: float
    recall_at_thr: float
    threshold: float
    n_pos: int
    n_neg: int
    per_action: dict[int, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1_at_thr": self.f1_at_thr,
            "precision_at_thr": self.precision_at_thr,
            "recall_at_thr": self.recall_at_thr,
            "threshold": self.threshold,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "per_action": self.per_action,
        }


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    actions: np.ndarray | None = None,
) -> WindowMetrics:
    """Compute the per-window metrics report for a trained detector."""
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} vs labels {labels.shape}")
    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError(f"labels must be 0/1, got {np.unique(labels)}")

    pred = (scores >= threshold).astype(np.int64)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        # Degenerate single-class evaluation, so AUCs are undefined
        roc = float("nan")
        pr = float("nan")
    else:
        roc = float(roc_auc_score(labels, scores))
        pr = float(average_precision_score(labels, scores))

    f1 = float(f1_score(labels, pred, zero_division=0))
    prec = float(precision_score(labels, pred, zero_division=0))
    rec = float(recall_score(labels, pred, zero_division=0))

    per_action: dict[int, dict[str, float]] = {}
    if actions is not None:
        for a in np.unique(actions):
            mask = actions == a
            if mask.sum() == 0:
                continue
            sub_lbl = labels[mask]
            sub_pred = pred[mask]
            row: dict[str, float] = {
                "n": int(mask.sum()),
                "n_pos": int(sub_lbl.sum()),
                "n_neg": int((1 - sub_lbl).sum()),
            }
            if row["n_pos"] > 0 and row["n_neg"] > 0:
                row["roc_auc"] = float(roc_auc_score(sub_lbl, scores[mask]))
                row["pr_auc"] = float(average_precision_score(sub_lbl, scores[mask]))
            row["f1"] = float(f1_score(sub_lbl, sub_pred, zero_division=0))
            row["precision"] = float(precision_score(sub_lbl, sub_pred, zero_division=0))
            row["recall"] = float(recall_score(sub_lbl, sub_pred, zero_division=0))
            per_action[int(a)] = row

    return WindowMetrics(
        roc_auc=roc,
        pr_auc=pr,
        f1_at_thr=f1,
        precision_at_thr=prec,
        recall_at_thr=rec,
        threshold=float(threshold),
        n_pos=int(labels.sum()),
        n_neg=int((1 - labels).sum()),
        per_action=per_action,
    )
