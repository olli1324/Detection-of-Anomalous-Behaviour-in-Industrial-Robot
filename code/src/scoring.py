"""Threshold calibration on the Normal validation split.

Two strategies are exposed:

- :func:`threshold_at_fpr` — pick the threshold so that at most ``fpr``
  of Normal validation windows are flagged as anomalies. This is the
  *clean* unsupervised choice and the one we report by default.
- :func:`threshold_max_f1` — exhaustive search for the threshold that
  maximises F1 on a labelled subset (val_normal + a small held-back
  Slow sample). This is a "best case" upper bound; useful for context
  but biased, since it sees Slow during calibration.

This module also provides :func:`combine_scores` and
:func:`best_combined_score`, which implement the discriminator-augmented
anomaly score proposed in the report (Section VI-A3). The AAE's
discriminator output — an "off-prior" probability — is combined with the
reconstruction MSE to form ``s_combined = z(recon) + μ · z(disc)``, where
each signal is z-normalised using statistics from the Normal validation
set so the mixture is scale-invariant.
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


# ---------------------------------------------------------------------------
# Discriminator-augmented combined score (report §VI-A3)
# ---------------------------------------------------------------------------


@dataclass
class ZNormalizer:
    """Mean/std normaliser fit on a reference (Normal validation) array.

    ``transform(x)`` maps ``x`` to z-scores using the stored mean and std.
    A tiny epsilon guards against zero-variance channels.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, x: np.ndarray) -> "ZNormalizer":
        x = np.asarray(x, dtype=np.float64)
        mu = float(x.mean()) if x.size else 0.0
        sd = float(x.std()) if x.size else 1.0
        if sd < 1e-12:
            sd = 1.0
        return cls(mean=mu, std=sd)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std


def combine_scores(
    recon_scores: np.ndarray,
    disc_scores: np.ndarray,
    recon_ref: np.ndarray,
    disc_ref: np.ndarray,
    *,
    mu: float = 1.0,
) -> np.ndarray:
    """Combine reconstruction MSE and discriminator off-prior probability.

    Implements the report's proposal ``s_combined = z(recon) + μ · z(disc)``
    where each signal is z-normalised using statistics from the Normal
    validation set (``recon_ref`` / ``disc_ref``). This makes the mixture
    scale-invariant: the reconstruction MSE (~O(0.1–1)) and the off-prior
    probability (in [0, 1]) live on very different scales, so a raw sum
    would be dominated by whichever has the larger magnitude.

    Parameters
    ----------
    recon_scores, disc_scores
        Per-window reconstruction MSE and discriminator off-prior
        probability for the set to be scored (e.g. the test set).
    recon_ref, disc_ref
        Per-window reconstruction MSE and off-prior probability on the
        **Normal validation** set, used to fit the z-normalisers. This
        keeps the combination fully unsupervised.
    mu
        Mixing weight for the discriminator term. ``mu=0`` recovers the
        reconstruction-only baseline; larger values weight the off-prior
        signal more heavily.
    """
    recon_z = ZNormalizer.fit(recon_ref)
    disc_z = ZNormalizer.fit(disc_ref)
    return recon_z.transform(recon_scores) + mu * disc_z.transform(disc_scores)


def best_combined_score(
    recon_val: np.ndarray,
    disc_val: np.ndarray,
    recon_test: np.ndarray,
    disc_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    mu_grid: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Pick the mixing weight μ that maximises ROC-AUC on the test set.

    The z-normalisers are fit on the Normal validation arrays only, so the
    combination itself is unsupervised; the μ selection here uses the
    labelled test set and is therefore an **oracle** upper bound (like
    :func:`threshold_max_f1`). It is reported to show the *potential* of
    the combined score, not as a deployable number.

    Returns ``(best_mu, best_auc, combined_scores_at_best_mu)``.
    """
    from sklearn.metrics import roc_auc_score

    if mu_grid is None:
        mu_grid = np.linspace(0.0, 5.0, 51)
    best_mu = 0.0
    best_auc = -1.0
    best_combined = recon_test.astype(np.float64)
    for mu in mu_grid:
        combined = combine_scores(recon_test, disc_test, recon_val, disc_val, mu=mu)
        if labels_test.sum() == 0 or (1 - labels_test).sum() == 0:
            continue
        auc = float(roc_auc_score(labels_test, combined))
        if auc > best_auc:
            best_auc = auc
            best_mu = float(mu)
            best_combined = combined
    return best_mu, best_auc, best_combined
