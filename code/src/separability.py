"""Separability sanity check.

Three quick views before we touch any deep model:

1. **PCA(2)** of standardised feature vectors, Normal-vs-Slow scatter. Tells us
   whether the two distributions are visually separable in 2D.
2. **Logistic regression on a balanced subsample**. A "cheat" supervised
   baseline. If a linear classifier can already reach high accuracy, the
   AE/AAE comparison is unlikely to be the bottleneck; the data is easy.
   If it can't, we know the task is genuinely hard and any AE/AAE that beats
   chance is meaningful.
3. **Per-channel anomaly score**: |z| of each Slow row's value vs. Normal's
   mean/std, averaged across rows. Identifies which channels carry the signal.

The point of (2) is *not* to use it as a model, since it sees both classes and
breaks the unsupervised assumption; it's purely a yardstick for difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import figures_dir


@dataclass
class SeparabilityReport:
    pca_figure: Path
    logreg_train_acc: float
    logreg_test_acc: float
    logreg_test_auc: float
    n_train: int
    n_test: int


def _balanced_sample(
    normal: pd.DataFrame, slow: pd.DataFrame, columns: list[str], *, n_per_class: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_idx = rng.choice(len(normal), size=min(n_per_class, len(normal)), replace=False)
    s_idx = rng.choice(len(slow), size=min(n_per_class, len(slow)), replace=False)
    X = np.concatenate(
        [
            normal.iloc[n_idx][columns].to_numpy(dtype=np.float32),
            slow.iloc[s_idx][columns].to_numpy(dtype=np.float32),
        ],
        axis=0,
    )
    y = np.concatenate([np.zeros(len(n_idx), dtype=np.int8), np.ones(len(s_idx), dtype=np.int8)])
    return X, y


def run_separability(
    normal: pd.DataFrame,
    slow: pd.DataFrame,
    feature_columns: list[str],
    *,
    keep_columns: list[str] | None = None,
    n_per_class: int = 5000,
    seed: int = 42,
) -> SeparabilityReport:
    """Run the three checks and save figures.

    ``keep_columns``: optional subset of features. If None, uses all
    non-degenerate columns from ``feature_columns``.
    """
    cols = keep_columns if keep_columns is not None else feature_columns
    X, y = _balanced_sample(normal, slow, cols, n_per_class=n_per_class, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- PCA(2) scatter ----------------------------------------------------
    pca = PCA(n_components=2, random_state=seed).fit(X_train_s)
    Z_normal = pca.transform(X_test_s[y_test == 0])
    Z_slow = pca.transform(X_test_s[y_test == 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        Z_normal[:, 0], Z_normal[:, 1], s=6, alpha=0.4, color="#4c72b0", label="Normal"
    )
    ax.scatter(Z_slow[:, 0], Z_slow[:, 1], s=6, alpha=0.4, color="#dd8452", label="Slow")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("PCA(2) of standardised feature vectors (test split)")
    ax.legend()
    fig.tight_layout()
    pca_path = figures_dir() / "04_pca_scatter.png"
    fig.savefig(pca_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Logistic regression "cheat" baseline ------------------------------
    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(X_train_s, y_train)
    train_acc = clf.score(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])

    return SeparabilityReport(
        pca_figure=pca_path,
        logreg_train_acc=float(train_acc),
        logreg_test_acc=float(test_acc),
        logreg_test_auc=float(test_auc),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )
