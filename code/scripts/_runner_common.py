"""Shared helpers for the train_ae / train_aae / compare scripts.

Avoids duplicating the dataloader+config+plotting boilerplate.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Make ``src`` importable from any of the scripts/ entry points
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_kuka  # noqa: E402
from src.datasets import make_loaders  # noqa: E402
from src.windows import make_windowed_splits  # noqa: E402


def load_config(config_path: Path | None = None) -> dict:
    cfg_path = config_path or Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_data(cfg: dict, batch_size: int | None = None):
    """Load dataset → window splits → DataLoader bundle."""
    ds = load_kuka()
    print(
        f"Loaded {ds.n_features} sensor features; "
        f"Normal={len(ds.normal):,}, Slow={len(ds.slow):,}"
    )
    splits = make_windowed_splits(
        ds.normal,
        ds.slow,
        ds.feature_columns,
        window=cfg["windows"]["window"],
        stride=cfg["windows"]["stride"],
        val_frac=cfg["windows"]["val_frac"],
        test_frac=cfg["windows"]["test_frac"],
        seed=cfg["seed"],
    )
    print(
        f"Windows ({splits.window}x{splits.n_features}, stride {splits.stride}): "
        f"train={len(splits.X_train):,}, val_normal={len(splits.X_val_normal):,}, "
        f"test_normal={len(splits.X_test_normal):,}, test_slow={len(splits.X_test_slow):,}"
    )
    loaders = make_loaders(
        splits,
        batch_size=batch_size or cfg["train"]["batch_size"],
        num_workers=0,
        seed=cfg["seed"],
    )
    return ds, splits, loaders


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_safe)


def _safe(o):
    """JSON-serialise numpy / dataclass payloads."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return asdict(o) if hasattr(o, "__dataclass_fields__") else o.__dict__
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def plot_loss_curves(result, *, title: str, fname: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.train_losses, label="train recon")
    ax.plot(result.val_losses, label="val recon")
    if result.disc_losses:
        ax.plot(result.disc_losses, label="disc loss", linestyle="--", alpha=0.7)
        ax.plot(result.gen_losses, label="gen loss", linestyle="--", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(scores, labels, *, title: str, fname: Path, threshold: float | None = None) -> None:
    """Histogram of anomaly scores split by label."""
    fig, ax = plt.subplots(figsize=(7, 4))
    s_norm = scores[labels == 0]
    s_slow = scores[labels == 1]
    bins = np.linspace(min(scores.min(), 0), scores.max(), 80)
    ax.hist(s_norm, bins=bins, alpha=0.55, color="#4c72b0", label=f"Normal (n={len(s_norm)})")
    ax.hist(s_slow, bins=bins, alpha=0.55, color="#dd8452", label=f"Slow (n={len(s_slow)})")
    if threshold is not None:
        ax.axvline(threshold, color="black", linewidth=1, linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_xlabel("per-window reconstruction MSE")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(scores, labels, *, title: str, fname_roc: Path, fname_pr: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return  # cannot draw ROC/PR with a single class
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4c72b0")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {title}")
    fig.tight_layout()
    fname_roc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname_roc, dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#dd8452")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall — {title}")
    fig.tight_layout()
    fig.savefig(fname_pr, dpi=130, bbox_inches="tight")
    plt.close(fig)
