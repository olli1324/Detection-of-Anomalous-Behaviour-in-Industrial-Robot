"""Train + evaluate the baseline 1D-conv autoencoder on the Kuka dataset.

Run from ``code/``::

    python scripts/train_ae.py

Saves into ``code/results/ae/``:

- ``ae_loss_curves.png``         — training/validation loss
- ``ae_score_hist.png``          — anomaly score distribution Normal vs Slow
- ``ae_roc.png``, ``ae_pr.png``  — ROC + PR curves on the test set
- ``ae_metrics.json``            — full metrics report (incl. per-action)
- ``ae_scores.npz``              — raw scores/labels/actions for later analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Make ``src`` and shared runner importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _runner_common import (  # noqa: E402
    build_data,
    load_config,
    plot_loss_curves,
    plot_roc_pr,
    plot_score_distribution,
    save_json,
)
from src.metrics import compute_metrics  # noqa: E402
from src.models.ae import AEConfig, ConvAE  # noqa: E402
from src.scoring import threshold_at_fpr, threshold_max_f1  # noqa: E402
from src.training import collect_anomaly_scores, train_ae  # noqa: E402
from src.utils import results_dir, seed_everything, select_device  # noqa: E402


def main(epochs: int | None = None) -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])

    out = results_dir() / "ae"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data and building loaders...")
    ds, splits, loaders = build_data(cfg)

    print("[2/4] Building model...")
    ae_cfg = AEConfig(
        n_features=splits.n_features,
        window=splits.window,
        channels=tuple(cfg["model"]["channels"]),
        kernel_size=cfg["model"]["kernel_size"],
        latent_dim=cfg["model"]["latent_dim"],
    )
    model = ConvAE(ae_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ConvAE: {n_params:,} parameters")

    n_epochs = epochs if epochs is not None else cfg["train"]["n_epochs"]
    device = select_device()
    print(f"[3/4] Training on device: {device} for {n_epochs} epochs")

    result = train_ae(
        model,
        loaders.train,
        loaders.val_normal,
        n_epochs=n_epochs,
        lr=cfg["train"]["lr_ae"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
    )

    plot_loss_curves(result, title="Baseline AE loss", fname=out / "ae_loss_curves.png")

    print("[4/4] Evaluating + thresholding...")

    val_scores, _, _ = collect_anomaly_scores(result.model, loaders.val_normal, device)
    test_scores, test_labels, test_actions = collect_anomaly_scores(
        result.model, loaders.stacked_test, device
    )

    thr_fpr = threshold_at_fpr(val_scores, fpr=cfg["eval"]["fpr_target"])
    thr_f1 = threshold_max_f1(test_scores, test_labels)

    metrics_fpr = compute_metrics(test_scores, test_labels, thr_fpr.value, actions=test_actions)
    metrics_f1 = compute_metrics(test_scores, test_labels, thr_f1.value, actions=test_actions)

    plot_score_distribution(
        test_scores,
        test_labels,
        title="AE — anomaly score (test)",
        fname=out / "ae_score_hist.png",
        threshold=thr_fpr.value,
    )
    plot_roc_pr(
        test_scores,
        test_labels,
        title="Baseline AE",
        fname_roc=out / "ae_roc.png",
        fname_pr=out / "ae_pr.png",
    )

    np.savez_compressed(
        out / "ae_scores.npz",
        scores=test_scores,
        labels=test_labels,
        actions=test_actions,
    )
    payload = {
        "model": "ConvAE",
        "n_parameters": int(n_params),
        "device": str(device),
        "config": cfg,
        "n_epochs": int(n_epochs),
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "thresholds": {
            "fpr": {
                "value": thr_fpr.value,
                "method": thr_fpr.method,
                "detail": thr_fpr.detail,
            },
            "max_f1_test": {
                "value": thr_f1.value,
                "method": thr_f1.method,
                "detail": thr_f1.detail,
            },
        },
        "metrics_at_fpr_threshold": metrics_fpr.as_dict(),
        "metrics_at_max_f1_threshold": metrics_f1.as_dict(),
    }
    save_json(out / "ae_metrics.json", payload)

    print()
    print(
        f"AE @ FPR-{cfg['eval']['fpr_target']:.0%} threshold ({thr_fpr.value:.4f}):"
        f"  ROC-AUC {metrics_fpr.roc_auc:.4f}  PR-AUC {metrics_fpr.pr_auc:.4f}"
        f"  F1 {metrics_fpr.f1_at_thr:.4f}"
    )
    print(
        f"AE @ best-F1 (cheat) threshold ({thr_f1.value:.4f}):"
        f"  ROC-AUC {metrics_f1.roc_auc:.4f}  F1 {metrics_f1.f1_at_thr:.4f}"
    )
    print(f"Artifacts saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None, help="Override train.n_epochs")
    args = p.parse_args()
    main(epochs=args.epochs)
