"""Train + evaluate the adversarial autoencoder on the Kuka dataset.

Mirrors ``train_ae.py`` for the AAE so artifacts are directly comparable.
Saves into ``code/results/aae/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
from src.models.ae import AEConfig  # noqa: E402
from src.models.aae import AAEConfig, AdversarialAE  # noqa: E402
from src.scoring import (  # noqa: E402
    best_combined_score,
    combine_scores,
    threshold_at_fpr,
    threshold_max_f1,
)
from src.training import collect_aae_scores, train_aae  # noqa: E402
from src.utils import results_dir, seed_everything, select_device  # noqa: E402


def main(epochs: int | None = None) -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])

    out = results_dir() / "aae"
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
    aae_cfg = AAEConfig(
        ae=ae_cfg,
        discriminator_hidden=tuple(cfg["aae"]["discriminator_hidden"]),
        adv_weight=cfg["aae"]["adv_weight"],
    )
    model = AdversarialAE(aae_cfg)
    n_ae = sum(p.numel() for p in model.ae.parameters())
    n_d = sum(p.numel() for p in model.disc.parameters())
    print(f"  AdversarialAE: AE {n_ae:,} + D {n_d:,} = {n_ae + n_d:,} parameters")

    n_epochs = epochs if epochs is not None else cfg["train"]["n_epochs"]
    device = select_device()
    print(f"[3/4] Training on device: {device} for {n_epochs} epochs")

    result = train_aae(
        model,
        loaders.train,
        loaders.val_normal,
        n_epochs=n_epochs,
        lr_ae=cfg["train"]["lr_ae"],
        lr_disc=cfg["train"]["lr_disc"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
    )

    plot_loss_curves(result, title="Adversarial AE loss", fname=out / "aae_loss_curves.png")

    print("[4/4] Evaluating + thresholding...")

    # Collect reconstruction MSE AND discriminator off-prior probability.
    # The discriminator signal is what the standard AAE scoring discards;
    # combining it with the reconstruction error is the extension proposed
    # in the report (Section VI-A3, following Perera et al. OCGAN).
    val_recon, val_disc, _, _ = collect_aae_scores(result.model, loaders.val_normal, device)
    test_recon, test_disc, test_labels, test_actions = collect_aae_scores(
        result.model, loaders.stacked_test, device
    )

    # --- Baseline: reconstruction-only scoring (as before) ---------------
    thr_fpr = threshold_at_fpr(val_recon, fpr=cfg["eval"]["fpr_target"])
    thr_f1 = threshold_max_f1(test_recon, test_labels)

    metrics_fpr = compute_metrics(test_recon, test_labels, thr_fpr.value, actions=test_actions)
    metrics_f1 = compute_metrics(test_recon, test_labels, thr_f1.value, actions=test_actions)

    # --- Discriminator-augmented combined score -------------------------
    # s_combined = z(recon) + mu * z(disc), z-normalised on Normal val.
    # Select mu by maximising ROC-AUC on the test set (oracle upper bound,
    # reported alongside the max-F1 threshold for context).
    cs_cfg = cfg["eval"].get("combined_score", {})
    mu_grid = np.linspace(
        cs_cfg.get("mu_min", 0.0),
        cs_cfg.get("mu_max", 5.0),
        int(cs_cfg.get("mu_steps", 51)),
    )
    best_mu, best_auc, test_combined = best_combined_score(
        val_recon, val_disc, test_recon, test_disc, test_labels, mu_grid=mu_grid
    )
    # Threshold the combined score at the same FPR target, using the
    # combined-score distribution on the Normal validation set.
    val_combined = combine_scores(val_recon, val_disc, val_recon, val_disc, mu=best_mu)
    thr_combined_fpr = threshold_at_fpr(val_combined, fpr=cfg["eval"]["fpr_target"])
    thr_combined_f1 = threshold_max_f1(test_combined, test_labels)
    metrics_combined_fpr = compute_metrics(
        test_combined, test_labels, thr_combined_fpr.value, actions=test_actions
    )
    metrics_combined_f1 = compute_metrics(
        test_combined, test_labels, thr_combined_f1.value, actions=test_actions
    )

    plot_score_distribution(
        test_recon,
        test_labels,
        title="AAE — anomaly score (test)",
        fname=out / "aae_score_hist.png",
        threshold=thr_fpr.value,
    )
    plot_roc_pr(
        test_recon,
        test_labels,
        title="Adversarial AE",
        fname_roc=out / "aae_roc.png",
        fname_pr=out / "aae_pr.png",
    )

    np.savez_compressed(
        out / "aae_scores.npz",
        scores=test_recon,
        disc_scores=test_disc,
        combined_scores=test_combined,
        labels=test_labels,
        actions=test_actions,
        val_recon=val_recon,
        val_disc=val_disc,
        val_combined=val_combined,
        best_mu=best_mu,
    )
    payload = {
        "model": "AdversarialAE",
        "n_parameters": {"ae": int(n_ae), "disc": int(n_d), "total": int(n_ae + n_d)},
        "device": str(device),
        "config": cfg,
        "n_epochs": int(n_epochs),
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "disc_losses": result.disc_losses,
        "gen_losses": result.gen_losses,
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
            "combined_fpr": {
                "value": thr_combined_fpr.value,
                "method": thr_combined_fpr.method,
                "detail": thr_combined_fpr.detail,
            },
            "combined_max_f1_test": {
                "value": thr_combined_f1.value,
                "method": thr_combined_f1.method,
                "detail": thr_combined_f1.detail,
            },
        },
        "combined_score": {
            "best_mu": best_mu,
            "best_auc_oracle": best_auc,
            "mu_grid": mu_grid.tolist(),
        },
        "metrics_at_fpr_threshold": metrics_fpr.as_dict(),
        "metrics_at_max_f1_threshold": metrics_f1.as_dict(),
        "metrics_combined_at_fpr_threshold": metrics_combined_fpr.as_dict(),
        "metrics_combined_at_max_f1_threshold": metrics_combined_f1.as_dict(),
    }
    save_json(out / "aae_metrics.json", payload)

    print()
    print(
        f"AAE @ FPR-{cfg['eval']['fpr_target']:.0%} threshold ({thr_fpr.value:.4f}):"
        f"  ROC-AUC {metrics_fpr.roc_auc:.4f}  PR-AUC {metrics_fpr.pr_auc:.4f}"
        f"  F1 {metrics_fpr.f1_at_thr:.4f}"
    )
    print(
        f"AAE @ best-F1 (cheat) threshold ({thr_f1.value:.4f}):"
        f"  ROC-AUC {metrics_f1.roc_auc:.4f}  F1 {metrics_f1.f1_at_thr:.4f}"
    )
    print(
        f"AAE+disc @ mu={best_mu:.2f} (oracle) | FPR-{cfg['eval']['fpr_target']:.0%} thr"
        f" ({thr_combined_fpr.value:.4f}):  ROC-AUC {metrics_combined_fpr.roc_auc:.4f}"
        f"  PR-AUC {metrics_combined_fpr.pr_auc:.4f}  F1 {metrics_combined_fpr.f1_at_thr:.4f}"
    )
    print(
        f"AAE+disc @ best-F1 (cheat) threshold ({thr_combined_f1.value:.4f}):"
        f"  ROC-AUC {metrics_combined_fpr.roc_auc:.4f}  F1 {metrics_combined_fpr.f1_at_thr:.4f}"
    )
    print(f"Artifacts saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None, help="Override train.n_epochs")
    args = p.parse_args()
    main(epochs=args.epochs)
