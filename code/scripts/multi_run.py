"""Multi-seed statistical comparison of AE vs AAE.

Trains both models across 6 fixed seeds, records ROC-AUC and PR-AUC
for each run, then applies a Wilcoxon signed-rank test to determine
whether the performance gap is statistically significant.

Run from code/:
    python scripts/multi_run.py [--epochs N]

Outputs to results/multi_run/:
    results_raw.csv          -- per-seed metrics for both models
    wilcoxon_report.txt      -- statistical test results
    multi_run_summary.png    -- boxplot + per-seed lines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _runner_common import build_data, load_config, save_json          # noqa: E402
from src.metrics import compute_metrics                                  # noqa: E402
from src.models.ae import AEConfig, ConvAE                              # noqa: E402
from src.models.aae import AAEConfig, AdversarialAE                     # noqa: E402
from src.scoring import threshold_at_fpr, threshold_max_f1              # noqa: E402
from src.training import collect_anomaly_scores, train_ae, train_aae    # noqa: E402
from src.utils import results_dir, seed_everything, select_device       # noqa: E402

# Six seeds — minimum needed for Wilcoxon p < 0.05 (min achievable p = 0.03125)
SEEDS = [42, 7, 13, 99, 2024, 314]


def run_one_seed(seed: int, cfg: dict, n_epochs: int, device) -> dict:
    """Train AE and AAE for a single seed, return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    seed_everything(seed)
    cfg_seed = {**cfg, "seed": seed}

    # ------------------------------------------------------------------ data
    ds, splits, loaders = build_data(cfg_seed)

    ae_cfg = AEConfig(
        n_features=splits.n_features,
        window=splits.window,
        channels=tuple(cfg["model"]["channels"]),
        kernel_size=cfg["model"]["kernel_size"],
        latent_dim=cfg["model"]["latent_dim"],
    )

    # ------------------------------------------------------------------ AE
    print(f"\n[Seed {seed}] Training AE for {n_epochs} epochs...")
    ae_model = ConvAE(ae_cfg)
    ae_result = train_ae(
        ae_model, loaders.train, loaders.val_normal,
        n_epochs=n_epochs,
        lr=cfg["train"]["lr_ae"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
        log_every=n_epochs,          # only print final epoch
    )
    val_scores_ae, _, _ = collect_anomaly_scores(
        ae_result.model, loaders.val_normal, device
    )
    test_scores_ae, test_labels, test_actions = collect_anomaly_scores(
        ae_result.model, loaders.stacked_test, device
    )
    thr_ae = threshold_at_fpr(val_scores_ae, fpr=cfg["eval"]["fpr_target"])
    m_ae = compute_metrics(test_scores_ae, test_labels, thr_ae.value,
                           actions=test_actions)
    print(f"  AE  ROC-AUC={m_ae.roc_auc:.4f}  PR-AUC={m_ae.pr_auc:.4f}  "
          f"F1@FPR5%={m_ae.f1_at_thr:.4f}  val_MSE={ae_result.val_losses[-1]:.4f}")

    # ------------------------------------------------------------------ AAE
    print(f"\n[Seed {seed}] Training AAE for {n_epochs} epochs...")
    aae_cfg = AAEConfig(
        ae=ae_cfg,
        discriminator_hidden=tuple(cfg["aae"]["discriminator_hidden"]),
        adv_weight=cfg["aae"]["adv_weight"],
    )
    aae_model = AdversarialAE(aae_cfg)
    aae_result = train_aae(
        aae_model, loaders.train, loaders.val_normal,
        n_epochs=n_epochs,
        lr_ae=cfg["train"]["lr_ae"],
        lr_disc=cfg["train"]["lr_disc"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
        log_every=n_epochs,
    )
    val_scores_aae, _, _ = collect_anomaly_scores(
        aae_result.model, loaders.val_normal, device
    )
    test_scores_aae, _, _ = collect_anomaly_scores(
        aae_result.model, loaders.stacked_test, device
    )
    thr_aae = threshold_at_fpr(val_scores_aae, fpr=cfg["eval"]["fpr_target"])
    m_aae = compute_metrics(test_scores_aae, test_labels, thr_aae.value,
                            actions=test_actions)
    print(f"  AAE ROC-AUC={m_aae.roc_auc:.4f}  PR-AUC={m_aae.pr_auc:.4f}  "
          f"F1@FPR5%={m_aae.f1_at_thr:.4f}  val_MSE={aae_result.val_losses[-1]:.4f}")

    return {
        "seed": seed,
        # AE
        "ae_roc_auc":   m_ae.roc_auc,
        "ae_pr_auc":    m_ae.pr_auc,
        "ae_f1_fpr":    m_ae.f1_at_thr,
        "ae_val_mse":   ae_result.val_losses[-1],
        # AAE
        "aae_roc_auc":  m_aae.roc_auc,
        "aae_pr_auc":   m_aae.pr_auc,
        "aae_f1_fpr":   m_aae.f1_at_thr,
        "aae_val_mse":  aae_result.val_losses[-1],
        # deltas (positive = AE wins)
        "delta_roc_auc": m_ae.roc_auc  - m_aae.roc_auc,
        "delta_pr_auc":  m_ae.pr_auc   - m_aae.pr_auc,
        "delta_f1_fpr":  m_ae.f1_at_thr - m_aae.f1_at_thr,
    }


def wilcoxon_report(df: pd.DataFrame) -> str:
    """Run Wilcoxon signed-rank tests and return a formatted report string."""
    from scipy.stats import wilcoxon

    lines = []
    lines.append("=" * 60)
    lines.append("  WILCOXON SIGNED-RANK TEST  (AE vs AAE, paired by seed)")
    lines.append("  H1: AE > AAE   |   n = {}".format(len(df)))
    lines.append("=" * 60)

    for metric, col_ae, col_aae in [
        ("ROC-AUC",   "ae_roc_auc",  "aae_roc_auc"),
        ("PR-AUC",    "ae_pr_auc",   "aae_pr_auc"),
        ("F1@FPR-5%", "ae_f1_fpr",   "aae_f1_fpr"),
    ]:
        ae_vals  = df[col_ae].values
        aae_vals = df[col_aae].values
        diff = ae_vals - aae_vals

        # AE wins / ties / loses count
        n_win  = int((diff > 0).sum())
        n_tie  = int((diff == 0).sum())
        n_lose = int((diff < 0).sum())

        # Wilcoxon needs at least one non-zero difference
        if (diff == 0).all():
            lines.append(f"\n{metric}: all differences are zero — cannot test.")
            continue

        try:
            stat, p = wilcoxon(ae_vals, aae_vals, alternative="greater",
                               zero_method="wilcox")
            sig = "*** SIGNIFICANT (p < 0.05)" if p < 0.05 else \
                  "~ marginal (p < 0.10)"     if p < 0.10 else \
                  "not significant"
        except Exception as e:
            lines.append(f"\n{metric}: test failed — {e}")
            continue

        lines.append(f"\n{metric}")
        lines.append(f"  AE  mean={ae_vals.mean():.4f}  std={ae_vals.std():.4f}")
        lines.append(f"  AAE mean={aae_vals.mean():.4f}  std={aae_vals.std():.4f}")
        lines.append(f"  AE wins {n_win}/{len(df)} seeds  "
                     f"(ties {n_tie}, losses {n_lose})")
        lines.append(f"  Wilcoxon W={stat:.1f}  p={p:.4f}  → {sig}")

    lines.append("\n" + "=" * 60)
    lines.append("Note: minimum achievable p with n=6 is 0.03125.")
    lines.append("=" * 60)
    return "\n".join(lines)


def plot_summary(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("ROC-AUC",   "ae_roc_auc",  "aae_roc_auc"),
        ("PR-AUC",    "ae_pr_auc",   "aae_pr_auc"),
        ("F1@FPR-5%", "ae_f1_fpr",   "aae_f1_fpr"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, (title, col_ae, col_aae) in zip(axes, metrics):
        ae_vals  = df[col_ae].values
        aae_vals = df[col_aae].values

        # boxplot
        bp = ax.boxplot(
            [ae_vals, aae_vals],
            labels=["AE", "AAE"],
            patch_artist=True,
            widths=0.4,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4c72b0")
        bp["boxes"][1].set_facecolor("#dd8452")

        # per-seed lines
        for ae_v, aae_v in zip(ae_vals, aae_vals):
            ax.plot([1, 2], [ae_v, aae_v],
                    color="gray", linewidth=0.8, alpha=0.6, zorder=3)

        # individual points
        ax.scatter([1] * len(ae_vals),  ae_vals,  color="#4c72b0",
                   zorder=4, s=40)
        ax.scatter([2] * len(aae_vals), aae_vals, color="#dd8452",
                   zorder=4, s=40)

        ax.set_title(title, fontsize=13)
        ax.set_ylabel(title)
        ax.set_xlim(0.5, 2.5)

    fig.suptitle(
        f"AE vs AAE — {len(df)} seeds  "
        f"({df.attrs.get('n_epochs', '?')} epochs each)",
        fontsize=14,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved → {out}")


def main(n_epochs: int | None = None) -> None:
    cfg = load_config()
    device = select_device()
    epochs = n_epochs if n_epochs is not None else cfg["train"]["n_epochs"]

    print(f"Device: {device}")
    print(f"Epochs per run: {epochs}")
    print(f"Seeds: {SEEDS}")
    print(f"Total training runs: {len(SEEDS) * 2} (AE + AAE per seed)")

    out_dir = results_dir() / "multi_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        row = run_one_seed(seed, cfg, epochs, device)
        rows.append(row)
        # Save incrementally so a crash doesn't lose everything
        pd.DataFrame(rows).to_csv(out_dir / "results_raw.csv", index=False)
        print(f"\n  [saved partial results after seed {seed}]")

    df = pd.DataFrame(rows)
    df.attrs["n_epochs"] = epochs
    df.to_csv(out_dir / "results_raw.csv", index=False)

    # Statistical report
    report = wilcoxon_report(df)
    print("\n" + report)
    (out_dir / "wilcoxon_report.txt").write_text(report)

    # Summary plot
    plot_summary(df, out_dir / "multi_run_summary.png")

    # Pretty console table
    print("\nPer-seed results:")
    print(df[[
        "seed",
        "ae_roc_auc", "aae_roc_auc", "delta_roc_auc",
        "ae_pr_auc",  "aae_pr_auc",  "delta_pr_auc",
    ]].to_string(index=False, float_format="{:.4f}".format))

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Epochs per model per seed (default: value in configs/default.yaml)"
    )
    args = p.parse_args()
    main(n_epochs=args.epochs)
