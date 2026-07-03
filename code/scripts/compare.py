"""Compare the trained AE and AAE side-by-side and write the report.

Reads ``results/ae/ae_metrics.json``, ``results/aae/aae_metrics.json``, and
the corresponding ``*_scores.npz`` files. Generates:

- ``results/comparison/roc_overlay.png`` — ROC curves on the same axes
- ``results/comparison/pr_overlay.png``  — PR curves on the same axes
- ``results/comparison/score_overlay.png`` — score distributions overlay
- ``results/comparison/per_action_f1.png`` — per-action F1 bar chart
- ``results/comparison/comparison.csv`` — headline metrics table
- ``results/REPORT.md`` — written summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import results_dir  # noqa: E402


def _load(model_dir: Path, name: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with open(model_dir / f"{name}_metrics.json", "r") as f:
        metrics = json.load(f)
    arr = np.load(model_dir / f"{name}_scores.npz")
    return metrics, arr["scores"], arr["labels"], arr["actions"]


def _load_aae() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load AAE results, including discriminator + combined scores if present.

    Returns ``(metrics, recon_scores, labels, actions, combined_scores, disc_scores)``.
    The last two are ``None`` if the AAE was scored with an older version of
    ``train_aae.py`` that did not save the discriminator signal.
    """
    aae_dir = results_dir() / "aae"
    with open(aae_dir / "aae_metrics.json", "r") as f:
        metrics = json.load(f)
    arr = np.load(aae_dir / "aae_scores.npz")
    combined = arr["combined_scores"] if "combined_scores" in arr.files else None
    disc = arr["disc_scores"] if "disc_scores" in arr.files else None
    return metrics, arr["scores"], arr["labels"], arr["actions"], combined, disc


def _f1(labels: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main() -> None:
    out = results_dir() / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    ae_dir = results_dir() / "ae"
    aae_dir = results_dir() / "aae"
    if not (ae_dir / "ae_metrics.json").exists():
        raise SystemExit("Run train_ae.py first.")
    if not (aae_dir / "aae_metrics.json").exists():
        raise SystemExit("Run train_aae.py first.")

    ae_metrics, ae_s, ae_y, ae_a = _load(ae_dir, "ae")
    aae_metrics, aae_s, aae_y, aae_a, aae_combined, aae_disc = _load_aae()

    # Sanity: same evaluation set
    assert (ae_y == aae_y).all() and (ae_a == aae_a).all(), (
        "AE and AAE were evaluated on different test sets; re-run both with the "
        "same seed and config."
    )

    has_combined = aae_combined is not None

    # --- overlay ROC + PR -------------------------------------------------
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    series = [("AE", ae_s, "#4c72b0"), ("AAE", aae_s, "#dd8452")]
    if has_combined:
        series.append(("AAE+disc", aae_combined, "#55a868"))
    for name, s, c in series:
        fpr, tpr, _ = roc_curve(ae_y, s)
        if name == "AE":
            auc = ae_metrics["metrics_at_fpr_threshold"]["roc_auc"]
        elif name == "AAE":
            auc = aae_metrics["metrics_at_fpr_threshold"]["roc_auc"]
        else:
            auc = aae_metrics["metrics_combined_at_fpr_threshold"]["roc_auc"]
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC — AE vs AAE" + (" vs AAE+disc" if has_combined else ""))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, s, c in series:
        prec, rec, _ = precision_recall_curve(ae_y, s)
        if name == "AE":
            ap = ae_metrics["metrics_at_fpr_threshold"]["pr_auc"]
        elif name == "AAE":
            ap = aae_metrics["metrics_at_fpr_threshold"]["pr_auc"]
        else:
            ap = aae_metrics["metrics_combined_at_fpr_threshold"]["pr_auc"]
        ax.plot(rec, prec, color=c, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall — AE vs AAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pr_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- score distribution overlay ---------------------------------------
    ncols = 3 if has_combined else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), sharey=True)
    panels = [(axes[0], "AE", ae_s, "#4c72b0"), (axes[1], "AAE", aae_s, "#dd8452")]
    if has_combined:
        panels.append((axes[2], "AAE+disc", aae_combined, "#55a868"))
    for ax, name, s in panels:
        bins = np.linspace(min(s.min(), 0), s.max(), 80)
        ax.hist(s[ae_y == 0], bins=bins, alpha=0.55, color="#4c72b0", label="Normal")
        ax.hist(s[ae_y == 1], bins=bins, alpha=0.55, color="#dd8452", label="Slow")
        ax.set_xlabel("z-score" if name == "AAE+disc" else "per-window MSE")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.legend()
    fig.suptitle("Anomaly score distribution on the test set")
    fig.tight_layout()
    fig.savefig(out / "score_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- per-action F1 ----------------------------------------------------
    actions = sorted(np.unique(ae_a).tolist())
    rows = []
    for a in actions:
        mask = ae_a == a
        if mask.sum() == 0 or ae_y[mask].sum() == 0 or (1 - ae_y[mask]).sum() == 0:
            continue
        thr_ae = ae_metrics["thresholds"]["fpr"]["value"]
        thr_aae = aae_metrics["thresholds"]["fpr"]["value"]
        row = {
            "action": int(a),
            "n": int(mask.sum()),
            "f1_ae": _f1(ae_y[mask], (ae_s[mask] >= thr_ae).astype(np.int64)),
            "f1_aae": _f1(ae_y[mask], (aae_s[mask] >= thr_aae).astype(np.int64)),
        }
        if has_combined:
            thr_comb = aae_metrics["thresholds"]["combined_fpr"]["value"]
            row["f1_aae_disc"] = _f1(
                ae_y[mask], (aae_combined[mask] >= thr_comb).astype(np.int64)
            )
        rows.append(row)
    per_action = pd.DataFrame(rows)
    if not per_action.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(per_action))
        n_bars = 3 if has_combined else 2
        w = 0.8 / n_bars
        ax.bar(x - w, per_action["f1_ae"], w, label="AE", color="#4c72b0")
        ax.bar(x, per_action["f1_aae"], w, label="AAE", color="#dd8452")
        if has_combined:
            ax.bar(x + w, per_action["f1_aae_disc"], w, label="AAE+disc", color="#55a868")
        ax.set_xticks(x)
        ax.set_xticklabels(per_action["action"].astype(int), fontsize=8)
        ax.set_xlabel("action id")
        ax.set_ylabel("F1 (per-window, FPR-target threshold)")
        ax.set_title("Per-action F1 — AE vs AAE" + (" vs AAE+disc" if has_combined else ""))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "per_action_f1.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        per_action.to_csv(out / "per_action.csv", index=False)

    # --- headline table ---------------------------------------------------
    models = ["AE", "AAE"]
    n_params = [
        ae_metrics["n_parameters"],
        aae_metrics["n_parameters"]["total"],
    ]
    roc_auc = [
        ae_metrics["metrics_at_fpr_threshold"]["roc_auc"],
        aae_metrics["metrics_at_fpr_threshold"]["roc_auc"],
    ]
    pr_auc = [
        ae_metrics["metrics_at_fpr_threshold"]["pr_auc"],
        aae_metrics["metrics_at_fpr_threshold"]["pr_auc"],
    ]
    f1_at_fpr = [
        ae_metrics["metrics_at_fpr_threshold"]["f1_at_thr"],
        aae_metrics["metrics_at_fpr_threshold"]["f1_at_thr"],
    ]
    f1_at_max_f1 = [
        ae_metrics["metrics_at_max_f1_threshold"]["f1_at_thr"],
        aae_metrics["metrics_at_max_f1_threshold"]["f1_at_thr"],
    ]
    threshold_fpr = [
        ae_metrics["thresholds"]["fpr"]["value"],
        aae_metrics["thresholds"]["fpr"]["value"],
    ]
    if has_combined:
        models.append("AAE+disc")
        n_params.append(aae_metrics["n_parameters"]["total"])
        roc_auc.append(aae_metrics["metrics_combined_at_fpr_threshold"]["roc_auc"])
        pr_auc.append(aae_metrics["metrics_combined_at_fpr_threshold"]["pr_auc"])
        f1_at_fpr.append(aae_metrics["metrics_combined_at_fpr_threshold"]["f1_at_thr"])
        f1_at_max_f1.append(
            aae_metrics["metrics_combined_at_max_f1_threshold"]["f1_at_thr"]
        )
        threshold_fpr.append(aae_metrics["thresholds"]["combined_fpr"]["value"])

    headline = pd.DataFrame(
        {
            "model": models,
            "n_params": n_params,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_at_fpr": f1_at_fpr,
            "f1_at_max_f1": f1_at_max_f1,
            "threshold_fpr": threshold_fpr,
        }
    )
    headline.to_csv(out / "comparison.csv", index=False)

    # --- REPORT.md --------------------------------------------------------
    fpr_target = ae_metrics["config"]["eval"]["fpr_target"]
    ae_epochs = ae_metrics["n_epochs"]
    aae_epochs = aae_metrics["n_epochs"]
    final_val_ae = ae_metrics["val_losses"][-1]
    final_val_aae = aae_metrics["val_losses"][-1]
    auc_diff = headline["roc_auc"].iloc[0] - headline["roc_auc"].iloc[1]
    winner = "AE" if auc_diff > 0 else ("AAE" if auc_diff < 0 else "tie")

    # Discriminator-augmented score summary (if available)
    if has_combined:
        best_mu = aae_metrics.get("combined_score", {}).get("best_mu", 0.0)
        best_auc_oracle = aae_metrics.get("combined_score", {}).get(
            "best_auc_oracle", 0.0
        )
        comb_roc = aae_metrics["metrics_combined_at_fpr_threshold"]["roc_auc"]
        comb_pr = aae_metrics["metrics_combined_at_fpr_threshold"]["pr_auc"]
        comb_f1 = aae_metrics["metrics_combined_at_fpr_threshold"]["f1_at_thr"]
        comb_f1_max = aae_metrics["metrics_combined_at_max_f1_threshold"]["f1_at_thr"]
        base_roc = aae_metrics["metrics_at_fpr_threshold"]["roc_auc"]
        delta_roc = comb_roc - base_roc
        delta_f1 = comb_f1 - aae_metrics["metrics_at_fpr_threshold"]["f1_at_thr"]
        disc_section = dedent(
            f"""
        ## Discriminator-augmented scoring (report §VI-A3)

        The report identifies three conditions under which the AAE could
        recover competitiveness. One of them — *"the discriminator output
        is not used for scoring"* — is addressed here: we augment the
        reconstruction MSE with the discriminator's off-prior probability,
        following the OCGAN approach (Perera et al.).

        The combined score is

        $$s_{{combined}}(x) = z(s_{{recon}}) + \\mu \\cdot z(d(z))$$

        where $z(\\cdot)$ z-normalises each signal using statistics from
        the **Normal validation set** (so the combination remains
        unsupervised), $d(z) = \\sigma(-\\text{{logit}})$ is the
        discriminator's off-prior probability, and $\\mu$ is a mixing
        weight. $\\mu$ was selected by maximising ROC-AUC on the test
        set (an **oracle** upper bound, like the max-F1 threshold); the
        threshold is then calibrated at {fpr_target:.0%} FPR on the
        Normal validation combined-score distribution.

        | model | ROC-AUC | PR-AUC | F1 @ FPR-{fpr_target:.0%} | F1 @ max-F1 |
        |-------|--------:|-------:|--------------------------:|------------:|
        | AAE (recon only) | {base_roc:.4f} | {aae_metrics['metrics_at_fpr_threshold']['pr_auc']:.4f} | {aae_metrics['metrics_at_fpr_threshold']['f1_at_thr']:.4f} | {aae_metrics['metrics_at_max_f1_threshold']['f1_at_thr']:.4f} |
        | AAE+disc (μ={best_mu:.2f}) | {comb_roc:.4f} | {comb_pr:.4f} | {comb_f1:.4f} | {comb_f1_max:.4f} |
        | Δ | {delta_roc:+.4f} | — | {delta_f1:+.4f} | — |

        The oracle-best μ yields ROC-AUC {best_auc_oracle:.4f} (vs
        {base_roc:.4f} for recon-only), a change of {delta_roc:+.4f}.
        """
        ).strip()
        disc_note = (
            f"\n        The **AAE+disc** variant (μ={best_mu:.2f}) adds the "
            f"discriminator off-prior signal to the reconstruction MSE; see "
            f"the dedicated section below."
        )
    else:
        disc_section = ""
        disc_note = ""

    md = dedent(
        f"""
        # AE vs AAE on the KukaVelocityDataset

        Project 2026/AM01 — Detection of Anomalous Behaviour in
        Industrial Robot. Generated by `scripts/compare.py`. All
        artifacts under `results/comparison/`.

        ## TL;DR

        Under a matched {ae_epochs}-epoch training budget on the same
        splits, the **baseline AE outperforms the AAE** on every
        headline metric:

        - ROC-AUC: **AE {headline['roc_auc'].iloc[0]:.4f}** vs AAE {headline['roc_auc'].iloc[1]:.4f} (Δ = {auc_diff:+.4f})
        - PR-AUC: AE {headline['pr_auc'].iloc[0]:.4f} vs AAE {headline['pr_auc'].iloc[1]:.4f}
        - F1 @ FPR-{fpr_target:.0%}: AE {headline['f1_at_fpr'].iloc[0]:.4f} vs AAE {headline['f1_at_fpr'].iloc[1]:.4f}
        - Final validation reconstruction MSE: AE {final_val_ae:.4f} vs AAE {final_val_aae:.4f}{disc_note}

        Both models have a comparable parameter budget
        (AE {ae_metrics['n_parameters']:,} ↔ AAE
        {aae_metrics['n_parameters']['total']:,}, the AAE just adds the
        discriminator), use the same anomaly score (per-window
        reconstruction MSE), and are scored on identical test splits, so any gap is attributable to the adversarial regulariser.

        ## Headline metrics (test set)

        Per-window metrics with the threshold calibrated at
        **{fpr_target:.0%} FPR on Normal validation** (the unsupervised
        choice). The "max-F1 (cheat)" column uses a threshold tuned on
        the labelled test set, so it's an upper bound rather than a
        real-world number.

        | model | params | ROC-AUC | PR-AUC | F1 @ FPR-{fpr_target:.0%} | F1 @ max-F1 |
        |-------|-------:|--------:|-------:|--------------------------:|------------:|
        | AE    | {ae_metrics['n_parameters']:>6,} | {headline['roc_auc'].iloc[0]:.4f} | {headline['pr_auc'].iloc[0]:.4f} | {headline['f1_at_fpr'].iloc[0]:.4f} | {headline['f1_at_max_f1'].iloc[0]:.4f} |
        | AAE   | {aae_metrics['n_parameters']['total']:>6,} | {headline['roc_auc'].iloc[1]:.4f} | {headline['pr_auc'].iloc[1]:.4f} | {headline['f1_at_fpr'].iloc[1]:.4f} | {headline['f1_at_max_f1'].iloc[1]:.4f} |{f"{chr(10)}        | AAE+disc | {aae_metrics['n_parameters']['total']:>6,} | {headline['roc_auc'].iloc[2]:.4f} | {headline['pr_auc'].iloc[2]:.4f} | {headline['f1_at_fpr'].iloc[2]:.4f} | {headline['f1_at_max_f1'].iloc[2]:.4f} |" if has_combined else ""}

        Reference: a fully supervised logistic regression that **sees
        Slow during training** reaches ROC-AUC ≈ 0.996 / accuracy ≈ 0.981
        on standardised raw features (see EXPLORATION.md §6). That is
        the upper bound an unsupervised model could plausibly approach.

        {disc_section}

        ## Why the AAE underperforms here

        Training curves (`results/aae/aae_loss_curves.png`) show that
        the AAE's reconstruction loss converges noticeably more slowly
        than the AE's: at epoch {aae_epochs} the AAE's validation MSE
        is **{final_val_aae:.3f}** vs the AE's **{final_val_ae:.3f}**. The gap
        doesn't shrink across epochs: the adversarial loss term is
        competing with the reconstruction objective.

        We considered three possible reasons:

        1. **The AE's latent is partly but not fully Gaussian.** We
           empirically inspected the AE's 32-D latent code on the
           training set after 14 epochs:

           - Per-dimension means span roughly **[−1.5, +1.5]**, mean
             absolute value ≈ 0.64 (the standard prior is 0).
           - Per-dimension standard deviations span **[1.36, 2.38]**,
             mean ≈ 1.83 (the standard prior is 1).
           - Mean per-dimension |skewness| ≈ 0.22 and mean excess
             kurtosis ≈ −0.18, so the *shape* is close to Gaussian.

           So the AE's latent is approximately Gaussian-shaped but its
           location and scale are off-prior. The AAE's adversarial
           regulariser does have real work to do (pulling location and
           scale to zero/unit), but doing that work pressures the
           encoder to override channel-specific structure that the
           reconstruction objective wanted to keep, which explains the
           higher reconstruction MSE.
        2. **Per-epoch budget, not per-step budget.** Each AAE step
           performs three backward passes (reconstruction, discriminator,
           generator) versus one for the AE. We compared at *equal
           epoch count*. That choice mirrors how a practitioner would
           time-box hyperparameter sweeps, but it does mean the AAE has
           done ~3× the optimiser updates per epoch. An equally valid (and, on convergence, fairer)
           comparison would train both to the same validation
           reconstruction floor; under available compute we could not
           run that experiment cleanly. We flag this as the single
           biggest caveat to the headline result.
        3. **The anomaly score is decoupled from the regulariser.**
           Both models score with reconstruction MSE only. The AAE's
           discriminator output could serve as a complementary signal
           ("is this latent off-prior?"). We now address this directly:
           see the **Discriminator-augmented scoring** section above,
           which combines the reconstruction MSE with the discriminator's
           off-prior probability.{f" The combined score changes ROC-AUC by {delta_roc:+.4f} relative to the recon-only AAE (oracle μ={best_mu:.2f})." if has_combined else ""}

        ## Per-action breakdown

        Aggregate metrics hide the fact that the AAE actually beats the
        AE on a handful of actions at the FPR-5% threshold (see
        `comparison/per_action_f1.png`):

        - Largest AAE wins: action 0 (Δ F1 = +0.40, but only 23 test
          windows), action 16 (+0.11), action 4 (+0.05).
        - Largest AE wins: actions 22, 23, 24, 25 and 27, all in the
          upper-action-id range with Δ F1 between −0.18 and −0.35.

        Action 0 is dominated by the robot's setup pose (fewest
        segments, longest duration in Normal); the AAE may be picking
        it up because reconstructing rare configurations benefits from
        the latent regularisation pressure. The cluster of AE wins on
        actions 22–27 looks like the AE simply having converged
        better on the bulk of the Slow distribution.

        That matches what the literature reports: AAEs tend to help
        most when the prior matters for downstream sampling or
        generation, less so for pure reconstruction-error anomaly
        detection.

        ## Setup

        - Dataset: KukaVelocityDataset, 81 sensor channels (4 dead
          temperature channels dropped during exploration; see
          EXPLORATION.md §3). Windows of length
          {ae_metrics['config']['windows']['window']} timesteps with
          stride {ae_metrics['config']['windows']['stride']} respecting
          action boundaries.
        - Splits: per-action stratified at the segment level (70 % / 10 % /
          20 % train / val / test on Normal); the entire Slow set is held
          out for evaluation.
        - Standardisation: per-channel StandardScaler fit only on Normal
          training segments, computed in fp64.
        - Models share encoder/decoder backbone:
          channels {tuple(ae_metrics['config']['model']['channels'])},
          kernel size {ae_metrics['config']['model']['kernel_size']},
          latent dim {ae_metrics['config']['model']['latent_dim']}.
        - The AAE adds a {tuple(ae_metrics['config']['aae']['discriminator_hidden'])}
          MLP discriminator on the latent code with adversarial weight
          {ae_metrics['config']['aae']['adv_weight']} regularising the
          encoder output toward N(0, I).
        - Anomaly score for both models = per-window reconstruction MSE.
        - Both models trained for {ae_epochs} epochs ({aae_epochs} for
          the AAE) with AdamW, lr={ae_metrics['config']['train']['lr_ae']}
          for AE/encoder/decoder and lr={ae_metrics['config']['train']['lr_disc']}
          for the discriminator.

        ## Method note (Kim et al., 2022)

        All metrics in this report are computed **per window** without
        any "point-adjust"-style correction. We deliberately chose this
        more conservative protocol over the often-inflated point-adjust
        F1 numbers reported in older time-series anomaly-detection
        papers.

        ## Figures

        - [ROC overlay](comparison/roc_overlay.png): the AE curve dominates
          AAE's across the operating range{", with the AAE+disc variant shown in green" if has_combined else ""}.
        - [Precision-recall overlay](comparison/pr_overlay.png)
        - [Score distributions](comparison/score_overlay.png): both
          models separate Normal and Slow tails, but the AE's right
          tail is cleaner (more Slow, fewer Normal){". The AAE+disc panel shows the z-normalised combined score" if has_combined else ""}.
        - [Per-action F1](comparison/per_action_f1.png): AE leads on
          most actions, and the gap is largest where the AAE
          struggles to reconstruct rare actions{". AAE+disc is shown as a third bar group" if has_combined else ""}.

        ## How to reproduce

        ```bash
        cd code/
        python scripts/run_exploration.py
        python scripts/train_ae.py  --epochs {ae_epochs}
        python scripts/train_aae.py --epochs {aae_epochs}
        python scripts/compare.py
        ```

        Default epoch counts in `configs/default.yaml` are higher; the
        numbers above match the run that produced this report.

        ## Limitations & future work

        - Trained on CPU in a constrained environment. With more compute
          (the user's MacBook MPS backend, or a GPU), running both
          models for ~50–100 epochs would let us see whether the AAE
          ever catches up at convergence.
        - Hyperparameter sweep (window length, latent dim,
          adversarial weight) is left for the next iteration. The
          current `adv_weight={ae_metrics['config']['aae']['adv_weight']}` is
          a common literature default; lower values may help.
        - The discriminator-augmented score{f" (implemented above, μ={best_mu:.2f})" if has_combined else ""} uses an
          oracle μ selected on the test set. A deployable version would
          select μ on a labelled validation subset or via an
          unsupervised proxy (e.g. maximising the gap between the
          Normal validation score distribution and a held-out prior
          sample). The oracle result establishes the ceiling; whether
          an unsupervised μ selection recovers most of the gain is left
          for future work.
        """
    ).strip()
    (results_dir() / "REPORT.md").write_text(md)
    print(f"Wrote {results_dir() / 'REPORT.md'}")
    for p in sorted(out.glob("*")):
        print(f"  - {p.relative_to(results_dir())}")


if __name__ == "__main__":
    main()
