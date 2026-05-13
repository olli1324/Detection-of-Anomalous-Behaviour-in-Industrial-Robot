"""Plotting helpers for the exploration phase.

All figures get saved into ``code/results/figures/`` via :func:`utils.figures_dir`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import figures_dir


def _save(fig: plt.Figure, name: str) -> Path:
    out = figures_dir() / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_action_counts(counts: pd.DataFrame) -> Path:
    """Bar chart: per-action sample counts, Normal vs Slow."""
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.4
    x = np.arange(len(counts))
    ax.bar(x - width / 2, counts["normal_samples"], width, label="Normal", color="#4c72b0")
    ax.bar(x + width / 2, counts["slow_samples"], width, label="Slow", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(a)}" for a in counts.index], fontsize=8)
    ax.set_xlabel("action id")
    ax.set_ylabel("number of samples")
    ax.set_title("Per-action sample counts: Normal vs. Slow")
    ax.legend()
    return _save(fig, "01_action_counts.png")


def plot_segment_lengths(seg_summary: pd.DataFrame) -> Path:
    """Per-action mean and median segment length, Normal vs Slow."""
    seg_summary = seg_summary.sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    width = 0.4
    x = np.arange(len(seg_summary))

    for ax, stat, title in zip(
        axes,
        ["mean", "median"],
        ["Mean segment length (timesteps)", "Median segment length (timesteps)"],
    ):
        ax.bar(
            x - width / 2,
            seg_summary[f"normal_{stat}"],
            width,
            label="Normal",
            color="#4c72b0",
        )
        ax.bar(
            x + width / 2,
            seg_summary[f"slow_{stat}"],
            width,
            label="Slow",
            color="#dd8452",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(a)}" for a in seg_summary.index], fontsize=8)
        ax.set_xlabel("action id")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Action-segment duration: Normal vs. Slow")
    fig.tight_layout()
    return _save(fig, "02_segment_lengths.png")


def plot_channel_signals(
    normal: pd.DataFrame,
    slow: pd.DataFrame,
    columns: list[str],
    *,
    n_steps: int = 4000,
    suptitle: str,
    fname: str,
) -> Path:
    """Time-series plot: first ``n_steps`` timesteps for each channel,
    Normal on top of Slow."""
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 1.6 * n_rows), sharex=True)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for i, col in enumerate(columns):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(
            normal[col].iloc[:n_steps].to_numpy(),
            color="#4c72b0",
            linewidth=0.6,
            alpha=0.85,
            label="Normal",
        )
        ax.plot(
            slow[col].iloc[:n_steps].to_numpy(),
            color="#dd8452",
            linewidth=0.6,
            alpha=0.85,
            label="Slow",
        )
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(len(columns), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout()
    return _save(fig, fname)


def plot_zshift_bar(summary: pd.DataFrame, top_k: int = 25) -> Path:
    """Top-K channels by absolute z-shift between Slow and Normal means."""
    df = summary.assign(abs_z=summary["z_shift"].abs())
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_z"])
    df = df.sort_values("abs_z", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4c72b0" if z >= 0 else "#dd8452" for z in df["z_shift"]]
    ax.barh(df.index[::-1], df["z_shift"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("(slow_mean − normal_mean) / normal_std")
    ax.set_title(f"Top {top_k} channels by mean shift between Slow and Normal")
    fig.tight_layout()
    return _save(fig, "03_top_zshift.png")
