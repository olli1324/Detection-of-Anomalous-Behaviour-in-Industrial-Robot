"""Per-channel statistics on Normal vs. Slow.

We compute basic summary stats for each column and a couple of "is this channel
informative?" checks that we'll use to decide what to feed into the
autoencoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _per_column_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean / std / min / max / fraction-zero / fraction-finite per column."""
    arr = df[columns].to_numpy(dtype=np.float64, copy=False)
    out = pd.DataFrame(
        {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0, ddof=0),
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "frac_zero": (arr == 0).mean(axis=0),
            "frac_finite": np.isfinite(arr).mean(axis=0),
        },
        index=columns,
    )
    return out


def channel_summary(normal: pd.DataFrame, slow: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Side-by-side channel summary for Normal and Slow.

    Adds a ``z_shift`` column = mean shift between Slow and Normal expressed
    in units of Normal's own standard deviation. Useful as a first-pass
    "is this channel responding to the anomaly?" signal.
    """
    n = _per_column_stats(normal, columns).add_prefix("normal_")
    s = _per_column_stats(slow, columns).add_prefix("slow_")
    out = pd.concat([n, s], axis=1)

    eps = 1e-12
    out["z_shift"] = (out["slow_mean"] - out["normal_mean"]) / (out["normal_std"] + eps)
    out["std_ratio"] = out["slow_std"] / (out["normal_std"] + eps)
    return out


def degenerate_channels(summary: pd.DataFrame, *, std_thresh: float = 1e-9) -> list[str]:
    """Return channels whose Normal-distribution standard deviation is below
    ``std_thresh`` (i.e. effectively constant in the training set).

    These channels carry no information for a reconstruction model and should
    be dropped (or, at minimum, not standardised).
    """
    mask = summary["normal_std"] <= std_thresh
    return summary.index[mask].tolist()
