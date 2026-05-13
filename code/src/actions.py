"""Action-level analysis.

The ``action`` column is an integer label for what motion the robot is
performing at each timestep. The dataset has 32 distinct actions in Normal
and 31 in Slow (action 9 is missing in Slow).

This module:

- Returns the per-action sample count for both splits.
- Segments runs of constant action and reports per-action *segment durations*
  in number of timesteps. Comparing the duration distributions Normal-vs-Slow
  is the most direct way to confirm that "slow" actually shows up as longer
  segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def action_counts(normal: pd.DataFrame, slow: pd.DataFrame) -> pd.DataFrame:
    """Per-action sample counts (rows) in Normal and Slow."""
    n = normal["action"].value_counts().sort_index()
    s = slow["action"].value_counts().sort_index()
    out = pd.DataFrame({"normal_samples": n, "slow_samples": s}).fillna(0).astype(int)
    out.index.name = "action"
    return out


def _segment_runs(action_series: np.ndarray) -> list[tuple[int, int, int]]:
    """Segment a 1-D action stream into (action, start_idx, length) triples.

    Each run of identical consecutive action values becomes one segment.
    """
    if len(action_series) == 0:
        return []
    # Find positions where the action value changes
    change_points = np.flatnonzero(np.diff(action_series) != 0) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(action_series)]))
    return [(int(action_series[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


@dataclass
class ActionSegments:
    """Per-action segment-duration distributions."""

    normal: pd.DataFrame  # columns: action, length
    slow: pd.DataFrame  # columns: action, length

    def summary(self) -> pd.DataFrame:
        """Per-action median/mean segment duration for both splits."""
        n = self.normal.groupby("action")["length"].agg(["count", "mean", "median"]).add_prefix(
            "normal_"
        )
        s = self.slow.groupby("action")["length"].agg(["count", "mean", "median"]).add_prefix(
            "slow_"
        )
        out = n.join(s, how="outer").fillna(0)
        out["median_ratio"] = out["slow_median"] / out["normal_median"].replace(0, np.nan)
        out["mean_ratio"] = out["slow_mean"] / out["normal_mean"].replace(0, np.nan)
        return out


def action_segments(normal: pd.DataFrame, slow: pd.DataFrame) -> ActionSegments:
    """Compute per-action segment durations for both splits."""
    n_runs = _segment_runs(normal["action"].to_numpy())
    s_runs = _segment_runs(slow["action"].to_numpy())
    n_df = pd.DataFrame(n_runs, columns=["action", "start", "length"]).drop(columns="start")
    s_df = pd.DataFrame(s_runs, columns=["action", "start", "length"]).drop(columns="start")
    return ActionSegments(normal=n_df, slow=s_df)
