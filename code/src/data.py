"""Loaders for the KukaVelocityDataset.

The dataset ships as three NumPy files:

- ``KukaColumnNames.npy`` — ``(87,)`` array of strings, the *full* column list
  with ``action`` first and ``anomaly`` last.
- ``KukaNormal.npy`` — ``(233_792, 86)`` of nominal data. **Important**: this
  array is *missing the trailing ``anomaly`` column*, so its column names are
  ``KukaColumnNames[:-1]``.
- ``KukaSlow.npy`` — ``(41_538, 87)`` of anomalous data, with the ``anomaly``
  column present and equal to 1 for every row.

This module returns pandas DataFrames with proper column names. We also
expose a "feature columns" helper that drops the bookkeeping columns
``action`` and ``anomaly`` so downstream models see only the 85 sensor
channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import dataset_dir

# Bookkeeping columns that should NOT be fed to the autoencoder as features.
NON_FEATURE_COLS = ("action", "anomaly")

# Sensors known (from exploration) to emit a single fixed value in BOTH
# KukaNormal and KukaSlow, so they carry no information.
DEAD_CHANNELS = (
    "sensor_id2_temp",
    "sensor_id5_temp",
    "sensor_id6_temp",
    "sensor_id7_temp",
)


@dataclass
class KukaDataset:
    """A pair of dataframes plus metadata.

    Attributes
    ----------
    normal : pd.DataFrame
        Nominal data, shape ``(N_normal, 87)``. The ``anomaly`` column is
        synthesised as 0 for consistency with ``slow``.
    slow : pd.DataFrame
        Anomalous data, shape ``(N_slow, 87)``. ``anomaly`` is 1.
    columns : list[str]
        The full 87-column list (action + 85 sensor channels + anomaly).
    feature_columns : list[str]
        The sensor channels (no ``action``, no ``anomaly``). With the
        default ``drop_dead_channels=True`` this is **81 channels**;
        passing ``drop_dead_channels=False`` keeps all 85.
    """

    normal: pd.DataFrame
    slow: pd.DataFrame
    columns: list[str]
    feature_columns: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)


def load_kuka(data_dir: Path | None = None, *, drop_dead_channels: bool = True) -> KukaDataset:
    """Load the three .npy files and return a :class:`KukaDataset`.

    Performs sanity checks and aligns ``KukaNormal`` to the same column schema
    as ``KukaSlow`` by appending an all-zero ``anomaly`` column.
    """
    data_dir = Path(data_dir) if data_dir is not None else dataset_dir()
    cols = np.load(data_dir / "KukaColumnNames.npy", allow_pickle=True).tolist()
    normal = np.load(data_dir / "KukaNormal.npy", allow_pickle=True)
    slow = np.load(data_dir / "KukaSlow.npy", allow_pickle=True)

    # --- shape sanity checks ------------------------------------------------
    if len(cols) != 87:
        raise ValueError(f"Expected 87 column names, got {len(cols)}")
    if cols[0] != "action" or cols[-1] != "anomaly":
        raise ValueError(
            "Expected 'action' as first column and 'anomaly' as last; got "
            f"first={cols[0]!r} last={cols[-1]!r}"
        )
    if normal.ndim != 2 or normal.shape[1] != 86:
        raise ValueError(f"KukaNormal should be (N, 86), got {normal.shape}")
    if slow.ndim != 2 or slow.shape[1] != 87:
        raise ValueError(f"KukaSlow should be (N, 87), got {slow.shape}")

    # KukaNormal has 86 columns: action + 85 sensors. Append an all-zero
    # anomaly column so the two arrays share the same schema.
    normal_full = np.concatenate(
        [normal, np.zeros((normal.shape[0], 1), dtype=normal.dtype)], axis=1
    )

    df_normal = pd.DataFrame(normal_full, columns=cols)
    df_slow = pd.DataFrame(slow, columns=cols)

    # KukaSlow's anomaly column should be all 1s; assert it.
    if not (df_slow["anomaly"] == 1).all():
        raise ValueError("Expected all KukaSlow rows to be labelled anomaly=1")

    feature_cols = [c for c in cols if c not in NON_FEATURE_COLS]
    if drop_dead_channels:
        feature_cols = [c for c in feature_cols if c not in DEAD_CHANNELS]

    return KukaDataset(
        normal=df_normal,
        slow=df_slow,
        columns=cols,
        feature_columns=feature_cols,
    )


def feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Group the 85 sensor columns by sensor type for plotting/ablations.

    Returns a dict mapping group name → list of feature column names.
    """
    groups: dict[str, list[str]] = {
        "electrical": [c for c in feature_columns if c.startswith("machine_name")],
        "accelerometer": [c for c in feature_columns if "_Acc" in c],
        "gyroscope": [c for c in feature_columns if "_Gyro" in c],
        "quaternion": [
            c
            for c in feature_columns
            if c.startswith("sensor_id") and c.rsplit("_", 1)[-1] in {"q1", "q2", "q3", "q4"}
        ],
        "temperature": [c for c in feature_columns if c.endswith("_temp")],
    }
    # Verify the partition covers all features exactly once. Drop empty groups
    # (e.g. "temperature" when dead channels have been removed) before
    # returning so callers don't have to handle them.
    covered = sorted(c for cols in groups.values() for c in cols)
    expected = sorted(feature_columns)
    if covered != expected:
        missing = set(expected) - set(covered)
        extra = set(covered) - set(expected)
        raise AssertionError(
            f"feature_groups partition mismatch — missing={missing} extra={extra}"
        )
    return {k: v for k, v in groups.items() if v}
