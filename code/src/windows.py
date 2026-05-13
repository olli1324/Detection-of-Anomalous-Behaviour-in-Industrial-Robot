"""Sliding-window construction and train / val / test splitting.

Design decisions (motivated in ``results/EXPLORATION.md``):

- Windows respect **action boundaries**: a window may not span two
  different actions. Crossing actions would mix dynamics from different
  trajectories and inject extra "boundary novelty" into the
  reconstruction error.
- Windows are built per *segment* (a maximal run of one action). For a
  segment of length ``L``, with window length ``w`` and stride ``s``, we
  emit ``floor((L - w) / s) + 1`` windows. Segments shorter than ``w`` are
  dropped.
- Splits operate at the **segment level**, not the window level, to keep
  evaluation on Normal honest (otherwise consecutive overlapping windows
  end up in train and val).
- Standardisation is fit *only* on the training segments. Channels with
  near-zero std in the training set are passed through unchanged
  (subtract the constant mean, no division). On Slow data, those
  channels can vary, which produces a strong, unscaled reconstruction
  signal, which is exactly what we want for the temperature columns
  identified in exploration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .actions import _segment_runs


@dataclass
class StandardScalerSafe:
    """A NumPy-only scaler that handles zero-std channels gracefully.

    Attributes
    ----------
    mean : np.ndarray of shape (F,)
    std : np.ndarray of shape (F,)
        Already clipped: any value below ``eps`` is replaced by 1 to
        disable division for that channel.
    is_constant : np.ndarray of bools, shape (F,)
        True for channels whose training std was below ``eps``.
    """

    mean: np.ndarray
    std: np.ndarray
    is_constant: np.ndarray
    eps: float = 1e-9

    @classmethod
    def fit(cls, X: np.ndarray, *, eps: float = 1e-9) -> "StandardScalerSafe":
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}")
        # IMPORTANT: compute stats in fp64. With ~100k+ near-constant fp32
        # values, the rounding error in the running sum is large enough
        # that the *true* per-channel std (~1e-13) is reported as ~0.1 if
        # we let NumPy reduce in fp32, which would let degenerate
        # channels through as if they were normal.
        X64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
        mean = X64.mean(axis=0)
        std = X64.std(axis=0, ddof=0)
        is_constant = std < eps
        std_safe = std.copy()
        std_safe[is_constant] = 1.0
        return cls(mean=mean, std=std_safe, is_constant=is_constant, eps=eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        scaler = self.fit(X)
        return scaler.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean


# -----------------------------------------------------------------------------
# Segment + windowing primitives
# -----------------------------------------------------------------------------


@dataclass
class Segment:
    """A maximal run of constant `action` in a dataframe."""

    action: int
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


def list_segments(action_array: np.ndarray) -> list[Segment]:
    return [Segment(a, s, l) for (a, s, l) in _segment_runs(action_array)]


def windows_from_segment(
    seg: Segment, *, window: int, stride: int
) -> list[tuple[int, int]]:
    """Return ``[(start, end)]`` pairs of windows fitting inside ``seg``.

    ``stride`` of 1 produces fully overlapping windows (one per timestep);
    larger strides reduce the count without changing the segment respect
    invariant.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if seg.length < window:
        return []
    n = (seg.length - window) // stride + 1
    return [(seg.start + i * stride, seg.start + i * stride + window) for i in range(n)]


def build_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    window: int,
    stride: int,
    segment_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise an array of windows from a dataframe.

    Returns
    -------
    X : np.ndarray of shape ``(N, F, T)`` (channels-first)
    actions : np.ndarray of shape ``(N,)`` — the action id for each window
    """
    arr = df[feature_columns].to_numpy(dtype=np.float32)
    actions_arr = df["action"].to_numpy()
    segs = list_segments(actions_arr)
    if segment_indices is not None:
        segs = [segs[i] for i in segment_indices]

    rows = []
    win_actions = []
    for seg in segs:
        for s, e in windows_from_segment(seg, window=window, stride=stride):
            rows.append(arr[s:e].T)  # (F, T)
            win_actions.append(seg.action)
    if not rows:
        return (
            np.zeros((0, len(feature_columns), window), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    X = np.stack(rows, axis=0)
    a = np.array(win_actions, dtype=np.int64)
    return X, a


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------


@dataclass
class SegmentSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    def sizes(self, segments: list[Segment]) -> dict[str, int]:
        return {
            "train": sum(segments[i].length for i in self.train),
            "val": sum(segments[i].length for i in self.val),
            "test": sum(segments[i].length for i in self.test),
        }


def stratified_segment_split(
    segments: list[Segment],
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> SegmentSplit:
    """Split segment indices stratified by action.

    Each action's segments are shuffled deterministically and split
    according to the given fractions. Actions with too few segments fall
    back to placing the segment in train.
    """
    rng = np.random.default_rng(seed)
    by_action: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        by_action.setdefault(seg.action, []).append(i)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for a, indices in by_action.items():
        idx = np.array(indices)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_test - n_val
        # Guarantee at least 1 segment in train if possible
        if n_train < 1 and n >= 1:
            n_train = 1
            n_val = max(0, min(n_val, n - n_train))
            n_test = n - n_train - n_val
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train : n_train + n_val].tolist())
        test.extend(idx[n_train + n_val :].tolist())

    return SegmentSplit(train=sorted(train), val=sorted(val), test=sorted(test))


# -----------------------------------------------------------------------------
# Top-level "build everything" helper
# -----------------------------------------------------------------------------


@dataclass
class WindowedSplits:
    """Materialised windows and labels for AE/AAE training.

    Conventions:
      - All ``X_*`` arrays are shape ``(N, F, T)``.
      - All ``y_*`` arrays are 0/1 anomaly labels of shape ``(N,)``.
      - All ``a_*`` arrays are action ids of shape ``(N,)``.
      - The scaler is fit on Normal training windows only.
    """

    X_train: np.ndarray
    a_train: np.ndarray

    X_val_normal: np.ndarray
    a_val_normal: np.ndarray

    X_test_normal: np.ndarray
    a_test_normal: np.ndarray

    X_test_slow: np.ndarray
    a_test_slow: np.ndarray

    feature_columns: list[str]
    scaler: StandardScalerSafe
    window: int
    stride: int

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def stacked_test(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, y, a)`` for the combined Normal-test + Slow-test set."""
        X = np.concatenate([self.X_test_normal, self.X_test_slow], axis=0)
        y = np.concatenate(
            [
                np.zeros(len(self.X_test_normal), dtype=np.int64),
                np.ones(len(self.X_test_slow), dtype=np.int64),
            ]
        )
        a = np.concatenate([self.a_test_normal, self.a_test_slow], axis=0)
        return X, y, a


def _scale_windows(X: np.ndarray, scaler: StandardScalerSafe) -> np.ndarray:
    """Apply standardisation to a (N, F, T) window batch.

    Reshapes to (N*T, F), scales, reshapes back. Constant-in-train channels
    are mean-centred but not divided (std == 1 in scaler.std for them).
    """
    if X.size == 0:
        return X
    n, f, t = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, f)
    flat = scaler.transform(flat)
    return flat.reshape(n, t, f).transpose(0, 2, 1).astype(np.float32, copy=False)


def make_windowed_splits(
    normal_df: pd.DataFrame,
    slow_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    window: int = 64,
    stride: int = 16,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> WindowedSplits:
    """Build the full windowed splits used by training / evaluation.

    - Splits Normal segments into train / val / test (segment-level,
      stratified by action).
    - Builds windows for each Normal split with the given stride.
    - Builds windows for the entire Slow set (it is only used at test
      time, so no segment split there).
    - Fits a :class:`StandardScalerSafe` on the Normal training timesteps
      (NOT on windows: per-timestep stats are what matter).
    - Applies the scaler to all four window batches.
    """
    n_actions = normal_df["action"].to_numpy()
    s_actions = slow_df["action"].to_numpy()
    n_segments = list_segments(n_actions)
    s_segments = list_segments(s_actions)

    split = stratified_segment_split(
        n_segments, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # 1. Fit scaler on the timesteps that make up the training segments
    train_indices = np.concatenate(
        [np.arange(n_segments[i].start, n_segments[i].end) for i in split.train]
    )
    train_arr = normal_df[feature_columns].to_numpy(dtype=np.float32)[train_indices]
    scaler = StandardScalerSafe.fit(train_arr)

    # 2. Build windows
    X_train, a_train = build_windows(
        normal_df,
        feature_columns,
        window=window,
        stride=stride,
        segment_indices=split.train,
    )
    X_val_normal, a_val_normal = build_windows(
        normal_df,
        feature_columns,
        window=window,
        stride=stride,
        segment_indices=split.val,
    )
    X_test_normal, a_test_normal = build_windows(
        normal_df,
        feature_columns,
        window=window,
        stride=stride,
        segment_indices=split.test,
    )
    X_test_slow, a_test_slow = build_windows(
        slow_df, feature_columns, window=window, stride=stride
    )

    # 3. Apply scaler to all batches
    X_train = _scale_windows(X_train, scaler)
    X_val_normal = _scale_windows(X_val_normal, scaler)
    X_test_normal = _scale_windows(X_test_normal, scaler)
    X_test_slow = _scale_windows(X_test_slow, scaler)

    return WindowedSplits(
        X_train=X_train,
        a_train=a_train,
        X_val_normal=X_val_normal,
        a_val_normal=a_val_normal,
        X_test_normal=X_test_normal,
        a_test_normal=a_test_normal,
        X_test_slow=X_test_slow,
        a_test_slow=a_test_slow,
        feature_columns=feature_columns,
        scaler=scaler,
        window=window,
        stride=stride,
    )
