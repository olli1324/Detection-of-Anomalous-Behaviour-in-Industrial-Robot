"""Thin PyTorch ``Dataset`` wrappers around the windowed splits.

Kept deliberately simple: the data is already materialised in memory as
NumPy arrays of shape ``(N, F, T)``, so each ``__getitem__`` is just an
indexing op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Importing torch is deferred to runtime to keep tests / static checks
# usable when the framework is not installed.


def _torch():
    import torch  # noqa: WPS433 (intentional local import)

    return torch


class WindowDataset:
    """A ``torch.utils.data.Dataset`` over preassembled windows.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(N, F, T)``, float32.
    y : np.ndarray | None
        Optional anomaly labels of shape ``(N,)``, int64. If None, returns
        ``-1`` so downstream code can detect "no label".
    a : np.ndarray | None
        Optional action ids of shape ``(N,)``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected (N, F, T), got {X.shape}")
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = (
            np.ascontiguousarray(y, dtype=np.int64) if y is not None else None
        )
        self.a = (
            np.ascontiguousarray(a, dtype=np.int64) if a is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        torch = _torch()
        x = torch.from_numpy(self.X[idx])  # (F, T)
        y = (
            torch.tensor(self.y[idx], dtype=torch.int64)
            if self.y is not None
            else torch.tensor(-1, dtype=torch.int64)
        )
        a = (
            torch.tensor(self.a[idx], dtype=torch.int64)
            if self.a is not None
            else torch.tensor(-1, dtype=torch.int64)
        )
        return x, y, a


@dataclass
class DataLoaderBundle:
    """Container of train / val / test DataLoaders with consistent settings.

    ``stacked_test`` is the (Normal-test ⊕ Slow-test) loader used for
    final anomaly evaluation.
    """

    train: object
    val_normal: object
    test_normal: object
    test_slow: object
    stacked_test: object


def make_loaders(
    splits,  # WindowedSplits, but typed loosely to avoid import cycle
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoaderBundle:
    """Build the full set of DataLoaders for a training run.

    All loaders are reproducible via the supplied seed (the train loader
    uses a generator-bound shuffle; eval loaders are unshuffled).
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = WindowDataset(splits.X_train, a=splits.a_train)
    val_ds = WindowDataset(splits.X_val_normal, a=splits.a_val_normal)
    test_n = WindowDataset(
        splits.X_test_normal,
        y=np.zeros(len(splits.X_test_normal), dtype=np.int64),
        a=splits.a_test_normal,
    )
    test_s = WindowDataset(
        splits.X_test_slow,
        y=np.ones(len(splits.X_test_slow), dtype=np.int64),
        a=splits.a_test_slow,
    )

    X_all, y_all, a_all = splits.stacked_test()
    test_all = WindowDataset(X_all, y=y_all, a=a_all)

    common = dict(batch_size=batch_size, num_workers=num_workers)
    return DataLoaderBundle(
        train=DataLoader(train_ds, shuffle=True, generator=g, drop_last=True, **common),
        val_normal=DataLoader(val_ds, shuffle=False, **common),
        test_normal=DataLoader(test_n, shuffle=False, **common),
        test_slow=DataLoader(test_s, shuffle=False, **common),
        stacked_test=DataLoader(test_all, shuffle=False, **common),
    )
