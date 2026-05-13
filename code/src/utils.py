"""Small utilities: device selection, seeding, paths."""

from __future__ import annotations

import os
import random
from pathlib import Path


def project_root() -> Path:
    """Return the absolute path of the repository root (the folder containing
    ``code/`` and ``KukaVelocityDataset/``)."""
    return Path(__file__).resolve().parents[2]


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    return project_root() / "KukaVelocityDataset"


def results_dir() -> Path:
    out = code_root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def figures_dir() -> Path:
    return results_dir() / "figures"


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_device():
    """Pick the best available PyTorch device: MPS on Apple Silicon, then CUDA, then CPU.

    Returns a ``torch.device`` instance. Importing torch is delayed so the rest
    of the codebase stays usable without it.
    """
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
