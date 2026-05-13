"""Training loops for the AE and AAE.

These are deliberately simple, single-file training loops, no Lightning
and no Hydra. The point is to keep the AE/AAE comparison transparent.

Both loops:

- Use the device returned by :func:`utils.select_device` (MPS on Apple
  Silicon, CUDA on NVIDIA, CPU otherwise).
- Run ``n_epochs`` minibatch passes over the training loader.
- Track per-epoch train loss and validation reconstruction loss.
- Return a :class:`TrainResult` with the trained model, the loss history,
  and the device used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .models.ae import (
    ConvAE,
    per_window_recon_error,
    reconstruction_loss,
)
from .models.aae import AdversarialAE


@dataclass
class TrainResult:
    model: object  # nn.Module
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    # AAE-specific extras (empty list for the AE)
    disc_losses: list[float] = field(default_factory=list)
    gen_losses: list[float] = field(default_factory=list)
    device: str = ""


def _eval_recon(model, loader, device) -> float:
    """Per-window MSE averaged over a loader. ``model`` should be in eval mode."""
    import torch

    model.eval()
    n = 0
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            x_hat, _ = model(x)
            err = per_window_recon_error(x, x_hat)
            total += float(err.sum().item())
            n += len(x)
    return total / max(n, 1)


def train_ae(
    model: ConvAE,
    train_loader,
    val_loader,
    *,
    n_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device=None,
    log_every: int = 1,
) -> TrainResult:
    """Train the baseline AE. Reports per-epoch train + val MSE."""
    import torch
    from torch.optim import AdamW

    from .utils import select_device

    device = device or select_device()
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for ep in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            x_hat, _ = model(x)
            loss = reconstruction_loss(x, x_hat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(x)
            n += len(x)
        train_loss = running / max(n, 1)
        val_loss = _eval_recon(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if ep % log_every == 0:
            print(
                f"  AE epoch {ep:>3d}/{n_epochs} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f}"
            )

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        device=str(device),
    )


def train_aae(
    model: AdversarialAE,
    train_loader,
    val_loader,
    *,
    n_epochs: int = 30,
    lr_ae: float = 1e-3,
    lr_disc: float = 5e-4,
    weight_decay: float = 1e-5,
    device=None,
    log_every: int = 1,
) -> TrainResult:
    """Train the AAE with the standard 3-step loop (recon, disc, gen).

    Uses two optimisers: one for the AE parameters (encoder + decoder)
    and one for the discriminator. The generator-side adversarial loss
    is back-propagated through the encoder only.
    """
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    from .utils import select_device

    device = device or select_device()
    model = model.to(device)

    ae_params = list(model.ae.parameters())
    disc_params = list(model.disc.parameters())
    opt_ae = AdamW(ae_params, lr=lr_ae, weight_decay=weight_decay)
    opt_disc = AdamW(disc_params, lr=lr_disc, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    disc_losses: list[float] = []
    gen_losses: list[float] = []

    adv_w = model.cfg.adv_weight

    for ep in range(1, n_epochs + 1):
        model.train()
        running_recon = 0.0
        running_disc = 0.0
        running_gen = 0.0
        n = 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            B = len(x)

            # 1. Reconstruction step (update AE)
            x_hat, _ = model(x)
            loss_recon = reconstruction_loss(x, x_hat)
            opt_ae.zero_grad(set_to_none=True)
            loss_recon.backward()
            opt_ae.step()

            # 2. Discriminator step (update D only)
            with torch.no_grad():
                z_fake = model.encode(x)
            z_real = model.sample_prior(B, device)
            d_real = model.disc(z_real)
            d_fake = model.disc(z_fake)
            ones = torch.ones_like(d_real)
            zeros = torch.zeros_like(d_fake)
            loss_disc = 0.5 * (
                F.binary_cross_entropy_with_logits(d_real, ones)
                + F.binary_cross_entropy_with_logits(d_fake, zeros)
            )
            opt_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            opt_disc.step()

            # 3. Generator step (update encoder via AE optimiser)
            z_fake = model.encode(x)
            d_fake_for_g = model.disc(z_fake)
            loss_gen = F.binary_cross_entropy_with_logits(
                d_fake_for_g, torch.ones_like(d_fake_for_g)
            )
            opt_ae.zero_grad(set_to_none=True)
            (adv_w * loss_gen).backward()
            opt_ae.step()

            running_recon += float(loss_recon.item()) * B
            running_disc += float(loss_disc.item()) * B
            running_gen += float(loss_gen.item()) * B
            n += B

        train_loss = running_recon / max(n, 1)
        d_loss = running_disc / max(n, 1)
        g_loss = running_gen / max(n, 1)
        val_loss = _eval_recon(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        disc_losses.append(d_loss)
        gen_losses.append(g_loss)

        if ep % log_every == 0:
            print(
                f"  AAE epoch {ep:>3d}/{n_epochs} | recon {train_loss:.5f} | "
                f"disc {d_loss:.4f} | gen {g_loss:.4f} | val recon {val_loss:.5f}"
            )

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        disc_losses=disc_losses,
        gen_losses=gen_losses,
        device=str(device),
    )


def collect_anomaly_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return ``(scores, labels, actions)``.

    ``scores`` is per-window reconstruction MSE (the anomaly score).
    ``labels`` is 0/1 (Normal / Slow). ``actions`` is the action id.
    """
    import torch

    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y, a = batch[0].to(device, non_blocking=True), batch[1], batch[2]
            x_hat, _ = model(x)
            err = per_window_recon_error(x, x_hat).cpu().numpy()
            scores.append(err)
            labels.append(y.numpy())
            actions.append(a.numpy())
    return (
        np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32),
        np.concatenate(labels) if labels else np.zeros((0,), dtype=np.int64),
        np.concatenate(actions) if actions else np.zeros((0,), dtype=np.int64),
    )
