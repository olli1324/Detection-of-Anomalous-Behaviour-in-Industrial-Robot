"""Adversarial autoencoder (Makhzani et al., 2015).

Same encoder/decoder backbone as ``ConvAE`` plus a discriminator that
operates on the latent code. The generator (encoder) is trained to fool
the discriminator into thinking the latent code came from a chosen prior
(default: standard Gaussian); that's the adversarial regulariser.

The training procedure has three steps per minibatch:

1. **Reconstruction.** Update encoder + decoder to minimise the MSE
   between input and reconstruction.
2. **Discriminator.** Sample fake latents = ``encode(x).detach()`` and
   real latents from the prior. Update the discriminator to tell them
   apart (BCE loss).
3. **Generator (regularisation).** Update only the encoder to make the
   discriminator classify ``encode(x)`` as "real", pushing the latent
   distribution towards the prior.

The anomaly score is still the per-window reconstruction MSE, exactly
as for the AE. This keeps the AE/AAE comparison apples-to-apples: any
detection-quality difference comes from how the adversarial regulariser
shapes the encoder, not from a different scoring rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ae import AEConfig, ConvAE


@dataclass
class AAEConfig:
    ae: AEConfig
    discriminator_hidden: tuple[int, ...] = (128, 128)
    adv_weight: float = 0.1  # weight of the generator-side adversarial loss


class LatentDiscriminator(nn.Module):
    """MLP that maps a latent code to a single logit (real vs fake)."""

    def __init__(self, latent_dim: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


class AdversarialAE(nn.Module):
    """AE + latent-discriminator wrapper.

    The forward pass behaves exactly like ``ConvAE.forward`` so existing
    eval code (anomaly scoring, threshold calibration) works unchanged.
    """

    def __init__(self, cfg: AAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ae = ConvAE(cfg.ae)
        self.disc = LatentDiscriminator(cfg.ae.latent_dim, cfg.discriminator_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ae(x)

    # ------------------------------------------------------------------
    # Helpers used by the AAE training loop
    # ------------------------------------------------------------------

    def sample_prior(self, batch: int, device: torch.device | str) -> torch.Tensor:
        """Sample real latents from the standard-Gaussian prior."""
        return torch.randn(batch, self.cfg.ae.latent_dim, device=device)
