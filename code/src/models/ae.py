"""Baseline 1D-convolutional autoencoder.

Architecture (channels-first, ``(B, F, T)``):

    Encoder:
        Conv1d(F        -> ch1, k, stride=2, pad)  + GELU + LayerNorm
        Conv1d(ch1      -> ch2, k, stride=2, pad)  + GELU + LayerNorm
        Conv1d(ch2      -> ch3, k, stride=2, pad)  + GELU + LayerNorm
        Flatten + Linear -> latent_dim

    Decoder (mirror):
        Linear -> reshape
        ConvTranspose1d ... x3 (last has no activation)

The encoder downsamples T by a factor of 8, so a default window of T=64
maps to a flat 8 * ch3 = 256 hidden vector before the latent projection.
The default latent_dim is 32, small enough to act as a real bottleneck
on 81 input channels, large enough to store basic dynamics.

The forward pass returns ``(x_hat, z)``: reconstruction and latent code.
``encode`` and ``decode`` are exposed for the AAE wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AEConfig:
    n_features: int
    window: int = 64
    channels: tuple[int, int, int] = (64, 96, 128)
    kernel_size: int = 5
    latent_dim: int = 32

    @property
    def downsampled_window(self) -> int:
        # Three stride-2 conv layers downsample T by 2**3 = 8.
        if self.window % 8 != 0:
            raise ValueError(
                f"window={self.window} must be divisible by 8 for the default conv stack"
            )
        return self.window // 8


def _conv_block(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=pad),
        nn.GELU(),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
    )


def _deconv_block(in_ch: int, out_ch: int, k: int, *, last: bool = False) -> nn.Sequential:
    pad = k // 2
    out_pad = 1  # exact upsampling by 2 with stride=2, kernel odd
    layers: list[nn.Module] = [
        nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=k, stride=2, padding=pad, output_padding=out_pad
        )
    ]
    if not last:
        layers.append(nn.GELU())
        layers.append(nn.GroupNorm(num_groups=8, num_channels=out_ch))
    return nn.Sequential(*layers)


class ConvAE(nn.Module):
    """Channels-first 1-D conv autoencoder for ``(B, F, T)`` windows."""

    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.channels

        self.enc_conv = nn.Sequential(
            _conv_block(cfg.n_features, c1, cfg.kernel_size),
            _conv_block(c1, c2, cfg.kernel_size),
            _conv_block(c2, c3, cfg.kernel_size),
        )
        self.enc_proj = nn.Linear(c3 * cfg.downsampled_window, cfg.latent_dim)

        self.dec_proj = nn.Linear(cfg.latent_dim, c3 * cfg.downsampled_window)
        self.dec_conv = nn.Sequential(
            _deconv_block(c3, c2, cfg.kernel_size),
            _deconv_block(c2, c1, cfg.kernel_size),
            _deconv_block(c1, cfg.n_features, cfg.kernel_size, last=True),
        )

    # --- public API --------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_conv(x)
        h = h.flatten(start_dim=1)
        z = self.enc_proj(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = self.cfg.channels
        h = self.dec_proj(z)
        h = h.view(-1, c3, self.cfg.downsampled_window)
        x_hat = self.dec_conv(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error per window, averaged over the batch."""
    return torch.mean((x_hat - x) ** 2)


def per_window_recon_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-window MSE, shape ``(B,)``. Used as the anomaly score."""
    return torch.mean((x_hat - x) ** 2, dim=(1, 2))
