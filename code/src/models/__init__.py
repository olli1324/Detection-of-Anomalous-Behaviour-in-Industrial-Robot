"""Model definitions: baseline AE and adversarial AE."""
from .ae import ConvAE
from .aae import AdversarialAE, LatentDiscriminator

__all__ = ["ConvAE", "AdversarialAE", "LatentDiscriminator"]
