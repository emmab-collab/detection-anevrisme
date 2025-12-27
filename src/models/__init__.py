"""
Models Module

Architectures de réseaux de neurones pour la détection d'anévrismes.
"""

from .unet3d import UNet3DClassifier, ConvBlock3D

__all__ = [
    "UNet3DClassifier",
    "ConvBlock3D",
]
