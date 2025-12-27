"""
UNet 3D Architecture

Implémentation d'un UNet 3D pour la classification de cubes médicaux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Bloc de convolution 3D double avec normalisation et activation.

    Parameters
    ----------
    in_ch : int
        Nombre de canaux en entrée
    out_ch : int
        Nombre de canaux en sortie
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DClassifier(nn.Module):
    """
    UNet 3D pour classification binaire et localisation.

    Architecture encoder-decoder avec skip connections.
    Sortie : 14 valeurs (13 positions + 1 label binaire)

    Parameters
    ----------
    in_ch : int, optional
        Nombre de canaux en entrée, par défaut 1
    base_ch : int, optional
        Nombre de canaux de base, par défaut 16

    Examples
    --------
    >>> model = UNet3DClassifier(in_ch=1, base_ch=32)
    >>> x = torch.randn(2, 1, 48, 48, 48)
    >>> output = model(x)  # (2, 14)
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)

        # Classifier pour 14 sorties (13 positions + 1 label)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(base_ch, 14)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor d'entrée (B, C, D, H, W)

        Returns
        -------
        torch.Tensor
            Logits de sortie (B, 14)
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Classification
        out = self.classifier(d1)  # (B, 14)

        return out
