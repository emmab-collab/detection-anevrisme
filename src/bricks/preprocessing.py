"""
Preprocessing Pipeline

Classe pour le preprocessing des volumes DICOM 3D.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional

from ..data import dicom_to_numpy
from ..config import TARGET_SPACING, CROP_THRESHOLD


class Preprocessor:
    """
    Pipeline de preprocessing pour volumes DICOM 3D.

    Ce pipeline effectue les étapes suivantes :
    1. Chargement du volume DICOM
    2. Resampling à un espacement cible
    3. Cropping pour retirer le fond noir
    4. Normalisation entre 0 et 1

    Parameters
    ----------
    target_spacing : tuple of float, optional
        Espacement cible en mm (dx, dy, dz), par défaut (0.4, 0.4, 0.4)
    crop_threshold : float, optional
        Seuil pour le cropping (ratio du max), par défaut 0.1

    Examples
    --------
    >>> preprocessor = Preprocessor()
    >>> volume = preprocessor.process_volume(patient_path)
    >>> volume, coords = preprocessor.process_volume_with_coords(patient_path, aneurysm_coords)
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = TARGET_SPACING,
        crop_threshold: float = CROP_THRESHOLD,
    ):
        self.target_spacing = target_spacing
        self.crop_threshold = crop_threshold

    def resample(
        self, volume: np.ndarray, spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Resample le volume à l'espacement cible.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D à resampler
        spacing : tuple of float
            Espacement actuel (dx, dy, dz) en mm

        Returns
        -------
        np.ndarray
            Volume resampleé
        """
        zoom_factors = [s / t for s, t in zip(spacing, self.target_spacing)]
        resampled = zoom(volume, zoom_factors, order=1)
        return resampled

    def crop(self, volume: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Coupe le volume pour retirer le fond noir.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D à cropper

        Returns
        -------
        cropped : np.ndarray
            Volume croppé
        crop_indices : tuple of int
            Indices du coin supérieur gauche (x_min, y_min, z_min)
        """
        # Créer un masque des voxels non nuls
        mask = volume > (volume.max() * self.crop_threshold)

        if not mask.any():
            return volume, (0, 0, 0)

        # Récupérer les indices min/max pour chaque dimension
        x_min, x_max = mask.any(axis=(1, 2)).nonzero()[0][[0, -1]]
        y_min, y_max = mask.any(axis=(0, 2)).nonzero()[0][[0, -1]]
        z_min, z_max = mask.any(axis=(0, 1)).nonzero()[0][[0, -1]]

        # Crop
        cropped = volume[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
        crop_indices = (x_min, y_min, z_min)

        return cropped, crop_indices

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalise le volume entre 0 et 1.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D à normaliser

        Returns
        -------
        np.ndarray
            Volume normalisé
        """
        v_min, v_max = volume.min(), volume.max()

        if v_max > v_min:
            normalized = (volume - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(volume)

        return normalized

    def resample_coordinates(
        self, coords: np.ndarray, spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Resample les coordonnées à l'espacement cible.

        Parameters
        ----------
        coords : np.ndarray
            Coordonnées (x, y, z) dans l'espace original
        spacing : tuple of float
            Espacement original (dx, dy, dz) en mm

        Returns
        -------
        np.ndarray
            Coordonnées dans l'espace resampleé
        """
        x, y, z = coords

        # Coordonnées physiques en mm
        coords_mm = np.array([x * spacing[0], y * spacing[1], z * spacing[2]])

        # Nouveaux indices
        new_coords = coords_mm / np.array(self.target_spacing)

        return new_coords

    def process_volume(self, patient_path: str) -> np.ndarray:
        """
        Pipeline complet de preprocessing d'un volume.

        Parameters
        ----------
        patient_path : str
            Chemin vers le dossier contenant les fichiers DICOM

        Returns
        -------
        np.ndarray
            Volume préprocessé et normalisé
        """
        # 1. Charger le volume DICOM
        volume, spacing = dicom_to_numpy(patient_path)

        # 2. Resample
        volume = self.resample(volume, spacing)

        # 3. Crop
        volume, _ = self.crop(volume)

        # 4. Normalize
        volume = self.normalize(volume)

        return volume

    def process_volume_with_coords(
        self, patient_path: str, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline complet avec transformation des coordonnées.

        Parameters
        ----------
        patient_path : str
            Chemin vers le dossier contenant les fichiers DICOM
        coords : np.ndarray
            Coordonnées de l'anévrisme (x, y, z)

        Returns
        -------
        volume : np.ndarray
            Volume préprocessé et normalisé
        transformed_coords : np.ndarray
            Coordonnées transformées dans le nouvel espace
        """
        # 1. Charger le volume DICOM
        volume, spacing = dicom_to_numpy(patient_path)

        # 2. Resample volume et coordonnées
        volume = self.resample(volume, spacing)
        coords = self.resample_coordinates(coords, spacing)

        # 3. Crop volume et ajuster coordonnées
        volume, crop_indices = self.crop(volume)
        coords = coords - np.array(crop_indices)

        # 4. Normalize
        volume = self.normalize(volume)

        return volume, coords

    def __repr__(self) -> str:
        return (
            f"Preprocessor("
            f"target_spacing={self.target_spacing}, "
            f"crop_threshold={self.crop_threshold})"
        )
