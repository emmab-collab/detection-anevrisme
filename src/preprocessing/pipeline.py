"""
Complete preprocessing pipelines for volume and coordinate processing.
"""

import numpy as np

from ..config import TARGET_SPACING
from ..data.dicom_loader import dicom_to_numpy
from .coordinates import get_center, resample_coordonnees
from .transforms import crop, normalization, resample


def preprocessing_volume_and_coords(
    series_path, patient_path, df_loc, target_spacing=None
):
    """
    Complete preprocessing pipeline for volume and aneurysm coordinates.

    This function performs the full preprocessing pipeline:
    1. Load DICOM volume and get aneurysm center coordinates
    2. Resample volume and coordinates to target spacing
    3. Crop volume to remove background and adjust coordinates
    4. Normalize volume to [0, 1] range

    Parameters
    ----------
    series_path : str
        Base path to the series directory
    patient_path : str
        Path to the patient directory
    df_loc : pd.DataFrame
        DataFrame containing location information
    target_spacing : tuple of float or None, optional
        Target voxel spacing (dx, dy, dz) in mm
        If None, uses TARGET_SPACING from config (default: None)

    Returns
    -------
    norm_volume : np.ndarray
        Preprocessed and normalized volume
    crop_coords : np.ndarray
        Adjusted aneurysm coordinates [x, y, z] after cropping

    Examples
    --------
    >>> volume, coords = preprocessing_volume_and_coords(
    ...     series_path, patient_path, df_loc, target_spacing=(0.4, 0.4, 0.4)
    ... )
    """
    if target_spacing is None:
        target_spacing = TARGET_SPACING

    # Get original data
    coords = get_center(series_path, patient_path, df_loc)
    volume, spacing = dicom_to_numpy(patient_path)

    # Resample volume and coordinates
    resample_volume = resample(volume, spacing, target_spacing=target_spacing)
    resample_coords = resample_coordonnees(
        spacing, coords, target_spacing=target_spacing
    )

    # Crop and adjust coordinates
    crop_volume, crop_indices = crop(resample_volume)
    crop_coords = resample_coords - np.array(crop_indices)

    # Normalize
    norm_volume = normalization(crop_volume)

    return norm_volume, crop_coords


def preprocessing_volume(patient_path, target_spacing=None):
    """
    Preprocessing pipeline for volume only (for inference).

    This function performs preprocessing without coordinate tracking:
    1. Load DICOM volume
    2. Resample to target spacing
    3. Crop to remove background
    4. Normalize to [0, 1] range

    Parameters
    ----------
    patient_path : str
        Path to the patient directory
    target_spacing : tuple of float or None, optional
        Target voxel spacing (dx, dy, dz) in mm
        If None, uses TARGET_SPACING from config (default: None)

    Returns
    -------
    np.ndarray
        Preprocessed and normalized volume

    Examples
    --------
    >>> volume = preprocessing_volume(patient_path, target_spacing=(0.4, 0.4, 0.4))
    """
    if target_spacing is None:
        target_spacing = TARGET_SPACING

    volume, spacing = dicom_to_numpy(patient_path)
    resample_volume = resample(volume, spacing, target_spacing=target_spacing)
    crop_volume, _ = crop(resample_volume)
    new_volume = normalization(crop_volume)

    return new_volume
