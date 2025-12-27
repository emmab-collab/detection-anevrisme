"""
Volume transformation functions for preprocessing.
"""

import numpy as np
from scipy.ndimage import zoom
from ..config import TARGET_SPACING, CROP_THRESHOLD


def resample(volume, spacing, target_spacing=None):
    """
    Resample a 3D volume to a target spacing.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume with shape (X, Y, Z)
    spacing : tuple of float
        Original voxel spacing (dx, dy, dz) in mm
    target_spacing : tuple of float or None, optional
        Target voxel spacing (dx, dy, dz) in mm
        If None, uses TARGET_SPACING from config (default: None)

    Returns
    -------
    np.ndarray
        Resampled volume

    Examples
    --------
    >>> volume, spacing = dicom_to_numpy(patient_path)
    >>> resampled = resample(volume, spacing, target_spacing=(0.4, 0.4, 0.4))
    """
    if target_spacing is None:
        target_spacing = TARGET_SPACING

    zoom_factors = [s / t for s, t in zip(spacing, target_spacing)]
    new_volume = zoom(volume, zoom_factors, order=1)

    return new_volume


def crop(volume, threshold=None):
    """
    Crop a volume to remove background regions.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume with shape (X, Y, Z)
    threshold : float or None, optional
        Threshold as a fraction of max intensity (default: uses CROP_THRESHOLD from config)

    Returns
    -------
    cropped_volume : np.ndarray
        Cropped volume
    crop_indices : tuple of int
        Starting indices (x_min, y_min, z_min) of the crop

    Notes
    -----
    Creates a mask of voxels above threshold * max_intensity and crops to the
    bounding box of this mask.
    """
    if threshold is None:
        threshold = CROP_THRESHOLD

    # Create mask of non-zero voxels
    mask = volume > (volume.max() * threshold)

    if not mask.any():
        return volume, (0, 0, 0)

    # Get min/max indices for each dimension
    x_min, x_max = mask.any(axis=(1, 2)).nonzero()[0][[0, -1]]
    y_min, y_max = mask.any(axis=(0, 2)).nonzero()[0][[0, -1]]
    z_min, z_max = mask.any(axis=(0, 1)).nonzero()[0][[0, -1]]

    # Crop
    cropped = volume[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]

    return cropped, (x_min, y_min, z_min)


def normalization(volume):
    """
    Normalize a volume to the [0, 1] range.

    Parameters
    ----------
    volume : np.ndarray
        Input volume

    Returns
    -------
    np.ndarray
        Normalized volume with values in [0, 1]

    Notes
    -----
    Uses min-max normalization. If max == min, returns zeros.
    """
    v_min, v_max = volume.min(), volume.max()

    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume
