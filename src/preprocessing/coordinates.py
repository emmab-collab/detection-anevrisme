"""
Coordinate transformation utilities for medical images.
"""

import numpy as np
import ast
from ..data.dicom_loader import get_instance_number, coordonnee_z
from ..config import TARGET_SPACING


def get_center(series_path, patient_path, df_loc):
    """
    Get the 3D center coordinates of an aneurysm.

    Parameters
    ----------
    series_path : str
        Base path to the series directory
    patient_path : str
        Path to the patient directory
    df_loc : pd.DataFrame
        DataFrame containing location information with columns:
        'SeriesInstanceUID', 'SOPInstanceUID', 'coordinates'

    Returns
    -------
    np.ndarray
        3D coordinates [x, y, z] of the aneurysm center

    Notes
    -----
    - Extracts coordinates from df_loc (stored as string dict)
    - Note: x and y are intentionally swapped in the original code
    - z is computed from the InstanceNumber
    """
    import os

    numero_patient = os.path.basename(patient_path)
    numero_coupe = df_loc[df_loc["SeriesInstanceUID"] == numero_patient][
        "SOPInstanceUID"
    ].iloc[0]

    instance_number = get_instance_number(patient_path, df_loc, series_path)
    z = coordonnee_z(patient_path, instance_number)

    coord_str = df_loc[df_loc["SOPInstanceUID"] == numero_coupe]["coordinates"].iloc[0]
    coord_dict = ast.literal_eval(coord_str)

    # Note: x and y are swapped intentionally
    x = coord_dict["y"]
    y = coord_dict["x"]

    center = np.array([x, y, z])
    return center


def resample_coordonnees(spacing, coords, target_spacing=None):
    """
    Resample coordinates from original spacing to target spacing.

    Parameters
    ----------
    spacing : tuple of float
        Original voxel spacing (dx, dy, dz) in mm
    coords : array-like
        Original coordinates [x, y, z] in voxel indices
    target_spacing : float or tuple of float, optional
        Target spacing. Can be:
        - float: isotropic spacing (same for all dimensions)
        - tuple: (dx, dy, dz)
        - None: uses TARGET_SPACING from config (default: None)

    Returns
    -------
    np.ndarray
        New coordinates [x, y, z] in the resampled space

    Examples
    --------
    >>> original_coords = np.array([100, 150, 80])
    >>> original_spacing = (0.8, 0.8, 1.0)
    >>> new_coords = resample_coordonnees(original_spacing, original_coords, 0.4)
    """
    if target_spacing is None:
        target_spacing = (
            TARGET_SPACING[0] if isinstance(TARGET_SPACING, tuple) else TARGET_SPACING
        )

    x, y, z = coords

    if isinstance(target_spacing, (int, float)):
        target_spacing = (target_spacing, target_spacing, target_spacing)

    # Convert to physical coordinates (mm)
    coords_mm = np.array([x * spacing[0], y * spacing[1], z * spacing[2]])

    # Convert to new voxel indices
    new_voxel = coords_mm / np.array(target_spacing)

    return new_voxel
