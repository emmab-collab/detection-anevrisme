"""
DICOM file loading and processing utilities.
"""

import glob

import numpy as np
import pydicom


def get_instance_number(patient_path, df_loc, series_path):
    """
    Retrieve the InstanceNumber from a DICOM file.

    Parameters
    ----------
    patient_path : str
        Path to the patient directory
    df_loc : pd.DataFrame
        DataFrame containing location information with columns:
        'SeriesInstanceUID', 'SOPInstanceUID'
    series_path : str
        Base path to the series directory

    Returns
    -------
    int
        The InstanceNumber from the DICOM file
    """
    import os

    numero_patient = os.path.basename(patient_path)
    numero_coupe = df_loc[df_loc['SeriesInstanceUID'] == numero_patient]['SOPInstanceUID'].iloc[0]
    exemple_path = os.path.join(
        series_path,
        f"{numero_patient}/{numero_coupe}.dcm"
    )

    ds = pydicom.dcmread(exemple_path)
    return int(ds.InstanceNumber)


def coordonnee_z(patient_path, instance_number=163):
    """
    Find the z-index in the volume corresponding to an InstanceNumber.

    Parameters
    ----------
    patient_path : str
        Path to the patient directory containing DICOM files
    instance_number : int, optional
        The InstanceNumber to search for (default: 163)

    Returns
    -------
    int
        The z-index in the sorted volume

    Raises
    ------
    IndexError
        If the InstanceNumber is not found in the slices
    """
    dicom_files = sorted(glob.glob(patient_path + '/*.dcm'))
    slices = [pydicom.dcmread(f) for f in dicom_files]

    # Sort slices by InstanceNumber
    slices.sort(key=lambda s: int(s.InstanceNumber))

    # Find the z-index in the volume
    z_indices = [i for i, s in enumerate(slices) if int(s.InstanceNumber) == instance_number]

    if not z_indices:
        raise ValueError(f"InstanceNumber {instance_number} not found in patient slices")

    return z_indices[0]


def dicom_to_numpy(patient_path):
    """
    Convert a directory of DICOM files to a 3D NumPy volume.

    Parameters
    ----------
    patient_path : str
        Path to the patient directory containing DICOM files

    Returns
    -------
    volume : np.ndarray
        3D volume with shape (X, Y, Z)
    spacing : tuple of float
        Voxel spacing (dx, dy, dz) in mm

    Notes
    -----
    - Slices are sorted by InstanceNumber
    - Only slices with matching dimensions are kept
    - Falls back to SliceThickness=1.0 if not present in DICOM
    """
    dicom_files = sorted(glob.glob(patient_path + '/*.dcm'))
    slices = [pydicom.dcmread(f) for f in dicom_files]

    # Sort slices by InstanceNumber
    slices.sort(key=lambda s: int(s.InstanceNumber))

    # Stack pixel arrays into a 3D volume (X, Y, Z)
    target_shape = slices[0].pixel_array.shape

    # Filter out slices with different dimensions
    slices = [s for s in slices if s.pixel_array.shape == target_shape]
    volume = np.stack([s.pixel_array for s in slices], axis=-1)

    # Get real voxel spacing
    pixel_spacing = slices[0].PixelSpacing
    dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])
    dz = float(getattr(slices[0], 'SliceThickness', 1.0))

    return volume, (dx, dy, dz)
