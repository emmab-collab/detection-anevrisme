"""
General utility functions for medical image processing.
"""

import pydicom


def get_pixelspacing(path):
    """
    Retrieve voxel spacing (mm) from a DICOM file.

    Parameters
    ----------
    path : str
        Path to the DICOM file

    Returns
    -------
    row_spacing : float or None
        Row spacing in mm/pixel
    col_spacing : float or None
        Column spacing in mm/pixel
    slice_thickness : float
        Slice thickness in mm (defaults to 1.0 if not present)

    Examples
    --------
    >>> row_sp, col_sp, slice_th = get_pixelspacing("path/to/file.dcm")
    >>> print(f"Spacing: {row_sp}mm x {col_sp}mm x {slice_th}mm")
    """
    dcm = pydicom.dcmread(path)

    # In-plane spacing (mm/pixel)
    if "PixelSpacing" in dcm:
        row_spacing, col_spacing = [float(x) for x in dcm.PixelSpacing]
    else:
        row_spacing, col_spacing = None, None

    # Slice thickness (mm)
    slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))

    return row_spacing, col_spacing, slice_thickness
