"""
Patient metadata extraction utilities.
"""

import os
import pandas as pd
import ast
from ..config import ANEURYSM_POSITIONS


def get_patient_ID(patient_path):
    """
    Extract the patient ID from the patient path.

    Parameters
    ----------
    patient_path : str
        Path to the patient directory

    Returns
    -------
    str
        The patient ID (basename of the path)
    """
    return str(os.path.basename(patient_path))


def get_position(df_train, patient_path):
    """
    Get the aneurysm position vector for a patient.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame containing SeriesInstanceUID and position columns
    patient_path : str
        Path to the patient directory

    Returns
    -------
    np.ndarray
        1D array with binary indicators for each possible aneurysm position

    Notes
    -----
    The positions are defined in config.ANEURYSM_POSITIONS
    """
    patient_id = get_patient_ID(patient_path)
    row = df_train[df_train["SeriesInstanceUID"] == patient_id][ANEURYSM_POSITIONS].values.flatten()
    return row


def parse_coordinates(coordinates_str):
    """
    Parse coordinates string to extract x, y, z values.

    Parameters
    ----------
    coordinates_str : str
        String representation of coordinates dict (e.g., "{'x': 1.0, 'y': 2.0}")

    Returns
    -------
    tuple
        (x, y, z) coordinates. z is 0.0 if not present.
    """
    try:
        coords_dict = ast.literal_eval(coordinates_str)
        x = coords_dict.get('x', 0.0)
        y = coords_dict.get('y', 0.0)
        z = coords_dict.get('z', 0.0)
        return x, y, z
    except:
        return 0.0, 0.0, 0.0


def expand_coordinates(df_localizers):
    """
    Expand the 'coordinates' column into separate x, y, z columns.

    Parameters
    ----------
    df_localizers : pd.DataFrame
        DataFrame with 'coordinates' column (string format)

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'x', 'y', 'z' columns
    """
    df = df_localizers.copy()

    # Parse coordinates
    coords = df['coordinates'].apply(parse_coordinates)

    # Add as separate columns
    df['x'] = coords.apply(lambda c: c[0])
    df['y'] = coords.apply(lambda c: c[1])
    df['z'] = coords.apply(lambda c: c[2])

    return df


def ajouter_Modality(df_main, df_info):
    """
    Add Modality column to main DataFrame by merging with info DataFrame.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main DataFrame containing SeriesInstanceUID
    df_info : pd.DataFrame
        Info DataFrame containing SeriesInstanceUID and Modality columns

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with Modality column added
    """
    df_merged = df_main.merge(
        df_info[['SeriesInstanceUID', 'Modality']],
        on='SeriesInstanceUID',
        how='left'
    )
    return df_merged
