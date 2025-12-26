"""
Patient metadata extraction utilities.
"""

import os
import pandas as pd
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
