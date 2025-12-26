"""
Exploratory Data Analysis

Classe pour l'analyse exploratoire des données DICOM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from tqdm import tqdm
import os
import pydicom

from ..data import ajouter_Modality


class EDA:
    """
    Analyse exploratoire des données.

    Cette classe permet d'analyser :
    - La distribution des modalités
    - Les séries défectueuses (tailles incohérentes)
    - La distribution des anévrismes par position
    - Les statistiques des volumes

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame principal avec les informations des séries
    df_localizers : pd.DataFrame
        DataFrame avec les localisations d'anévrismes
    series_dir : str
        Chemin vers le dossier contenant les séries DICOM

    Examples
    --------
    >>> eda = EDA(df_train, df_localizers, series_dir)
    >>> eda.analyze_modalities()
    >>> defective = eda.detect_defective_series()
    >>> eda.plot_aneurysm_distribution()
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_localizers: pd.DataFrame,
        series_dir: str
    ):
        self.df_train = df_train
        self.df_localizers = df_localizers
        self.series_dir = series_dir

        # Ajouter modalité aux localizers
        self.df_loc_with_modality = ajouter_Modality(
            self.df_localizers, self.df_train
        )

    def analyze_modalities(self) -> pd.Series:
        """
        Analyse la distribution des modalités.

        Returns
        -------
        pd.Series
            Comptage par modalité
        """
        print("\n" + "="*60)
        print("MODALITY DISTRIBUTION")
        print("="*60)

        modality_counts = self.df_train['Modality'].value_counts()
        print(modality_counts)

        return modality_counts

    def analyze_aneurysm_distribution(self) -> pd.Series:
        """
        Analyse la distribution des anévrismes.

        Returns
        -------
        pd.Series
            Comptage présence/absence
        """
        print("\n" + "="*60)
        print("ANEURYSM DISTRIBUTION")
        print("="*60)

        aneurysm_counts = self.df_train['Aneurysm Present'].value_counts()
        print(f"Without aneurysm: {aneurysm_counts.get(0, 0)}")
        print(f"With aneurysm: {aneurysm_counts.get(1, 0)}")
        print(f"Total: {len(self.df_train)}")

        return aneurysm_counts

    def analyze_positions(self) -> pd.DataFrame:
        """
        Analyse la distribution des positions anatomiques.

        Returns
        -------
        pd.DataFrame
            Statistiques par position
        """
        print("\n" + "="*60)
        print("ANEURYSM POSITIONS")
        print("="*60)

        # Colonnes de positions
        position_cols = [col for col in self.df_train.columns
                        if col not in ['SeriesInstanceUID', 'PatientAge',
                                      'PatientSex', 'Modality', 'Aneurysm Present']]

        position_counts = self.df_train[position_cols].sum().sort_values(ascending=False)

        print(position_counts)

        return position_counts

    def detect_defective_series(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Détecte les séries avec des tailles d'images incohérentes.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame à analyser, utilise df_loc_with_modality si None

        Returns
        -------
        list of str
            Liste des SeriesInstanceUID défectueux
        """
        if df is None:
            df = self.df_loc_with_modality

        print("\n" + "="*60)
        print("DETECTING DEFECTIVE SERIES")
        print("="*60)

        defective_series = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking series"):
            series_uid = row['SeriesInstanceUID']
            patient_path = os.path.join(self.series_dir, series_uid)

            if not os.path.exists(patient_path):
                continue

            try:
                dicom_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

                if not dicom_files:
                    continue

                shapes = set()
                for dcm_file in dicom_files[:10]:  # Check first 10 files
                    dcm_path = os.path.join(patient_path, dcm_file)
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    shapes.add((ds.Rows, ds.Columns))

                if len(shapes) > 1:
                    print(f"⚠️  {series_uid}: multiple shapes {shapes}")
                    defective_series.append(series_uid)

            except Exception as e:
                print(f"Error checking {series_uid}: {e}")
                continue

        print(f"\nFound {len(defective_series)} defective series")

        return defective_series

    def analyze_slice_counts(self) -> Dict:
        """
        Analyse le nombre de slices par série.

        Returns
        -------
        dict
            Statistiques sur le nombre de slices
        """
        print("\n" + "="*60)
        print("SLICE COUNT STATISTICS")
        print("="*60)

        slice_counts = []

        for _, row in tqdm(self.df_train.iterrows(),
                          total=min(100, len(self.df_train)),
                          desc="Counting slices"):

            series_uid = row['SeriesInstanceUID']
            patient_path = os.path.join(self.series_dir, series_uid)

            if not os.path.exists(patient_path):
                continue

            try:
                dicom_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]
                slice_counts.append(len(dicom_files))
            except:
                continue

        if slice_counts:
            stats = {
                'mean': np.mean(slice_counts),
                'median': np.median(slice_counts),
                'min': np.min(slice_counts),
                'max': np.max(slice_counts),
                'std': np.std(slice_counts)
            }

            print(f"Mean: {stats['mean']:.1f}")
            print(f"Median: {stats['median']:.1f}")
            print(f"Min: {stats['min']}")
            print(f"Max: {stats['max']}")
            print(f"Std: {stats['std']:.1f}")

            return stats

        return {}

    def plot_aneurysm_distribution(self):
        """Visualise la distribution des anévrismes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Présence/Absence
        aneurysm_counts = self.df_train['Aneurysm Present'].value_counts()
        axes[0].bar(['Absent', 'Present'], aneurysm_counts.values)
        axes[0].set_title('Aneurysm Presence Distribution')
        axes[0].set_ylabel('Count')

        # Plot 2: Distribution par modalité
        modality_counts = self.df_train['Modality'].value_counts()
        axes[1].bar(modality_counts.index, modality_counts.values)
        axes[1].set_title('Modality Distribution')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_report(self, output_path: Optional[str] = None):
        """
        Génère un rapport complet d'analyse.

        Parameters
        ----------
        output_path : str, optional
            Chemin pour sauvegarder le rapport HTML
        """
        print("\n" + "="*60)
        print("EDA REPORT")
        print("="*60)

        self.analyze_modalities()
        self.analyze_aneurysm_distribution()
        self.analyze_positions()
        self.analyze_slice_counts()

        # TODO: Générer rapport HTML si output_path fourni

        print("\n" + "="*60)

    def __repr__(self) -> str:
        return (
            f"EDA("
            f"n_series={len(self.df_train)}, "
            f"n_localizers={len(self.df_localizers)})"
        )
