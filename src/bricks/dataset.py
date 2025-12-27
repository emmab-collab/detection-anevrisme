"""
Dataset Builder

Classe pour construire des datasets d'entraînement à partir de volumes DICOM.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import ANEURYSM_POSITIONS
from ..data import get_patient_ID, get_position
from .preprocessing import Preprocessor


class DatasetBuilder:
    """
    Construction de datasets pour l'entraînement.

    Cette classe permet de :
    - Extraire des cubes positifs (contenant des anévrismes)
    - Extraire des cubes négatifs (sans anévrisme)
    - Construire un dataset complet avec labels et positions
    - Sauvegarder au format .npz

    Parameters
    ----------
    preprocessor : Preprocessor
        Instance de Preprocessor pour le preprocessing des volumes
    cube_size : int, optional
        Taille des cubes à extraire, par défaut 48
    series_dir : str, optional
        Chemin vers le dossier contenant les séries DICOM

    Examples
    --------
    >>> preprocessor = Preprocessor()
    >>> builder = DatasetBuilder(preprocessor, cube_size=48)
    >>> dataset = builder.build_dataset(df_localizers, modality='CTA')
    >>> builder.save(dataset, 'results/processed/cta_dataset.npz')
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        cube_size: int = 48,
        series_dir: Optional[str] = None,
    ):
        self.preprocessor = preprocessor
        self.cube_size = cube_size
        self.series_dir = series_dir
        self.position_mapping = {pos: idx for idx, pos in enumerate(ANEURYSM_POSITIONS)}

    def extract_cube(
        self, volume: np.ndarray, center: np.ndarray, size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extrait un cube centré sur des coordonnées données.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D source
        center : np.ndarray
            Coordonnées du centre (x, y, z)
        size : int, optional
            Taille du cube, utilise self.cube_size si non spécifié

        Returns
        -------
        np.ndarray
            Cube extrait de taille (size, size, size)
        """
        if size is None:
            size = self.cube_size

        half = size // 2
        x, y, z = np.round(center).astype(int)

        # Calcul des bornes
        x_min, x_max = max(0, x - half), min(volume.shape[0], x + half)
        y_min, y_max = max(0, y - half), min(volume.shape[1], y + half)
        z_min, z_max = max(0, z - half), min(volume.shape[2], z + half)

        # Extraire le cube
        cube = volume[x_min:x_max, y_min:y_max, z_min:z_max]

        # Padding si nécessaire
        if cube.shape != (size, size, size):
            cube = self._pad_cube(cube, size)

        return cube

    def _pad_cube(self, cube: np.ndarray, target_size: int) -> np.ndarray:
        """Pad un cube pour atteindre la taille cible."""
        pad_x = max(0, target_size - cube.shape[0])
        pad_y = max(0, target_size - cube.shape[1])
        pad_z = max(0, target_size - cube.shape[2])

        # Centrer le padding
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        pad_z_before = pad_z // 2
        pad_z_after = pad_z - pad_z_before

        padded = np.pad(
            cube,
            (
                (pad_x_before, pad_x_after),
                (pad_y_before, pad_y_after),
                (pad_z_before, pad_z_after),
            ),
            mode="constant",
            constant_values=0,
        )

        # Vérifier la taille finale
        if padded.shape != (target_size, target_size, target_size):
            # Fallback: crop ou pad pour atteindre exactement la taille cible
            result = np.zeros(
                (target_size, target_size, target_size), dtype=padded.dtype
            )
            min_x = min(padded.shape[0], target_size)
            min_y = min(padded.shape[1], target_size)
            min_z = min(padded.shape[2], target_size)
            result[:min_x, :min_y, :min_z] = padded[:min_x, :min_y, :min_z]
            return result

        return padded

    def extract_non_overlapping_cubes(
        self, volume: np.ndarray, stride: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extrait des cubes non chevauchants d'un volume.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D source
        stride : int, optional
            Pas entre les cubes, utilise cube_size si non spécifié

        Returns
        -------
        list of np.ndarray
            Liste de cubes extraits
        """
        if stride is None:
            stride = self.cube_size

        D, H, W = volume.shape
        cubes = []

        for z in range(0, D - self.cube_size + 1, stride):
            for y in range(0, H - self.cube_size + 1, stride):
                for x in range(0, W - self.cube_size + 1, stride):
                    cube = volume[
                        z : z + self.cube_size,
                        y : y + self.cube_size,
                        x : x + self.cube_size,
                    ]

                    if cube.shape == (self.cube_size, self.cube_size, self.cube_size):
                        cubes.append(cube)

        return cubes

    def create_position_vector(self, position_name: str) -> np.ndarray:
        """
        Crée un vecteur one-hot pour la position de l'anévrisme.

        Parameters
        ----------
        position_name : str
            Nom de la position anatomique

        Returns
        -------
        np.ndarray
            Vecteur one-hot de taille (13,)
        """
        vector = np.zeros(13, dtype=float)

        if position_name in self.position_mapping:
            idx = self.position_mapping[position_name]
            vector[idx] = 1.0

        return vector

    def extract_positive_cubes(
        self, df_localizers: pd.DataFrame, n_cubes_per_patient: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Extrait les cubes positifs (contenant des anévrismes).

        Parameters
        ----------
        df_localizers : pd.DataFrame
            DataFrame avec les informations de localisation
        n_cubes_per_patient : int, optional
            Nombre de cubes à extraire par patient, par défaut 1

        Returns
        -------
        dict
            Dictionnaire contenant:
            - 'cubes': array (N, cube_size, cube_size, cube_size)
            - 'labels': array (N,) de 1
            - 'positions': array (N, 13) one-hot encodé
            - 'patient_ids': list de SeriesInstanceUID
        """
        cubes_list = []
        positions_list = []
        patient_ids = []

        for idx, row in tqdm(
            df_localizers.iterrows(),
            total=len(df_localizers),
            desc="Extracting positive cubes",
        ):

            try:
                series_uid = row["SeriesInstanceUID"]
                patient_path = os.path.join(self.series_dir, series_uid)

                # Get coordinates et position
                coords = np.array([row["x"], row["y"], row["z"]])
                position_name = row.get("location", "Unknown")

                # Preprocessing
                volume, transformed_coords = (
                    self.preprocessor.process_volume_with_coords(patient_path, coords)
                )

                # Extract cube
                cube = self.extract_cube(volume, transformed_coords)

                # Position vector
                position_vector = self.create_position_vector(position_name)

                cubes_list.append(cube)
                positions_list.append(position_vector)
                patient_ids.append(series_uid)

            except Exception as e:
                print(f"Error processing {series_uid}: {e}")
                continue

        return {
            "cubes": np.array(cubes_list),
            "labels": np.ones(len(cubes_list), dtype=float),
            "positions": np.array(positions_list),
            "patient_ids": patient_ids,
        }

    def extract_negative_cubes(
        self, df_negatives: pd.DataFrame, n_cubes_per_volume: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Extrait les cubes négatifs (sans anévrisme).

        Parameters
        ----------
        df_negatives : pd.DataFrame
            DataFrame avec les séries sans anévrisme
        n_cubes_per_volume : int, optional
            Nombre de cubes à extraire par volume, par défaut 5

        Returns
        -------
        dict
            Dictionnaire avec cubes, labels, positions, patient_ids
        """
        cubes_list = []
        patient_ids = []

        for idx, row in tqdm(
            df_negatives.iterrows(),
            total=len(df_negatives),
            desc="Extracting negative cubes",
        ):

            try:
                series_uid = row["SeriesInstanceUID"]
                patient_path = os.path.join(self.series_dir, series_uid)

                # Preprocessing
                volume = self.preprocessor.process_volume(patient_path)

                # Extract non-overlapping cubes
                cubes = self.extract_non_overlapping_cubes(volume)

                # Sample n_cubes_per_volume
                if len(cubes) > n_cubes_per_volume:
                    indices = np.random.choice(
                        len(cubes), n_cubes_per_volume, replace=False
                    )
                    cubes = [cubes[i] for i in indices]

                cubes_list.extend(cubes)
                patient_ids.extend([series_uid] * len(cubes))

            except Exception as e:
                print(f"Error processing {series_uid}: {e}")
                continue

        n_cubes = len(cubes_list)

        return {
            "cubes": np.array(cubes_list),
            "labels": np.zeros(n_cubes, dtype=float),
            "positions": np.zeros((n_cubes, 13), dtype=float),
            "patient_ids": patient_ids,
        }

    def _filter_available_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre le DataFrame pour ne garder que les séries disponibles localement.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame avec colonne 'SeriesInstanceUID'

        Returns
        -------
        pd.DataFrame
            DataFrame filtré avec seulement les séries existantes
        """
        if self.series_dir is None:
            return df

        available_series = []
        for series_uid in df["SeriesInstanceUID"].unique():
            patient_path = os.path.join(self.series_dir, series_uid)
            if os.path.exists(patient_path):
                available_series.append(series_uid)

        return df[df["SeriesInstanceUID"].isin(available_series)].reset_index(drop=True)

    def build_dataset(
        self, df_train: pd.DataFrame, df_localizers: pd.DataFrame, modality: str = "CTA"
    ) -> Dict[str, np.ndarray]:
        """
        Construit un dataset complet pour une modalité.

        Parameters
        ----------
        df_train : pd.DataFrame
            DataFrame principal avec les informations des séries
        df_localizers : pd.DataFrame
            DataFrame avec les localisations d'anévrismes
        modality : str, optional
            Modalité à traiter ('CTA', 'MRA', 'MRI T1post', 'MRI T2')

        Returns
        -------
        dict
            Dataset complet avec cubes, labels, positions
        """
        print(f"\n{'='*60}")
        print(f"Building dataset for {modality}")
        print(f"{'='*60}\n")

        # Filter by modality
        df_modality = df_train[df_train["Modality"] == modality].reset_index(drop=True)
        print(f"Total {modality} series in CSV: {len(df_modality)}")

        # Filter only locally available series
        df_modality = self._filter_available_series(df_modality)
        print(f"Available {modality} series locally: {len(df_modality)}")

        # Split positives and negatives
        df_positives = df_modality[df_modality["Aneurysm Present"] == 1]
        df_negatives = df_modality[df_modality["Aneurysm Present"] == 0]

        # Join localizers for positives
        df_pos_loc = df_localizers[
            df_localizers["SeriesInstanceUID"].isin(df_positives["SeriesInstanceUID"])
        ]

        print(f"Positive series: {len(df_positives)}")
        print(f"Negative series: {len(df_negatives)}")

        # Extract cubes
        positive_data = self.extract_positive_cubes(df_pos_loc)
        negative_data = self.extract_negative_cubes(df_negatives)

        # Combine - handle empty cases
        if len(positive_data["cubes"]) == 0 and len(negative_data["cubes"]) == 0:
            print("\n⚠️ No cubes extracted for this modality")
            return {
                "cubes": np.array([]),
                "labels": np.array([]),
                "positions": np.array([]).reshape(0, 13),
                "patient_ids": [],
            }
        elif len(positive_data["cubes"]) == 0:
            dataset = {
                "cubes": negative_data["cubes"],
                "labels": negative_data["labels"],
                "positions": negative_data["positions"],
                "patient_ids": negative_data["patient_ids"],
            }
        elif len(negative_data["cubes"]) == 0:
            dataset = {
                "cubes": positive_data["cubes"],
                "labels": positive_data["labels"],
                "positions": positive_data["positions"],
                "patient_ids": positive_data["patient_ids"],
            }
        else:
            dataset = {
                "cubes": np.concatenate(
                    [positive_data["cubes"], negative_data["cubes"]]
                ),
                "labels": np.concatenate(
                    [positive_data["labels"], negative_data["labels"]]
                ),
                "positions": np.concatenate(
                    [positive_data["positions"], negative_data["positions"]]
                ),
                "patient_ids": positive_data["patient_ids"]
                + negative_data["patient_ids"],
            }

        print(f"\nDataset created:")
        print(f"  Total cubes: {len(dataset['cubes'])}")
        print(f"  Positive: {int(dataset['labels'].sum())}")
        print(f"  Negative: {int((1 - dataset['labels']).sum())}")

        return dataset

    def save(self, dataset: Dict[str, np.ndarray], output_path: str):
        """
        Sauvegarde le dataset au format .npz.

        Parameters
        ----------
        dataset : dict
            Dataset à sauvegarder
        output_path : str
            Chemin de sortie
        """
        # Convert patient_ids to array for npz
        dataset_to_save = {
            "cubes": dataset["cubes"],
            "labels": dataset["labels"],
            "positions": dataset["positions"],
            "patient_ids": np.array(dataset["patient_ids"], dtype=object),
        }

        np.savez_compressed(output_path, **dataset_to_save)
        print(f"\nDataset saved to: {output_path}")

    @staticmethod
    def load(input_path: str) -> Dict[str, np.ndarray]:
        """
        Charge un dataset depuis un fichier .npz.

        Parameters
        ----------
        input_path : str
            Chemin du fichier à charger

        Returns
        -------
        dict
            Dataset chargé
        """
        loaded = np.load(input_path, allow_pickle=True)

        dataset = {
            "cubes": loaded["cubes"],
            "labels": loaded["labels"],
            "positions": loaded["positions"],
            "patient_ids": loaded["patient_ids"].tolist(),
        }

        print(f"Dataset loaded from: {input_path}")
        print(f"  Total cubes: {len(dataset['cubes'])}")

        return dataset

    def __repr__(self) -> str:
        return (
            f"DatasetBuilder("
            f"cube_size={self.cube_size}, "
            f"preprocessor={self.preprocessor})"
        )
