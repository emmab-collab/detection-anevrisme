"""
Data Augmentation

Classe pour l'augmentation de données sur les cubes 3D.
"""

import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from ..augmentation import random_deformation, data_augmentation
from ..config import (
    DEFAULT_N_AUGMENTATIONS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_DISPLACEMENT,
)


class Augmentor:
    """
    Augmentation de données pour cubes 3D.

    Applique des déformations élastiques aléatoires pour augmenter
    la taille du dataset d'entraînement.

    Parameters
    ----------
    n_augmentations : int, optional
        Nombre de versions augmentées par cube original, par défaut 12
    grid_size : int, optional
        Taille de la grille de déformation, par défaut 3
    max_displacement : float, optional
        Déplacement maximal en voxels, par défaut 3.0

    Examples
    --------
    >>> augmentor = Augmentor(n_augmentations=12)
    >>> augmented_dataset = augmentor.augment_dataset(dataset)
    >>> augmentor.save(augmented_dataset, 'output.npz')
    """

    def __init__(
        self,
        n_augmentations: int = DEFAULT_N_AUGMENTATIONS,
        grid_size: int = DEFAULT_GRID_SIZE,
        max_displacement: float = DEFAULT_MAX_DISPLACEMENT,
    ):
        self.n_augmentations = n_augmentations
        self.grid_size = grid_size
        self.max_displacement = max_displacement

    def augment_cube(self, cube: np.ndarray) -> np.ndarray:
        """
        Génère plusieurs versions augmentées d'un cube.

        Parameters
        ----------
        cube : np.ndarray
            Cube 3D à augmenter (shape: cube_size x cube_size x cube_size)

        Returns
        -------
        np.ndarray
            Cubes augmentés (shape: n_augmentations x cube_size x cube_size x cube_size)
        """
        augmented_cubes = data_augmentation(
            cube,
            grid_size=self.grid_size,
            max_displacement=self.max_displacement,
            n_augmentations=self.n_augmentations,
        )

        return augmented_cubes

    def augment_dataset(
        self, dataset: Dict[str, np.ndarray], augment_negatives: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Augmente un dataset complet.

        Parameters
        ----------
        dataset : dict
            Dataset avec clés 'cubes', 'labels', 'positions', 'patient_ids'
        augment_negatives : bool, optional
            Si True, augmente aussi les cubes négatifs, par défaut False

        Returns
        -------
        dict
            Dataset augmenté avec les mêmes clés
        """
        cubes = dataset["cubes"]
        labels = dataset["labels"]
        positions = dataset["positions"]
        patient_ids = dataset["patient_ids"]

        augmented_cubes = []
        augmented_labels = []
        augmented_positions = []
        augmented_patient_ids = []

        print(f"Augmenting dataset...")
        print(f"Original size: {len(cubes)} cubes")

        for i, (cube, label, position, patient_id) in enumerate(
            tqdm(
                zip(cubes, labels, positions, patient_ids),
                total=len(cubes),
                desc="Augmenting cubes",
            )
        ):
            # Garder l'original
            augmented_cubes.append(cube)
            augmented_labels.append(label)
            augmented_positions.append(position)
            augmented_patient_ids.append(patient_id)

            # Augmenter seulement les positifs (ou tous si augment_negatives=True)
            if label == 1 or augment_negatives:
                # Générer n_augmentations versions
                aug_cubes = self.augment_cube(cube)

                # Ajouter chaque version
                for aug_cube in aug_cubes:
                    augmented_cubes.append(aug_cube)
                    augmented_labels.append(label)
                    augmented_positions.append(position)
                    augmented_patient_ids.append(f"{patient_id}_aug")

        augmented_dataset = {
            "cubes": np.array(augmented_cubes),
            "labels": np.array(augmented_labels),
            "positions": np.array(augmented_positions),
            "patient_ids": augmented_patient_ids,
        }

        print(f"\nAugmentation complete:")
        print(f"  Original cubes: {len(cubes)}")
        print(f"  Augmented cubes: {len(augmented_cubes)}")
        print(f"  Augmentation factor: {len(augmented_cubes) / len(cubes):.1f}x")

        return augmented_dataset

    def augment_positives_only(self, cubes_positive: np.ndarray) -> np.ndarray:
        """
        Augmente uniquement les cubes positifs.

        Parameters
        ----------
        cubes_positive : np.ndarray
            Cubes positifs (N, cube_size, cube_size, cube_size)

        Returns
        -------
        np.ndarray
            Cubes augmentés incluant originaux
        """
        all_cubes = []

        for cube in tqdm(cubes_positive, desc="Augmenting positive cubes"):
            # Original
            all_cubes.append(cube)

            # Augmentations
            aug_cubes = self.augment_cube(cube)
            all_cubes.extend(aug_cubes)

        return np.array(all_cubes)

    def save(self, dataset: Dict[str, np.ndarray], output_path: str):
        """
        Sauvegarde le dataset augmenté.

        Parameters
        ----------
        dataset : dict
            Dataset augmenté
        output_path : str
            Chemin de sortie
        """
        dataset_to_save = {
            "cubes": dataset["cubes"],
            "labels": dataset["labels"],
            "positions": dataset["positions"],
            "patient_ids": np.array(dataset["patient_ids"], dtype=object),
        }

        np.savez_compressed(output_path, **dataset_to_save)
        print(f"\nAugmented dataset saved to: {output_path}")

    @staticmethod
    def load(input_path: str) -> Dict[str, np.ndarray]:
        """
        Charge un dataset augmenté.

        Parameters
        ----------
        input_path : str
            Chemin du fichier

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

        print(f"Augmented dataset loaded from: {input_path}")
        print(f"  Total cubes: {len(dataset['cubes'])}")

        return dataset

    def __repr__(self) -> str:
        return (
            f"Augmentor("
            f"n_augmentations={self.n_augmentations}, "
            f"grid_size={self.grid_size}, "
            f"max_displacement={self.max_displacement})"
        )
