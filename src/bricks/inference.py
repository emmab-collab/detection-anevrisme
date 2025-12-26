"""
Model Inference

Classe pour l'inférence sur nouvelles données.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .preprocessing import Preprocessor


class Predictor:
    """
    Inférence sur nouvelles données.

    Cette classe gère :
    - Le chargement de modèles entraînés
    - L'inférence sur volumes complets
    - L'agrégation de prédictions sur cubes multiples

    Parameters
    ----------
    model : nn.Module
        Modèle PyTorch
    preprocessor : Preprocessor
        Preprocessor pour les volumes
    device : str, optional
        Device ('cuda' ou 'cpu'), par défaut 'cuda'
    cube_size : int, optional
        Taille des cubes, par défaut 48
    stride : int, optional
        Pas pour extraction de cubes, par défaut 24

    Examples
    --------
    >>> model = UNet3DClassifier()
    >>> preprocessor = Preprocessor()
    >>> predictor = Predictor(model, preprocessor)
    >>> predictor.load_model('best_model.pth')
    >>> prediction = predictor.predict_volume(patient_path)
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: Preprocessor,
        device: str = 'cuda',
        cube_size: int = 48,
        stride: int = 24
    ):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device
        self.cube_size = cube_size
        self.stride = stride

        self.model.eval()

    def load_model(self, path: str):
        """
        Charge les poids d'un modèle.

        Parameters
        ----------
        path : str
            Chemin vers le fichier .pth
        """
        state_dict = torch.load(path, map_location=self.device)

        # Si c'est un checkpoint complet
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from: {path}")

    def extract_sliding_cubes(
        self,
        volume: np.ndarray
    ) -> Tuple[list, list]:
        """
        Extrait des cubes chevauchants d'un volume.

        Parameters
        ----------
        volume : np.ndarray
            Volume 3D preprocessé

        Returns
        -------
        cubes : list
            Liste de cubes
        positions : list
            Liste de positions (x, y, z) des coins supérieurs gauches
        """
        D, H, W = volume.shape
        cubes = []
        positions = []

        for z in range(0, max(D - self.cube_size + 1, 1), self.stride):
            for y in range(0, max(H - self.cube_size + 1, 1), self.stride):
                for x in range(0, max(W - self.cube_size + 1, 1), self.stride):
                    cube = volume[z:z+self.cube_size,
                                 y:y+self.cube_size,
                                 x:x+self.cube_size]

                    # Pad si nécessaire
                    if cube.shape != (self.cube_size, self.cube_size, self.cube_size):
                        cube = self._pad_cube(cube)

                    cubes.append(cube)
                    positions.append((x, y, z))

        return cubes, positions

    def _pad_cube(self, cube: np.ndarray) -> np.ndarray:
        """Pad un cube à la taille cible."""
        pad_z = max(0, self.cube_size - cube.shape[0])
        pad_y = max(0, self.cube_size - cube.shape[1])
        pad_x = max(0, self.cube_size - cube.shape[2])

        padded = np.pad(
            cube,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode='constant',
            constant_values=0
        )

        return padded

    def predict_cube(
        self,
        cube: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, int]:
        """
        Prédit sur un cube unique.

        Parameters
        ----------
        cube : np.ndarray
            Cube 3D
        threshold : float, optional
            Seuil de décision, par défaut 0.5

        Returns
        -------
        prob : float
            Probabilité prédite
        label : int
            Label prédit (0 ou 1)
        """
        # Convertir en tensor
        cube_tensor = torch.tensor(cube, dtype=torch.float32)
        cube_tensor = cube_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(cube_tensor)

            # Si multi-output, prendre dernière colonne
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                prob = torch.sigmoid(logits[:, -1]).item()
            else:
                prob = torch.sigmoid(logits).item()

        label = 1 if prob > threshold else 0

        return prob, label

    def predict_volume(
        self,
        patient_path: str,
        threshold: float = 0.5,
        top_k: int = 5,
        aggregation: str = 'mean'
    ) -> Dict:
        """
        Prédit sur un volume complet.

        Parameters
        ----------
        patient_path : str
            Chemin vers les fichiers DICOM
        threshold : float, optional
            Seuil de décision, par défaut 0.5
        top_k : int, optional
            Nombre de cubes top à considérer, par défaut 5
        aggregation : str, optional
            Méthode d'agrégation ('mean', 'max', 'percentile'), par défaut 'mean'

        Returns
        -------
        dict
            Dictionnaire avec:
            - 'volume_prob': probabilité agrégée
            - 'volume_label': label prédit
            - 'n_cubes': nombre de cubes analysés
            - 'top_k_probs': probabilités des top-k cubes
        """
        # 1. Preprocessing
        volume = self.preprocessor.process_volume(patient_path)

        # 2. Extract cubes
        cubes, positions = self.extract_sliding_cubes(volume)

        # 3. Predict sur chaque cube
        all_probs = []

        for cube in tqdm(cubes, desc="Predicting cubes", disable=True):
            prob, _ = self.predict_cube(cube, threshold)
            all_probs.append(prob)

        all_probs = np.array(all_probs)

        # 4. Agrégation
        k = min(top_k, len(all_probs))
        top_k_idx = np.argsort(all_probs)[-k:]
        top_k_probs = all_probs[top_k_idx]

        if aggregation == 'max':
            volume_prob = np.max(top_k_probs)
        elif aggregation == 'percentile':
            volume_prob = np.percentile(top_k_probs, 90)
        else:  # mean
            volume_prob = np.mean(top_k_probs)

        volume_label = 1 if volume_prob > threshold else 0

        return {
            'volume_prob': volume_prob,
            'volume_label': volume_label,
            'n_cubes': len(cubes),
            'top_k_probs': top_k_probs.tolist()
        }

    def predict_batch(
        self,
        patient_paths: list,
        threshold: float = 0.5
    ) -> list:
        """
        Prédit sur un batch de volumes.

        Parameters
        ----------
        patient_paths : list
            Liste de chemins vers volumes
        threshold : float, optional
            Seuil de décision

        Returns
        -------
        list
            Liste de prédictions
        """
        predictions = []

        for patient_path in tqdm(patient_paths, desc="Predicting volumes"):
            pred = self.predict_volume(patient_path, threshold)
            predictions.append(pred)

        return predictions

    def __repr__(self) -> str:
        return (
            f"Predictor("
            f"device={self.device}, "
            f"cube_size={self.cube_size}, "
            f"stride={self.stride})"
        )
