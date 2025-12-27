"""
Bricks - Pipeline Components

Ce module contient les composants de pipeline réutilisables pour le projet
de détection d'anévrismes.

Classes disponibles
-------------------
Preprocessor
    Pipeline de preprocessing pour volumes DICOM
DatasetBuilder
    Construction de datasets d'entraînement
Augmentor
    Augmentation de données
EDA
    Analyse exploratoire des données
Trainer
    Entraînement de modèles
Predictor
    Inférence sur nouvelles données
"""

from .augmentation import Augmentor
from .dataset import DatasetBuilder
from .eda import EDA
from .inference import Predictor
from .preprocessing import Preprocessor
from .training import Trainer

__all__ = [
    "Preprocessor",
    "DatasetBuilder",
    "Augmentor",
    "EDA",
    "Trainer",
    "Predictor",
]
