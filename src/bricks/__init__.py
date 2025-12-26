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

from .preprocessing import Preprocessor
from .dataset import DatasetBuilder
from .augmentation import Augmentor
from .eda import EDA
from .training import Trainer
from .inference import Predictor

__all__ = [
    'Preprocessor',
    'DatasetBuilder',
    'Augmentor',
    'EDA',
    'Trainer',
    'Predictor',
]
