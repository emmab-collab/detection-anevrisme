"""
Aneurysm Detection Package

A comprehensive package for medical image processing, focusing on
aneurysm detection from DICOM images.

Modules
-------
config
    Configuration constants
data
    DICOM loading and metadata utilities
preprocessing
    Volume preprocessing and coordinate transformations
augmentation
    Data augmentation techniques
visualization
    Volume visualization tools
utils
    General utility functions
bricks
    Pipeline components (Preprocessor, DatasetBuilder, Trainer, etc.)
models
    Deep learning model architectures
"""

# Import augmentation utilities
from .augmentation import data_augmentation, dataset_augmented, random_deformation

# Import bricks (pipeline components)
from .bricks import EDA, Augmentor, DatasetBuilder, Predictor, Preprocessor, Trainer

# Import main configuration
from .config import (
    ANEURYSM_POSITIONS,
    CROP_THRESHOLD,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_DISPLACEMENT,
    DEFAULT_N_AUGMENTATIONS,
    TARGET_SPACING,
)

# Import data utilities
from .data import (
    ajouter_Modality,
    coordonnee_z,
    dicom_to_numpy,
    get_instance_number,
    get_patient_ID,
    get_position,
)

# Import models
from .models import ConvBlock3D, UNet3DClassifier

# Import paths configuration
from .paths import (
    CHECKPOINTS_DIR,
    IS_KAGGLE,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    SERIES_DIR,
    TRAIN_CSV,
    TRAIN_LOCALIZERS_CSV,
    print_config,
)

# Import preprocessing utilities
from .preprocessing import (
    crop,
    get_center,
    normalization,
    preprocessing_volume,
    preprocessing_volume_and_coords,
    resample,
    resample_coordonnees,
)

# Import general utilities
from .utils import get_pixelspacing

# Import visualization utilities
from .visualization import show_middle_slices, show_slice_with_point

__version__ = "0.2.0"

__all__ = [
    # Config
    "TARGET_SPACING",
    "CROP_THRESHOLD",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_MAX_DISPLACEMENT",
    "DEFAULT_N_AUGMENTATIONS",
    "ANEURYSM_POSITIONS",
    # Paths
    "SERIES_DIR",
    "TRAIN_CSV",
    "TRAIN_LOCALIZERS_CSV",
    "OUTPUT_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "CHECKPOINTS_DIR",
    "IS_KAGGLE",
    "print_config",
    # Data
    "dicom_to_numpy",
    "get_instance_number",
    "coordonnee_z",
    "get_patient_ID",
    "get_position",
    "ajouter_Modality",
    # Preprocessing
    "resample",
    "crop",
    "normalization",
    "get_center",
    "resample_coordonnees",
    "preprocessing_volume_and_coords",
    "preprocessing_volume",
    # Augmentation
    "random_deformation",
    "data_augmentation",
    "dataset_augmented",
    # Visualization
    "show_middle_slices",
    "show_slice_with_point",
    # Utils
    "get_pixelspacing",
    # Bricks (Pipeline Components)
    "Preprocessor",
    "DatasetBuilder",
    "Augmentor",
    "EDA",
    "Trainer",
    "Predictor",
    # Models
    "UNet3DClassifier",
    "ConvBlock3D",
]
