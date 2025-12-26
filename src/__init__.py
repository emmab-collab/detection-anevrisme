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

# Import main configuration
from .config import (
    TARGET_SPACING,
    CROP_THRESHOLD,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_DISPLACEMENT,
    DEFAULT_N_AUGMENTATIONS,
    ANEURYSM_POSITIONS
)

# Import paths configuration
from .paths import (
    SERIES_DIR,
    TRAIN_CSV,
    TRAIN_LOCALIZERS_CSV,
    OUTPUT_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    IS_KAGGLE,
    print_config
)

# Import data utilities
from .data import (
    dicom_to_numpy,
    get_instance_number,
    coordonnee_z,
    get_patient_ID,
    get_position,
    ajouter_Modality
)

# Import preprocessing utilities
from .preprocessing import (
    resample,
    crop,
    normalization,
    get_center,
    resample_coordonnees,
    preprocessing_volume_and_coords,
    preprocessing_volume
)

# Import augmentation utilities
from .augmentation import (
    random_deformation,
    data_augmentation,
    dataset_augmented
)

# Import visualization utilities
from .visualization import (
    show_middle_slices,
    show_slice_with_point
)

# Import general utilities
from .utils import get_pixelspacing

# Import bricks (pipeline components)
from .bricks import (
    Preprocessor,
    DatasetBuilder,
    Augmentor,
    EDA,
    Trainer,
    Predictor
)

# Import models
from .models import UNet3DClassifier, ConvBlock3D

__version__ = "0.2.0"

__all__ = [
    # Config
    'TARGET_SPACING',
    'CROP_THRESHOLD',
    'DEFAULT_GRID_SIZE',
    'DEFAULT_MAX_DISPLACEMENT',
    'DEFAULT_N_AUGMENTATIONS',
    'ANEURYSM_POSITIONS',

    # Paths
    'SERIES_DIR',
    'TRAIN_CSV',
    'TRAIN_LOCALIZERS_CSV',
    'OUTPUT_DIR',
    'PROCESSED_DIR',
    'MODELS_DIR',
    'CHECKPOINTS_DIR',
    'IS_KAGGLE',
    'print_config',

    # Data
    'dicom_to_numpy',
    'get_instance_number',
    'coordonnee_z',
    'get_patient_ID',
    'get_position',
    'ajouter_Modality',

    # Preprocessing
    'resample',
    'crop',
    'normalization',
    'get_center',
    'resample_coordonnees',
    'preprocessing_volume_and_coords',
    'preprocessing_volume',

    # Augmentation
    'random_deformation',
    'data_augmentation',
    'dataset_augmented',

    # Visualization
    'show_middle_slices',
    'show_slice_with_point',

    # Utils
    'get_pixelspacing',

    # Bricks (Pipeline Components)
    'Preprocessor',
    'DatasetBuilder',
    'Augmentor',
    'EDA',
    'Trainer',
    'Predictor',

    # Models
    'UNet3DClassifier',
    'ConvBlock3D',
]
