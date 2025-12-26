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
model
    Deep learning models (to be implemented)
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

__version__ = "0.1.0"

__all__ = [
    # Config
    'TARGET_SPACING',
    'CROP_THRESHOLD',
    'DEFAULT_GRID_SIZE',
    'DEFAULT_MAX_DISPLACEMENT',
    'DEFAULT_N_AUGMENTATIONS',
    'ANEURYSM_POSITIONS',

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
]
