"""
Configuration constants for aneurysm detection project.
"""

# Preprocessing constants
TARGET_SPACING = (0.4, 0.4, 0.4)  # mm per voxel (dx, dy, dz)
CROP_THRESHOLD = 0.1  # Percentage of max intensity for cropping mask

# Data augmentation constants
DEFAULT_GRID_SIZE = 3
DEFAULT_MAX_DISPLACEMENT = 3
DEFAULT_N_AUGMENTATIONS = 12

# Aneurysm positions
ANEURYSM_POSITIONS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]
