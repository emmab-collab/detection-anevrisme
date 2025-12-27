"""
Elastic deformation augmentation for 3D medical volumes.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from ..config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_DISPLACEMENT,
    DEFAULT_N_AUGMENTATIONS,
)


def random_deformation(volume, grid_size=None, max_displacement=None):
    """
    Apply random elastic deformation to a 3D volume.

    This function creates a smooth deformation field using a sparse grid of
    control points and applies it to the volume using spline interpolation.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume with shape (D, H, W)
    grid_size : int, optional
        Number of control points along each dimension (default: from config)
        Smaller values create smoother deformations
    max_displacement : float, optional
        Maximum displacement in voxels for control points (default: from config)

    Returns
    -------
    np.ndarray
        Deformed volume with same shape as input

    Notes
    -----
    - Uses cubic spline interpolation for smooth deformations
    - Reflects at boundaries to avoid edge artifacts
    - Computational cost increases with grid_size^3

    Examples
    --------
    >>> volume = np.random.rand(48, 48, 48)
    >>> deformed = random_deformation(volume, grid_size=3, max_displacement=3)
    """
    if grid_size is None:
        grid_size = DEFAULT_GRID_SIZE
    if max_displacement is None:
        max_displacement = DEFAULT_MAX_DISPLACEMENT

    shape = volume.shape
    assert len(shape) == 3, "Le volume doit être 3D"

    # Original volume coordinates
    dz, dy, dx = shape
    z, y, x = np.meshgrid(
        np.linspace(0, dz - 1, shape[0]),
        np.linspace(0, dy - 1, shape[1]),
        np.linspace(0, dx - 1, shape[2]),
        indexing="ij",
    )

    # Generate control point grid
    grid_z = np.linspace(0, dz - 1, grid_size)
    grid_y = np.linspace(0, dy - 1, grid_size)
    grid_x = np.linspace(0, dx - 1, grid_size)

    # Create random displacement field at control points
    displacement = [
        np.random.uniform(
            -max_displacement, max_displacement, size=(grid_size, grid_size, grid_size)
        )
        for _ in range(3)
    ]

    # Interpolate displacement field to full volume using cubic splines
    disp_interp = [
        RegularGridInterpolator(
            (grid_z, grid_y, grid_x), d, bounds_error=False, fill_value=0
        )
        for d in displacement
    ]

    # Compute deformed coordinates
    coords = np.array([z, y, x])
    points = np.array([z.flatten(), y.flatten(), x.flatten()]).T

    dz_new = disp_interp[0](points).reshape(shape)
    dy_new = disp_interp[1](points).reshape(shape)
    dx_new = disp_interp[2](points).reshape(shape)

    z_new = z + dz_new
    y_new = y + dy_new
    x_new = x + dx_new

    # Apply deformation to volume
    deformed = map_coordinates(volume, [z_new, y_new, x_new], order=3, mode="reflect")

    return deformed


def data_augmentation(volume, n_aug=None):
    """
    Generate multiple augmented versions of a volume.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume
    n_aug : int, optional
        Number of augmented versions to generate (default: from config)

    Returns
    -------
    np.ndarray
        Stacked augmented volumes with shape (n_aug, *volume.shape)

    Examples
    --------
    >>> volume = np.random.rand(48, 48, 48)
    >>> augmented = data_augmentation(volume, n_aug=12)
    >>> augmented.shape
    (12, 48, 48, 48)
    """
    if n_aug is None:
        n_aug = DEFAULT_N_AUGMENTATIONS

    augmented_volumes = []

    for i in range(n_aug):
        vol_def = random_deformation(volume)
        augmented_volumes.append(vol_def)

    augmented_volumes = np.stack(augmented_volumes)

    return augmented_volumes


def dataset_augmented(liste_volumes, n_aneurysm):
    """
    Create an augmented dataset from a list of volumes.

    Parameters
    ----------
    liste_volumes : list of np.ndarray
        List of input volumes to augment
    n_aneurysm : int
        Number of augmented versions per volume

    Returns
    -------
    np.ndarray
        Concatenated augmented dataset with shape (len(liste_volumes) * n_aneurysm, *volume.shape)

    Examples
    --------
    >>> volumes = [np.random.rand(48, 48, 48) for _ in range(10)]
    >>> dataset = dataset_augmented(volumes, n_aneurysm=12)
    >>> dataset.shape
    (120, 48, 48, 48)
    """
    liste_cubes = []

    for i in tqdm(range(len(liste_volumes)), desc="Augmenting volumes"):
        volume = liste_volumes[i]

        # Generate augmentations
        aug = data_augmentation(volume, n_aneurysm)

        liste_cubes.append(aug)

    # Concatenate all results
    liste_cubes = np.concatenate(liste_cubes, axis=0)

    return liste_cubes
