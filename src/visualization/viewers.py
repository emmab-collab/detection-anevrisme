"""
3D volume visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_middle_slices(volume):
    """
    Display the middle slices of a 3D volume in three orthogonal planes.

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape (X, Y, Z)

    Notes
    -----
    Displays three views:
    - Axial (XY plane at mid Z)
    - Coronal (XZ plane at mid Y)
    - Sagittal (YZ plane at mid X)

    Examples
    --------
    >>> volume = np.random.rand(128, 128, 64)
    >>> show_middle_slices(volume)
    """
    mid_x = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_z = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial slice (XY plane at depth z)
    axes[0].imshow(volume[:, :, mid_z].T, cmap='gray')
    axes[0].set_title(f'Axiale (z={mid_z})')
    axes[0].axis('on')

    # Coronal slice (XZ plane at y coordinate)
    axes[1].imshow(volume[:, mid_y, :].T, cmap='gray')
    axes[1].set_title(f'Coronale (y={mid_y})')
    axes[1].axis('on')

    # Sagittal slice (YZ plane at x coordinate)
    axes[2].imshow(volume[mid_x, :, :].T, cmap='gray')
    axes[2].set_title(f'Sagittale (x={mid_x})')
    axes[2].axis('on')

    plt.tight_layout()
    plt.show()


def show_slice_with_point(volume, coord, plane="axial"):
    """
    Display a slice with a marked point of interest.

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape (X, Y, Z)
    coord : array-like
        Coordinates [x, y, z] of the point to mark
    plane : str, optional
        Which plane to display: 'axial', 'coronal', or 'sagittal' (default: 'axial')

    Raises
    ------
    ValueError
        If plane is not one of 'axial', 'coronal', or 'sagittal'

    Notes
    -----
    Shows two subplots side by side:
    - Left: the slice without markers
    - Right: the same slice with a red cross marking the point

    Examples
    --------
    >>> volume = np.random.rand(128, 128, 64)
    >>> coords = np.array([64, 64, 32])
    >>> show_slice_with_point(volume, coords, plane="axial")
    """
    x, y, z = coord.astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    if plane == "axial":       # XY plane at depth z
        img = volume[:, :, z].T
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        ax[1].scatter(x, y, c="r", s=40, marker="x")
        title = f"Axial z={z}"

    elif plane == "sagittal":  # YZ plane at x coordinate
        img = volume[x, :, :].T
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        ax[1].scatter(y, z, c="r", s=40, marker="x")
        title = f"Sagittal x={x}"

    elif plane == "coronal":   # XZ plane at y coordinate
        img = volume[:, y, :].T
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        ax[1].scatter(x, z, c="r", s=40, marker="x")
        title = f"Coronal y={y}"

    else:
        raise ValueError("plane doit être 'axial', 'sagittal' ou 'coronal'")

    for a in ax:
        a.set_title(title)
        a.axis("on")

    plt.tight_layout()
    plt.show()
