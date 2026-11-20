"""Shared HW2 helpers: pyramid construction, one LK step and image warping."""
import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import griddata


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])


X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])


Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Return the original image followed by num_levels filtered decimations."""
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale image")
    if not isinstance(num_levels, (int, np.integer)) or num_levels < 0:
        raise ValueError("num_levels must be a non-negative integer")

    pyramid = [image.copy()]
    for _ in range(num_levels):
        filtered = signal.convolve2d(
            pyramid[-1], PYRAMID_FILTER, boundary='symm', mode='same')
        # Keep the even-indexed rows and columns, including for odd sizes.
        pyramid.append(filtered[::2, ::2].copy())
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one LK increment; du is horizontal and dv is vertical.

    Use the assignment's derivative filters on I2 without rescaling them.
    Each complete window supplies A @ [du, dv] = -(I2 - I1).
    Border pixels and windows without two independent gradient directions
    retain zero flow. Inputs are never modified.
    """
    if I1.ndim != 2 or I2.ndim != 2 or I1.shape != I2.shape:
        raise ValueError("I1 and I2 must be grayscale images of the same shape")
    if (not isinstance(window_size, (int, np.integer)) or
            window_size < 1 or window_size % 2 == 0):
        raise ValueError("window_size must be a positive odd integer")

    # Cast before subtraction: uint8 arithmetic would wrap negative values.
    first = I1.astype(np.float64)
    second = I2.astype(np.float64)
    Ix = signal.convolve2d(second, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(second, Y_DERIVATIVE_FILTER, mode='same')
    It = second - first
    du = np.zeros(first.shape, dtype=np.float64)
    dv = np.zeros(first.shape, dtype=np.float64)
    half = window_size // 2
    height, width = first.shape

    for row in range(half, height - half):
        for col in range(half, width - half):
            region = (slice(row - half, row + half + 1),
                      slice(col - half, col + half + 1))
            A = np.column_stack((Ix[region].ravel(), Iy[region].ravel()))
            b = -It[region].ravel()
            try:
                # Solve directly, avoiding an explicit inverse of A.T @ A.
                shift, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank == 2 and np.all(np.isfinite(shift)):
                du[row, col], dv[row, col] = shift
            # A zero/undefined result at one pixel must not skip the row.
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Back-project by sampling image at (column + u, row + v).

    Flow values are measured in pixels of their own grid. Resize and scale
    each component when its resolution differs from the image resolution.
    Out-of-bounds interpolation samples use the original image at that pixel,
    as specified in the assignment.
    """
    if image.ndim != 2 or u.ndim != 2 or v.ndim != 2:
        raise ValueError("image, u and v must be two-dimensional arrays")
    if u.size == 0 or v.size == 0:
        raise ValueError("flow arrays must not be empty")

    height, width = image.shape
    horizontal = u.astype(np.float64)
    vertical = v.astype(np.float64)
    if horizontal.shape != image.shape:
        scale_x = width / horizontal.shape[1]
        horizontal = cv2.resize(horizontal, (width, height)) * scale_x
    if vertical.shape != image.shape:
        scale_y = height / vertical.shape[0]
        vertical = cv2.resize(vertical, (width, height)) * scale_y

    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    points = np.column_stack((columns.ravel(), rows.ravel()))
    samples = np.column_stack(((columns + horizontal).ravel(),
                               (rows + vertical).ravel()))
    image_warp = griddata(
        points, image.astype(np.float64).ravel(), samples,
        method='linear', fill_value=np.nan).reshape(image.shape)
    holes = np.isnan(image_warp)
    image_warp[holes] = image[holes]
    return image_warp
