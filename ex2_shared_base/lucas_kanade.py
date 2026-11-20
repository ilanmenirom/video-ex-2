# Shared starting point for HW2 2026/2027: only the three basic functions are implemented.
# Adapted from the shared 2020 solution; see README.md for provenance and changes.

import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


# FILL IN YOUR ID
ID1 = 123456789
ID2 = 987654321


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


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



def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels)) X M * (2^(num_levels)).
        Where: K is the ceil(h / (2^(num_levels)),
        and M is ceil(h / (2^(num_levels)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels)) X M * (2^(num_levels)).
        Where: K is the ceil(h / (2^(num_levels)),
        and M is ceil(h / (2^(num_levels)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    pass


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    pass


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    pass


