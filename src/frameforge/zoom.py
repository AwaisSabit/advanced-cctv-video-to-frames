"""Digital zoom: zoom in on a region, zoom out, and pan."""

import cv2
import numpy as np


def clamp_region(x, y, w, h, width, height):
    """Keep a region inside the frame, at least 1 pixel wide."""
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def zoom_in(image: np.ndarray, x: int, y: int, w: int, h: int,
            output_size=None) -> np.ndarray:
    """Crop a region of interest and return it.

    With ``output_size`` (width, height) the cropped region is
    enlarged back to that size; without it, the crop is returned
    at its own size. Digital zoom cannot add detail that is not
    in the pixels.
    """
    height, width = image.shape[:2]
    x, y, w, h = clamp_region(x, y, w, h, width, height)
    cropped = image[y : y + h, x : x + w]
    if output_size is not None:
        return cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)
    return cropped


def zoom_centre(image: np.ndarray, factor: float) -> np.ndarray:
    """Zoom in on the centre of the frame by a factor (2 = half size).

    The result is enlarged back to the full frame size.
    """
    if factor <= 1:
        raise ValueError("Zoom factor must be more than 1")
    height, width = image.shape[:2]
    w, h = int(width / factor), int(height / factor)
    x, y = (width - w) // 2, (height - h) // 2
    return zoom_in(image, x, y, w, h, output_size=(width, height))
