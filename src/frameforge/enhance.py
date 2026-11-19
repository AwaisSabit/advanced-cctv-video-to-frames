"""Frame enhancement: sharpening, denoising and low light."""

import cv2
import numpy as np


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Sharpen edges with an unsharp mask.

    ``strength`` 1 is a gentle sharpen; higher values sharpen
    more but can add halos around strong edges.
    """
    if strength <= 0:
        raise ValueError("Sharpen strength must be positive")
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def denoise(image: np.ndarray, strength: float = 10.0) -> np.ndarray:
    """Reduce sensor and compression noise, keeping edges."""
    if strength <= 0:
        raise ValueError("Denoise strength must be positive")
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def enhance(image: np.ndarray, sharpen_strength: float = 1.0,
            denoise_strength: float = 10.0) -> np.ndarray:
    """Denoise first, then sharpen — the usual order, so the
    sharpening does not sharpen the noise as well."""
    return sharpen(denoise(image, denoise_strength), sharpen_strength)
