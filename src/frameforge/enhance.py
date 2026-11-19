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


def low_light(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Brighten a dark night frame with CLAHE on the lightness.

    Works on the L channel of LAB colour space, so colours stay
    natural while dark areas gain contrast. A dark frame comes
    out much brighter; a normal frame changes only a little.
    """
    if clip_limit <= 0:
        raise ValueError("Clip limit must be positive")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correct(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Brighten (gamma > 1) or darken (gamma < 1) a frame."""
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)
