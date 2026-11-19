"""Frame quality scoring: blur, brightness, contrast, clear frames."""

import cv2
import numpy as np

from frameforge.models import Frame, QualityScores


def blur_score(image: np.ndarray) -> float:
    """Sharpness score: variance of the Laplacian.

    A sharp frame has strong edges and a high score; a blurred
    frame has soft edges and a low score.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def brightness_score(image: np.ndarray) -> float:
    """Mean grey level, 0 (black) to 255 (white)."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grey))


def contrast_score(image: np.ndarray) -> float:
    """Standard deviation of grey levels: a flat frame scores low."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(grey))


def score_frame(frame: Frame) -> QualityScores:
    """Score one frame for sharpness, brightness and contrast."""
    return QualityScores(
        sharpness=blur_score(frame.image),
        brightness=brightness_score(frame.image),
        contrast=contrast_score(frame.image),
    )


def clear_frames(frames, min_sharpness=100.0):
    """Keep only the sharp frames, dropping blurred ones.

    Each kept frame gets its scores attached in ``frame.scores``.
    """
    kept = []
    for frame in frames:
        scores = score_frame(frame)
        frame.scores = {
            "sharpness": scores.sharpness,
            "brightness": scores.brightness,
            "contrast": scores.contrast,
        }
        if scores.sharpness >= min_sharpness:
            kept.append(frame)
    return kept
