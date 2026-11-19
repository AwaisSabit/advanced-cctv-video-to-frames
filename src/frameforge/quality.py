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


def score_frame(frame: Frame) -> QualityScores:
    """Score one frame for sharpness (brightness and contrast
    arrive in a later step)."""
    return QualityScores(
        sharpness=blur_score(frame.image), brightness=0.0, contrast=0.0
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
