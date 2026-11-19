"""Timestamp and label overlays drawn on frames."""

import cv2
import numpy as np


def _format_time(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def add_timestamp(image: np.ndarray, seconds: float,
                  position=(10, 30)) -> np.ndarray:
    """Draw a HH:MM:SS timestamp in the corner of a frame."""
    out = image.copy()
    cv2.putText(
        out, _format_time(seconds), position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def add_label(image: np.ndarray, text: str, position=(10, 60)) -> np.ndarray:
    """Draw a text label on a frame, with a dark bar behind it."""
    out = image.copy()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = position
    cv2.rectangle(out, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(
        out, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def add_overlay(image: np.ndarray, seconds: float, label: str = "") -> np.ndarray:
    """Timestamp plus an optional label, the standard CCTV look."""
    out = add_timestamp(image, seconds)
    if label:
        out = add_label(out, label)
    return out
