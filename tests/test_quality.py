"""Tests for blur detection and frame quality scoring."""

import cv2
import numpy as np
import pytest

from frameforge.models import Frame
from frameforge.quality import blur_score, clear_frames, score_frame


def sharp_image():
    """A frame with hard edges: a checkerboard."""
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(0, 240, 20):
        for x in range(0, 320, 20):
            if (x // 20 + y // 20) % 2 == 0:
                image[y : y + 20, x : x + 20] = 255
    return image


def test_blur_score_is_higher_for_a_sharp_frame():
    sharp = sharp_image()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert blur_score(sharp) > blur_score(blurred)
    assert blur_score(blurred) < blur_score(sharp) * 0.2


def test_score_frame_reports_sharpness():
    frame = Frame(image=sharp_image(), index=0, timestamp=0.0)
    scores = score_frame(frame)
    assert scores.sharpness == pytest.approx(blur_score(sharp_image()))
    assert scores.is_clear


def test_clear_filtering_keeps_sharp_and_drops_blurred():
    sharp = Frame(image=sharp_image(), index=0, timestamp=0.0)
    blurred = Frame(
        image=cv2.GaussianBlur(sharp_image(), (21, 21), 0),
        index=1,
        timestamp=0.1,
    )
    kept = clear_frames([sharp, blurred], min_sharpness=100.0)
    assert [f.index for f in kept] == [0]
    assert kept[0].scores["sharpness"] > 100.0
