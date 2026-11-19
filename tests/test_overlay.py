"""Tests for timestamp and label overlays."""

import numpy as np

from frameforge.overlay import add_label, add_overlay, add_timestamp


def test_timestamp_overlay_draws_text():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    out = add_timestamp(image, 3723.0)  # 01:02:03
    assert out.shape == image.shape
    assert out.sum() > 0  # white text pixels were drawn


def test_label_overlay_draws_a_dark_bar_and_text():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    out = add_label(image, "Gate 2")
    assert out.shape == image.shape
    assert out.sum() > 0


def test_overlay_with_label_draws_both():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    out = add_overlay(image, 65.0, label="Gate 2")
    assert out.sum() > add_timestamp(image, 65.0).sum()
