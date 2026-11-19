"""Tests for digital zoom in, zoom out and panning."""

import numpy as np
import pytest

from frameforge.zoom import pan, pan_path, zoom_centre, zoom_in, zoom_levels, zoom_out


def test_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[100:140, 100:180] = 255  # a white block off-centre
    return image


def test_zoom_in_crops_to_the_region():
    zoomed = zoom_in(test_image(), 100, 100, 80, 40)
    assert zoomed.shape == (40, 80, 3)
    assert zoomed.mean() == pytest.approx(255.0)


def test_zoom_in_enlarges_back_to_full_size():
    zoomed = zoom_in(test_image(), 100, 100, 80, 40, output_size=(320, 240))
    assert zoomed.shape == (240, 320, 3)


def test_zoom_centre_keeps_full_frame_size():
    zoomed = zoom_centre(test_image(), 2.0)
    assert zoomed.shape == test_image().shape


def test_zoom_centre_rejects_factor_of_one_or_less():
    with pytest.raises(ValueError):
        zoom_centre(test_image(), 1.0)


def test_zoom_out_returns_the_full_frame():
    zoomed = zoom_out(test_image(), 1.0)
    assert zoomed.shape == test_image().shape
    assert np.array_equal(zoomed, test_image())


def test_zoom_out_shrinks_the_frame():
    zoomed = zoom_out(test_image(), 0.5)
    assert zoomed.shape == (120, 160, 3)


def test_zoom_levels_cover_full_and_zoomed_views():
    views = zoom_levels(test_image())
    assert set(views) == {1.0, 1.5, 2.0, 4.0}
    assert views[1.0].shape == test_image().shape
    assert views[4.0].shape == test_image().shape
    assert not np.array_equal(views[4.0], test_image())


def test_pan_moves_the_zoom_window():
    left = pan(test_image(), 2.0, -60, 0)
    right = pan(test_image(), 2.0, 60, 0)
    assert left.shape == test_image().shape
    assert right.shape == test_image().shape
    assert not np.array_equal(left, right)


def test_pan_clamps_at_the_frame_edge():
    panned = pan(test_image(), 2.0, 10000, 10000)
    assert panned.shape == test_image().shape


def test_pan_path_sweeps_left_to_right():
    views = pan_path(test_image(), 2.0, steps=3)
    assert len(views) == 3
    assert all(v.shape == test_image().shape for v in views)
    assert not np.array_equal(views[0], views[2])
