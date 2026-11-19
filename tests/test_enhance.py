"""Tests for frame enhancement: sharpening, denoising, low light."""

import numpy as np
import pytest

from frameforge.enhance import denoise, enhance, sharpen


def noisy_image(seed=7):
    rng = np.random.default_rng(seed)
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    noise = rng.normal(0, 15, base.shape)
    return np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)


def edge_image():
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    image[:, 50:] = 200
    return image


def test_sharpen_changes_pixels_without_changing_size():
    image = edge_image()
    out = sharpen(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert not np.array_equal(out, image)


def test_denoise_reduces_noise_without_changing_size():
    image = noisy_image()
    out = denoise(image)
    assert out.shape == image.shape
    assert float(np.std(out)) < float(np.std(image))


def test_enhance_changes_pixels_without_changing_size():
    image = noisy_image()
    out = enhance(image)
    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_sharpen_rejects_zero_strength():
    with pytest.raises(ValueError):
        sharpen(edge_image(), 0)
