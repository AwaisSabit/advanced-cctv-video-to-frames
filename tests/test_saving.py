"""Tests for frame saving in PNG/JPG with metadata JSON."""

import json

import cv2
import numpy as np
import pytest

from frameforge.models import Frame
from frameforge.saving import make_contact_sheet, save_frame, save_frames, write_metadata


def a_frame(index=0):
    image = np.full((100, 160, 3), 128, dtype=np.uint8)
    return Frame(image=image, index=index, timestamp=index * 0.1)


def test_save_frame_writes_png_and_jpg(tmp_path):
    png = save_frame(a_frame(), tmp_path / "frame.png")
    jpg = save_frame(a_frame(), tmp_path / "frame.jpg")
    assert png.exists() and jpg.exists()
    assert cv2.imread(str(png)).shape == (100, 160, 3)
    assert cv2.imread(str(jpg)).shape == (100, 160, 3)


def test_save_frame_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save_frame(a_frame(), tmp_path / "frame.bmp")


def test_save_frames_writes_numbered_files(tmp_path):
    paths = save_frames([a_frame(0), a_frame(5)], tmp_path, fmt="png")
    assert [p.name for p in paths] == ["frame_000000.png", "frame_000005.png"]
    assert all(p.exists() for p in paths)


def test_write_metadata_json(tmp_path):
    frame = a_frame(3)
    frame.scores = {"sharpness": 120.5}
    path = write_metadata([frame], tmp_path / "metadata.json", video_path="cctv.mp4")
    data = json.loads(path.read_text())
    assert data["video"] == "cctv.mp4"
    assert data["frame_count"] == 1
    assert data["frames"][0]["index"] == 3
    assert data["frames"][0]["scores"]["sharpness"] == 120.5


def test_make_contact_sheet_lays_frames_in_a_grid(tmp_path):
    from PIL import Image

    frames = [a_frame(i) for i in range(5)]
    path = make_contact_sheet(frames, tmp_path / "sheet.png", columns=2)
    image = Image.open(path)
    assert image.size == (640, 3 * 260)  # 2 columns, 3 rows
    assert path.exists()


def test_make_contact_sheet_needs_frames(tmp_path):
    with pytest.raises(ValueError):
        make_contact_sheet([], tmp_path / "sheet.png")
