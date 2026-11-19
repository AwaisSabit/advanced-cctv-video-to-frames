"""Shared helpers: build small synthetic videos with OpenCV."""

import cv2
import numpy as np
import pytest


def make_video(path, frames, fps=10, size=(320, 240)):
    """Write a list of BGR numpy frames to an .avi file with OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened(), f"VideoWriter could not open {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def plain_frame(size=(320, 240), colour=(40, 40, 40)):
    """A flat grey frame, width x height in (width, height) order."""
    width, height = size
    return np.full((height, width, 3), colour, dtype=np.uint8)


def moving_rectangle_video(path, count=30, fps=10, size=(320, 240)):
    """A white rectangle sliding across a dark background."""
    frames = []
    for i in range(count):
        frame = plain_frame(size)
        x = 10 + i * 8
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)
        frames.append(frame)
    return make_video(path, frames, fps=fps, size=size)


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.avi"
    moving_rectangle_video(path, count=30, fps=10)
    return path
