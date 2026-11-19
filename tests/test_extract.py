"""Tests for extraction by interval."""

import pytest

from frameforge.extract import extract_by_interval
from frameforge.video import VideoFile


def test_extract_every_second_returns_right_count(sample_video):
    video = VideoFile(sample_video)  # 30 frames at 10 fps
    frames = extract_by_interval(video, every_seconds=1.0)
    assert [f.index for f in frames] == [0, 10, 20]


def test_extract_every_nth_frame(sample_video):
    video = VideoFile(sample_video)
    frames = extract_by_interval(video, every_nth=15)
    assert [f.index for f in frames] == [0, 15]


def test_extract_interval_needs_one_option(sample_video):
    video = VideoFile(sample_video)
    with pytest.raises(ValueError):
        extract_by_interval(video)
    with pytest.raises(ValueError):
        extract_by_interval(video, every_seconds=1, every_nth=5)
    with pytest.raises(ValueError):
        extract_by_interval(video, every_seconds=0)
