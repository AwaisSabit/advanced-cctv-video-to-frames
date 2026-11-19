"""Tests for extraction by interval."""

import pytest

from frameforge.extract import (
    extract_at_timestamps,
    extract_by_interval,
    extract_by_range,
)
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


def test_extract_by_range_returns_frames_in_range(sample_video):
    video = VideoFile(sample_video)  # 30 frames at 10 fps, 3 seconds
    frames = extract_by_range(video, 1.0, 2.0)
    assert [f.index for f in frames] == list(range(10, 21))


def test_extract_by_range_with_interval(sample_video):
    video = VideoFile(sample_video)
    frames = extract_by_range(video, 0.0, 2.0, every_seconds=1.0)
    assert [f.index for f in frames] == [0, 10, 20]


def test_extract_by_range_rejects_bad_range(sample_video):
    video = VideoFile(sample_video)
    with pytest.raises(ValueError):
        extract_by_range(video, 2.0, 1.0)
    with pytest.raises(ValueError):
        extract_by_range(video, 0.0, 99.0)


def test_extract_at_timestamps(sample_video):
    video = VideoFile(sample_video)
    frames = extract_at_timestamps(video, [0.0, 1.5, 2.9])
    assert [f.index for f in frames] == [0, 15, 29]
    assert frames[1].timestamp == pytest.approx(1.5)
