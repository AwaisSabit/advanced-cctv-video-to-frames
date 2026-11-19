"""Tests for video loading and metadata reading."""

import pytest

from frameforge.video import VideoError, VideoFile, read_metadata


def test_metadata_on_synthetic_video(sample_video):
    meta = read_metadata(sample_video)
    assert meta.frame_count == 30
    assert meta.fps == pytest.approx(10.0)
    assert meta.duration_seconds == pytest.approx(3.0)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.codec


def test_video_file_opens_and_reads_frames(sample_video):
    video = VideoFile(sample_video)
    assert len(video) == 30
    frame = video.frame_at_index(0)
    assert frame.image.shape == (240, 320, 3)
    assert frame.timestamp == pytest.approx(0.0)
    frame = video.frame_at_time(1.0)
    assert frame.index == 10
    assert frame.timestamp == pytest.approx(1.0)


def test_frame_at_bad_index_raises(sample_video):
    video = VideoFile(sample_video)
    with pytest.raises(VideoError):
        video.frame_at_index(30)
    with pytest.raises(VideoError):
        video.frame_at_time(99.0)


def test_missing_video_raises(tmp_path):
    with pytest.raises(VideoError):
        read_metadata(tmp_path / "no-such-video.avi")


def test_iter_frames_yields_all(sample_video):
    video = VideoFile(sample_video)
    frames = list(video.iter_frames())
    assert len(frames) == 30
    assert [f.index for f in frames] == list(range(30))
