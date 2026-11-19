"""Tests for extraction by interval."""

import pytest

from frameforge.extract import (
    detect_motion,
    detect_scene_changes,
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


def test_scene_change_detection_finds_a_cut(tmp_path):
    from tests.conftest import make_video, plain_frame

    dark = [plain_frame(colour=(30, 30, 30)) for _ in range(10)]
    bright = [plain_frame(colour=(220, 220, 220)) for _ in range(10)]
    path = make_video(tmp_path / "cut.avi", dark + bright, fps=10)
    video = VideoFile(path)
    cuts = detect_scene_changes(video)
    assert [f.index for f in cuts] == [10]


def test_scene_change_detection_ignores_a_steady_video(sample_video):
    video = VideoFile(sample_video)
    cuts = detect_scene_changes(video, threshold=0.9)
    assert cuts == []


def test_motion_detection_finds_moving_frames(sample_video):
    video = VideoFile(sample_video)  # rectangle moves every frame
    moving = detect_motion(video)
    assert len(moving) > 0
    assert moving[0].index == 1


def test_motion_detection_skips_a_still_video(tmp_path):
    from tests.conftest import make_video, plain_frame

    still = [plain_frame() for _ in range(10)]
    path = make_video(tmp_path / "still.avi", still, fps=10)
    video = VideoFile(path)
    assert detect_motion(video) == []
