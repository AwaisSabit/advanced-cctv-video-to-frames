"""End to end tests: extraction, quality, zoom and enhancement
together, on synthetic videos made with OpenCV."""

import json

import cv2
import numpy as np
import pytest

from frameforge.cli import main
from frameforge.enhance import enhance, low_light
from frameforge.extract import (
    detect_motion,
    detect_scene_changes,
    extract_at_timestamps,
    extract_by_interval,
    extract_by_range,
)
from frameforge.quality import blur_score, clear_frames
from frameforge.saving import save_frame, write_metadata
from frameforge.video import VideoFile
from frameforge.zoom import zoom_centre, zoom_in, zoom_out
from tests.conftest import make_video, plain_frame
from tests.test_quality import sharp_image


def test_full_pipeline_on_a_synthetic_video(tmp_path, sample_video):
    video = VideoFile(sample_video)
    assert video.metadata.frame_count == 30
    frames = extract_by_interval(video, every_seconds=1.0)
    assert len(frames) == 3
    ranged = extract_by_range(video, 1.0, 2.0)
    assert len(ranged) == 11
    at = extract_at_timestamps(video, [0.5, 2.5])
    assert [f.index for f in at] == [5, 25]


def test_scene_cut_and_motion_on_synthetic_videos(tmp_path):
    dark = [plain_frame(colour=(30, 30, 30)) for _ in range(10)]
    bright = [plain_frame(colour=(220, 220, 220)) for _ in range(10)]
    cut = VideoFile(make_video(tmp_path / "cut.avi", dark + bright, fps=10))
    assert [f.index for f in detect_scene_changes(cut)] == [10]
    still = VideoFile(make_video(tmp_path / "still.avi",
                                 [plain_frame() for _ in range(10)], fps=10))
    assert detect_motion(still) == []


def test_quality_zoom_and_enhancement_together(sample_video):
    sharp = sharp_image()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert blur_score(sharp) > blur_score(blurred)
    from frameforge.models import Frame

    frames = [
        Frame(image=sharp, index=0, timestamp=0.0),
        Frame(image=blurred, index=1, timestamp=0.1),
    ]
    assert [f.index for f in clear_frames(frames, min_sharpness=100.0)] == [0]
    cropped = zoom_in(sharp, 0, 0, 160, 120)
    assert cropped.shape == (120, 160, 3)
    assert zoom_out(sharp).shape == sharp.shape
    assert zoom_centre(sharp, 2.0).shape == sharp.shape
    enhanced = enhance(sharp)
    assert enhanced.shape == sharp.shape
    assert not np.array_equal(enhanced, sharp)
    assert low_light(np.full((50, 50, 3), 20, dtype=np.uint8)).mean() > 20


def test_saving_and_cli_together(tmp_path, sample_video):
    video = VideoFile(sample_video)
    frame = video.frame_at_index(5)
    png = save_frame(frame, tmp_path / "frame.png")
    jpg = save_frame(frame, tmp_path / "frame.jpg")
    assert png.exists() and jpg.exists()
    meta = write_metadata([frame], tmp_path / "metadata.json",
                          video_path=str(sample_video))
    assert json.loads(meta.read_text())["frame_count"] == 1
    assert main(["info", str(sample_video)]) == 0
    assert main(["extract", str(sample_video), "--every-nth", "10",
                 "--output-dir", str(tmp_path / "frames")]) == 0
    assert main(["clear", str(sample_video), "--every-seconds", "1",
                 "--min-sharpness", "10000",
                 "--output-dir", str(tmp_path / "clear")]) == 0
    out = tmp_path / "zoom.png"
    assert main(["zoom", str(sample_video), "--at", "1.0", "--zoom", "2",
                 "--output", str(out)]) == 0
    assert cv2.imread(str(out)) is not None
