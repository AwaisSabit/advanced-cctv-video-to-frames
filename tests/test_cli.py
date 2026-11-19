"""Tests for the command line interface."""

from pathlib import Path

import cv2
import pytest

from frameforge.cli import main


def test_cli_info_exits_zero(sample_video, capsys):
    assert main(["info", str(sample_video)]) == 0
    out = capsys.readouterr().out
    assert "Frames:     30" in out
    assert "320 x 240" in out


def test_cli_extract_by_interval(tmp_path, sample_video):
    out = tmp_path / "frames"
    assert main(["extract", str(sample_video), "--every-seconds", "1",
                 "--output-dir", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == 3
    assert (out / "metadata.json").exists()


def test_cli_clear_keeps_only_sharp_frames(tmp_path, sample_video):
    out = tmp_path / "clear"
    assert main(["clear", str(sample_video), "--every-seconds", "1",
                 "--min-sharpness", "10000",
                 "--output-dir", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == 0


def test_cli_zoom_writes_a_zoomed_frame(tmp_path, sample_video):
    out = tmp_path / "zoomed.png"
    assert main(["zoom", str(sample_video), "--at", "1.0",
                 "--region", "100", "100", "80", "40",
                 "--output", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (240, 320, 3)


def test_cli_enhance_writes_an_enhanced_frame(tmp_path, sample_video):
    out = tmp_path / "enhanced.png"
    assert main(["enhance", str(sample_video), "--at", "1.0",
                 "--low-light", "--output", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (240, 320, 3)


def test_cli_batch_processes_a_list(tmp_path, sample_video):
    list_file = tmp_path / "videos.txt"
    list_file.write_text(f"{sample_video}\n")
    out = tmp_path / "all"
    assert main(["batch", str(list_file), "--every-seconds", "1",
                 "--output-dir", str(out)]) == 0
    assert (out / "sample" / "metadata.json").exists()
