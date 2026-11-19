"""Batch processing for multiple videos."""

from pathlib import Path

from frameforge.extract import extract_by_interval
from frameforge.saving import save_frames, write_metadata
from frameforge.video import VideoFile


def read_video_list(path) -> list:
    """Read one video path per line, skipping blanks and comments."""
    videos = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            videos.append(line)
    return videos


def process_video(video_path, output_dir, every_seconds=10.0, fmt="png"):
    """Extract frames from one video into its own folder."""
    video = VideoFile(video_path)
    frames = extract_by_interval(video, every_seconds=every_seconds)
    folder = Path(output_dir) / Path(video_path).stem
    paths = save_frames(frames, folder, fmt=fmt)
    write_metadata(frames, folder / "metadata.json", video_path=str(video_path))
    return {"video": str(video_path), "frames": len(frames), "paths": paths}


def process_batch(video_paths, output_dir, every_seconds=10.0, fmt="png"):
    """Extract frames from several videos, one folder each."""
    results = []
    for video_path in video_paths:
        results.append(
            process_video(video_path, output_dir, every_seconds, fmt)
        )
    return results


def process_batch_file(list_path, output_dir, every_seconds=10.0, fmt="png"):
    """Read a list file and process every video in it."""
    return process_batch(
        read_video_list(list_path), output_dir, every_seconds, fmt
    )
