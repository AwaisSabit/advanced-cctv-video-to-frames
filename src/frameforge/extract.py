"""Frame extraction: by interval (range, timestamps, scene and
motion arrive in later steps)."""

from frameforge.models import Frame


def extract_by_interval(video, every_seconds=None, every_nth=None):
    """Pick one frame every N seconds, or every Nth frame.

    Pass exactly one of ``every_seconds`` or ``every_nth``.
    """
    if (every_seconds is None) == (every_nth is None):
        raise ValueError("Pass exactly one of every_seconds or every_nth")
    if every_seconds is not None:
        if every_seconds <= 0:
            raise ValueError("every_seconds must be positive")
        step = max(1, int(round(every_seconds * video.metadata.fps)))
    else:
        if every_nth <= 0:
            raise ValueError("every_nth must be positive")
        step = int(every_nth)
    return [
        frame for frame in video.iter_frames() if frame.index % step == 0
    ]


def extract_by_range(video, start_seconds, end_seconds, every_seconds=None):
    """Pick frames between two timestamps, optionally every N seconds."""
    if start_seconds < 0 or end_seconds < start_seconds:
        raise ValueError("Bad time range: start must be >= 0 and end >= start")
    if end_seconds > video.metadata.duration_seconds:
        raise ValueError("End time is past the end of the video")
    start_index = int(start_seconds * video.metadata.fps)
    end_index = min(
        int(end_seconds * video.metadata.fps), video.metadata.frame_count - 1
    )
    step = 1
    if every_seconds is not None:
        if every_seconds <= 0:
            raise ValueError("every_seconds must be positive")
        step = max(1, int(round(every_seconds * video.metadata.fps)))
    frames = []
    for frame in video.iter_frames():
        if frame.index < start_index or frame.index > end_index:
            continue
        if (frame.index - start_index) % step == 0:
            frames.append(frame)
    return frames


def extract_at_timestamps(video, timestamps):
    """Pick the frame closest to each timestamp in seconds."""
    return [video.frame_at_time(float(t)) for t in timestamps]


def detect_scene_changes(video, threshold=0.3):
    """Find frames where the picture changes suddenly (a cut).

    Compares the mean absolute difference between consecutive
    frames; a frame is a scene change when the difference is
    above ``threshold`` times the maximum possible difference.
    Returns the frames just after each cut.
    """
    import cv2
    import numpy as np

    cuts = []
    previous = None
    for frame in video.iter_frames():
        grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, (80, 60))
        if previous is not None:
            diff = float(np.mean(cv2.absdiff(small, previous))) / 255.0
            if diff > threshold:
                cuts.append(frame)
        previous = small
    return cuts


def detect_motion(video, threshold=25, min_pixels=500):
    """Find frames where something moved.

    Compares each frame with the previous one; a frame counts as
    motion when more than ``min_pixels`` pixels changed by more
    than ``threshold`` grey levels.
    """
    import cv2
    import numpy as np

    moving = []
    previous = None
    for frame in video.iter_frames():
        grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            diff = cv2.absdiff(grey, previous)
            changed = int(np.count_nonzero(diff > threshold))
            if changed >= min_pixels:
                moving.append(frame)
        previous = grey
    return moving
