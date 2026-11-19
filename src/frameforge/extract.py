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
