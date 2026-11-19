"""Video loading and metadata reading with OpenCV."""

from pathlib import Path

import cv2

from frameforge.models import Frame, VideoMetadata


class VideoError(Exception):
    """Raised when a video cannot be opened or read."""


def _fourcc_to_text(fourcc: float) -> str:
    code = int(fourcc)
    chars = bytes(
        (code >> (8 * i)) & 0xFF for i in range(4)
    ).decode("ascii", errors="replace")
    return chars.strip("\x00").strip() or "unknown"


def read_metadata(path: str | Path) -> VideoMetadata:
    """Read duration, fps, frame count, resolution and codec."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = _fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    duration = count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=path,
        duration_seconds=duration,
        fps=fps,
        frame_count=count,
        width=width,
        height=height,
        codec=codec,
    )


class VideoFile:
    """A video file opened with OpenCV, with random frame access."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.metadata = read_metadata(self.path)

    def __len__(self) -> int:
        return self.metadata.frame_count

    def frame_at_index(self, index: int) -> Frame:
        """Read one frame by its index (0-based)."""
        if index < 0 or index >= self.metadata.frame_count:
            raise VideoError(
                f"Frame index {index} is out of range "
                f"(0-{self.metadata.frame_count - 1})"
            )
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
        finally:
            cap.release()
        if not ok or image is None:
            raise VideoError(f"Could not read frame {index} from {self.path}")
        timestamp = index / self.metadata.fps if self.metadata.fps else 0.0
        return Frame(image=image, index=index, timestamp=timestamp, source=self.path)

    def frame_at_time(self, seconds: float) -> Frame:
        """Read the frame closest to a timestamp in seconds."""
        if seconds < 0 or seconds > self.metadata.duration_seconds:
            raise VideoError(
                f"Timestamp {seconds}s is out of range "
                f"(0-{self.metadata.duration_seconds:.2f}s)"
            )
        index = min(
            int(round(seconds * self.metadata.fps)),
            self.metadata.frame_count - 1,
        )
        return self.frame_at_index(index)

    def iter_frames(self):
        """Yield every frame in order, with its index and timestamp."""
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise VideoError(f"Could not open video: {self.path}")
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok or image is None:
                    break
                timestamp = index / self.metadata.fps if self.metadata.fps else 0.0
                yield Frame(
                    image=image, index=index, timestamp=timestamp, source=self.path
                )
                index += 1
        finally:
            cap.release()
