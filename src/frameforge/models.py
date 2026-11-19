"""Shared data types for FrameForge."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VideoMetadata:
    """Facts about a video file, read with OpenCV."""

    path: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class Frame:
    """One extracted video frame and where it came from."""

    image: np.ndarray
    index: int
    timestamp: float
    source: str = ""
    scores: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class QualityScores:
    """Sharpness, brightness and contrast scores for one frame."""

    sharpness: float
    brightness: float
    contrast: float

    @property
    def is_clear(self) -> bool:
        return self.sharpness > 0
