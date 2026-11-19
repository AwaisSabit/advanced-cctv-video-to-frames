"""FrameForge — advanced video to frames for CCTV footage."""

from frameforge.extract import extract_by_interval
from frameforge.models import Frame, VideoMetadata
from frameforge.quality import clear_frames
from frameforge.video import VideoFile

__version__ = "1.0.0"
__all__ = [
    "Frame",
    "VideoFile",
    "VideoMetadata",
    "clear_frames",
    "extract_by_interval",
    "__version__",
]
