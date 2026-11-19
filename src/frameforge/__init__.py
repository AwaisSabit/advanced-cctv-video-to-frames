"""FrameForge — advanced video to frames for CCTV footage."""

from frameforge.models import Frame, VideoMetadata
from frameforge.video import VideoFile

__version__ = "1.0.0"
__all__ = ["Frame", "VideoFile", "VideoMetadata", "__version__"]
