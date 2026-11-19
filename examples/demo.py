"""Demo: make a small synthetic CCTV-style video, then extract
clear zoomed frames from it.

Run from the project root:

    python examples/demo.py

Writes examples/output/cctv-demo.avi, a few extracted frames,
one zoomed frame, one enhanced low-light frame and a contact
sheet into examples/output/.
"""

from pathlib import Path

import cv2
import numpy as np

from frameforge import VideoFile, clear_frames, extract_by_interval
from frameforge.enhance import low_light
from frameforge.overlay import add_overlay
from frameforge.quality import score_frame
from frameforge.saving import make_contact_sheet, save_frame, save_frames
from frameforge.zoom import zoom_in

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 10, 6


def make_demo_video(path):
    """A dark corridor scene: a bright rectangle (a person)
    walks across, the camera shakes a little, and one stretch
    is blurred as if the camera lost focus."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    for i in range(FPS * SECONDS):
        frame = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
        cv2.rectangle(frame, (0, 280), (WIDTH, HEIGHT), (60, 60, 60), -1)
        x = 40 + i * 9
        cv2.rectangle(frame, (x, 180), (x + 50, 280), (220, 220, 220), -1)
        cv2.circle(frame, (x + 25, 165), 18, (220, 220, 220), -1)
        if 30 <= i < 40:  # a blurred stretch in the middle
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        writer.write(frame)
    writer.release()
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    video_path = OUT / "cctv-demo.avi"
    make_demo_video(video_path)
    video = VideoFile(video_path)
    meta = video.metadata
    print(f"Demo video: {meta.frame_count} frames, {meta.fps} fps, "
          f"{meta.duration_seconds:.1f}s, {meta.resolution}")

    frames = extract_by_interval(video, every_seconds=1.0)
    print(f"Extracted {len(frames)} frames, one per second")
    for frame in frames:
        scores = score_frame(frame)
        print(f"  frame {frame.index} at {frame.timestamp:.1f}s: "
              f"sharpness {scores.sharpness:.1f}")

    sharp = clear_frames(frames, min_sharpness=50.0)
    print(f"Kept {len(sharp)} clear frames")
    save_frames(sharp, OUT / "clear", fmt="png")
    write_paths = save_frames(frames, OUT / "all", fmt="jpg")
    print(f"Saved {len(write_paths)} frames to {OUT / 'all'}")

    if sharp:
        first = sharp[0]
        zoomed = zoom_in(first.image, 200, 120, 240, 180,
                         output_size=(640, 360))
        from frameforge.models import Frame

        zoom_frame = Frame(image=add_overlay(zoomed, first.timestamp, "Gate 2"),
                           index=first.index, timestamp=first.timestamp)
        save_frame(zoom_frame, OUT / "zoomed.png")
        print(f"Zoomed frame saved to {OUT / 'zoomed.png'}")
        dark = low_light(first.image)
        save_frame(Frame(image=dark, index=first.index,
                         timestamp=first.timestamp), OUT / "enhanced.png")
        print(f"Enhanced frame saved to {OUT / 'enhanced.png'}")

    make_contact_sheet(frames, OUT / "contact-sheet.png", columns=3)
    print(f"Contact sheet saved to {OUT / 'contact-sheet.png'}")


if __name__ == "__main__":
    main()
