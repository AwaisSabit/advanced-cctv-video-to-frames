"""Saving frames as PNG/JPG with a metadata JSON file."""

import json
from pathlib import Path

import cv2

from frameforge.models import Frame


def save_frame(frame: Frame, path, fmt=None, quality=95) -> Path:
    """Save one frame as PNG or JPG.

    ``fmt`` is "png" or "jpg"; without it, the extension decides.
    Returns the path written.
    """
    path = Path(path)
    suffix = (fmt or path.suffix.lstrip(".")).lower()
    if suffix in ("jpg", "jpeg"):
        ok = cv2.imwrite(
            str(path), frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
    elif suffix == "png":
        ok = cv2.imwrite(str(path), frame.image)
    else:
        raise ValueError(f"Unsupported image format: {suffix!r}")
    if not ok:
        raise OSError(f"Could not write frame to {path}")
    return path


def save_frames(frames, output_dir, fmt="png", prefix="frame") -> list:
    """Save frames as frame_000001.png and friends, in order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        name = f"{prefix}_{frame.index:06d}.{fmt}"
        paths.append(save_frame(frame, output_dir / name, fmt=fmt))
    return paths


def write_metadata(frames, path, video_path="") -> Path:
    """Write a JSON file describing the saved frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": video_path,
        "frame_count": len(frames),
        "frames": [
            {
                "index": f.index,
                "timestamp": round(f.timestamp, 3),
                "source": f.source,
                "scores": f.scores,
            }
            for f in frames
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def make_contact_sheet(frames, path, columns=4, thumb_size=(320, 240)) -> Path:
    """Lay frames out in a grid on one image, with Pillow.

    Each thumbnail is labelled with its timestamp. Returns the
    path written.
    """
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("No frames to put on a contact sheet")
    rows = (len(frames) + columns - 1) // columns
    thumb_w, thumb_h = thumb_size
    label_h = 20
    sheet = Image.new(
        "RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb).resize((thumb_w, thumb_h))
        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_h + 3), f"{frame.timestamp:.2f}s", fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path
