"""Command line interface for FrameForge."""

import argparse
import sys
from pathlib import Path

from frameforge.batch import process_batch_file
from frameforge.enhance import denoise, enhance, low_light, sharpen
from frameforge.extract import (
    detect_motion,
    detect_scene_changes,
    extract_at_timestamps,
    extract_by_interval,
    extract_by_range,
)
from frameforge.models import Frame
from frameforge.overlay import add_overlay
from frameforge.quality import clear_frames
from frameforge.saving import make_contact_sheet, save_frame, save_frames, write_metadata
from frameforge.video import VideoFile
from frameforge.zoom import pan, zoom_centre, zoom_in


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameforge",
        description="Extract clear frames from CCTV footage.",
    )
    parser.add_argument("--version", action="version", version="frameforge 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Show video metadata.")
    info.add_argument("video")

    extract = sub.add_parser("extract", help="Extract frames from a video.")
    extract.add_argument("video")
    extract.add_argument("--every-seconds", type=float, default=None)
    extract.add_argument("--every-nth", type=int, default=None)
    extract.add_argument("--from", dest="start", type=float, default=None)
    extract.add_argument("--to", dest="end", type=float, default=None)
    extract.add_argument("--at", dest="at", type=float, nargs="*", default=None)
    extract.add_argument("--motion", action="store_true")
    extract.add_argument("--scene", action="store_true")
    extract.add_argument("--output-dir", default="frames")
    extract.add_argument("--format", default="png", choices=["png", "jpg"])
    extract.add_argument("--contact-sheet", action="store_true")

    clear = sub.add_parser("clear", help="Keep only clear frames.")
    clear.add_argument("video")
    clear.add_argument("--every-seconds", type=float, default=5.0)
    clear.add_argument("--min-sharpness", type=float, default=100.0)
    clear.add_argument("--output-dir", default="clear")
    clear.add_argument("--format", default="png", choices=["png", "jpg"])

    zoom = sub.add_parser("zoom", help="Zoom in on a region of a frame.")
    zoom.add_argument("video")
    zoom.add_argument("--at", type=float, required=True)
    zoom.add_argument("--region", type=int, nargs=4, default=None,
                      metavar=("X", "Y", "W", "H"))
    zoom.add_argument("--zoom", type=float, default=None)
    zoom.add_argument("--pan", type=int, nargs=2, default=None,
                      metavar=("DX", "DY"))
    zoom.add_argument("--label", default="")
    zoom.add_argument("--output", required=True)

    enhance_p = sub.add_parser("enhance", help="Enhance a frame.")
    enhance_p.add_argument("video")
    enhance_p.add_argument("--at", type=float, required=True)
    enhance_p.add_argument("--low-light", action="store_true")
    enhance_p.add_argument("--sharpen", action="store_true")
    enhance_p.add_argument("--denoise", action="store_true")
    enhance_p.add_argument("--label", default="")
    enhance_p.add_argument("--output", required=True)

    batch = sub.add_parser("batch", help="Process several videos.")
    batch.add_argument("list_file")
    batch.add_argument("--every-seconds", type=float, default=10.0)
    batch.add_argument("--output-dir", default="all-frames")
    batch.add_argument("--format", default="png", choices=["png", "jpg"])
    return parser


def _print_info(video):
    meta = video.metadata
    print(f"File:       {meta.path}")
    print(f"Duration:   {meta.duration_seconds:.2f} seconds")
    print(f"FPS:        {meta.fps:.2f}")
    print(f"Frames:     {meta.frame_count}")
    print(f"Resolution: {meta.width} x {meta.height}")
    print(f"Codec:      {meta.codec}")


def _save_selection(video, frames, output_dir, fmt, contact_sheet=False):
    paths = save_frames(frames, output_dir, fmt=fmt)
    write_metadata(frames, Path(output_dir) / "metadata.json",
                   video_path=video.path)
    if contact_sheet and frames:
        make_contact_sheet(frames, Path(output_dir) / "contact-sheet.png")
    for path in paths:
        print(f"Saved {path}")
    print(f"{len(frames)} frames saved to {output_dir}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        _print_info(VideoFile(args.video))
        return 0

    if args.command == "extract":
        video = VideoFile(args.video)
        if args.motion:
            frames = detect_motion(video)
        elif args.scene:
            frames = detect_scene_changes(video)
        elif args.at is not None:
            frames = extract_at_timestamps(video, args.at)
        elif args.start is not None and args.end is not None:
            frames = extract_by_range(
                video, args.start, args.end, every_seconds=args.every_seconds
            )
        else:
            frames = extract_by_interval(
                video, every_seconds=args.every_seconds, every_nth=args.every_nth
            )
        return _save_selection(
            video, frames, args.output_dir, args.format, args.contact_sheet
        )

    if args.command == "clear":
        video = VideoFile(args.video)
        frames = extract_by_interval(video, every_seconds=args.every_seconds)
        kept = clear_frames(frames, min_sharpness=args.min_sharpness)
        print(f"Kept {len(kept)} of {len(frames)} frames")
        return _save_selection(video, kept, args.output_dir, args.format)

    if args.command == "zoom":
        video = VideoFile(args.video)
        frame = video.frame_at_time(args.at)
        if args.region is not None:
            x, y, w, h = args.region
            image = zoom_in(frame.image, x, y, w, h,
                            output_size=(frame.width, frame.height))
        elif args.pan is not None:
            factor = args.zoom or 2.0
            image = pan(frame.image, factor, args.pan[0], args.pan[1])
        elif args.zoom is not None:
            image = zoom_centre(frame.image, args.zoom)
        else:
            raise SystemExit("Pass --region, --zoom or --pan to zoom.")
        out = Frame(image=image, index=frame.index, timestamp=frame.timestamp,
                    source=video.path)
        if args.label:
            out.image = add_overlay(out.image, out.timestamp, args.label)
        save_frame(out, args.output)
        print(f"Saved {args.output}")
        return 0

    if args.command == "enhance":
        video = VideoFile(args.video)
        frame = video.frame_at_time(args.at)
        image = frame.image
        if args.low_light:
            image = low_light(image)
        if args.sharpen and args.denoise:
            image = enhance(image)
        elif args.sharpen:
            image = sharpen(image)
        elif args.denoise:
            image = denoise(image)
        out = Frame(image=image, index=frame.index, timestamp=frame.timestamp,
                    source=video.path)
        if args.label:
            out.image = add_overlay(out.image, out.timestamp, args.label)
        save_frame(out, args.output)
        print(f"Saved {args.output}")
        return 0

    if args.command == "batch":
        results = process_batch_file(
            args.list_file, args.output_dir,
            every_seconds=args.every_seconds, fmt=args.format,
        )
        for result in results:
            print(f"{result['video']}: {result['frames']} frames")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
