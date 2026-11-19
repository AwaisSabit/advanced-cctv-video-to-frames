"""Command line interface for FrameForge (grows with the project)."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameforge",
        description="Extract clear frames from CCTV footage.",
    )
    parser.add_argument("--version", action="version", version="frameforge 1.0.0")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="Show video metadata.")
    sub.add_parser("extract", help="Extract frames from a video.")
    sub.add_parser("clear", help="Keep only clear frames.")
    sub.add_parser("zoom", help="Zoom in on a region of a frame.")
    sub.add_parser("enhance", help="Enhance a frame.")
    sub.add_parser("batch", help="Process several videos.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    print(f"Command '{args.command}' is not wired up yet in this early build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
