# FrameForge — Advanced Video to Frames

Extract clear, usable frames from CCTV footage with Python and
OpenCV: pick frames by interval, time range or motion, keep only
the sharp ones, zoom in on a region of interest, enhance low light
night footage, and save the results as PNG or JPG with a metadata
file and a contact sheet.

> Built as part of a 30-day portfolio challenge, Nov–Dec 2026.

## Problem

CCTV footage is hours long, and finding one clear frame of a
person or a number plate is hard. Scrubbing through a video by
hand takes too long, and most of the frames are blurred, dark or
empty. FrameForge does the searching for you: it pulls out frames
at the right moments, scores them for sharpness and brightness,
keeps only the clear ones, and lets you zoom in on the part that
matters.

## Features

- Video metadata: duration, fps, frame count, resolution, codec
- Frame extraction by interval, time range, timestamps, scene
  changes and motion
- Blur, brightness and contrast scoring, with clear-frame filtering
- Digital zoom in, zoom out and panning across a zoomed region
- Enhancement: sharpening, denoising and low-light night footage
- Timestamp and label overlays on saved frames
- PNG and JPG saving with a metadata JSON file and a contact sheet
- Batch processing for multiple videos
- Command line interface with `info`, `extract`, `clear`, `zoom`,
  `enhance` and `batch` subcommands

## Stack

- Python 3.11+
- [OpenCV](https://opencv.org/) (`opencv-python-headless`) for all
  video reading, writing and image work
- [NumPy](https://numpy.org/) for frame maths
- [Pillow](https://python-pillow.org/) for image saving and
  contact sheets
- pytest for tests

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/AwaisSabit/advanced-video-to-frames.git
cd advanced-video-to-frames
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[test]           # editable install, including pytest
```

## Usage

### Inspect a video

```bash
frameforge info cctv.mp4
```

```text
File:       cctv.mp4
Duration:   120.50 seconds
FPS:        25.00
Frames:     3012
Resolution: 1280 x 720
Codec:      mp4v
```

### Extract frames every few seconds

```bash
frameforge extract cctv.mp4 --every-seconds 10 --output-dir frames
frameforge extract cctv.mp4 --every-nth 150 --output-dir frames
frameforge extract cctv.mp4 --from 60 --to 120 --every-seconds 5 --output-dir frames
frameforge extract cctv.mp4 --at 12.5 45.0 90.0 --output-dir frames
```

### Keep only clear frames

```bash
frameforge clear cctv.mp4 --every-seconds 5 --min-sharpness 80 --output-dir clear
```

### Zoom in on a region

```bash
frameforge zoom cctv.mp4 --at 45.0 --region 400 200 800 500 --output zoomed.png
frameforge zoom cctv.mp4 --at 45.0 --zoom 2 --output zoomed.png
```

### Enhance a dark frame

```bash
frameforge enhance cctv.mp4 --at 90.0 --low-light --sharpen --output enhanced.png
```

### Batch process several videos

```bash
frameforge batch videos.txt --every-seconds 10 --output-dir all-frames
```

`videos.txt` holds one video path per line. See
`examples/batch-list.txt` for an example.

### As a library

```python
from frameforge import VideoFile, extract_by_interval, clear_frames

video = VideoFile("cctv.mp4")
print(video.metadata.duration_seconds, video.metadata.fps)
frames = extract_by_interval(video, every_seconds=10)
sharp = clear_frames(frames, min_sharpness=80)
```

### Demo

```bash
python examples/demo.py        # makes a synthetic CCTV-style video
                               # and extracts clear zoomed frames
```

## Tests

```bash
python -m pytest -q
```

Tests generate their own small synthetic videos with OpenCV
(a moving rectangle on a plain background, a sharp frame and a
blurred copy, a sudden scene cut) — no real video files are
committed.

## Limitations

- Digital zoom cannot add detail that is not in the pixels; it
  crops and enlarges what is already there, so a heavily zoomed
  frame will look softer.
- Enhancement helps visibility but does not recover detail lost
  to blur, darkness or compression.
- Scene change and motion thresholds need tuning per camera:
  a busy street needs a higher threshold than a quiet corridor.
- FrameForge reads videos with OpenCV; a codec OpenCV cannot
  decode cannot be processed.
- FrameForge extracts frames; it cannot read a QR code or a
  number plate back from a frame — that needs a phone camera or
  a recognition library.

## License

MIT — see [LICENSE](LICENSE).
