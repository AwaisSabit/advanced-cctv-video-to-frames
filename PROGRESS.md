# FrameForge — Progress

FrameForge is an advanced video to frames tool for CCTV footage,
built with Python and OpenCV. Version 1.0.0 is the finished tool:
video metadata, extraction by interval, range, timestamps, scene
changes and motion, sharpness and quality scoring, clear-frame
filtering, digital zoom and panning, enhancement for low light,
overlays, PNG/JPG saving with metadata, contact sheets, batch
processing and a full command line interface, all covered by
pytest tests on synthetic videos.

## Version 1.0.0

- Project set up with package layout, pyproject, README and gitignore
- Video loading and metadata reading (duration, fps, frame count,
  resolution, codec)
- Frame extraction by interval, timestamp range and specific
  timestamps
- Scene change detection and motion detection for frame extraction
- Blur, brightness and contrast scoring, with clear-frame filtering
- Digital zoom in, zoom out and panning across a zoomed region
- Frame enhancement with sharpening, denoising and low-light
  enhancement for night CCTV footage
- Timestamp and label overlays on frames
- Frame saving in PNG and JPG with a metadata JSON file
- A contact sheet of extracted frames
- Batch processing for multiple videos
- Command line interface with `info`, `extract`, `clear`, `zoom`,
  `enhance` and `batch` subcommands
- Tests for extraction, quality, zoom and enhancement
- Documentation, examples and final polish for version 1.0.0
