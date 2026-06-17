# Humanoid

Computer-vision demos that use your webcam to count things in real time — red balls and bicep flexes.

Built with [OpenCV](https://opencv.org/) and [MediaPipe](https://developers.google.com/mediapipe) (gun counter only).

## Requirements

- Python 3.10+
- Two webcams mounted side-by-side (fixed baseline, same height, parallel)

## Setup

From the project root:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

### Red ball counter

Counts red balls using **two cameras** in a stereo setup (like human eyes). A ball is counted only when it appears consistently in both views, which filters single-camera false positives such as fingertips.

```bash
python camera.py
```

Two windows open: a side-by-side stereo view with circle overlays and an info panel with the count.

| Key | Action |
|-----|--------|
| `d` | Toggle debug overlay (search mask + unmatched detections in orange) |
| `q` | Quit |

By default `camera.py` uses camera index `0` for the left eye and `1` for the right. Change `LEFT_CAMERA_INDEX` and `RIGHT_CAMERA_INDEX` at the top of the file if needed. Swap them if left and right are reversed.

**Mounting tips**

- Fix both cameras side-by-side; do not move them relative to each other after tuning
- Same height, pointing the same direction
- If real balls are missed, widen `MIN_DISPARITY` / `MAX_DISPARITY` in `ball_counter/stereo.py` for your baseline and typical distance
- If one camera shows no detections, press `d` and check whether a green mask appears on that pane. Frames are auto color-corrected per camera; matching camera models and USB bandwidth help

### Gun counter

Counts bicep flexes using pose landmarks. Face the camera and flex.

```bash
python guns.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |

### Red ball counter — offline tests

Runs detection checks against synthetic images (no camera required):

```bash
python -m ball_counter.test_images
```

Prints `All image tests passed.` on success, or exits with code `1` if any test fails.

Stereo fusion tests (no camera required):

```bash
python -m ball_counter.test_stereo
```

Tracker tests:

```bash
python -m ball_counter.test_tracker
```

Preprocess and washed-color detector tests:

```bash
python -m ball_counter.test_preprocess
python -m ball_counter.test_detector_washed
```

## Project layout

```
Humanoid/
├── camera.py              # Red ball counter (webcam)
├── guns.py                # Gun counter (webcam)
├── ball_counter/          # Red ball detection and overlay
├── gun_counter/           # Pose-based flex detection
└── requirements.txt
```
