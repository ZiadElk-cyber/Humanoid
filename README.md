# Humanoid

Computer-vision demos that use your webcam to count things in real time — red balls and bicep flexes.

Built with [OpenCV](https://opencv.org/) and [MediaPipe](https://developers.google.com/mediapipe) (gun counter only).

## Requirements

- Python 3.10+
- A working webcam

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

Counts red balls using the webcam. No calibration needed — works on any background with a moving camera.

```bash
python camera.py
```

Two windows open: a live camera view with circle overlays and an info panel with the count.

| Key | Action |
|-----|--------|
| `d` | Toggle debug overlay (red search mask) |
| `q` | Quit |

By default `camera.py` uses camera index `1`. Change `CAMERA_INDEX` at the top of the file if needed.

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

## Project layout

```
Humanoid/
├── camera.py              # Red ball counter (webcam)
├── guns.py                # Gun counter (webcam)
├── ball_counter/          # Red ball detection and overlay
├── gun_counter/           # Pose-based flex detection
└── requirements.txt
```
