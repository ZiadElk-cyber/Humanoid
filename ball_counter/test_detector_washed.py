"""Tests for detecting washed-out red balls after normalization."""

import cv2
import numpy as np

from ball_counter.detector import RedBallDetector


def _blue_cast(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame.astype(np.float32))
    b *= 1.65
    g *= 1.1
    r *= 0.82
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def _cast_red_ball_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 32, (0, 0, 220), -1)
    return _blue_cast(frame)


def _skin_tone_blobs(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
    for center in ((180, 200), (420, 260)):
        cv2.circle(frame, center, 22, (170, 190, 220), -1)
    return _blue_cast(frame)


def test_detects_cast_red_ball() -> bool:
    frame = _cast_red_ball_frame()
    detector = RedBallDetector()
    count = detector.detect(frame).count

    ok = count == 1
    print(f"Blue-cast red ball: count={count}")
    print("  PASS" if ok else "  FAIL (expected 1)")
    return ok


def test_rejects_skin_tone_after_normalization() -> bool:
    frame = _skin_tone_blobs()
    detector = RedBallDetector()
    count = detector.detect(frame).count

    ok = count == 0
    print(f"Skin-tone blobs normalized: count={count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def main() -> None:
    results = [
        test_detects_cast_red_ball(),
        test_rejects_skin_tone_after_normalization(),
    ]

    if all(results):
        print("All washed-color detector tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
