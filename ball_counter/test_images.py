"""Offline validation with synthetic red-ball images."""

import cv2
import numpy as np

from ball_counter.detector import RedBallDetector

RED_BGR = (0, 0, 220)
WHITE = (255, 255, 255)


def _blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), WHITE, dtype=np.uint8)


def _draw_ball(frame: np.ndarray, center: tuple[int, int], radius: int) -> None:
    cv2.circle(frame, center, radius, RED_BGR, -1)
    cv2.circle(frame, center, radius, (0, 0, 180), 2)


def _wood_background(width: int = 640, height: int = 480) -> np.ndarray:
    base = np.full((height, width, 3), (72, 98, 130), dtype=np.uint8)
    noise = np.random.default_rng(42).integers(-18, 19, (height, width, 3), dtype=np.int16)
    frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def _gradient_background(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        shade = int(80 + (y / height) * 140)
        frame[y, :] = (shade, shade, shade)
    return frame


def test_three_balls(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    _draw_ball(frame, (160, 240), 40)
    _draw_ball(frame, (320, 240), 35)
    _draw_ball(frame, (480, 240), 45)

    result = detector.detect(frame)
    ok = result.count == 3
    print(f"Three balls: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 3)")
    return ok


def test_empty_frame(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Empty frame: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_rejects_elongated_red_shape(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    cv2.rectangle(frame, (200, 180), (440, 220), RED_BGR, -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Elongated red shape: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_overlapping_two_balls(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    _draw_ball(frame, (300, 240), 40)
    _draw_ball(frame, (350, 240), 40)

    result = detector.detect(frame)
    ok = result.count == 2
    print(f"Overlapping two balls: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 2)")
    return ok


def test_overlapping_three_balls(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    _draw_ball(frame, (260, 240), 35)
    _draw_ball(frame, (310, 240), 35)
    _draw_ball(frame, (360, 240), 35)

    result = detector.detect(frame)
    ok = result.count == 3
    print(f"Overlapping three balls: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 3)")
    return ok


def test_wood_background(detector: RedBallDetector) -> bool:
    frame = _wood_background()
    _draw_ball(frame, (200, 240), 38)
    _draw_ball(frame, (440, 240), 42)

    result = detector.detect(frame)
    ok = result.count == 2
    print(f"Wood background: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 2)")
    return ok


def test_gradient_background(detector: RedBallDetector) -> bool:
    frame = _gradient_background()
    _draw_ball(frame, (200, 240), 38)
    _draw_ball(frame, (440, 240), 42)

    result = detector.detect(frame)
    ok = result.count == 2
    print(f"Gradient background: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 2)")
    return ok


def test_no_false_positives_on_orange_bg(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    cv2.rectangle(frame, (120, 120), (520, 360), (0, 140, 255), -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Orange background patch: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


SKIN_BGR = (200, 120, 80)


def test_rejects_skin_tone_blobs(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    for center in [(180, 200), (320, 260), (460, 220), (400, 340)]:
        cv2.circle(frame, center, 18, SKIN_BGR, -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Skin-tone blobs: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_rejects_small_red_dots(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    for center in [(200, 200), (300, 250), (400, 180), (500, 300)]:
        cv2.circle(frame, center, 8, RED_BGR, -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Small red dots: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_rejects_dull_maroon(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    for center in [(200, 240), (400, 240)]:
        cv2.circle(frame, center, 35, (0, 0, 80), -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Dull maroon: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_rejects_pale_red(detector: RedBallDetector) -> bool:
    frame = _blank_frame()
    for center in [(220, 240), (420, 240)]:
        cv2.circle(frame, center, 30, (200, 200, 255), -1)

    result = detector.detect(frame)
    ok = result.count == 0
    print(f"Pale red: count={result.count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def main() -> None:
    detector = RedBallDetector()
    results = [
        test_three_balls(detector),
        test_empty_frame(detector),
        test_rejects_elongated_red_shape(detector),
        test_overlapping_two_balls(detector),
        test_overlapping_three_balls(detector),
        test_wood_background(detector),
        test_gradient_background(detector),
        test_no_false_positives_on_orange_bg(detector),
        test_rejects_skin_tone_blobs(detector),
        test_rejects_small_red_dots(detector),
        test_rejects_dull_maroon(detector),
        test_rejects_pale_red(detector),
    ]

    if all(results):
        print("All image tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
