"""Offline validation against example pen images."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pen_counter.detector import DetectionMode, PenDetector

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
BLUE_PEN = ASSETS_DIR / (
    "c__Users_Siddig_AppData_Roaming_Cursor_User_workspaceStorage_"
    "ef668f5c446eaef19a95cdaa0b3ef5f4_images_image-51fbf3be-cc86-46f5-a697-a913deae8f00.png"
)
BLACK_PEN = ASSETS_DIR / (
    "c__Users_Siddig_AppData_Roaming_Cursor_User_workspaceStorage_"
    "ef668f5c446eaef19a95cdaa0b3ef5f4_images_image-6eaa1046-0234-40f6-a0ee-745104d9d6ae.png"
)


def synthetic_background(frame: np.ndarray) -> np.ndarray:
    """Approximate an empty table by taking the median of blurred frame variants."""
    blurred_frames = [
        cv2.GaussianBlur(frame, (kernel, kernel), 0)
        for kernel in (31, 41, 51, 61, 71)
    ]
    return np.median(np.stack(blurred_frames, axis=0), axis=0).astype(np.uint8)


def run_image_test(detector: PenDetector, image_path: Path, expected_color: str) -> bool:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"FAIL: could not read {image_path.name}")
        return False

    detector.calibrate(synthetic_background(frame))
    result = detector.detect(frame)

    total = sum(result.counts.values())
    colors = [d.color for d in result.detections]
    ok = total >= 1 and expected_color in colors

    print(f"{image_path.name}: total={total}, colors={colors}, counts={result.counts}")
    if ok:
        print(f"  PASS (found {expected_color})")
    else:
        print(f"  FAIL (expected at least one {expected_color})")
    return ok


def _run_calibration_frames(detector: PenDetector, frames: list[np.ndarray]):
    """Feed frames with simulated elapsed time so min-duration gate is satisfied."""
    start = 1000.0
    with patch("pen_counter.detector.time.monotonic") as mock_time:
        mock_time.return_value = start
        detector.start_calibration()
        result = None
        for i, frame in enumerate(frames):
            mock_time.return_value = start + (i + 1) * 0.05
            result = detector.add_calibration_frame(frame)
            if result is not None:
                break
        return result


def test_rejects_motion(detector: PenDetector, frame: np.ndarray) -> bool:
    """Shaken frames should fail multi-frame calibration."""
    detector.clear_calibration()
    frames = [
        np.roll(np.roll(frame, (i % 5) * 4, axis=0), (i % 5) * 4, axis=1)
        for i in range(PenDetector.CALIBRATION_FRAME_COUNT)
    ]
    result = _run_calibration_frames(detector, frames)
    if result is None:
        print("Motion rejection: FAIL (calibration never finalized)")
        return False
    ok = not result.success
    print(f"Motion rejection: success={result.success}, message={result.message!r}")
    return ok


def test_diagonal_pen(detector: PenDetector, frame: np.ndarray) -> bool:
    """A pen rotated ~45 degrees should still be detected."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    rotated = cv2.warpAffine(
        frame,
        cv2.getRotationMatrix2D(center, 45, 1.0),
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(220, 220, 220),
    )
    bg = synthetic_background(rotated)
    detector.calibrate(bg)
    result = detector.detect(rotated)

    total = sum(result.counts.values())
    colors = [d.color for d in result.detections]
    ok = total >= 1
    print(f"Diagonal pen: total={total}, colors={colors}")
    if ok:
        print("  PASS (diagonal pen detected)")
    else:
        print("  FAIL (expected diagonal pen detection)")
    return ok


def test_rejects_pen_in_frame(detector: PenDetector, frame: np.ndarray) -> bool:
    """Calibrating while a pen is visible should fail validation."""
    detector.clear_calibration()
    frames = [frame] * PenDetector.CALIBRATION_FRAME_COUNT
    result = _run_calibration_frames(detector, frames)
    if result is None:
        print("Pen-in-frame rejection: FAIL (calibration never finalized)")
        return False
    ok = not result.success and not detector.calibrated
    print(f"Pen-in-frame rejection: success={result.success}, calibrated={detector.calibrated}")
    return ok


def test_accepts_stable_empty(detector: PenDetector, frame: np.ndarray) -> bool:
    """Identical empty-table frames should pass multi-frame calibration."""
    h, w = frame.shape[:2]
    empty_table = np.full((h, w, 3), 220, dtype=np.uint8)
    detector.clear_calibration()
    frames = [empty_table] * PenDetector.CALIBRATION_FRAME_COUNT
    result = _run_calibration_frames(detector, frames)
    if result is None:
        print("Stable calibration: FAIL (calibration never finalized)")
        return False
    ok = result.success and detector.calibrated
    print(f"Stable calibration: success={result.success}, calibrated={detector.calibrated}")
    return ok


def main() -> None:
    detector = PenDetector(mode=DetectionMode.TABLE)
    try:
        blue_frame = cv2.imread(str(BLUE_PEN))
        results = [
            run_image_test(detector, BLUE_PEN, "blue"),
            run_image_test(detector, BLACK_PEN, "black"),
        ]
        if blue_frame is not None:
            results.extend([
                test_diagonal_pen(detector, blue_frame),
                test_rejects_motion(detector, blue_frame),
                test_accepts_stable_empty(detector, blue_frame),
                test_rejects_pen_in_frame(detector, blue_frame),
            ])
    finally:
        detector.close()

    if all(results):
        print("All image tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

