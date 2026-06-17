"""Tests for frame normalization before detection."""

import cv2
import numpy as np

from ball_counter.detector import RedBallDetector
from ball_counter.preprocess import gray_world_white_balance, normalize_frame


def _blue_cast(frame: np.ndarray, blue_scale: float = 1.65) -> np.ndarray:
    b, g, r = cv2.split(frame.astype(np.float32))
    b *= blue_scale
    g *= 1.1
    r *= 0.82
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def _channel_mean_spread(frame: np.ndarray) -> float:
    b, g, r = cv2.split(frame)
    means = [float(b.mean()), float(g.mean()), float(r.mean())]
    return float(np.std(means))


def test_gray_world_balances_channels() -> bool:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    frame[:, :, 0] = 245
    balanced = gray_world_white_balance(frame)

    before = _channel_mean_spread(frame)
    after = _channel_mean_spread(balanced)
    ok = after < before
    print(f"Channel spread: before={before:.2f} after={after:.2f}")
    print("  PASS" if ok else "  FAIL (expected lower spread after gray-world)")
    return ok


def test_normalize_helps_cast_red_ball_mask() -> bool:
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 35, (0, 0, 220), -1)
    cast = _blue_cast(frame)

    detector = RedBallDetector()
    raw_mask = int(cv2.countNonZero(detector._red_mask(cast)))
    norm_mask = int(cv2.countNonZero(detector._red_mask(normalize_frame(cast))))

    ok = norm_mask >= raw_mask and norm_mask > 0
    print(f"Cast ball mask pixels: raw={raw_mask} normalized={norm_mask}")
    print("  PASS" if ok else "  FAIL (expected non-zero normalized mask)")
    return ok


def test_normalize_preserves_standard_detection() -> bool:
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 35, (0, 0, 220), -1)
    cast = _blue_cast(frame)

    count = RedBallDetector().detect(cast).count
    ok = count == 1
    print(f"Cast red ball detect count={count}")
    print("  PASS" if ok else "  FAIL (expected 1)")
    return ok


def main() -> None:
    results = [
        test_gray_world_balances_channels(),
        test_normalize_helps_cast_red_ball_mask(),
        test_normalize_preserves_standard_detection(),
    ]

    if all(results):
        print("All preprocess tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
