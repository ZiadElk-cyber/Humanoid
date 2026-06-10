"""HSV color buckets for common pen colors (OpenCV H: 0-179, S/V: 0-255)."""

from typing import NamedTuple

import cv2
import numpy as np


class HsvRange(NamedTuple):
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


COLOR_RANGES: dict[str, list[HsvRange]] = {
    "black": [HsvRange((0, 0, 0), (179, 255, 90))],
    "blue": [HsvRange((90, 20, 25), (135, 255, 255))],
    "red": [
        HsvRange((0, 50, 40), (10, 255, 255)),
        HsvRange((170, 50, 40), (179, 255, 255)),
    ],
    "green": [HsvRange((35, 40, 40), (85, 255, 255))],
    "yellow": [HsvRange((18, 60, 80), (34, 255, 255))],
    "other": [HsvRange((0, 0, 91), (179, 19, 255))],
}

DISPLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (40, 40, 40),
    "blue": (255, 120, 0),
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "yellow": (0, 220, 255),
    "other": (200, 100, 200),
}

COLOR_ORDER = ["black", "blue", "red", "green", "yellow", "other"]


def mask_for_color(hsv: np.ndarray, color_name: str) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in COLOR_RANGES[color_name]:
        lower = np.array(hsv_range.lower, dtype=np.uint8)
        upper = np.array(hsv_range.upper, dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
    return combined


def _center_crop(bgr_roi: np.ndarray) -> np.ndarray:
    height, width = bgr_roi.shape[:2]
    margin_x = int(width * 0.2)
    margin_y = int(height * 0.2)
    if margin_x * 2 >= width or margin_y * 2 >= height:
        return bgr_roi
    return bgr_roi[margin_y : height - margin_y, margin_x : width - margin_x]


def classify_dominant_color(bgr_roi: np.ndarray, mask: np.ndarray | None = None) -> str:
    center = _center_crop(bgr_roi)
    if center.size == 0:
        return "other"

    if mask is not None:
        center_mask = _center_crop(mask)
        if center_mask.shape[:2] != center.shape[:2]:
            return "other"
        pixels = center[center_mask > 0]
        if pixels.size == 0:
            return "other"
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    else:
        hsv_pixels = cv2.cvtColor(center, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab_pixels = cv2.cvtColor(center, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    hsv = hsv_pixels.reshape(-1, 1, 3)
    lab = lab_pixels.reshape(-1, 1, 3)

    best_color = "other"
    best_count = 0
    for color_name in COLOR_ORDER:
        if color_name == "other":
            continue
        count = int(np.count_nonzero(mask_for_color(hsv, color_name)))
        if count > best_count:
            best_count = count
            best_color = color_name

    if best_count > 0:
        return best_color

    mean_h = float(np.mean(hsv_pixels[:, 0]))
    mean_s = float(np.mean(hsv_pixels[:, 1]))
    mean_v = float(np.mean(hsv_pixels[:, 2]))
    mean_l = float(np.mean(lab_pixels[:, 0]))

    if mean_v < 70 or mean_l < 80:
        return "black"
    if 90 <= mean_h <= 135 and mean_s >= 15:
        return "blue"

    return "other"
