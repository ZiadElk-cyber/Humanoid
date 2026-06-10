from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pen_counter.colors import COLOR_ORDER, classify_dominant_color

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

RotatedRect = tuple[tuple[float, float], tuple[float, float], float]


def rect_to_box_points(rect: RotatedRect) -> np.ndarray:
    return cv2.boxPoints(rect).astype(np.int32)


def rect_to_aabb(rect: RotatedRect) -> tuple[int, int, int, int]:
    points = rect_to_box_points(rect)
    x, y, w, h = cv2.boundingRect(points)
    return x, y, w, h


class DetectionMode(str, Enum):
    TABLE = "table"
    HAND = "hand"


@dataclass
class Detection:
    color: str
    contour: np.ndarray
    rect: RotatedRect


@dataclass
class DetectResult:
    counts: dict[str, int]
    detections: list[Detection]
    search_mask: np.ndarray | None


@dataclass
class CalibrationResult:
    success: bool
    message: str


class PenDetector:
    MIN_AREA = 150
    MIN_ASPECT_RATIO = 1.8
    MIN_SOLIDITY = 0.35
    MIN_EXTENT = 0.16
    MASK_EXPAND_ITERATIONS = 2
    MASK_KERNEL_SIZE = 15
    FG_THRESHOLD = 20
    NORMALIZED_DIFF_EPSILON = 15.0

    CALIBRATION_FRAME_COUNT = 20
    CALIBRATION_FRAME_MAX = 30
    CALIBRATION_MIN_MS = 500
    CALIBRATION_MAX_MS = 1000
    CALIBRATION_MAX_MOTION = 6.0
    CALIBRATION_MOTION_STD = 8.0
    CALIBRATION_MOTION_FRACTION = 0.05
    CALIBRATION_MAX_FOREGROUND_RATIO = 0.025
    CALIBRATION_MIN_BLOB_AREA = 400

    def __init__(self, mode: DetectionMode = DetectionMode.TABLE) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {MODEL_PATH}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
        self.mode = mode
        self._background: np.ndarray | None = None
        self._noise_level: float | None = None
        self._calibration_frames: list[np.ndarray] = []
        self._is_calibrating = False
        self._calibration_started_at: float | None = None
        self._calibration_message = ""

    @property
    def calibrated(self) -> bool:
        return self._background is not None

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return len(self._calibration_frames), self.CALIBRATION_FRAME_COUNT

    @property
    def calibration_message(self) -> str:
        return self._calibration_message

    @property
    def calibration_warning(self) -> str | None:
        if self.calibrated or self._is_calibrating:
            return None
        if self._calibration_message.startswith("Calibration failed:"):
            return self._calibration_message.removeprefix("Calibration failed: ").strip()
        return None

    def set_mode(self, mode: DetectionMode) -> None:
        self.mode = mode

    def start_calibration(self) -> None:
        self._background = None
        self._noise_level = None
        self._calibration_frames.clear()
        self._is_calibrating = True
        self._calibration_started_at = time.monotonic()
        self._calibration_message = (
            f"Calibrating... hold still (0/{self.CALIBRATION_FRAME_COUNT})"
        )

    def add_calibration_frame(self, frame: np.ndarray) -> CalibrationResult | None:
        if not self._is_calibrating:
            return None

        self._calibration_frames.append(frame.copy())
        current, total = self.calibration_progress
        self._calibration_message = f"Calibrating... hold still ({current}/{total})"

        elapsed_ms = 0.0
        if self._calibration_started_at is not None:
            elapsed_ms = (time.monotonic() - self._calibration_started_at) * 1000

        reached_frames = current >= total
        reached_min_time = elapsed_ms >= self.CALIBRATION_MIN_MS
        timed_out = elapsed_ms >= self.CALIBRATION_MAX_MS
        max_frames = current >= self.CALIBRATION_FRAME_MAX
        if not ((reached_frames and reached_min_time) or timed_out or max_frames):
            return None

        result = self._finalize_calibration()
        self._is_calibrating = False
        self._calibration_started_at = None
        self._calibration_message = result.message
        return result

    def calibrate(self, frame: np.ndarray) -> CalibrationResult:
        """Single-frame calibration for offline/testing use."""
        self._is_calibrating = False
        self._calibration_started_at = None
        self._calibration_frames.clear()
        self._background = frame.copy()
        self._noise_level = self._estimate_noise_level(frame)
        self._calibration_message = "Calibrated (single frame)"
        return CalibrationResult(success=True, message=self._calibration_message)

    def clear_calibration(self) -> None:
        self._background = None
        self._noise_level = None
        self._calibration_frames.clear()
        self._is_calibrating = False
        self._calibration_started_at = None
        self._calibration_message = ""

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame: np.ndarray) -> DetectResult:
        search_mask = self._search_mask(frame)
        if search_mask is None:
            counts = {color: 0 for color in COLOR_ORDER}
            return DetectResult(counts=counts, detections=[], search_mask=None)

        candidates = self._find_pen_candidates(search_mask)
        detections: list[Detection] = []
        counts = {color: 0 for color in COLOR_ORDER}

        for contour, rect in candidates:
            color = self._classify_contour_color(frame, contour)
            detections.append(Detection(color=color, contour=contour, rect=rect))
            counts[color] += 1

        return DetectResult(counts=counts, detections=detections, search_mask=search_mask)

    def _finalize_calibration(self) -> CalibrationResult:
        frames = self._calibration_frames
        if not frames:
            self._background = None
            self._noise_level = None
            return CalibrationResult(
                success=False,
                message="Calibration failed: no frames captured.",
            )

        stack = np.stack(frames, axis=0)
        background = np.median(stack, axis=0).astype(np.uint8)

        gray_stack = np.stack(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames],
            axis=0,
        )
        std_map = np.std(gray_stack.astype(np.float32), axis=0)
        noise_level = float(np.median(std_map))

        motion = self._mean_frame_motion(frames)
        if motion > self.CALIBRATION_MAX_MOTION:
            self._background = None
            self._noise_level = None
            return CalibrationResult(
                success=False,
                message=(
                    "Calibration failed: camera motion detected - hold still and press c again."
                ),
            )

        motion_fraction = float(np.mean(std_map > self.CALIBRATION_MOTION_STD))
        if motion_fraction > self.CALIBRATION_MOTION_FRACTION:
            self._background = None
            self._noise_level = None
            return CalibrationResult(
                success=False,
                message=(
                    "Calibration failed: camera motion detected - hold still and press c again."
                ),
            )

        fg_ratio = self._max_foreground_ratio(frames, background, noise_level)
        if fg_ratio > self.CALIBRATION_MAX_FOREGROUND_RATIO:
            self._background = None
            self._noise_level = None
            return CalibrationResult(
                success=False,
                message=(
                    "Calibration failed: objects still in frame - remove all pens and press c again."
                ),
            )

        if self._median_has_foreground_blobs(background):
            self._background = None
            self._noise_level = None
            return CalibrationResult(
                success=False,
                message=(
                    "Calibration failed: objects still in frame - remove all pens and press c again."
                ),
            )

        self._background = background
        self._noise_level = noise_level
        return CalibrationResult(
            success=True,
            message="Calibration complete. Place pens on the table.",
        )

    @staticmethod
    def _estimate_noise_level(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return float(np.median(cv2.absdiff(gray, blurred)))

    @staticmethod
    def _mean_frame_motion(frames: list[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.0
        motions = [
            float(np.mean(cv2.absdiff(frames[i], frames[i - 1])))
            for i in range(1, len(frames))
        ]
        return float(np.mean(motions))

    def _median_has_foreground_blobs(self, median_bg: np.ndarray) -> bool:
        blurred = cv2.GaussianBlur(median_bg, (51, 51), 0)
        gray = self._normalized_diff_gray(median_bg, blurred)
        mask = self._adaptive_threshold_mask(gray)
        mask = self._cleanup_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= self.CALIBRATION_MIN_BLOB_AREA:
                return True
        return False

    def _normalized_diff_gray(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(frame, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)
        normalized = diff_gray / (bg_gray + self.NORMALIZED_DIFF_EPSILON) * 255.0
        gray = np.clip(normalized, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _noise_floor_threshold(self, noise_level: float | None = None) -> int:
        level = noise_level if noise_level is not None else self._noise_level
        if level is None:
            return self.FG_THRESHOLD
        return max(self.FG_THRESHOLD, int(level * 2.5 + 10))

    def _max_foreground_ratio(
        self,
        frames: list[np.ndarray],
        background: np.ndarray,
        noise_level: float,
    ) -> float:
        noise_floor = self._noise_floor_threshold(noise_level)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        max_ratio = 0.0
        for frame in frames:
            gray = self._normalized_diff_gray(frame, background)
            _, mask = cv2.threshold(gray, noise_floor, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            max_ratio = max(max_ratio, float(np.mean(mask > 0)))
        return max_ratio

    def _search_mask(self, frame: np.ndarray) -> np.ndarray | None:
        if self.mode == DetectionMode.TABLE:
            if self._background is None:
                return None
            return self._foreground_mask(frame)

        return self._hand_mask(frame)

    def _foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        assert self._background is not None
        gray = self._normalized_diff_gray(frame, self._background)
        mask = self._adaptive_threshold_mask(gray)
        return self._cleanup_mask(mask)

    def _adaptive_threshold_mask(self, gray: np.ndarray) -> np.ndarray:
        noise_floor = self._noise_floor_threshold()
        otsu_val, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pct_val = float(np.percentile(gray, 90))

        candidates = [value for value in (otsu_val, pct_val) if value >= noise_floor]
        threshold = int(min(candidates)) if candidates else noise_floor

        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        if np.mean(mask > 0) > 0.3:
            _, mask = cv2.threshold(gray, noise_floor, 255, cv2.THRESH_BINARY)
            return mask

        if 8 < otsu_val < 180 and np.mean(otsu_mask > 0) <= 0.3:
            return otsu_mask

        return mask

    def _hand_mask(self, frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            return None

        height, width = frame.shape[:2]
        landmarks = result.hand_landmarks[0]
        points = np.array(
            [(int(lm.x * width), int(lm.y * height)) for lm in landmarks],
            dtype=np.int32,
        )
        hull = cv2.convexHull(points)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MASK_KERNEL_SIZE, self.MASK_KERNEL_SIZE),
        )
        mask = cv2.dilate(mask, kernel, iterations=self.MASK_EXPAND_ITERATIONS)
        return mask

    def _find_pen_candidates(
        self,
        search_mask: np.ndarray,
    ) -> list[tuple[np.ndarray, RotatedRect]]:
        mask = self._cleanup_mask(search_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[np.ndarray, RotatedRect]] = []
        seen_rects: list[RotatedRect] = []

        for contour in contours:
            if not self._is_pen_like(contour):
                continue
            rect = cv2.minAreaRect(contour)
            if self._overlaps_existing(rect, seen_rects):
                continue
            seen_rects.append(rect)
            candidates.append((contour, rect))

        return candidates

    @staticmethod
    def _classify_contour_color(frame: np.ndarray, contour: np.ndarray) -> str:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return "other"
        roi = frame[y : y + h, x : x + w]
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted], -1, 255, -1)
        return classify_dominant_color(roi, mask)

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _is_pen_like(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        if area < self.MIN_AREA:
            return False

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < self.MIN_ASPECT_RATIO:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = area / hull_area
        if solidity < self.MIN_SOLIDITY:
            return False

        rect_area = width * height
        if rect_area == 0:
            return False
        extent = area / rect_area
        if extent < self.MIN_EXTENT:
            return False

        return True

    def _overlaps_existing(
        self,
        rect: RotatedRect,
        existing: list[RotatedRect],
        iou_threshold: float = 0.3,
    ) -> bool:
        bbox = rect_to_aabb(rect)
        for other in existing:
            if self._iou(bbox, rect_to_aabb(other)) > iou_threshold:
                return True
        return False

    @staticmethod
    def _iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        union_area = aw * ah + bw * bh - inter_area
        return inter_area / union_area
