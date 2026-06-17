from dataclasses import dataclass

import cv2
import numpy as np

from ball_counter.preprocess import normalize_frame

MASK_SATURATION_MIN = 70
MASK_VALUE_MIN = 55
MASK_RED_DOMINANCE_MARGIN = 20

RED_LOWER_1 = (0, MASK_SATURATION_MIN, MASK_VALUE_MIN)
RED_UPPER_1 = (8, 255, 255)
RED_LOWER_2 = (172, MASK_SATURATION_MIN, MASK_VALUE_MIN)
RED_UPPER_2 = (179, 255, 255)

RED_DOMINANCE_MARGIN = 28
MASK_FILL_RATIO = 0.72
MASK_EXTENT_RATIO = 0.80
MIN_MEAN_SATURATION = 110
MIN_MEAN_VALUE = 100
MIN_RED_GREEN_GAP = 50
CORE_RADIUS_RATIO = 0.55
MIN_CONTOUR_CIRCULARITY = 0.70
DEFAULT_REFERENCE_RADIUS = 30
MERGED_BLOB_AREA_FACTOR = 1.8
LOW_CIRCULARITY_THRESHOLD = 0.55


@dataclass
class BallDetection:
    center: tuple[int, int]
    radius: int


@dataclass
class DetectResult:
    count: int
    balls: list[BallDetection]
    search_mask: np.ndarray | None


class RedBallDetector:
    MIN_AREA = 400
    MAX_AREA = 50_000
    DEDUP_DISTANCE_RATIO = 0.6
    MIN_WATERSHED_AREA = 200

    def detect(self, frame: np.ndarray) -> DetectResult:
        normalized = normalize_frame(frame)
        mask = self._red_mask(normalized)
        candidates = self._find_ball_candidates(normalized, mask)
        balls = self._deduplicate(candidates)
        return DetectResult(count=len(balls), balls=balls, search_mask=mask)

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array(RED_LOWER_1, dtype=np.uint8), np.array(RED_UPPER_1, dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array(RED_LOWER_2, dtype=np.uint8), np.array(RED_UPPER_2, dtype=np.uint8))
        hsv_mask = cv2.bitwise_or(mask1, mask2)

        b, g, r = cv2.split(frame)
        red_dominant = (
            (r.astype(np.int16) > g.astype(np.int16) + MASK_RED_DOMINANCE_MARGIN)
            & (r.astype(np.int16) > b.astype(np.int16) + MASK_RED_DOMINANCE_MARGIN)
        ).astype(np.uint8) * 255

        mask = cv2.bitwise_and(hsv_mask, red_dominant)
        return self._cleanup_mask(mask)

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _find_ball_candidates(self, frame: np.ndarray, mask: np.ndarray) -> list[BallDetection]:
        balls = self._find_circles_hough(frame, mask)
        balls.extend(self._split_merged_blobs(frame, mask, balls))
        return balls

    def _find_circles_hough(self, frame: np.ndarray, mask: np.ndarray) -> list[BallDetection]:
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=50,
            param2=26,
            minRadius=14,
            maxRadius=120,
        )
        if circles is None:
            return []

        balls: list[BallDetection] = []
        for cx, cy, radius in np.round(circles[0]).astype(int):
            if self._validate_circle(mask, frame, int(cx), int(cy), int(radius)):
                balls.append(BallDetection(center=(int(cx), int(cy)), radius=int(radius)))
        return balls

    def _split_merged_blobs(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        existing: list[BallDetection],
    ) -> list[BallDetection]:
        ref_radius = DEFAULT_REFERENCE_RADIUS
        single_ball_area = np.pi * ref_radius * ref_radius

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extra: list[BallDetection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if area <= MERGED_BLOB_AREA_FACTOR * single_ball_area and circularity >= LOW_CIRCULARITY_THRESHOLD:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(mask.shape[1], x + w + pad)
            y2 = min(mask.shape[0], y + h + pad)
            roi_mask = mask[y1:y2, x1:x2]

            for ball in self._watershed_circles(frame, mask, roi_mask, x1, y1):
                if any(self._centers_overlap(ball, other) for other in existing):
                    continue
                if any(self._centers_overlap(ball, other) for other in extra):
                    continue
                extra.append(ball)

        return extra

    def _watershed_circles(
        self,
        frame: np.ndarray,
        full_mask: np.ndarray,
        roi_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[BallDetection]:
        if cv2.countNonZero(roi_mask) == 0:
            return []

        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []

        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(roi_mask, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_input = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_input, markers)

        balls: list[BallDetection] = []
        for label in range(2, int(markers.max()) + 1):
            label_mask = np.uint8(markers == label) * 255
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.MIN_WATERSHED_AREA:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                if circularity < MIN_CONTOUR_CIRCULARITY:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                cx = int(cx) + offset_x
                cy = int(cy) + offset_y
                radius = max(int(radius), 1)
                if self._validate_circle(full_mask, frame, cx, cy, radius):
                    balls.append(BallDetection(center=(cx, cy), radius=radius))
        return balls

    @staticmethod
    def _ball_core_mask(
        height: int,
        width: int,
        cx: int,
        cy: int,
        radius: int,
    ) -> np.ndarray:
        core_radius = max(1, int(radius * CORE_RADIUS_RATIO))
        core_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(core_mask, (cx, cy), core_radius, 255, -1)
        return core_mask > 0

    def _validate_circle(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> bool:
        if radius < 1:
            return False

        area = np.pi * radius * radius
        if area < self.MIN_AREA or area > self.MAX_AREA:
            return False

        height, width = mask.shape[:2]
        if cx - radius < 0 or cy - radius < 0 or cx + radius >= width or cy + radius >= height:
            return False

        circle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

        inside = circle_mask > 0
        mask_pixels = int(np.count_nonzero(mask[inside]))
        total_pixels = int(np.count_nonzero(inside))
        if total_pixels == 0 or mask_pixels / total_pixels < MASK_FILL_RATIO:
            return False

        core_inside = self._ball_core_mask(height, width, cx, cy, radius)
        if not np.any(core_inside):
            return False

        core_roi = frame[core_inside]
        mean_b, mean_g, mean_r = core_roi.mean(axis=0)
        if mean_r <= mean_g + RED_DOMINANCE_MARGIN or mean_r <= mean_b + RED_DOMINANCE_MARGIN:
            return False
        if mean_r - mean_g < MIN_RED_GREEN_GAP:
            return False

        hsv_core = cv2.cvtColor(core_roi.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        if float(np.mean(hsv_core[:, 1])) < MIN_MEAN_SATURATION:
            return False
        if float(np.mean(hsv_core[:, 2])) < MIN_MEAN_VALUE:
            return False

        masked_region = cv2.bitwise_and(mask, mask, mask=circle_mask)
        contours, _ = cv2.findContours(masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        if contour_area / area < MASK_EXTENT_RATIO:
            return False

        perimeter = cv2.arcLength(largest, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
        if circularity < MIN_CONTOUR_CIRCULARITY:
            return False

        return True

    def _deduplicate(self, candidates: list[BallDetection]) -> list[BallDetection]:
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda ball: ball.radius, reverse=True)
        kept: list[BallDetection] = []

        for ball in ordered:
            if any(self._centers_overlap(ball, other) for other in kept):
                continue
            kept.append(ball)

        return kept

    def _centers_overlap(self, a: BallDetection, b: BallDetection) -> bool:
        ax, ay = a.center
        bx, by = b.center
        distance = np.hypot(ax - bx, ay - by)
        min_sep = self.DEDUP_DISTANCE_RATIO * (a.radius + b.radius)
        return distance < min_sep
