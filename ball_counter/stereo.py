from dataclasses import dataclass

from ball_counter.detector import BallDetection

Y_TOLERANCE = 18
MIN_DISPARITY = 8
MAX_DISPARITY = 180
RADIUS_TOLERANCE = 0.35


@dataclass
class _CandidateMatch:
    left_idx: int
    right_idx: int
    error: float


class StereoFusion:
    def __init__(self) -> None:
        self.last_pairs: list[tuple[BallDetection, BallDetection]] = []

    def fuse(
        self,
        left: list[BallDetection],
        right: list[BallDetection],
    ) -> list[BallDetection]:
        candidates = self._enumerate_candidates(left, right)
        candidates.sort(key=lambda match: match.error)

        used_left: set[int] = set()
        used_right: set[int] = set()
        pairs: list[tuple[BallDetection, BallDetection]] = []

        for match in candidates:
            if match.left_idx in used_left or match.right_idx in used_right:
                continue
            used_left.add(match.left_idx)
            used_right.add(match.right_idx)
            pairs.append((left[match.left_idx], right[match.right_idx]))

        self.last_pairs = pairs
        return [left_ball for left_ball, _ in pairs]

    def _enumerate_candidates(
        self,
        left: list[BallDetection],
        right: list[BallDetection],
    ) -> list[_CandidateMatch]:
        candidates: list[_CandidateMatch] = []

        for left_idx, left_ball in enumerate(left):
            for right_idx, right_ball in enumerate(right):
                error = self._match_error(left_ball, right_ball)
                if error is None:
                    continue
                candidates.append(
                    _CandidateMatch(left_idx=left_idx, right_idx=right_idx, error=error)
                )

        return candidates

    @staticmethod
    def _match_error(left: BallDetection, right: BallDetection) -> float | None:
        lx, ly = left.center
        rx, ry = right.center

        if abs(ly - ry) > Y_TOLERANCE:
            return None

        disparity = lx - rx
        if disparity < MIN_DISPARITY or disparity > MAX_DISPARITY:
            return None

        max_radius = max(left.radius, right.radius, 1)
        if abs(left.radius - right.radius) / max_radius > RADIUS_TOLERANCE:
            return None

        row_error = abs(ly - ry) / max(Y_TOLERANCE, 1)
        radius_error = abs(left.radius - right.radius) / max_radius
        disparity_mid = (MIN_DISPARITY + MAX_DISPARITY) / 2
        disparity_error = abs(disparity - disparity_mid) / max(
            disparity_mid - MIN_DISPARITY, 1
        )
        return row_error + radius_error + disparity_error
