from dataclasses import dataclass

import numpy as np

from ball_counter.detector import BallDetection

CONFIRM_HITS = 2
MAX_MISSES = 12
POSITION_ALPHA = 0.35
VELOCITY_ALPHA = 0.45
PREDICT_MATCH_RATIO = 1.0
RECOVERY_MATCH_RATIO = 1.2


@dataclass
class _Track:
    center: tuple[float, float]
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0


class BallTracker:
    def __init__(self) -> None:
        self._tracks: list[_Track] = []

    def update(self, detections: list[BallDetection]) -> list[BallDetection]:
        matched_tracks: set[int] = set()

        for detection in detections:
            track_idx = self._find_best_match(
                detection,
                matched_tracks,
                match_ratio=PREDICT_MATCH_RATIO,
            )
            if track_idx is None:
                track_idx = self._find_best_match(
                    detection,
                    matched_tracks,
                    match_ratio=RECOVERY_MATCH_RATIO,
                )
            if track_idx is None:
                self._tracks.append(
                    _Track(
                        center=(float(detection.center[0]), float(detection.center[1])),
                        radius=float(detection.radius),
                        hits=1,
                        misses=0,
                    )
                )
                continue

            matched_tracks.add(track_idx)
            self._update_track(self._tracks[track_idx], detection)

        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.center = (track.center[0] + track.vx, track.center[1] + track.vy)
                track.misses += 1

        self._tracks = [track for track in self._tracks if track.misses <= MAX_MISSES]
        return self._confirmed_balls()

    def _update_track(self, track: _Track, detection: BallDetection) -> None:
        tcx, tcy = track.center
        dcx, dcy = detection.center
        dx = dcx - tcx
        dy = dcy - tcy
        track.vx = (1 - VELOCITY_ALPHA) * track.vx + VELOCITY_ALPHA * dx
        track.vy = (1 - VELOCITY_ALPHA) * track.vy + VELOCITY_ALPHA * dy

        alpha = POSITION_ALPHA
        track.center = (
            (1 - alpha) * tcx + alpha * dcx,
            (1 - alpha) * tcy + alpha * dcy,
        )
        track.radius = (1 - alpha) * track.radius + alpha * detection.radius
        track.misses = 0
        track.hits += 1

    def _find_best_match(
        self,
        detection: BallDetection,
        matched_tracks: set[int],
        *,
        match_ratio: float,
    ) -> int | None:
        best_idx: int | None = None
        best_distance = float("inf")

        for idx, track in enumerate(self._tracks):
            if idx in matched_tracks:
                continue

            distance = self._match_distance(detection, track)
            min_sep = match_ratio * (detection.radius + track.radius)
            if distance <= min_sep and distance < best_distance:
                best_distance = distance
                best_idx = idx

        return best_idx

    @staticmethod
    def _match_distance(detection: BallDetection, track: _Track) -> float:
        px = track.center[0] + track.vx
        py = track.center[1] + track.vy
        dist_predicted = np.hypot(detection.center[0] - px, detection.center[1] - py)
        dist_current = np.hypot(
            detection.center[0] - track.center[0],
            detection.center[1] - track.center[1],
        )
        return float(min(dist_current, dist_predicted))

    def _confirmed_balls(self) -> list[BallDetection]:
        confirmed: list[BallDetection] = []
        for track in self._tracks:
            if track.hits < CONFIRM_HITS:
                continue
            confirmed.append(
                BallDetection(
                    center=(int(round(track.center[0])), int(round(track.center[1]))),
                    radius=max(1, int(round(track.radius))),
                )
            )
        return confirmed
