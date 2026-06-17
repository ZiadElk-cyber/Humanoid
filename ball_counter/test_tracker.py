"""Unit tests for temporal ball tracking."""

import math

from ball_counter.detector import BallDetection
from ball_counter.tracker import BallTracker, CONFIRM_HITS, MAX_MISSES


def _ball(x: int, y: int, r: int = 30) -> BallDetection:
    return BallDetection(center=(x, y), radius=r)


def test_holds_ball_through_brief_miss() -> bool:
    tracker = BallTracker()

    for _ in range(CONFIRM_HITS):
        tracker.update([_ball(100, 100)])

    counts: list[int] = []
    for _ in range(5):
        counts.append(len(tracker.update([])))

    counts.append(len(tracker.update([_ball(102, 98)])))

    ok = all(count == 1 for count in counts)
    print(f"Brief miss hold: counts={counts}")
    print("  PASS" if ok else "  FAIL (expected count 1 throughout)")
    return ok


def test_drops_after_long_miss() -> bool:
    tracker = BallTracker()

    for _ in range(CONFIRM_HITS):
        tracker.update([_ball(200, 200)])

    for _ in range(MAX_MISSES + 3):
        count = len(tracker.update([]))

    ok = count == 0
    print(f"Long miss drop: final count={count}")
    print("  PASS" if ok else "  FAIL (expected 0)")
    return ok


def test_smooths_position() -> bool:
    tracker = BallTracker()

    positions: list[tuple[int, int]] = []
    for x in (100, 130, 160, 190):
        balls = tracker.update([_ball(x, 100)])
        if balls:
            positions.append(balls[0].center)

    ok = len(positions) >= 2 and positions[-1][0] < 190
    print(f"Smoothed positions: {positions}")
    print("  PASS" if ok else "  FAIL (expected lagging smoothed x < 190)")
    return ok


def test_single_ball_circular_motion() -> bool:
    tracker = BallTracker()
    counts: list[int] = []

    for i in range(20):
        angle = i * 0.5
        x = 320 + int(80 * math.cos(angle))
        y = 240 + int(80 * math.sin(angle))
        counts.append(len(tracker.update([_ball(x, y)])))

    ok = max(counts[CONFIRM_HITS:]) == 1 and counts[-1] == 1
    print(f"Circular motion: max_count={max(counts)}, final={counts[-1]}")
    print("  PASS" if ok else "  FAIL (expected count 1 after confirm)")
    return ok


def test_fast_linear_motion() -> bool:
    tracker = BallTracker()
    counts: list[int] = []

    for i in range(10):
        x = 100 + i * 60
        counts.append(len(tracker.update([_ball(x, 200)])))

    ok = max(counts[CONFIRM_HITS:]) == 1 and counts[-1] == 1
    print(f"Fast linear motion: counts={counts}")
    print("  PASS" if ok else "  FAIL (expected count 1)")
    return ok


def main() -> None:
    results = [
        test_holds_ball_through_brief_miss(),
        test_drops_after_long_miss(),
        test_smooths_position(),
        test_single_ball_circular_motion(),
        test_fast_linear_motion(),
    ]

    if all(results):
        print("All tracker tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
