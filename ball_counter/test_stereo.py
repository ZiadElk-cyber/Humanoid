"""Unit tests for stereo correspondence fusion."""

from ball_counter.detector import BallDetection
from ball_counter.stereo import MAX_DISPARITY, StereoFusion


def _ball(x: int, y: int, r: int = 30) -> BallDetection:
    return BallDetection(center=(x, y), radius=r)


def test_matches_valid_pair() -> bool:
    fusion = StereoFusion()
    fused = fusion.fuse([_ball(200, 120)], [_ball(170, 118)])

    ok = len(fused) == 1 and fused[0].center == (200, 120)
    print(f"Valid pair: fused={len(fused)}, center={fused[0].center if fused else None}")
    print("  PASS" if ok else "  FAIL (expected 1 fused ball at left position)")
    return ok


def test_rejects_left_only() -> bool:
    fusion = StereoFusion()
    fused = fusion.fuse([_ball(200, 120)], [])

    ok = len(fused) == 0
    print(f"Left only: fused={len(fused)}")
    print("  PASS" if ok else "  FAIL (expected 0 fused)")
    return ok


def test_rejects_bad_disparity() -> bool:
    fusion = StereoFusion()
    too_small = fusion.fuse([_ball(200, 120)], [_ball(198, 120)])
    too_large = fusion.fuse(
        [_ball(200, 120)],
        [_ball(200 - MAX_DISPARITY - 10, 120)],
    )

    ok = len(too_small) == 0 and len(too_large) == 0
    print(f"Bad disparity: small={len(too_small)}, large={len(too_large)}")
    print("  PASS" if ok else "  FAIL (expected 0 fused for out-of-range disparity)")
    return ok


def test_two_balls_both_views() -> bool:
    fusion = StereoFusion()
    left = [_ball(150, 100), _ball(350, 200)]
    right = [_ball(120, 102), _ball(320, 198)]
    fused = fusion.fuse(left, right)

    ok = len(fused) == 2 and len(fusion.last_pairs) == 2
    print(f"Two balls: fused={len(fused)}, pairs={len(fusion.last_pairs)}")
    print("  PASS" if ok else "  FAIL (expected 2 fused balls)")
    return ok


def main() -> None:
    results = [
        test_matches_valid_pair(),
        test_rejects_left_only(),
        test_rejects_bad_disparity(),
        test_two_balls_both_views(),
    ]

    if all(results):
        print("All stereo tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
