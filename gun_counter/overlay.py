import cv2
import numpy as np

from gun_counter.detector import GunFlexResult

GUN_COLOR = (0, 200, 255)
ARM_COLOR = (0, 255, 0)
INACTIVE_COLOR = (120, 120, 120)

HELP_LINES = [
    "Flex your biceps at the camera",
    "q = quit",
]


def draw_overlay(frame: np.ndarray, result: GunFlexResult) -> np.ndarray:
    output = frame.copy()

    if result.landmarks is not None:
        _draw_pose_skeleton(output, result)

    for gun in result.guns:
        cv2.line(output, gun.shoulder_px, gun.elbow_px, GUN_COLOR, 4)
        cv2.line(output, gun.elbow_px, gun.wrist_px, GUN_COLOR, 4)
        cv2.circle(output, gun.elbow_px, 8, GUN_COLOR, -1)
        label = f"gun ({gun.side})"
        cv2.putText(
            output,
            label,
            (gun.elbow_px[0] + 10, gun.elbow_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GUN_COLOR,
            2,
            cv2.LINE_AA,
        )

    _draw_count(output, result.count)
    _draw_help(output)
    return output


def _draw_pose_skeleton(frame: np.ndarray, result: GunFlexResult) -> None:
    flex_sides = {gun.side for gun in result.guns}
    height, width = frame.shape[:2]

    arm_segments = [
        ("left", 11, 13),
        ("left", 13, 15),
        ("right", 12, 14),
        ("right", 14, 16),
    ]
    landmarks = result.landmarks
    for side, start_idx, end_idx in arm_segments:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        if start.visibility < 0.5 or end.visibility < 0.5:
            continue
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        color = GUN_COLOR if side in flex_sides else INACTIVE_COLOR
        thickness = 3 if side in flex_sides else 1
        cv2.line(frame, p1, p2, color, thickness)


def _draw_count(frame: np.ndarray, count: int) -> None:
    text = f"guns: {count}"
    cv2.putText(
        frame,
        text,
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_help(frame: np.ndarray) -> None:
    height = frame.shape[0]
    for i, line in enumerate(HELP_LINES):
        y = height - 16 - (len(HELP_LINES) - 1 - i) * 20
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
