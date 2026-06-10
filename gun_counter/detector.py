from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"

# MediaPipe pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

MIN_VISIBILITY = 0.5
MAX_FLEX_ANGLE_DEG = 85.0


@dataclass
class GunFlex:
    side: str
    elbow_px: tuple[int, int]
    shoulder_px: tuple[int, int]
    wrist_px: tuple[int, int]
    angle_deg: float


@dataclass
class GunFlexResult:
    count: int
    guns: list[GunFlex]
    landmarks: list | None


class GunFlexDetector:
    def __init__(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Pose landmarker model not found at {MODEL_PATH}. "
                "Download pose_landmarker_lite.task into gun_counter/models/."
            )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame: np.ndarray) -> GunFlexResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.pose_landmarks:
            return GunFlexResult(count=0, guns=[], landmarks=None)

        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks[0]
        guns: list[GunFlex] = []

        left = self._check_arm(
            landmarks,
            width,
            height,
            side="left",
            shoulder_idx=LEFT_SHOULDER,
            elbow_idx=LEFT_ELBOW,
            wrist_idx=LEFT_WRIST,
        )
        if left is not None:
            guns.append(left)

        right = self._check_arm(
            landmarks,
            width,
            height,
            side="right",
            shoulder_idx=RIGHT_SHOULDER,
            elbow_idx=RIGHT_ELBOW,
            wrist_idx=RIGHT_WRIST,
        )
        if right is not None:
            guns.append(right)

        return GunFlexResult(count=len(guns), guns=guns, landmarks=landmarks)

    def _check_arm(
        self,
        landmarks,
        width: int,
        height: int,
        *,
        side: str,
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
    ) -> GunFlex | None:
        shoulder = landmarks[shoulder_idx]
        elbow = landmarks[elbow_idx]
        wrist = landmarks[wrist_idx]

        if min(shoulder.visibility, elbow.visibility, wrist.visibility) < MIN_VISIBILITY:
            return None

        shoulder_px = self._to_pixel(shoulder, width, height)
        elbow_px = self._to_pixel(elbow, width, height)
        wrist_px = self._to_pixel(wrist, width, height)

        angle = self._elbow_angle(shoulder_px, elbow_px, wrist_px)
        if angle > MAX_FLEX_ANGLE_DEG:
            return None

        return GunFlex(
            side=side,
            elbow_px=elbow_px,
            shoulder_px=shoulder_px,
            wrist_px=wrist_px,
            angle_deg=angle,
        )

    @staticmethod
    def _to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _elbow_angle(
        shoulder: tuple[int, int],
        elbow: tuple[int, int],
        wrist: tuple[int, int],
    ) -> float:
        a = np.array(shoulder, dtype=np.float32)
        b = np.array(elbow, dtype=np.float32)
        c = np.array(wrist, dtype=np.float32)

        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 180.0

        cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
