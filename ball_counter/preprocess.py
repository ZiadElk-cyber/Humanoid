import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def gray_world_white_balance(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame.astype(np.float32))
    mean_b = max(float(b.mean()), 1.0)
    mean_g = max(float(g.mean()), 1.0)
    mean_r = max(float(r.mean()), 1.0)
    gray = (mean_b + mean_g + mean_r) / 3.0

    b = np.clip(b * (gray / mean_b), 0, 255)
    g = np.clip(g * (gray / mean_g), 0, 255)
    r = np.clip(r * (gray / mean_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def enhance_hsv_saturation_value(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = _CLAHE.apply(s)
    v = _CLAHE.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(frame)
    return enhance_hsv_saturation_value(balanced)
