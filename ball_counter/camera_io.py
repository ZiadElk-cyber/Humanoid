import sys

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            _apply_capture_settings(cap)
            return cap
        cap.release()

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. "
            "Check that both webcams are connected or try different camera indices."
        )
    _apply_capture_settings(cap)
    return cap


def _apply_capture_settings(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
