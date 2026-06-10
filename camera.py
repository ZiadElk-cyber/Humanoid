import cv2

from pen_counter.colors import COLOR_ORDER
from pen_counter.detector import DetectionMode, DetectResult, PenDetector
from pen_counter.overlay import draw_overlay

_EMPTY_RESULT = DetectResult(
    counts={color: 0 for color in COLOR_ORDER},
    detections=[],
    search_mask=None,
)

WINDOW_NAME = "Pen Counter"


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check that a webcam is connected.")

    detector = PenDetector(mode=DetectionMode.TABLE)
    debug_mode = False

    print("Pen counter started (table mode).")
    print("Remove all pens, hold camera still, press 'c' to calibrate (~1s).")
    print("Press 'h' to switch hand/table mode, 'd' for debug view, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            if detector.is_calibrating:
                cal_result = detector.add_calibration_frame(frame)
                if cal_result is not None:
                    print(cal_result.message)
                detect_result = _EMPTY_RESULT
            else:
                detect_result = detector.detect(frame)

            current, total = detector.calibration_progress
            display = draw_overlay(
                frame,
                detect_result.counts,
                detect_result.detections,
                mode=detector.mode,
                calibrated=detector.calibrated,
                calibrating=detector.is_calibrating,
                calibration_progress=(current, total) if detector.is_calibrating else None,
                calibration_warning=detector.calibration_warning,
                debug=debug_mode,
                search_mask=detect_result.search_mask,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and detector.mode == DetectionMode.TABLE and not detector.is_calibrating:
                detector.start_calibration()
                print("Calibrating... remove all pens and hold the camera still.")
            if key == ord("h"):
                if detector.mode == DetectionMode.TABLE:
                    detector.set_mode(DetectionMode.HAND)
                    detector.clear_calibration()
                    print("Switched to hand mode.")
                else:
                    detector.set_mode(DetectionMode.TABLE)
                    detector.clear_calibration()
                    print("Switched to table mode. Remove pens and press 'c' to calibrate.")
            if key == ord("d"):
                debug_mode = not debug_mode
                print(f"Debug view: {'ON' if debug_mode else 'OFF'}")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
