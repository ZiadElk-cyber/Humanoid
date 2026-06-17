import cv2

from ball_counter.detector import RedBallDetector
from ball_counter.overlay import draw_camera_view, draw_info_panel
from ball_counter.tracker import BallTracker

CAMERA_WINDOW = "Red Ball Counter - Camera"
INFO_WINDOW = "Red Ball Counter - Info"
CAMERA_INDEX = 1


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. "
            "Check that a webcam is connected or try a different CAMERA_INDEX."
        )

    detector = RedBallDetector()
    tracker = BallTracker()
    debug_mode = False

    cv2.namedWindow(CAMERA_WINDOW)
    cv2.namedWindow(INFO_WINDOW, cv2.WINDOW_AUTOSIZE)

    print(f"Red ball counter started (camera {CAMERA_INDEX}).")
    print("Point the camera at red balls. Press 'd' for debug view, 'q' to quit.")

    info_window_positioned = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            result = detector.detect(frame)
            stable_balls = tracker.update(result.balls)
            camera_display = draw_camera_view(
                frame,
                stable_balls,
                debug=debug_mode,
                search_mask=result.search_mask,
            )
            info_display = draw_info_panel(len(stable_balls), debug=debug_mode)

            cv2.imshow(CAMERA_WINDOW, camera_display)
            cv2.imshow(INFO_WINDOW, info_display)

            if not info_window_positioned:
                cv2.moveWindow(INFO_WINDOW, camera_display.shape[1] + 20, 0)
                info_window_positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_mode = not debug_mode
                print(f"Debug view: {'ON' if debug_mode else 'OFF'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
