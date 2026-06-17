import cv2

from ball_counter.camera_io import open_camera
from ball_counter.detector import RedBallDetector
from ball_counter.overlay import draw_info_panel, draw_stereo_view
from ball_counter.stereo import StereoFusion
from ball_counter.tracker import BallTracker

STEREO_WINDOW = "Red Ball Counter - Stereo"
INFO_WINDOW = "Red Ball Counter - Info"
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 1


def _align_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height))


def main() -> None:
    cap_left = open_camera(LEFT_CAMERA_INDEX)
    cap_right = open_camera(RIGHT_CAMERA_INDEX)

    detector = RedBallDetector()
    stereo = StereoFusion()
    tracker = BallTracker()
    debug_mode = False

    cv2.namedWindow(STEREO_WINDOW)
    cv2.namedWindow(INFO_WINDOW, cv2.WINDOW_AUTOSIZE)

    print(
        f"Red ball counter started (cameras {LEFT_CAMERA_INDEX} left, "
        f"{RIGHT_CAMERA_INDEX} right)."
    )
    print("Point both cameras at red balls. Press 'd' for debug view, 'q' to quit.")

    info_window_positioned = False

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            if not ret_l or not ret_r:
                print("Failed to read frame from one or both cameras.")
                break

            height, width = frame_l.shape[:2]
            frame_r = _align_frame(frame_r, width, height)

            left_result = detector.detect(frame_l)
            right_result = detector.detect(frame_r)
            fused = stereo.fuse(left_result.balls, right_result.balls)
            stable_balls = tracker.update(fused)

            stereo_display = draw_stereo_view(
                frame_l,
                frame_r,
                left_result.balls,
                right_result.balls,
                stable_balls,
                stereo.last_pairs,
                debug=debug_mode,
                search_mask_l=left_result.search_mask,
                search_mask_r=right_result.search_mask,
            )
            info_display = draw_info_panel(
                len(stable_balls),
                debug=debug_mode,
                stereo=True,
                left_count=len(left_result.balls),
                right_count=len(right_result.balls),
                fused_count=len(fused),
            )

            cv2.imshow(STEREO_WINDOW, stereo_display)
            cv2.imshow(INFO_WINDOW, info_display)

            if not info_window_positioned:
                cv2.moveWindow(INFO_WINDOW, stereo_display.shape[1] + 20, 0)
                info_window_positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_mode = not debug_mode
                print(f"Debug view: {'ON' if debug_mode else 'OFF'}")
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
