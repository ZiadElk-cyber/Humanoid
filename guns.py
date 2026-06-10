import cv2

from gun_counter.detector import GunFlexDetector
from gun_counter.overlay import draw_overlay

WINDOW_NAME = "Gun Counter"


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check that a webcam is connected.")

    detector = GunFlexDetector()

    print("Gun counter started.")
    print("Flex your biceps at the camera. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            result = detector.detect(frame)
            display = draw_overlay(frame, result)
            cv2.imshow(WINDOW_NAME, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
