import cv2
import numpy as np

from pen_counter.colors import COLOR_ORDER, DISPLAY_COLORS
from pen_counter.detector import Detection, DetectionMode, rect_to_box_points


HELP_LINES = [
    "Remove all pens, hold camera still, press c",
    "Place pens spread apart | h = hand mode | q = quit",
]


def draw_overlay(
    frame: np.ndarray,
    counts: dict[str, int],
    detections: list[Detection],
    *,
    mode: DetectionMode = DetectionMode.TABLE,
    calibrated: bool = False,
    calibrating: bool = False,
    calibration_progress: tuple[int, int] | None = None,
    calibration_warning: str | None = None,
    debug: bool = False,
    search_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = frame.copy()

    if debug and search_mask is not None:
        tint = np.zeros_like(output)
        tint[:, :, 1] = search_mask
        output = cv2.addWeighted(output, 0.7, tint, 0.3, 0)

    for detection in detections:
        color = DISPLAY_COLORS.get(detection.color, (255, 255, 255))
        box = rect_to_box_points(detection.rect)
        cv2.drawContours(output, [box], 0, color, 2)
        label = detection.color
        label_x = int(np.min(box[:, 0]))
        label_y = int(np.min(box[:, 1]))
        cv2.putText(
            output,
            label,
            (label_x, max(label_y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_legend(output, counts, debug=debug)
    _draw_status(
        output,
        mode=mode,
        calibrated=calibrated,
        calibrating=calibrating,
        calibration_progress=calibration_progress,
        calibration_warning=calibration_warning,
        debug=debug,
    )
    if calibrating and calibration_progress is not None:
        current, total = calibration_progress
        _draw_centered_banner(
            output,
            f"Calibrating... hold still ({current}/{total})",
            (0, 180, 255),
        )
    elif calibration_warning:
        _draw_centered_banner(output, calibration_warning, (0, 0, 255))
    _draw_help(output)
    return output


def _draw_status(
    frame: np.ndarray,
    *,
    mode: DetectionMode,
    calibrated: bool,
    calibrating: bool,
    calibration_progress: tuple[int, int] | None,
    calibration_warning: str | None,
    debug: bool,
) -> None:
    if calibrating and calibration_progress is not None:
        current, total = calibration_progress
        status = f"Calibrating... hold still ({current}/{total})"
        text_color = (255, 255, 255)
    elif calibration_warning:
        status = f"Calibration failed: {calibration_warning}"
        text_color = (0, 0, 255)
    elif mode == DetectionMode.TABLE and not calibrated:
        status = "Not calibrated - remove pens, hold still, press c"
        text_color = (255, 255, 255)
    elif mode == DetectionMode.TABLE:
        status = "mode: table | calibrated: yes"
        text_color = (255, 255, 255)
    else:
        status = "mode: hand"
        text_color = (255, 255, 255)

    if debug:
        status += " | debug"

    y = frame.shape[0] - 56
    cv2.putText(
        frame,
        status,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_legend(frame: np.ndarray, counts: dict[str, int], *, debug: bool) -> None:
    x0, y0 = 10, 10
    line_height = 24
    title = "Pen counts (debug)" if debug else "Pen counts"
    cv2.putText(
        frame,
        title,
        (x0, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        title,
        (x0, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    total = 0
    for i, color_name in enumerate(COLOR_ORDER):
        count = counts.get(color_name, 0)
        total += count
        y = y0 + 36 + i * line_height
        swatch = DISPLAY_COLORS[color_name]
        cv2.rectangle(frame, (x0, y - 12), (x0 + 14, y + 2), swatch, -1)
        text = f"{color_name}: {count}"
        cv2.putText(
            frame,
            text,
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    y_total = y0 + 36 + len(COLOR_ORDER) * line_height + 4
    total_text = f"total: {total}"
    cv2.putText(
        frame,
        total_text,
        (x0, y_total),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        total_text,
        (x0, y_total),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_centered_banner(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (frame.shape[1] - text_w) // 2)
    y = frame.shape[0] // 2
    cv2.rectangle(
        frame,
        (x - 12, y - text_h - 12),
        (x + text_w + 12, y + 12),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


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
