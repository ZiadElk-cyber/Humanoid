import cv2
import numpy as np

from ball_counter.detector import BallDetection

HELP_LINES = [
    "d = debug | q = quit",
]

PANEL_BG = (30, 30, 30)
PANEL_PAD = 12
LINE_GAP = 8
INFO_PANEL_WIDTH = 400
INFO_PANEL_HEIGHT = 320

TEXT_BODY = (235, 235, 235)
TEXT_TITLE = (255, 255, 255)
TEXT_COUNT = (80, 80, 255)
TEXT_MUTED = (180, 180, 180)

BALL_OUTLINE = (0, 0, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_camera_view(
    frame: np.ndarray,
    balls: list[BallDetection],
    *,
    debug: bool = False,
    search_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = frame.copy()

    if debug and search_mask is not None:
        tint = np.zeros_like(output)
        tint[:, :, 1] = search_mask
        output = cv2.addWeighted(output, 0.7, tint, 0.3, 0)

    for ball in balls:
        cv2.circle(output, ball.center, ball.radius, BALL_OUTLINE, 2)
        cv2.circle(output, ball.center, 3, BALL_OUTLINE, -1)

    return output


def draw_info_panel(
    count: int,
    *,
    debug: bool = False,
    width: int = INFO_PANEL_WIDTH,
    height: int = INFO_PANEL_HEIGHT,
) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    text_width = width - 2 * PANEL_PAD
    y = PANEL_PAD

    y = _draw_text_block(
        panel,
        ["Red Ball Counter"],
        x=PANEL_PAD,
        y=y,
        scale=0.75,
        color=TEXT_TITLE,
        thickness=2,
        line_gap=0,
    )
    y += LINE_GAP + 8

    count_text = f"Red balls: {count}"
    _draw_text(panel, count_text, (PANEL_PAD, y + 36), FONT, 1.0, TEXT_COUNT, 2)
    y += 56

    y = _draw_divider(panel, y)
    y += LINE_GAP

    status = "Ready — point camera at balls"
    if debug:
        status += " | debug"
    y = _draw_text_block(
        panel,
        _wrap_text(status, text_width, FONT, 0.5),
        x=PANEL_PAD,
        y=y,
        scale=0.5,
        color=TEXT_BODY,
        thickness=1,
    )
    y += LINE_GAP + 4

    help_lines: list[str] = []
    for line in HELP_LINES:
        help_lines.extend(_wrap_text(line, text_width, FONT, 0.48))
    _draw_text_block(
        panel,
        help_lines,
        x=PANEL_PAD,
        y=y,
        scale=0.48,
        color=TEXT_MUTED,
        thickness=1,
    )
    return panel


def _draw_divider(frame: np.ndarray, y: int) -> int:
    cv2.line(
        frame,
        (PANEL_PAD, y),
        (frame.shape[1] - PANEL_PAD, y),
        (70, 70, 70),
        1,
    )
    return y + 1


def _draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    *,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    line_gap: int = LINE_GAP,
) -> int:
    line_height = _line_height(FONT, scale)
    for line in lines:
        _draw_text(frame, line, (x, y + line_height - 4), FONT, scale, color, thickness)
        y += line_height + line_gap
    return y


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _line_height(font: int, scale: float) -> int:
    (_, text_h), baseline = cv2.getTextSize("Ag", font, scale, 1)
    return text_h + baseline + 2


def _wrap_text(text: str, max_width: int, font: int, scale: float) -> list[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (width, _), _ = cv2.getTextSize(candidate, font, scale, 1)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
