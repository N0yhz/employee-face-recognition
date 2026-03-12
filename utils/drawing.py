"""OpenCV drawing utilities for bounding boxes, labels and FPS overlay."""

from typing import List

import cv2
import numpy as np

# Colours (BGR)
_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_YELLOW = (0, 200, 220)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.6
_FONT_SCALE_FPS = 0.8
_THICKNESS = 2


def draw_results(frame: np.ndarray, results: List[dict]) -> np.ndarray:
    """Draw bounding boxes and identity labels on *frame* in-place.

    Recognised employees receive a **green** bounding box with their name and
    cosine similarity score.  Unrecognised faces receive a **red** box with
    the label ``"Unknown"``.

    Args:
        frame: BGR image as a NumPy array (H×W×3).  Modified in-place.
        results: List of result dicts as returned by
            :meth:`RecognitionPipeline.process_frame`.

    Returns:
        The same *frame* array with annotations applied.
    """
    for res in results:
        x1, y1, x2, y2 = res["bbox"]
        recognized: bool = res.get("recognized", False)
        name: str = res.get("name", "Unknown")
        similarity: float = res.get("similarity", 0.0)

        colour = _GREEN if recognized else _RED

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, _THICKNESS)

        # Label text
        if recognized:
            label = f"{name}  {similarity:.2f}"
        else:
            label = "Unknown"

        # Background pill for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE_LABEL, _THICKNESS
        )
        label_y = max(y1 - 6, text_h + baseline)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - baseline),
            (x1 + text_w + 4, label_y + baseline),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y),
            _FONT,
            _FONT_SCALE_LABEL,
            _WHITE,
            _THICKNESS,
            cv2.LINE_AA,
        )

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw an FPS counter in the top-left corner of *frame* in-place.

    Args:
        frame: BGR image as a NumPy array (H×W×3).  Modified in-place.
        fps: Current frames-per-second value to display.

    Returns:
        The same *frame* array with the FPS overlay applied.
    """
    label = f"FPS: {fps:.1f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE_FPS, _THICKNESS
    )
    # Semi-transparent black background
    cv2.rectangle(
        frame,
        (8, 8),
        (14 + text_w, 14 + text_h + baseline),
        _BLACK,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (10, 10 + text_h),
        _FONT,
        _FONT_SCALE_FPS,
        _YELLOW,
        _THICKNESS,
        cv2.LINE_AA,
    )
    return frame
