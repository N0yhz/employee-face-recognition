"""YOLOv8-based face / person detector."""

from typing import List

import numpy as np


class FaceDetector:
    """Detects faces (or persons) in a frame using a YOLOv8 model.

    The detector first tries to load a face-specific YOLO model
    (``yolov8n-face.pt``).  If that is unavailable it falls back to the
    standard ``yolov8n.pt`` model and filters detections to the *person*
    class (class index 0 in COCO).
    """

    # COCO class index for "person"
    _PERSON_CLASS = 0

    def __init__(self, model_path: str = "yolov8n-face.pt", confidence: float = 0.5) -> None:
        """Load the YOLO model.

        Args:
            model_path: Path or name of the YOLOv8 model file.
            confidence: Minimum confidence score for a detection to be kept.
        """
        from ultralytics import YOLO  # local import to keep startup fast

        self.confidence = confidence
        self._is_face_model = True

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            print(f"[FaceDetector] Could not load '{model_path}': {exc}")
            print("[FaceDetector] Falling back to yolov8n.pt (person class).")
            try:
                self.model = YOLO("yolov8n.pt")
                self._is_face_model = False
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"[FaceDetector] Failed to load fallback model: {fallback_exc}"
                ) from fallback_exc

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a NumPy array (H×W×3).

        Returns:
            A list of dicts, each with keys:

            * ``"bbox"`` – ``[x1, y1, x2, y2]`` in pixel coordinates (ints).
            * ``"confidence"`` – detection confidence as a float.
        """
        try:
            results = self.model(frame, verbose=False)[0]
        except Exception as exc:
            print(f"[FaceDetector] Inference error: {exc}")
            return []

        detections: List[dict] = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            # Filter to person class when using the generic model
            if not self._is_face_model:
                cls = int(box.cls[0])
                if cls != self._PERSON_CLASS:
                    continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

        return detections
