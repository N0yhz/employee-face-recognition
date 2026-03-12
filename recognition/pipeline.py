"""Full face recognition pipeline: detect → embed → match."""

from typing import List

import numpy as np

from database.employee_db import EmployeeDatabase
from recognition.detector import FaceDetector
from recognition.embedder import FaceEmbedder


class RecognitionPipeline:
    """Orchestrates face detection, embedding extraction and employee matching.

    For every frame the pipeline:

    1. Runs :class:`FaceDetector` to locate faces.
    2. Runs :class:`FaceEmbedder` to extract an ArcFace embedding for each face.
    3. Queries :class:`EmployeeDatabase` to find the closest known employee.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        db: EmployeeDatabase,
        threshold: float = 0.5,
    ) -> None:
        """Initialise the pipeline with pre-constructed components.

        Args:
            detector: Initialised :class:`FaceDetector` instance.
            embedder: Initialised :class:`FaceEmbedder` instance.
            db: Initialised :class:`EmployeeDatabase` instance.
            threshold: Cosine similarity threshold for a positive match.
        """
        self.detector = detector
        self.embedder = embedder
        self.db = db
        self.threshold = threshold

    def process_frame(self, frame: np.ndarray) -> List[dict]:
        """Run the full recognition pipeline on a single video frame.

        Args:
            frame: BGR image as a NumPy array (H×W×3).

        Returns:
            A list of result dicts, one per detected face, each containing:

            * ``"bbox"`` – ``[x1, y1, x2, y2]``.
            * ``"name"`` – employee name or ``"Unknown"``.
            * ``"employee_id"`` – employee ID string or ``None``.
            * ``"similarity"`` – cosine similarity (0–1); ``0.0`` if unknown.
            * ``"recognized"`` – ``True`` if similarity ≥ threshold.
            * ``"confidence"`` – YOLO detection confidence.
        """
        results: List[dict] = []

        try:
            detections = self.detector.detect(frame)
        except Exception as exc:
            print(f"[RecognitionPipeline] Detection error: {exc}")
            return results

        for det in detections:
            bbox = det["bbox"]
            det_conf = det["confidence"]

            embedding = self.embedder.get_embedding(frame, bbox)

            if embedding is None:
                results.append(
                    {
                        "bbox": bbox,
                        "name": "Unknown",
                        "employee_id": None,
                        "similarity": 0.0,
                        "recognized": False,
                        "confidence": det_conf,
                    }
                )
                continue

            match = self.db.find_match(embedding, self.threshold)

            if match:
                results.append(
                    {
                        "bbox": bbox,
                        "name": match["name"],
                        "employee_id": match["id"],
                        "similarity": round(match["similarity"], 4),
                        "recognized": True,
                        "confidence": det_conf,
                    }
                )
            else:
                results.append(
                    {
                        "bbox": bbox,
                        "name": "Unknown",
                        "employee_id": None,
                        "similarity": 0.0,
                        "recognized": False,
                        "confidence": det_conf,
                    }
                )

        return results
