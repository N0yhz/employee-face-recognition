"""ArcFace-based face embedding extractor using insightface."""

from typing import List, Optional

import numpy as np


class FaceEmbedder:
    """Extracts 512-dimensional ArcFace face embeddings via insightface.

    The insightface :class:`FaceAnalysis` app is initialised with the
    ``buffalo_l`` model pack by default.  It first tries GPU (``ctx_id=0``)
    and falls back to CPU (``ctx_id=-1``) automatically.
    """

    def __init__(self, model_name: str = "buffalo_l") -> None:
        """Initialise the insightface FaceAnalysis app.

        Args:
            model_name: Name of the insightface model pack to use.
        """
        import insightface  # local import

        self.app = insightface.app.FaceAnalysis(name=model_name)
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            # Fall back to CPU if GPU is unavailable
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def get_embedding(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """Crop a face region and return its normalised ArcFace embedding.

        The face crop is expanded by 20 % on each side before being passed
        to the insightface app so that the model receives adequate context.

        Args:
            frame: Full BGR frame as a NumPy array (H×W×3).
            bbox: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.

        Returns:
            A 512-dimensional ``float32`` NumPy array (L2-normalised), or
            ``None`` if the embedding could not be extracted.
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Expand crop by 20 % for better ArcFace accuracy
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            faces = self.app.get(crop)
            if not faces:
                return None

            # Use the face with the highest detection score
            face = max(faces, key=lambda f: f.det_score)
            embedding: np.ndarray = face.normed_embedding
            return embedding.astype(np.float32)
        except Exception as exc:
            print(f"[FaceEmbedder] Embedding error: {exc}")
            return None
