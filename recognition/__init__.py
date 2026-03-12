"""Recognition package for face detection, embedding and pipeline."""

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .pipeline import RecognitionPipeline

__all__ = ["FaceDetector", "FaceEmbedder", "RecognitionPipeline"]
