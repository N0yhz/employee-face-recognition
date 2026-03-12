"""Central configuration for the employee face recognition system."""

# YOLO model path or name. Falls back to yolov8n.pt if yolov8n-face.pt is unavailable.
YOLO_MODEL = "yolov8n-face.pt"

# ArcFace model name (insightface buffalo_l pack).
ARCFACE_MODEL = "buffalo_l"

# Cosine similarity threshold for a face to be considered a match.
SIMILARITY_THRESHOLD = 0.5

# WebSocket server settings.
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765

# Path to the employee embeddings database file.
DB_PATH = "database/employee_embeddings.json"

# Default camera index for OpenCV capture.
CAMERA_INDEX = 0

# Frame dimensions.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Minimum detection confidence for YOLO.
DETECTION_CONFIDENCE = 0.5
