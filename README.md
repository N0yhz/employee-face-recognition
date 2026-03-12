# Employee Face Recognition

Real-time **employee face recognition** demo using YOLOv8 for detection and ArcFace for identity matching, with live result streaming over WebSocket.

---

## Architecture

```
Camera / Video File
        │
        ▼
  ┌─────────────┐
  │  OpenCV     │  capture & preprocess frames
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  YOLOv8     │  detect faces / persons in each frame
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  ArcFace    │  extract 512-dim face embeddings
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  Employee DB     │  cosine-similarity lookup against registered employees
  └──────┬───────────┘
         │
         ├──► OpenCV imshow  (annotated live preview)
         │
         └──► WebSocket      (JSON broadcast to any connected client)
```

---

## Prerequisites

- Python 3.9 or newer
- `pip`
- A webcam **or** a video file to use as the demo source

---

## Installation

```bash
# Clone the repository
git clone https://github.com/N0yhz/employee-face-recognition.git
cd employee-face-recognition

# Install dependencies
pip install -r requirements.txt
```

> **YOLO model** – `ultralytics` downloads `yolov8n-face.pt` (or `yolov8n.pt` as fallback) automatically on first use.
>
> **ArcFace model** – `insightface` downloads the `buffalo_l` pack automatically on first use (~200 MB).

---

## Registering Employees

Before running the demo you can add employee faces to the database.  A clear,
front-facing photo gives the best accuracy.

```bash
python register_employee.py \
    --image  path/to/alice.jpg \
    --id     EMP001 \
    --name   "Alice Smith"

python register_employee.py \
    --image  path/to/bob.jpg \
    --id     EMP002 \
    --name   "Bob Jones"
```

Embeddings are stored in `database/employee_embeddings.json`.

> **Tip** – The demo works without any registered employees: all detected faces
> will simply be labelled *Unknown*.

---

## Running the Demo

```bash
# Default webcam (index 0)
python demo.py

# Specific camera index
python demo.py --source 1

# Video file
python demo.py --source /path/to/video.mp4

# Without WebSocket broadcasting
python demo.py --no-websocket
```

Press **Q** inside the preview window to quit.

---

## WebSocket Client

When the demo is running the WebSocket server listens on `ws://localhost:8765`
by default.  Any WebSocket client can subscribe and receive live recognition
results.

### Quick JavaScript example

```html
<!DOCTYPE html>
<html>
<head><title>Employee Recognition Feed</title></head>
<body>
<pre id="output"></pre>
<script>
  const ws = new WebSocket("ws://localhost:8765");
  ws.onmessage = (event) => {
    document.getElementById("output").textContent = event.data;
  };
</script>
</body>
</html>
```

### Message format

```json
{
  "timestamp": "2026-03-12T22:00:00",
  "faces": [
    {
      "name": "Alice Smith",
      "employee_id": "EMP001",
      "similarity": 0.92,
      "recognized": true,
      "bbox": [100, 50, 250, 200]
    },
    {
      "name": "Unknown",
      "employee_id": null,
      "similarity": 0.0,
      "recognized": false,
      "bbox": [400, 80, 530, 230]
    }
  ]
}
```

---

## Configuration

All tuneable parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `"yolov8n-face.pt"` | YOLO model file / name |
| `ARCFACE_MODEL` | `"buffalo_l"` | insightface model pack |
| `SIMILARITY_THRESHOLD` | `0.5` | Cosine similarity cutoff for a match |
| `WEBSOCKET_HOST` | `"localhost"` | WebSocket bind address |
| `WEBSOCKET_PORT` | `8765` | WebSocket port |
| `DB_PATH` | `"database/employee_embeddings.json"` | Employee DB path |
| `CAMERA_INDEX` | `0` | Default camera index |
| `FRAME_WIDTH` | `1280` | Capture width |
| `FRAME_HEIGHT` | `720` | Capture height |
| `DETECTION_CONFIDENCE` | `0.5` | YOLO detection threshold |

---

## Project Structure

```
employee-face-recognition/
├── README.md
├── requirements.txt
├── demo.py                   # Main runnable demo
├── register_employee.py      # CLI tool to register employee faces
├── config.py                 # Central configuration
├── database/
│   ├── __init__.py
│   ├── employee_db.py        # Face embedding storage & lookup
│   └── sample_employees/     # Place sample face images here
├── recognition/
│   ├── __init__.py
│   ├── detector.py           # YOLOv8 face detection
│   ├── embedder.py           # ArcFace embedding extraction
│   └── pipeline.py           # End-to-end recognition pipeline
├── websocket_server/
│   ├── __init__.py
│   └── server.py             # Async WebSocket broadcast server
└── utils/
    ├── __init__.py
    └── drawing.py            # OpenCV drawing helpers
```
