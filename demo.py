"""Main runnable demo for the employee face recognition system.

Opens a webcam or video file, runs the recognition pipeline on each frame,
overlays results and optionally streams them via WebSocket.

Usage::

    # Webcam (default)
    python demo.py

    # Video file
    python demo.py --source /path/to/video.mp4

    # Disable WebSocket broadcasting
    python demo.py --no-websocket
"""

import argparse
import asyncio
import sys
import threading
import time

import cv2

import config
from database.employee_db import EmployeeDatabase
from recognition.detector import FaceDetector
from recognition.embedder import FaceEmbedder
from recognition.pipeline import RecognitionPipeline
from utils.drawing import draw_fps, draw_results
from websocket_server.server import RecognitionWebSocketServer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Employee face recognition demo."
    )
    parser.add_argument(
        "--source",
        default=str(config.CAMERA_INDEX),
        help=(
            "Video source: camera index (integer) or path to a video file. "
            f"Default: {config.CAMERA_INDEX}."
        ),
    )
    parser.add_argument(
        "--no-websocket",
        action="store_true",
        help="Disable the WebSocket broadcast server.",
    )
    parser.add_argument(
        "--db",
        default=config.DB_PATH,
        help=f"Path to the employee database (default: {config.DB_PATH}).",
    )
    return parser.parse_args()


def _start_websocket_server(ws_server: RecognitionWebSocketServer) -> None:
    """Target function for the WebSocket server background thread.

    Runs a new asyncio event loop dedicated to the WebSocket server so that
    the synchronous OpenCV loop is not blocked.

    Args:
        ws_server: The server instance to start.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_server._loop = loop
    try:
        loop.run_until_complete(ws_server.start())
    except Exception as exc:
        print(f"[WebSocketServer] Server stopped: {exc}")
    finally:
        loop.close()


def _resolve_source(source_str: str):
    """Convert the ``--source`` CLI argument to an OpenCV-compatible value.

    Args:
        source_str: Raw string from the command line.

    Returns:
        An integer camera index or a file path string.
    """
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main() -> None:
    """Main entry point for the demo application."""
    args = parse_args()
    source = _resolve_source(args.source)

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    print("[demo] Initialising components…")

    print("[demo] Loading YOLO detector…")
    try:
        detector = FaceDetector(
            model_path=config.YOLO_MODEL,
            confidence=config.DETECTION_CONFIDENCE,
        )
    except Exception as exc:
        print(f"[demo] ERROR: Could not load detector: {exc}")
        sys.exit(1)

    print("[demo] Loading ArcFace embedder…")
    try:
        embedder = FaceEmbedder(model_name=config.ARCFACE_MODEL)
    except Exception as exc:
        print(f"[demo] ERROR: Could not load embedder: {exc}")
        sys.exit(1)

    db = EmployeeDatabase(db_path=args.db)
    employees = db.list_employees()
    print(f"[demo] Employee database loaded — {len(employees)} employee(s) registered.")
    if employees:
        for emp in employees:
            print(f"  • {emp['id']}: {emp['name']}")
    else:
        print("[demo] No employees registered — all faces will show as 'Unknown'.")

    pipeline = RecognitionPipeline(
        detector=detector,
        embedder=embedder,
        db=db,
        threshold=config.SIMILARITY_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Optionally start WebSocket server
    # ------------------------------------------------------------------
    ws_server: RecognitionWebSocketServer | None = None
    if not args.no_websocket:
        ws_server = RecognitionWebSocketServer(
            host=config.WEBSOCKET_HOST,
            port=config.WEBSOCKET_PORT,
        )
        ws_thread = threading.Thread(
            target=_start_websocket_server,
            args=(ws_server,),
            daemon=True,
            name="websocket-server",
        )
        ws_thread.start()
        print(
            f"[demo] WebSocket server starting on "
            f"ws://{config.WEBSOCKET_HOST}:{config.WEBSOCKET_PORT}"
        )
        # Brief pause to allow the server to bind before the first frame
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Open video source
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[demo] ERROR: Could not open video source '{source}'.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("[demo] Starting capture loop.  Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[demo] End of video stream or capture error.")
                break

            # ----------------------------------------------------------
            # Recognition
            # ----------------------------------------------------------
            results = pipeline.process_frame(frame)

            # ----------------------------------------------------------
            # Drawing
            # ----------------------------------------------------------
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_results(frame, results)
            draw_fps(frame, fps)

            cv2.imshow("Employee Face Recognition — press Q to quit", frame)

            # ----------------------------------------------------------
            # WebSocket broadcast
            # ----------------------------------------------------------
            if ws_server is not None:
                payload = {
                    "faces": [
                        {
                            "name": r["name"],
                            "employee_id": r["employee_id"],
                            "similarity": r["similarity"],
                            "recognized": r["recognized"],
                            "bbox": r["bbox"],
                        }
                        for r in results
                    ]
                }
                ws_server.broadcast(payload)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[demo] Quit requested by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[demo] Shutdown complete.")


if __name__ == "__main__":
    main()
