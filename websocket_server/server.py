"""Async WebSocket server that broadcasts recognition results to all clients."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Set

logger = logging.getLogger(__name__)


class RecognitionWebSocketServer:
    """Async WebSocket server for broadcasting face recognition results.

    The server keeps track of all active client connections and fans out
    every :meth:`broadcast` call to each of them as a JSON message.

    Example JSON payload sent to clients::

        {
            "timestamp": "2026-03-12T22:00:00",
            "faces": [
                {
                    "name": "John Doe",
                    "employee_id": "EMP001",
                    "similarity": 0.92,
                    "recognized": true,
                    "bbox": [100, 50, 250, 200]
                }
            ]
        }
    """

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        """Initialise the server configuration.

        Args:
            host: Interface to bind to.
            port: TCP port to listen on.
        """
        self.host = host
        self.port = port
        self._clients: Set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None

    async def _handler(self, websocket) -> None:
        """Handle a single WebSocket connection lifecycle.

        Registers the client on connect and unregisters it when the
        connection is closed.

        Args:
            websocket: The connected WebSocket object.
        """
        self._clients.add(websocket)
        logger.info(
            "[WebSocketServer] Client connected: %s (total: %d)",
            websocket.remote_address,
            len(self._clients),
        )
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info(
                "[WebSocketServer] Client disconnected (total: %d)",
                len(self._clients),
            )

    async def start(self) -> None:
        """Start the WebSocket server and listen for connections.

        This coroutine runs indefinitely until cancelled.
        """
        import websockets  # local import

        self._loop = asyncio.get_event_loop()
        self._server = await websockets.serve(self._handler, self.host, self.port)
        logger.info(
            "[WebSocketServer] Listening on ws://%s:%d", self.host, self.port
        )
        await self._server.wait_closed()

    def broadcast(self, data: dict) -> None:
        """Broadcast a recognition result to all connected WebSocket clients.

        This method is *thread-safe*: it can be called from a synchronous
        thread (e.g. the OpenCV loop) while the event loop runs on a
        separate thread.

        Args:
            data: Recognition result dict.  A ``"timestamp"`` field is added
                automatically if not already present.
        """
        if self._loop is None or not self._clients:
            return

        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

        message = json.dumps(data)

        async def _send_all():
            if not self._clients:
                return
            # Copy to avoid mutation during iteration
            clients = set(self._clients)
            for ws in clients:
                try:
                    await ws.send(message)
                except Exception as exc:
                    logger.debug("[WebSocketServer] Send error: %s", exc)
                    self._clients.discard(ws)

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
