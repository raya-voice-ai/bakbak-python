"""A programmable WebSocket server for regression tests, and a peer that
completes the handshake and then never answers again."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

Handler = Callable[["Server", ServerConnection, int], Awaitable[None]]


def chunk(context_id: str, audio: bytes, request_id: str = "req_1") -> str:
    return json.dumps(
        {
            "type": "chunk",
            "status_code": 206,
            "done": False,
            "data": base64.b64encode(audio).decode(),
            "context_id": context_id,
            "request_id": request_id,
        }
    )


def done(context_id: str, request_id: str = "req_1") -> str:
    return json.dumps(
        {
            "type": "done",
            "status_code": 200,
            "done": True,
            "context_id": context_id,
            "request_id": request_id,
        }
    )


def cancelled(context_id: str) -> str:
    return json.dumps({"type": "cancelled", "done": True, "context_id": context_id})


class Server:
    """Runs ``handler(server, ws, index)`` for each connection, in a thread
    with its own event loop, so it serves sync and async clients alike."""

    def __init__(self, handler: Handler, process_request: Any = None) -> None:
        self.handler = handler
        self.process_request = process_request
        self.connections = 0
        self.received: list[tuple[int, dict[str, Any]]] = []
        self.headers: list[dict[str, str]] = []
        self.close_codes: list[int | None] = []
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._stop: asyncio.Future[None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.host = ""
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> Server:
        self._thread.start()
        assert self._ready.wait(5)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
        self._thread.join(5)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())
        self._loop.close()

    async def _main(self) -> None:
        self._stop = self._loop.create_future()
        async with serve(
            self._handle,
            "localhost",
            0,
            family=socket.AF_INET,
            compression=None,
            ping_interval=None,
            process_request=self._process_request,
        ) as server:
            self.host, self.port = next(iter(server.sockets)).getsockname()[:2]
            self._ready.set()
            await self._stop

    async def _process_request(self, connection: ServerConnection, request: Any) -> Any:
        self.headers.append({k.lower(): v for k, v in request.headers.raw_items()})
        if self.process_request is not None:
            result = self.process_request(connection, request)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        index = self.connections
        self.connections += 1
        try:
            await self.handler(self, ws, index)
        finally:
            await ws.wait_closed()
            self.close_codes.append(ws.close_code)

    async def recv(self, ws: ServerConnection, index: int, timeout: float = 5) -> dict[str, Any]:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        self.received.append((index, message))
        return message

    def texts(self) -> list[str]:
        return [m["text"] for _, m in self.received if "text" in m]


class SilentPeer:
    """Completes the WebSocket handshake, then never reads or answers: a
    peer that stopped responding (a dead link)."""

    _GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self) -> None:
        self._listener = socket.create_server(("localhost", 0))
        self.host, self.port = self._listener.getsockname()[:2]
        self._clients: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> SilentPeer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._listener.close()
        for client in self._clients:
            client.close()

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            self._clients.append(client)
            request = b""
            while b"\r\n\r\n" not in request:
                request += client.recv(4096)
            key = next(
                line.split(b":", 1)[1].strip()
                for line in request.split(b"\r\n")
                if line.lower().startswith(b"sec-websocket-key:")
            )
            accept = base64.b64encode(hashlib.sha1(key + self._GUID.encode()).digest())
            client.sendall(
                b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
            )
