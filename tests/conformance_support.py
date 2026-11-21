"""Mock servers and matchers for the shared conformance fixtures.

The matching rules are those of ``conformance/README.md``:

- subset match for request headers and WebSocket ``recv`` steps (header names
  compared case-insensitively); the HTTP request body must match exactly;
- ``sleeps``: the waits the client asked for, each within ``[min, max]``;
- errors are matched on class name, then ``status``, ``error_code``,
  ``retryable``, ``request_id`` (``null`` means absent), ``message`` (equal),
  ``message_contains``, ``audio_bytes`` and ``body_fields`` (subset of the body).
"""

from __future__ import annotations

import asyncio
import base64
import http
import json
import socket
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

CONFORMANCE = Path(__file__).resolve().parent.parent / "conformance"


def load_scenarios(kind: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text("utf-8")) for p in sorted((CONFORMANCE / kind).glob("*.json"))]


# Matching


def subset_mismatches(
    expected: Mapping[str, Any], actual: Mapping[str, Any], what: str
) -> list[str]:
    problems = []
    for key, value in expected.items():
        if key not in actual:
            problems.append(f"{what}: missing {key!r}")
        elif actual[key] != value:
            problems.append(f"{what}: {key!r} is {actual[key]!r}, expected {value!r}")
    return problems


def check_sleeps(expected: list[list[float]], actual: list[float]) -> None:
    assert len(actual) == len(expected), f"sleeps {actual}, expected {expected}"
    for got, (low, high) in zip(actual, expected, strict=True):
        assert low - 1e-9 <= got <= high + 1e-9, f"sleep {got} not in [{low}, {high}]"


def check_error(expected: Mapping[str, Any], error: BaseException | None) -> None:
    assert error is not None, f"expected {expected['class']}, got no error"
    assert type(error).__name__ == expected["class"], f"got {type(error).__name__}: {error!r}"
    for key in ("status", "error_code", "retryable", "request_id"):
        if key in expected:
            assert getattr(error, key) == expected[key], (
                f"{key} is {getattr(error, key)!r}, expected {expected[key]!r}"
            )
    message = getattr(error, "message", None)
    if "message" in expected:
        assert message == expected["message"], f"message {message!r}"
    if "message_contains" in expected:
        assert message is not None
        assert expected["message_contains"] in message, f"message {message!r}"
    if "audio_bytes" in expected:
        assert getattr(error, "audio_bytes", None) == expected["audio_bytes"]
    if "body_fields" in expected:
        body = getattr(error, "body", None)
        assert isinstance(body, Mapping), f"body {body!r}"
        problems = subset_mismatches(expected["body_fields"], body, "body")
        assert not problems, problems


# HTTP mock server


class MockHTTPServer:
    """Plays a scenario's exchanges, one per HTTP attempt, on a real socket."""

    def __init__(self, exchanges: list[dict[str, Any]]) -> None:
        self.exchanges = exchanges
        self.index = 0
        self.failures: list[str] = []
        self._lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                mock.handle(self)

            def do_POST(self) -> None:
                mock.handle(self)

        self._server = ThreadingHTTPServer(("localhost", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> MockHTTPServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        with self._lock:
            index = self.index
            self.index += 1
        if index >= len(self.exchanges):
            self.failures.append(
                f"unexpected request #{index + 1}: {handler.command} {handler.path}"
            )
            self._send(handler, 500, {"content-type": "application/json"}, b"{}")
            return
        exchange = self.exchanges[index]
        self._check(index, exchange["expect_request"], handler, raw)
        response = exchange["response"]
        if response.get("network_error"):
            handler.close_connection = True
            return
        headers = dict(response.get("headers", {}))
        if "json" in response:
            self._send(handler, response["status"], headers, json.dumps(response["json"]).encode())
        elif "sse" in response:
            self._send(handler, response["status"], headers, response["sse"].encode(), stream=True)
        else:
            body = base64.b64decode(response["body_base64"])
            self._send(handler, response["status"], headers, body)

    def _check(
        self, index: int, expected: dict[str, Any], handler: BaseHTTPRequestHandler, raw: bytes
    ) -> None:
        where = f"request #{index + 1}"
        if handler.command != expected["method"]:
            self.failures.append(f"{where}: method {handler.command}")
        if handler.path != expected["path"]:
            self.failures.append(f"{where}: path {handler.path}")
        actual_headers = {k.lower(): v for k, v in handler.headers.items()}
        self.failures += subset_mismatches(expected.get("headers", {}), actual_headers, where)
        if "json" in expected:
            try:
                body = json.loads(raw)
            except ValueError:
                self.failures.append(f"{where}: body is not JSON")
                return
            if body != expected["json"]:
                self.failures.append(f"{where}: body {body!r} != {expected['json']!r}")

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        status: int,
        headers: dict[str, str],
        body: bytes,
        stream: bool = False,
    ) -> None:
        handler.send_response(status)
        for name, value in headers.items():
            handler.send_header(name, value)
        if not stream:
            handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
        handler.close_connection = True

    def assert_complete(self) -> None:
        assert not self.failures, self.failures
        assert self.index == len(self.exchanges), (
            f"client made {self.index} attempts, scenario has {len(self.exchanges)}"
        )


# WebSocket mock server


class MockWSServer:
    """Runs a scenario's scripted connections on a real WebSocket server, in a
    background thread with its own event loop (so it serves sync clients too)."""

    def __init__(self, connections: list[dict[str, Any]]) -> None:
        self.connections = connections
        self.accepted = 0
        self.failures: list[str] = []
        self._index: dict[int, int] = {}
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._stop: asyncio.Future[None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.host = ""
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> MockWSServer:
        self._thread.start()
        assert self._ready.wait(5), "mock WebSocket server did not start"
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
            self._handler,
            "localhost",
            0,
            family=socket.AF_INET,
            process_request=self._process_request,
            compression=None,
            ping_interval=None,
        ) as server:
            self.host, self.port = next(iter(server.sockets)).getsockname()[:2]
            self._ready.set()
            await self._stop

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        index = self.accepted
        self.accepted += 1
        if index >= len(self.connections):
            self.failures.append(f"unexpected connection #{index + 1}")
            return connection.respond(http.HTTPStatus.INTERNAL_SERVER_ERROR, "unexpected\n")
        entry = self.connections[index]
        self._index[id(connection)] = index
        for name, value in entry.get("expect_headers", {}).items():
            if request.headers.get(name) != value:
                self.failures.append(
                    f"connection #{index + 1}: header {name} is {request.headers.get(name)!r}"
                )
        reject = entry.get("reject")
        if reject is not None:
            body = json.dumps(reject["json"]).encode()
            headers = Headers()
            for name, value in reject.get("headers", {}).items():
                headers[name] = value
            headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            status = reject["status"]
            return Response(status, http.HTTPStatus(status).phrase, headers, body)
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        index = self._index[id(ws)]
        where = f"connection #{index + 1}"
        for n, step in enumerate(self.connections[index]["script"]):
            here = f"{where} step {n + 1}"
            try:
                if "recv" in step:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 5)
                    except asyncio.TimeoutError:
                        self.failures.append(f"{here}: no message within 5 s")
                        return
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        self.failures.append(f"{here}: message is not an object")
                        return
                    self.failures += subset_mismatches(step["recv"], message, here)
                elif "send" in step:
                    await ws.send(json.dumps(step["send"], ensure_ascii=False))
                elif "close" in step:
                    await ws.close(step["close"]["code"], step["close"].get("reason", ""))
                    return
                elif step.get("expect_no_more"):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 0.3)
                    except asyncio.TimeoutError:
                        pass
                    else:
                        self.failures.append(f"{here}: unexpected message {raw!r}")
            except ConnectionClosed:
                if n < len(self.connections[index]["script"]) - 1 or "recv" in step:
                    self.failures.append(f"{here}: the client closed the connection")
                return
        await ws.wait_closed()

    def assert_complete(self, expected_connections: int | None) -> None:
        assert not self.failures, self.failures
        if expected_connections is not None:
            assert self.accepted == expected_connections, (
                f"client opened {self.accepted} connections, expected {expected_connections}"
            )
