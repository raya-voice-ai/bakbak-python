"""The threaded WebSocket client: ``TTSWebSocket`` and its contexts.

A background reader thread per connection routes every message through the
core (the same one the asyncio client uses) under one lock; waiting callers
sleep on conditions of that lock. Timers (retry backoff and reconnects) run
on short-lived daemon threads, and a watchdog thread enforces the read timeout.
"""

# The driver is the client's own internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Self, Unpack, override
from websockets.exceptions import ConnectionClosed
from websockets.sync import client as _sync_client
from websockets.sync.client import ClientConnection, connect

from .._constants import WS_CLOSE_TIMEOUT, WS_MAX_MESSAGE_BYTES
from .._exceptions import APIStatusError, BakbakError
from .._retry import retry_delay
from .._utils import logger, redact_url, user_agent
from ..text import Segmenter
from ..types import RequestDefaults, SendParams
from ._common import (
    WEBSOCKET_LOGGER,
    build_request,
    check_context_id,
    connect_error,
    open_failure,
    text_delta,
)
from ._core import Connection, Core, Outcome, Request, State
from .events import Chunk, Failed, RequestResult, WebSocketEvent

if TYPE_CHECKING:
    from .._client import Bakbak

__all__ = ["Context", "RequestHandle", "TTSWebSocket"]


# From websockets 17.1 the threaded client follows redirects, through the
# `reconnect` connecter that `connect` uses. A redirect would carry X-API-Key
# to another origin (§8), so the SDK builds that connecter itself with
# redirects refused. Earlier releases never follow redirects.
_Reconnect: Any = getattr(_sync_client, "reconnect", None)
_FOLLOWS_REDIRECTS = _Reconnect is not None and "legacy" in inspect.signature(connect).parameters


def _connect(url: str, **kwargs: Any) -> ClientConnection:
    if not _FOLLOWS_REDIRECTS:
        return cast("ClientConnection", connect(url, **kwargs))
    connecter = _Reconnect(url, **kwargs)
    connecter.process_redirect = _refuse_redirect
    connection = cast("ClientConnection", connecter.connect())
    # The SDK manages the connection's lifetime itself (websockets' "legacy" use).
    connection.pending_legacy_warning = False  # pyright: ignore[reportAttributeAccessIssue]
    return connection


def _refuse_redirect(exc: Exception) -> Exception | str:
    """Never follow a redirect: a 3xx is raised as ``InvalidStatus`` and
    becomes an ``APIStatusError``."""
    return exc


class _Signal:
    """A condition on the WebSocket's lock. Use it with the lock held."""

    __slots__ = ("_cond",)

    def __init__(self, lock: threading.RLock) -> None:
        self._cond = threading.Condition(lock)

    def notify(self) -> None:
        self._cond.notify_all()

    def wait(self, timeout: float | None = None) -> None:
        self._cond.wait(timeout)


def _signal(core_signal: object) -> _Signal:
    assert isinstance(core_signal, _Signal)
    return core_signal


class RequestHandle:
    """One request sent on a context.

    ``handle.result()`` waits for its outcome and raises its error if it
    failed. ``for audio in handle`` yields its audio; that reads the context's
    events up to the end of this request, so use either the handles or
    ``ctx.events()``/``ctx.audio()``, not both.
    """

    __slots__ = ("_context", "_request")

    def __init__(self, request: Request, context: Context) -> None:
        self._request = request
        self._context = context

    @property
    def context_id(self) -> str:
        return self._context.context_id

    @property
    def request_id(self) -> str | None:
        """The server's id for the request, once its first message arrived."""
        return self._request.request_id

    @property
    def audio_bytes(self) -> int:
        """Audio delivered so far."""
        return self._request.audio_bytes

    @property
    def done(self) -> bool:
        """Whether the request has an outcome."""
        return self._request.state is State.DONE

    def result(self, timeout: float | None = None) -> RequestResult:
        """Wait for the outcome.

        Raises:
            BakbakError: the request's error, if it failed.
            TimeoutError: if ``timeout`` seconds pass first.
        """
        request = self._request
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._context._ws._lock:
            while request.state is not State.DONE:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("The request did not finish in time.")
                _signal(request.signal).wait(remaining)
        if request.outcome is Outcome.FAILED and request.error is not None:
            raise request.error
        return RequestResult(
            request.request_id, request.audio_bytes, cancelled=request.outcome is Outcome.CANCELLED
        )

    def __iter__(self) -> Iterator[bytes]:
        ws = self._context._ws
        core = ws._core
        request = self._request
        while True:
            with ws._lock:
                while True:
                    event, finished = core.pop_request_event(request)
                    if event is not None or finished:
                        break
                    _signal(request.context.signal).wait()
            if event is None:
                if request.outcome is Outcome.FAILED and request.error is not None:
                    raise request.error
                return
            if isinstance(event, Chunk):
                yield event.audio  # pyright: ignore[reportUnknownMemberType]
            elif isinstance(event, Failed):
                raise event.error
            else:
                return

    @override
    def __repr__(self) -> str:
        return (
            f"RequestHandle(context_id={self.context_id!r}, request_id={self.request_id!r}, "
            f"done={self.done!r}, audio_bytes={self.audio_bytes})"
        )


class Context:
    """A sequence of requests whose audio plays in order (BEHAVIOR.md §6.2).

    Requests on one context run one after another; different contexts run at
    the same time. Get one from ``ws.context()``. Its methods are thread-safe.
    """

    __slots__ = ("_ctx", "_defaults", "_ws", "context_id")

    def __init__(self, ws: TTSWebSocket, context_id: str, defaults: dict[str, object]) -> None:
        self._ws = ws
        self.context_id = context_id
        self._defaults = defaults
        with ws._lock:
            self._ctx = ws._core.context(context_id)

    def send(
        self, *, extra_body: dict[str, object] | None = None, **params: Unpack[SendParams]
    ) -> RequestHandle:
        """Send one request. Fields left out come from the context's and the
        WebSocket's defaults; ``extra_body`` adds fields the SDK does not
        model yet.

        It blocks only while the connection's limits are reached (64
        unfinished requests, or 16 busy contexts); a server-side failure does
        not raise here but becomes the request's outcome.
        """
        handle = self._send(params, extra_body, None)
        assert handle is not None
        return handle

    def _send(
        self,
        params: Mapping[str, object],
        extra_body: Mapping[str, object] | None,
        epoch: int | None,
    ) -> RequestHandle | None:
        """Send one request; with ``epoch``, send nothing (and return None) if
        the context was cancelled since that epoch."""
        ws = self._ws
        core = ws._core
        ctx = self._ctx
        wire, width = build_request(
            {**ws._defaults, **self._defaults}, params, extra_body, self.context_id
        )
        with ws._lock:
            while True:
                if core.closed:
                    raise BakbakError("This WebSocket is closed.")
                if ctx.input_ended:
                    raise BakbakError("end_input() was called on this context; use a new context.")
                if epoch is not None and ctx.cancel_epoch != epoch:
                    return None
                if not core.at_capacity(ctx):
                    break
                _signal(core.capacity).wait()
            request = core.submit(ctx, wire, width, lambda r: RequestHandle(r, self))
        ws._flush()
        handle: RequestHandle = request.handle
        return handle

    def send_stream(
        self,
        tokens: Iterable[object],
        *,
        end_input: bool = True,
        segmenter: Segmenter | None = None,
        **params: Unpack[RequestDefaults],
    ) -> list[RequestHandle]:
        """Send text that arrives in pieces, such as LLM tokens, sentence by
        sentence as each sentence is final.

        ``tokens`` is an iterable of ``str``, or of stream events from an LLM
        SDK (OpenAI, Anthropic or Gemini style; the text delta is found by
        duck typing). It blocks until ``tokens`` is exhausted, so read the
        audio on another thread. A ``cancel()`` of the context stops it: it
        sends nothing more and returns (after the next token arrives). Calls
        ``end_input()`` at the end unless ``end_input=False``, also when
        ``tokens`` raises.

        Args:
            segmenter: a configured :class:`bakbak.text.Segmenter`, e.g. with
                ``eager_first_clause=True`` for an earlier first request.
        """
        splitter = segmenter or Segmenter()
        handles: list[RequestHandle] = []
        epoch = self._ctx.cancel_epoch
        try:
            for item in tokens:
                if self._ctx.cancel_epoch != epoch:
                    return handles
                text = text_delta(item)
                if not text:
                    continue
                for sentence in splitter.push(text):
                    handle = self._send({**params, "text": sentence}, None, epoch)
                    if handle is None:
                        return handles
                    handles.append(handle)
            for sentence in splitter.flush():
                handle = self._send({**params, "text": sentence}, None, epoch)
                if handle is None:
                    return handles
                handles.append(handle)
        finally:
            if end_input:
                self.end_input()
        return handles

    def end_input(self) -> None:
        """No more requests: ``events()`` and ``audio()`` end once every
        request has an outcome."""
        with self._ws._lock:
            self._ws._core.end_input(self._ctx)

    def cancel(self) -> None:
        """Cancel every unfinished request of this context.

        Their audio stops at once (what arrives before the server's
        acknowledgement is discarded) and each ends as ``Cancelled``. Returns
        when the server has acknowledged; the context is then ready for new
        requests.
        """
        ws = self._ws
        core = ws._core
        with ws._lock:
            conn = core.cancel(self._ctx)
            message = core.cancel_message(self._ctx)
        if conn is not None:
            ws._write(conn, message)
        with ws._lock:
            while self._ctx.cancels_pending and not core.closed:
                _signal(self._ctx.cancel_signal).wait()

    def events(self) -> Iterator[WebSocketEvent[RequestHandle]]:
        """Yield ``Chunk``, ``Done``, ``Failed`` and ``Cancelled`` in order.

        Ends after ``end_input()`` once every request has its outcome."""
        ws = self._ws
        core = ws._core
        ctx = self._ctx
        while True:
            with ws._lock:
                while True:
                    event = core.pop_event(ctx)
                    if event is not None:
                        break
                    if core.events_finished(ctx):
                        return
                    _signal(ctx.signal).wait()
            yield event

    def audio(self) -> Iterator[bytes]:
        """Yield only the audio, and raise the first failure."""
        for event in self.events():
            if isinstance(event, Chunk):
                yield event.audio
            elif isinstance(event, Failed):
                raise event.error

    @override
    def __repr__(self) -> str:
        return f"Context(context_id={self.context_id!r})"


class TTSWebSocket:
    """Many streaming requests over one WebSocket (BEHAVIOR.md §6).

    ``ws.connect()`` (or ``with``) opens it ahead of time so the first request
    pays no handshake; otherwise the first send opens it. Reconnects, resends
    and holds are handled for you; every unfinished request gets an outcome
    when a connection is lost.
    """

    def __init__(
        self,
        client: Bakbak,
        defaults: RequestDefaults,
        on_error: Callable[[APIStatusError], None] | None,
    ) -> None:
        self._client = client
        self._defaults: dict[str, object] = dict(defaults)
        self._on_error = on_error
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._notices: list[APIStatusError] = []
        self._watchdog: threading.Thread | None = None
        self._wake = threading.Event()
        self._readers: list[threading.Thread] = []
        self._core = Core(self, max_retries=client.max_retries, read_timeout=client.timeout.read)

    # Public API

    @property
    def url(self) -> str:
        """The WebSocket URL (it never contains the API key)."""
        return self._client.websocket_url

    @property
    def connected(self) -> bool:
        """Whether a connection is open and accepting requests."""
        with self._lock:
            return self._core.usable()

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> Self:
        """Open the connection now, so the first request pays no handshake.

        Raises:
            AuthenticationError: the key was refused (or another
                ``APIStatusError`` for another refusal).
            APIConnectionError: the server could not be reached.
        """
        with self._lock:
            if self._core.closed:
                raise BakbakError("This WebSocket is closed.")
        with self._connect_lock:
            with self._lock:
                usable = self._core.usable()
            if not usable:
                self._open()
        return self

    def context(
        self, context_id: str | None = None, **defaults: Unpack[RequestDefaults]
    ) -> Context:
        """A context: requests on it play in order.

        Args:
            context_id: 1 to 128 characters of ``A-Z a-z 0-9 . _ -``; a random
                ``ctx_…`` id when left out.
            **defaults: request fields for this context, over the WebSocket's.
        """
        with self._lock:
            if self._core.closed:
                raise BakbakError("This WebSocket is closed.")
        return Context(self, check_context_id(context_id), dict(defaults))

    def close(self) -> None:
        """Close the WebSocket. Unfinished requests fail with
        ``WebSocketClosedError``."""
        with self._lock:
            connections = self._core.close()
            readers = list(self._readers)
        self._wake.set()
        # Close every connection at once, and never wait much more than
        # WS_CLOSE_TIMEOUT for peers that do not answer.
        closers = [
            threading.Thread(target=self._close_transport, args=(c.transport,), daemon=True)
            for c in connections
        ]
        for closer in closers:
            closer.start()
        deadline = time.monotonic() + WS_CLOSE_TIMEOUT + 0.25
        current = threading.current_thread()
        for thread in [*closers, *readers]:
            if thread is not current:
                thread.join(max(deadline - time.monotonic(), 0.0))

    @override
    def __repr__(self) -> str:
        return f"TTSWebSocket(url={self.url!r}, connected={self.connected!r})"

    # Driver: called by the core with the lock held, never blocking

    def new_signal(self) -> _Signal:
        return _Signal(self._lock)

    def schedule_flush(self) -> None:
        self._spawn(self._flush)

    def schedule_retry(self, request: Request, delay: float) -> None:
        def run() -> None:
            self._client._sleep(delay)
            with self._lock:
                self._core.retry_due(request)

        self._spawn(run)

    def schedule_reconnect(self, delay: float, *, force: bool) -> None:
        def run() -> None:
            if delay:
                self._client._sleep(delay)
            self._flush(force_connect=force)

        self._spawn(run)

    def abort(self, connection: Connection) -> None:
        self._spawn(lambda: self._close_transport(connection.transport))

    def notice(self, error: APIStatusError) -> None:
        self._notices.append(error)

    # Internals

    @staticmethod
    def _spawn(target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="bakbak-ws", daemon=True).start()

    def _dispatch_notices(self) -> None:
        with self._lock:
            notices, self._notices = self._notices, []
        for error in notices:
            if self._on_error is None:
                logger.debug("WebSocket error about no request: %s", error.message)
                continue
            try:
                self._on_error(error)
            except Exception:
                logger.debug("the on_error callback raised", exc_info=True)

    def _open(self) -> None:
        """Open a connection, retrying what can be retried."""
        url = self.url
        retries = 0
        while True:
            started = time.monotonic()
            try:
                transport = _connect(
                    url,
                    additional_headers=self._client.websocket_headers(),
                    user_agent_header=user_agent(),
                    compression=None,
                    open_timeout=self._client.timeout.connect,
                    close_timeout=WS_CLOSE_TIMEOUT,
                    max_size=WS_MAX_MESSAGE_BYTES,
                    logger=WEBSOCKET_LOGGER,
                )
            except Exception as exc:
                mapped = connect_error(exc)
                if mapped is None:
                    raise
                error, retry_after = mapped
            else:
                break
            # Raised outside the except block: the error carries no chained
            # exception from the WebSocket library.
            if not getattr(error, "retryable", False) or retries >= self._core.max_retries:
                raise error
            retries += 1
            delay = retry_delay(retries, retry_after)
            logger.debug("WebSocket connect failed (%s); retrying in %.2f s", error.message, delay)
            self._client._sleep(delay)
        logger.debug(
            "WebSocket connected to %s in %.0f ms",
            redact_url(url),
            (time.monotonic() - started) * 1000,
        )
        with self._lock:
            if self._core.closed:
                closed = True
            else:
                closed = False
                conn = self._core.opened(transport)
                reader = threading.Thread(
                    target=self._read, args=(conn,), name="bakbak-ws-reader", daemon=True
                )
                self._readers = [r for r in self._readers if r.is_alive()]
                self._readers.append(reader)
                reader.start()
                self._ensure_watchdog()
        if closed:
            self._close_transport(transport)
        return

    def _flush(self, force_connect: bool = False) -> None:
        """Open a connection if a request needs one, then send what can go."""
        core = self._core

        def needed() -> bool:
            with self._lock:
                return (
                    not core.closed
                    and not core.usable()
                    and (force_connect or core.needs_connection())
                )

        if needed():
            with self._connect_lock:
                if needed():
                    try:
                        self._open()
                    except Exception as exc:
                        with self._lock:
                            core.fail_waiting(open_failure(exc))
                        return
        with self._send_lock:
            with self._lock:
                batch = core.take_sendable()
            for conn, wire in batch:
                if not self._send(conn, wire):
                    break

    def _write(self, conn: Connection, wire: str) -> None:
        with self._send_lock:
            self._send(conn, wire)

    def _send(self, conn: Connection, wire: str) -> bool:
        transport: ClientConnection = conn.transport
        try:
            transport.send(wire)
        except Exception:
            # The reader sees the close and resends or fails what was in flight.
            logger.debug("WebSocket send failed on connection %d", conn.id, exc_info=True)
            self._close_transport(transport)
            return False
        return True

    def _read(self, conn: Connection) -> None:
        transport: ClientConnection = conn.transport
        try:
            for message in transport:
                if isinstance(message, str):
                    with self._lock:
                        self._core.on_message(conn, message)
                    if self._notices:
                        self._dispatch_notices()
        except ConnectionClosed:
            pass
        except Exception:
            logger.debug("WebSocket reader failed on connection %d", conn.id, exc_info=True)
        finally:
            if transport.close_code is None:
                self._close_transport(transport)
            with self._lock:
                self._core.on_close(
                    conn, transport.close_code or 1006, transport.close_reason or ""
                )
            self._dispatch_notices()

    @staticmethod
    def _close_transport(transport: ClientConnection) -> None:
        try:
            transport.close()
        except Exception:
            logger.debug("closing a WebSocket failed", exc_info=True)

    def _ensure_watchdog(self) -> None:
        if self._core.read_timeout is None:
            return
        if self._watchdog is None or not self._watchdog.is_alive():
            self._wake.clear()
            self._watchdog = threading.Thread(
                target=self._watch, name="bakbak-ws-watchdog", daemon=True
            )
            self._watchdog.start()

    def _watch(self) -> None:
        """Fail requests that hear nothing for the read timeout."""
        core = self._core
        idle = core.read_timeout or 60.0
        while True:
            with self._lock:
                if core.closed or not core.connections:
                    self._watchdog = None
                    return
                wait, sends = core.check_timeouts()
            for conn, wire in sends:
                self._write(conn, wire)
            self._wake.wait(idle if wait is None else min(wait, idle))
