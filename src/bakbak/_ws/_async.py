"""The asyncio WebSocket client: ``AsyncTTSWebSocket`` and its contexts.

One reader task per connection routes every message through the core, which
also keeps the connection answering the server's pings. Requests are written
under one send lock, so a context's messages leave in the order they were sent.
"""

# The driver is the client's own internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
)
from types import TracebackType
from typing import TYPE_CHECKING, Any

from typing_extensions import Self, Unpack, override
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

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
from ._core import Connection, Core, Driver, Outcome, Request, State
from .events import Chunk, Failed, RequestResult, WebSocketEvent

if TYPE_CHECKING:
    from .._client import AsyncBakbak

__all__ = ["AsyncContext", "AsyncRequestHandle", "AsyncTTSWebSocket"]


class _Connect(connect):
    """``connect`` that never follows a redirect: the API never redirects, and
    a redirect would carry ``X-API-Key`` to another origin (§8). A 3xx is
    raised as ``InvalidStatus`` and becomes an ``APIStatusError``."""

    @override
    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


class _Signal:
    """An ``asyncio.Event`` used as a wake-up: check, then ``wait()``."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def notify(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        # Callers check their condition first; nothing can run between that
        # check and this clear, so no wake-up is lost.
        self._event.clear()
        await self._event.wait()


def _signal(core_signal: object) -> _Signal:
    assert isinstance(core_signal, _Signal)
    return core_signal


_STOP = object()


async def _iterate(
    tokens: Iterable[object] | AsyncIterable[object],
) -> AsyncGenerator[object, None]:
    if isinstance(tokens, AsyncIterable):
        async for item in tokens:
            yield item
    else:
        for item in tokens:
            yield item


class AsyncRequestHandle:
    """One request sent on a context.

    ``await handle`` (or ``await handle.result()``) waits for its outcome and
    raises its error if it failed. ``async for audio in handle`` yields its
    audio; that reads the context's events up to the end of this request, so
    use either the handles or ``ctx.events()``/``ctx.audio()``, not both.
    """

    __slots__ = ("_context", "_request")

    def __init__(self, request: Request, context: AsyncContext) -> None:
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

    async def result(self, timeout: float | None = None) -> RequestResult:
        """Wait for the outcome.

        Raises:
            BakbakError: the request's error, if it failed.
            TimeoutError: if ``timeout`` seconds pass first.
        """
        request = self._request

        async def finished() -> None:
            while request.state is not State.DONE:
                await _signal(request.signal).wait()

        try:
            await asyncio.wait_for(finished(), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("The request did not finish in time.") from None
        if request.outcome is Outcome.FAILED and request.error is not None:
            raise request.error
        return RequestResult(
            request.request_id, request.audio_bytes, cancelled=request.outcome is Outcome.CANCELLED
        )

    def __await__(self) -> Generator[Any, None, RequestResult]:
        return self.result().__await__()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._audio()

    async def _audio(self) -> AsyncIterator[bytes]:
        core = self._context._ws._core
        request = self._request
        while True:
            event, finished = core.pop_request_event(request)
            if event is not None:
                if isinstance(event, Chunk):
                    yield event.audio  # pyright: ignore[reportUnknownMemberType]
                elif isinstance(event, Failed):
                    raise event.error
                else:
                    return
                continue
            if finished:
                if request.outcome is Outcome.FAILED and request.error is not None:
                    raise request.error
                return
            await _signal(request.context.signal).wait()

    @override
    def __repr__(self) -> str:
        return (
            f"AsyncRequestHandle(context_id={self.context_id!r}, request_id={self.request_id!r}, "
            f"done={self.done!r}, audio_bytes={self.audio_bytes})"
        )


class AsyncContext:
    """A sequence of requests whose audio plays in order (BEHAVIOR.md §6.2).

    Requests on one context run one after another; different contexts run at
    the same time. Get one from ``ws.context()``.
    """

    __slots__ = ("_ctx", "_defaults", "_ws", "context_id")

    def __init__(self, ws: AsyncTTSWebSocket, context_id: str, defaults: dict[str, object]) -> None:
        self._ws = ws
        self.context_id = context_id
        self._defaults = defaults
        self._ctx = ws._core.context(context_id)

    async def send(
        self, *, extra_body: dict[str, object] | None = None, **params: Unpack[SendParams]
    ) -> AsyncRequestHandle:
        """Send one request. Fields left out come from the context's and the
        WebSocket's defaults; ``extra_body`` adds fields the SDK does not
        model yet.

        It waits only while the connection's limits are reached (64 unfinished
        requests, or 16 busy contexts); a server-side failure does not raise
        here but becomes the request's outcome.
        """
        handle = await self._send(params, extra_body, None)
        assert handle is not None
        return handle

    async def _send(
        self,
        params: Mapping[str, object],
        extra_body: Mapping[str, object] | None,
        epoch: int | None,
    ) -> AsyncRequestHandle | None:
        """Send one request; with ``epoch``, send nothing (and return None) if
        the context was cancelled since that epoch."""
        ws = self._ws
        core = ws._core
        ctx = self._ctx
        wire, width = build_request(
            {**ws._defaults, **self._defaults}, params, extra_body, self.context_id
        )
        while True:
            if core.closed:
                raise BakbakError("This WebSocket is closed.")
            if ctx.input_ended:
                raise BakbakError("end_input() was called on this context; use a new context.")
            if epoch is not None and ctx.cancel_epoch != epoch:
                return None
            if not core.at_capacity(ctx):
                break
            await _signal(core.capacity).wait()
        request = core.submit(ctx, wire, width, lambda r: AsyncRequestHandle(r, self))
        try:
            # The flush runs in its own task: cancelling this call cannot cut a
            # batch of writes short, so the core always matches the wire.
            await asyncio.shield(ws._spawn(ws._flush()))
        except asyncio.CancelledError:
            # The caller gave up before getting the handle. If the request was
            # not written yet, take it back: nothing it abandoned is sent or
            # billed. If it was, its outcome still reaches the context.
            core.withdraw(request)
            raise
        handle: AsyncRequestHandle = request.handle
        return handle

    async def send_stream(
        self,
        tokens: Iterable[object] | AsyncIterable[object],
        *,
        end_input: bool = True,
        segmenter: Segmenter | None = None,
        **params: Unpack[RequestDefaults],
    ) -> list[AsyncRequestHandle]:
        """Send text that arrives in pieces, such as LLM tokens, sentence by
        sentence as each sentence is final.

        ``tokens`` may be an iterable or an async iterable of ``str``, or of
        stream events from an LLM SDK (OpenAI, Anthropic or Gemini style; the
        text delta is found by duck typing). A sync iterable is read on the
        event loop, so it should not block. Calls ``end_input()`` at the end
        unless ``end_input=False``, also when ``tokens`` raises.

        Args:
            segmenter: a configured :class:`bakbak.text.Segmenter`, e.g. with
                ``eager_first_clause=True`` for an earlier first request.
        """
        splitter = segmenter or Segmenter()
        handles: list[AsyncRequestHandle] = []
        epoch = self._ctx.cancel_epoch
        items = _iterate(tokens)
        racing = isinstance(tokens, AsyncIterable)
        try:
            while True:
                item = await self._next_item(items, epoch, racing)
                if item is _STOP:
                    break
                text = text_delta(item)
                if not text:
                    continue
                for sentence in splitter.push(text):
                    handle = await self._send({**params, "text": sentence}, None, epoch)
                    if handle is None:
                        return handles
                    handles.append(handle)
            for sentence in splitter.flush():
                handle = await self._send({**params, "text": sentence}, None, epoch)
                if handle is None:
                    return handles
                handles.append(handle)
        finally:
            await items.aclose()
            if end_input:
                self.end_input()
        return handles

    async def _next_item(
        self, items: AsyncGenerator[object, None], epoch: int, racing: bool
    ) -> object:
        """The next token, or ``_STOP`` when the tokens end or the context is
        cancelled: a cancel stops ``send_stream`` even while it waits for a
        slow LLM (§6.3)."""
        ctx = self._ctx
        if ctx.cancel_epoch != epoch:
            return _STOP
        if not racing:
            try:
                return await items.__anext__()
            except StopAsyncIteration:
                return _STOP
        getter = asyncio.ensure_future(items.__anext__())
        watcher = asyncio.ensure_future(self._cancelled_since(epoch))
        try:
            await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            watcher.cancel()
            if not getter.done():
                # Stop waiting for the token, and let the generator finish
                # before it is closed.
                getter.cancel()
                await asyncio.gather(getter, return_exceptions=True)
        if ctx.cancel_epoch != epoch or getter.cancelled():
            return _STOP
        try:
            return getter.result()
        except StopAsyncIteration:
            return _STOP

    async def _cancelled_since(self, epoch: int) -> None:
        core = self._ws._core
        while self._ctx.cancel_epoch == epoch and not core.closed:
            await _signal(self._ctx.signal).wait()

    def end_input(self) -> None:
        """No more requests: ``events()`` and ``audio()`` end once every
        request has an outcome."""
        self._ws._core.end_input(self._ctx)

    async def cancel(self) -> None:
        """Cancel every unfinished request of this context.

        Their audio stops at once (what arrives before the server's
        acknowledgement is discarded) and each ends as ``Cancelled``. Returns
        when the server has acknowledged; the context is then ready for new
        requests.
        """
        ws = self._ws
        core = ws._core
        conn = core.cancel(self._ctx)
        if conn is not None:
            # The cancel must reach the wire even if this task is cancelled
            # (barge-in often lands while an error is being handled).
            await asyncio.shield(ws._spawn(ws._write(conn, core.cancel_message(self._ctx))))
        while self._ctx.cancels_pending and not core.closed:
            await _signal(self._ctx.cancel_signal).wait()

    async def events(self) -> AsyncIterator[WebSocketEvent[AsyncRequestHandle]]:
        """Yield ``Chunk``, ``Done``, ``Failed`` and ``Cancelled`` in order.

        Ends after ``end_input()`` once every request has its outcome."""
        core = self._ws._core
        ctx = self._ctx
        while True:
            event = core.pop_event(ctx)
            if event is not None:
                yield event
                continue
            if core.events_finished(ctx):
                return
            await _signal(ctx.signal).wait()

    async def audio(self) -> AsyncIterator[bytes]:
        """Yield only the audio, and raise the first failure."""
        async for event in self.events():
            if isinstance(event, Chunk):
                yield event.audio
            elif isinstance(event, Failed):
                raise event.error

    @override
    def __repr__(self) -> str:
        return f"AsyncContext(context_id={self.context_id!r})"


class AsyncTTSWebSocket:
    """Many streaming requests over one WebSocket (BEHAVIOR.md §6).

    ``await client.tts.websocket()`` (or ``await ws.connect()``, or
    ``async with``) opens it ahead of time so the first request pays no
    handshake; otherwise the first send opens it. Reconnects, resends and
    holds are handled for you; every unfinished request gets an outcome when a
    connection is lost.
    """

    def __init__(
        self,
        client: AsyncBakbak,
        defaults: RequestDefaults,
        on_error: Callable[[APIStatusError], None] | None,
    ) -> None:
        self._client = client
        self._defaults: dict[str, object] = dict(defaults)
        self._on_error = on_error
        driver = Driver(
            new_signal=self._new_signal,
            schedule_flush=self._schedule_flush,
            schedule_retry=self._schedule_retry,
            schedule_reconnect=self._schedule_reconnect,
            abort=self._abort,
            notice=self._notice,
        )
        self._core = Core(driver, max_retries=client.max_retries, read_timeout=client.timeout.read)
        self._send_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._watchdog: asyncio.Task[None] | None = None

    # Public API

    @property
    def url(self) -> str:
        """The WebSocket URL (it never contains the API key)."""
        return self._client.websocket_url

    @property
    def connected(self) -> bool:
        """Whether a connection is open and accepting requests."""
        return self._core.usable()

    def __await__(self) -> Generator[Any, None, Self]:
        return self.connect().__await__()

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def connect(self) -> Self:
        """Open the connection now, so the first request pays no handshake.

        Raises:
            AuthenticationError: the key was refused (or another
                ``APIStatusError`` for another refusal).
            APIConnectionError: the server could not be reached.
        """
        if self._core.closed:
            raise BakbakError("This WebSocket is closed.")
        async with self._connect_lock:
            if not self._core.usable():
                await self._open()
        return self

    def context(
        self, context_id: str | None = None, **defaults: Unpack[RequestDefaults]
    ) -> AsyncContext:
        """A context: requests on it play in order.

        Args:
            context_id: 1 to 128 characters of ``A-Z a-z 0-9 . _ -``; a random
                ``ctx_…`` id when left out.
            **defaults: request fields for this context, over the WebSocket's.
        """
        if self._core.closed:
            raise BakbakError("This WebSocket is closed.")
        return AsyncContext(self, check_context_id(context_id), dict(defaults))

    async def close(self) -> None:
        """Close the WebSocket. Unfinished requests fail with
        ``WebSocketClosedError``."""
        connections = self._core.close()
        await asyncio.gather(*(self._close_transport(c.transport) for c in connections))
        current = asyncio.current_task()
        pending = [t for t in self._tasks if t is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @override
    def __repr__(self) -> str:
        return f"AsyncTTSWebSocket(url={self.url!r}, connected={self.connected!r})"

    # Driver: called by the core, never blocking

    def _new_signal(self) -> _Signal:
        return _Signal()

    def _schedule_flush(self) -> None:
        self._spawn(self._flush())

    def _schedule_retry(self, request: Request, delay: float) -> None:
        self._spawn(self._retry_later(request, delay))

    def _schedule_reconnect(self, delay: float, *, force: bool) -> None:
        self._spawn(self._reconnect_later(delay, force))

    def _abort(self, connection: Connection) -> None:
        self._spawn(self._close_transport(connection.transport))

    def _notice(self, error: APIStatusError) -> None:
        if self._on_error is None:
            logger.debug("WebSocket error about no request: %s", error.message)
            return
        try:
            self._on_error(error)
        except Exception:
            logger.debug("the on_error callback raised", exc_info=True)

    # Internals

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _open(self) -> None:
        """Open a connection, retrying what can be retried."""
        url = self.url
        retries = 0
        while True:
            started = time.monotonic()
            try:
                transport = await _Connect(
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
            await self._client._sleep(delay)
        logger.debug(
            "WebSocket connected to %s in %.0f ms",
            redact_url(url),
            (time.monotonic() - started) * 1000,
        )
        if self._core.closed:  # closed while the handshake ran
            await self._close_transport(transport)
            return
        conn = self._core.opened(transport)
        self._spawn(self._read(conn))
        self._ensure_watchdog()

    async def _flush(self, force_connect: bool = False) -> None:
        """Open a connection if a request needs one, then send what can go."""
        core = self._core

        def needed() -> bool:
            return (
                not core.closed and not core.usable() and (force_connect or core.needs_connection())
            )

        if needed():
            async with self._connect_lock:
                if needed():
                    try:
                        await self._open()
                    except Exception as exc:
                        core.fail_waiting(open_failure(exc))
                        return
        async with self._send_lock:
            for conn, wire in core.take_sendable():
                if not await self._send(conn, wire):
                    break

    async def _write(self, conn: Connection, wire: str) -> None:
        async with self._send_lock:
            await self._send(conn, wire)

    async def _send(self, conn: Connection, wire: str) -> bool:
        transport: ClientConnection = conn.transport
        try:
            await transport.send(wire)
        except Exception:
            # The reader sees the close and resends or fails what was in flight.
            logger.debug("WebSocket send failed on connection %d", conn.id, exc_info=True)
            await self._close_transport(transport)
            return False
        return True

    async def _read(self, conn: Connection) -> None:
        transport: ClientConnection = conn.transport
        try:
            async for message in transport:
                if isinstance(message, str):
                    self._core.on_message(conn, message)
        except ConnectionClosed:
            pass
        except Exception:
            logger.debug("WebSocket reader failed on connection %d", conn.id, exc_info=True)
        finally:
            if transport.close_code is None:
                await self._close_transport(transport)
            self._core.on_close(conn, transport.close_code or 1006, transport.close_reason or "")

    async def _close_transport(self, transport: ClientConnection) -> None:
        """Close within about a second, even if the peer never answers."""
        try:
            await asyncio.wait_for(transport.close(), WS_CLOSE_TIMEOUT + 0.25)
        except Exception:
            logger.debug("closing a WebSocket failed; aborting it", exc_info=True)
            transport.transport.abort()

    async def _retry_later(self, request: Request, delay: float) -> None:
        await self._client._sleep(delay)
        self._core.retry_due(request)

    async def _reconnect_later(self, delay: float, force: bool) -> None:
        if delay:
            await self._client._sleep(delay)
        await self._flush(force_connect=force)

    def _ensure_watchdog(self) -> None:
        if self._core.read_timeout is None:
            return
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.get_running_loop().create_task(self._watch())
            self._tasks.add(self._watchdog)
            self._watchdog.add_done_callback(self._tasks.discard)

    async def _watch(self) -> None:
        """Fail requests that hear nothing for the read timeout."""
        core = self._core
        idle = core.read_timeout or 60.0
        while not core.closed and core.connections:
            wait, sends = core.check_timeouts()
            for conn, wire in sends:
                await self._write(conn, wire)
            await asyncio.sleep(idle if wait is None else min(wait, idle))
