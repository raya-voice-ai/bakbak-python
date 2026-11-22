"""The WebSocket protocol (BEHAVIOR.md §6) as a state machine without I/O.

The async and the sync clients drive this one implementation, so they cannot
differ in how they route messages, cancel, hold, resend or fail requests. The
core never blocks: it changes state, appends events, and asks its driver to do
anything that waits (open a connection, sleep, close a connection) through
the ``Driver`` callbacks, which only schedule work.

Vocabulary:

- A *request* is one synthesis request. It is ``QUEUED`` (not sent yet),
  ``INFLIGHT`` (sent, no terminal message yet), ``BACKOFF`` (waiting to be
  resent after a retryable error), ``HELD`` (refused by a draining connection,
  waiting until its context has nothing left there) or ``DONE``.
- A *context* sends its requests in order. Its unsent requests wait in
  ``pending`` (sorted by send order); its in-flight ones are in ``inflight``,
  all on the connection the context is ``bound`` to.
- Messages for a context go to its oldest in-flight request: the server runs a
  context's requests one after another and answers each in order. A request
  that already has its outcome may stay at the head of ``inflight`` (its audio
  was invalid): the server's remaining messages for it are then dropped.
- Only *busy* contexts (something unfinished, in flight or being cancelled)
  are in the routing table, so a long-lived WebSocket does not grow with the
  number of contexts it has served (§6.6).
"""

from __future__ import annotations

import binascii
import enum
import json
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .._constants import WS_HEALTHY_AFTER, WS_MAX_CONTEXTS, WS_MAX_UNFINISHED
from .._exceptions import (
    APIStatusError,
    APITimeoutError,
    BakbakError,
    WebSocketClosedError,
    event_status,
    make_interrupted_error,
    make_status_error,
)
from .._retry import backoff_delay
from .._streaming import SampleAligner, decode_audio
from .._utils import as_mapping, logger
from .events import Cancelled, Chunk, Done, Failed

__all__ = ["Connection", "Context", "Core", "Driver", "Request", "Signal", "State"]

_DRAINING_PHRASE = "Reconnect and send the request again"


class State(enum.Enum):
    QUEUED = "queued"
    INFLIGHT = "inflight"
    BACKOFF = "backoff"
    HELD = "held"
    DONE = "done"


class Outcome(enum.Enum):
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WITHDRAWN = "withdrawn"  # its send() was cancelled before it was written


class Signal(Protocol):
    """Wakes whoever waits on a context, a request or the send capacity."""

    def notify(self) -> None: ...


class _Reconnect(Protocol):
    def __call__(self, delay: float, *, force: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class Driver:
    """What the core asks of the async or sync client. None of it blocks.

    The client passes its own private methods, so they stay off its public
    surface.
    """

    new_signal: Callable[[], Signal]
    schedule_flush: Callable[[], None]
    """Send whatever became sendable (opening a connection if needed)."""
    schedule_retry: Callable[[Request, float], None]
    """After ``delay`` (with the injectable sleep), call ``retry_due``."""
    schedule_reconnect: _Reconnect
    """After ``delay``, flush; with ``force``, open a connection even if
    nothing is waiting to be sent."""
    abort: Callable[[Connection], None]
    """Close a connection from the client side (code 1000)."""
    notice: Callable[[APIStatusError], None]
    """Deliver an error that is about no request."""


class Connection:
    """One WebSocket connection."""

    __slots__ = ("closed", "closing", "draining", "id", "opened_at", "received", "transport")

    def __init__(self, conn_id: int, transport: Any, opened_at: float) -> None:
        self.id = conn_id
        self.transport: Any = transport
        self.opened_at = opened_at
        self.draining = False
        self.closing = False  # the client asked to close it
        self.closed = False
        self.received = False

    def __repr__(self) -> str:
        return f"<connection {self.id}>"


class Request:
    """One request, as the core tracks it. ``handle`` is the driver's view."""

    __slots__ = (
        "aligner",
        "audio_bytes",
        "conn",
        "context",
        "drained",
        "error",
        "had_audio",
        "handle",
        "outcome",
        "queued_events",
        "request_id",
        "retries",
        "seq",
        "signal",
        "state",
        "wire",
    )

    def __init__(self, seq: int, context: Context, wire: str, width: int, signal: Signal) -> None:
        self.seq = seq
        self.context = context
        self.wire = wire
        self.aligner = SampleAligner(width)
        self.signal = signal
        self.state = State.QUEUED
        self.conn: Connection | None = None
        self.audio_bytes = 0
        self.had_audio = False
        self.request_id: str | None = None
        self.retries = 0
        self.drained = False  # a draining connection refused it once already
        self.outcome: Outcome | None = None
        self.error: BakbakError | None = None
        self.queued_events = 0
        self.handle: Any = None


class Context:
    """One context, as the core tracks it."""

    __slots__ = (
        "__weakref__",
        "bound",
        "cancel_conn",
        "cancel_epoch",
        "cancel_signal",
        "cancels_pending",
        "events",
        "id",
        "inflight",
        "input_ended",
        "last_activity",
        "pending",
        "signal",
        "unfinished",
    )

    def __init__(self, context_id: str, signal: Signal, cancel_signal: Signal) -> None:
        self.id = context_id
        self.signal = signal
        self.cancel_signal = cancel_signal
        self.pending: deque[Request] = deque()
        self.inflight: deque[Request] = deque()
        self.bound: Connection | None = None
        self.cancels_pending = 0
        self.cancel_conn: Connection | None = None
        self.cancel_epoch = 0  # bumped by every cancel(): stops a send_stream feeding it
        self.events: deque[tuple[Request, Any]] = deque()
        self.input_ended = False
        self.last_activity = 0.0
        self.unfinished = 0

    @property
    def busy(self) -> bool:
        return self.unfinished > 0 or self.cancels_pending > 0 or bool(self.inflight)


def _insert_in_order(pending: deque[Request], request: Request) -> None:
    """Put a request back among the unsent ones, keeping send order."""
    for i, other in enumerate(pending):
        if other.seq > request.seq:
            pending.insert(i, request)
            return
    pending.append(request)


def _is_draining(status: int | None, message: object) -> bool:
    """A draining refusal: 503 and the reconnect phrase. ``error_code`` alone
    never is: ``service_unavailable`` also means billing is down (§6.4)."""
    return status == 503 and isinstance(message, str) and _DRAINING_PHRASE in message


class Core:
    """Routing, ordering, cancels, holds, resends and failures."""

    def __init__(
        self,
        driver: Driver,
        *,
        max_retries: int,
        read_timeout: float | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.driver = driver
        self.max_retries = max_retries
        self.read_timeout = read_timeout
        self.clock = clock
        # The routing table: busy contexts only. Every live context, busy or
        # not, can still be found by id while its owner holds it.
        self.contexts: dict[str, Context] = {}
        self._known: weakref.WeakValueDictionary[str, Context] = weakref.WeakValueDictionary()
        self.connections: list[Connection] = []
        self.current: Connection | None = None
        self.unfinished = 0
        self.failures = 0  # consecutive connections lost before they proved healthy
        self.closed = False
        self.capacity = driver.new_signal()
        self._seq = 0
        self._conn_seq = 0

    # Contexts and capacity

    def context(self, context_id: str) -> Context:
        ctx = self._known.get(context_id)
        if ctx is None:
            ctx = Context(context_id, self.driver.new_signal(), self.driver.new_signal())
            self._known[context_id] = ctx
        return ctx

    def at_capacity(self, ctx: Context) -> bool:
        """Whether a new request on ``ctx`` must wait (§6.2 back-pressure)."""
        if self.unfinished >= WS_MAX_UNFINISHED:
            return True
        if ctx.busy:
            return False
        return sum(1 for c in self.contexts.values() if c.busy) >= WS_MAX_CONTEXTS

    def submit(
        self, ctx: Context, wire: str, width: int, make_handle: Callable[[Request], Any]
    ) -> Request:
        self._seq += 1
        request = Request(self._seq, ctx, wire, width, self.driver.new_signal())
        request.handle = make_handle(request)
        ctx.pending.append(request)
        ctx.unfinished += 1
        self.unfinished += 1
        self.contexts[ctx.id] = ctx
        return request

    def withdraw(self, request: Request) -> bool:
        """Take back a request whose ``send()`` was abandoned, if it was not
        written yet. Its caller never got a handle: no event is emitted."""
        if request.state is not State.QUEUED:
            return False
        ctx = request.context
        ctx.pending.remove(request)
        request.state = State.DONE
        request.outcome = Outcome.WITHDRAWN
        ctx.unfinished -= 1
        self.unfinished -= 1
        request.signal.notify()
        ctx.signal.notify()
        self.capacity.notify()
        self._forget_if_idle(ctx)
        return True

    def end_input(self, ctx: Context) -> None:
        ctx.input_ended = True
        ctx.signal.notify()

    def _forget_if_idle(self, ctx: Context) -> None:
        if not ctx.busy and self.contexts.get(ctx.id) is ctx:
            del self.contexts[ctx.id]

    # Sending

    def usable(self) -> bool:
        conn = self.current
        return conn is not None and not conn.closed and not conn.draining

    def _waiting_for_connection(self, ctx: Context) -> bool:
        return (
            not ctx.cancels_pending
            and bool(ctx.pending)
            and ctx.pending[0].state is State.QUEUED
            and ctx.bound is None
        )

    def needs_connection(self) -> bool:
        """Whether a request is ready to go and there is no usable connection."""
        return (
            not self.closed
            and not self.usable()
            and any(self._waiting_for_connection(c) for c in self.contexts.values())
        )

    def opened(self, transport: Any) -> Connection:
        self._conn_seq += 1
        conn = Connection(self._conn_seq, transport, self.clock())
        self.connections.append(conn)
        self.current = conn
        return conn

    def take_sendable(self) -> list[tuple[Connection, str]]:
        """Mark every request that may be sent now as in flight on the current
        connection, and return the messages, in send order. The driver must
        write all of them (it shields the writes from cancellation)."""
        conn = self.current
        if self.closed or conn is None or conn.closed or conn.draining:
            return []
        heads: list[Context] = [
            c
            for c in self.contexts.values()
            if not c.cancels_pending
            and c.pending
            and c.pending[0].state is State.QUEUED
            and (c.bound is None or c.bound is conn)
        ]
        out: list[tuple[Connection, str]] = []
        now = self.clock()
        while heads:
            ctx = min(heads, key=lambda c: c.pending[0].seq)
            request = ctx.pending.popleft()
            if not ctx.inflight:
                ctx.last_activity = now
            request.state = State.INFLIGHT
            request.conn = conn
            ctx.inflight.append(request)
            ctx.bound = conn
            out.append((conn, request.wire))
            if not ctx.pending or ctx.pending[0].state is not State.QUEUED:
                heads.remove(ctx)
        return out

    def fail_waiting(self, error: BakbakError) -> None:
        """No connection could be opened: fail every request waiting for one."""
        for ctx in list(self.contexts.values()):
            if ctx.bound is not None:
                continue
            for request in [r for r in ctx.pending if r.state is State.QUEUED]:
                ctx.pending.remove(request)
                self._finish(request, Outcome.FAILED, error)

    # Receiving

    def on_message(self, conn: Connection, text: str) -> None:
        try:
            message = as_mapping(json.loads(text))
        except ValueError:
            logger.debug("ignoring a WebSocket message that is not JSON")
            return
        if message is None:
            return
        conn.received = True
        self.failures = 0
        kind = message.get("type")
        context_id = message.get("context_id")
        ctx = self.contexts.get(context_id) if isinstance(context_id, str) else None
        if ctx is None:
            if kind == "error" and context_id is None:
                self.driver.notice(make_status_error(event_status(message), dict(message)))
            else:
                logger.debug("dropping a %s message for a context with nothing in flight", kind)
            return
        ctx.last_activity = self.clock()
        if kind == "cancelled":
            self._cancel_acknowledged(ctx, conn)
            return
        if ctx.cancels_pending:
            return  # cancelled: discard everything until the acknowledgement
        if ctx.bound is not conn or not ctx.inflight:
            logger.debug("dropping a %s message with no request in flight", kind)
            return
        request = ctx.inflight[0]
        if request.state is State.DONE:
            # It already failed (invalid audio); drop the rest of its stream.
            if kind in ("done", "error"):
                self._pop_inflight(ctx)
            return
        request_id = message.get("request_id")
        if isinstance(request_id, str) and request_id:
            request.request_id = request_id
        if kind == "chunk":
            self._on_chunk(ctx, request, message)
        elif kind == "done":
            self._pop_inflight(ctx)
            self._finish(request, Outcome.DONE)
        elif kind == "error":
            self._on_error(conn, ctx, request, dict(message))

    def _on_chunk(self, ctx: Context, request: Request, message: Any) -> None:
        try:
            raw = decode_audio(message)
        except (binascii.Error, ValueError):
            # The stream broke here (§6.4): fail the request now, and keep it
            # at the head of the context until its stream ends, so its other
            # messages are not taken for the next request's.
            self._finish(
                request,
                Outcome.FAILED,
                make_interrupted_error(
                    None,
                    audio_bytes=request.audio_bytes,
                    header_request_id=request.request_id,
                    default_message="The server sent audio that is not valid base64.",
                ),
            )
            return
        if not raw:
            return
        request.had_audio = True
        audio = request.aligner.feed(raw)
        if audio:
            request.audio_bytes += len(audio)
            self._emit(ctx, request, Chunk(audio, request.handle))

    def _on_error(
        self, conn: Connection, ctx: Context, request: Request, message: dict[str, object]
    ) -> None:
        self._pop_inflight(ctx)
        if request.had_audio:
            self._finish(
                request,
                Outcome.FAILED,
                make_interrupted_error(message, audio_bytes=request.audio_bytes),
            )
            return
        error = make_status_error(event_status(message), message)
        if _is_draining(error.status, message.get("message")):
            # Stop sending here; new requests go to a new connection.
            logger.debug("connection %d is draining", conn.id)
            conn.draining = True
            if self.current is conn:
                self.current = None
            if not request.drained:
                # Hold it (and, through ordering, everything after it in its
                # context) until the context has nothing left on this
                # connection, then resend it on a new one. No retry is spent.
                request.drained = True
                request.state = State.HELD
                request.conn = conn
                _insert_in_order(ctx.pending, request)
                if ctx.bound is None:
                    self._release_held(ctx)
                self._close_if_drained(conn)
                return
            # Refused while draining a second time: an ordinary retryable
            # error, resent on a new connection. Later requests of a draining
            # connection are refused too, so this keeps the context's order.
            self._close_if_drained(conn)
            if request.retries < self.max_retries:
                self._back_off(request, error)
                return
            self._finish(request, Outcome.FAILED, error)
            return
        if error.retryable and not ctx.inflight and request.retries < self.max_retries:
            # Nothing later of this context was sent, so resending keeps order.
            self._back_off(request, error)
            return
        self._finish(request, Outcome.FAILED, error)

    def _back_off(self, request: Request, error: APIStatusError) -> None:
        request.retries += 1
        request.state = State.BACKOFF
        request.conn = None
        _insert_in_order(request.context.pending, request)
        delay = backoff_delay(request.retries)
        logger.debug("retrying a request in %.2f s after: %s", delay, error.message)
        self.driver.schedule_retry(request, delay)

    def _pop_inflight(self, ctx: Context) -> None:
        conn = ctx.bound
        ctx.inflight.popleft()
        if ctx.inflight:
            return
        ctx.bound = None
        if conn is not None and conn.draining:
            self._release_held(ctx)
            self._close_if_drained(conn)
        self._forget_if_idle(ctx)

    def _release_held(self, ctx: Context) -> None:
        """``ctx`` has nothing left on a draining connection: what it holds,
        and what it queued meanwhile, goes to a new connection now. Like the
        resends after a 1001, this costs no retry."""
        for request in ctx.pending:
            if request.state is State.HELD:
                request.state = State.QUEUED
                request.conn = None
        if self._waiting_for_connection(ctx):
            self.driver.schedule_flush()

    def _close_if_drained(self, conn: Connection) -> None:
        """Close a draining connection once no context needs it any more,
        rather than wait for the server (§6.4)."""
        if conn.closed or conn.closing or not conn.draining:
            return
        if any(c.bound is conn or c.cancel_conn is conn for c in self.contexts.values()):
            return
        conn.closing = True
        self.driver.abort(conn)

    def retry_due(self, request: Request) -> None:
        if request.state is State.BACKOFF:
            request.state = State.QUEUED
            self.driver.schedule_flush()

    # Closing and reconnecting

    def on_close(self, conn: Connection, code: int, reason: str) -> None:
        if conn.closed:
            return
        conn.closed = True
        if conn in self.connections:
            self.connections.remove(conn)
        if self.current is conn:
            self.current = None
        if self.closed:
            return
        logger.debug("connection %d closed with %d", conn.id, code)
        if code in (1008, 1009):
            error = WebSocketClosedError(
                f"The server closed the WebSocket ({code}): {reason or 'policy violation'}",
                code=code,
                reason=reason,
                retryable=False,
            )
            self._fail_all(error)
            return
        resend = 0
        unfinished = False
        for ctx in list(self.contexts.values()):
            if ctx.cancel_conn is conn:
                self._settle_cancels(ctx)
            if ctx.bound is conn:
                for request in list(ctx.inflight):
                    if request.state is State.DONE:
                        continue  # already failed; its stream is gone with the connection
                    unfinished = True
                    if request.had_audio:
                        self._finish(
                            request,
                            Outcome.FAILED,
                            make_interrupted_error(
                                None,
                                audio_bytes=request.audio_bytes,
                                header_request_id=request.request_id,
                                default_message="The WebSocket closed before the audio was complete.",
                            ),
                        )
                    else:
                        request.state = State.QUEUED
                        request.conn = None
                        _insert_in_order(ctx.pending, request)
                        resend += 1
                ctx.inflight.clear()
                ctx.bound = None
            for request in ctx.pending:
                if request.state is State.HELD and request.conn is conn:
                    request.state = State.QUEUED
                    request.conn = None
                    resend += 1
            self._forget_if_idle(ctx)
        # Going away (1001), a draining connection, or a 1000 that arrives with
        # work unfinished: reconnect at once and resend; this costs no retry.
        if code == 1001 or conn.draining or conn.closing or (code == 1000 and unfinished):
            self.driver.schedule_reconnect(0.0, force=code == 1001)
            return
        if resend == 0:
            # Nothing was in flight here, but requests may be waiting for a
            # connection (queued behind a cancel this close just settled).
            if self.needs_connection():
                self.driver.schedule_reconnect(0.0, force=False)
            return
        healthy = conn.received or self.clock() - conn.opened_at >= WS_HEALTHY_AFTER
        self.failures = 0 if healthy else self.failures + 1
        if self.failures > self.max_retries:
            self.failures = 0
            self.fail_waiting(
                WebSocketClosedError(
                    f"The WebSocket kept closing ({code}) before any answer arrived.",
                    code=code,
                    reason=reason,
                    retryable=True,
                )
            )
            return
        delay = backoff_delay(self.failures) if self.failures else 0.0
        self.driver.schedule_reconnect(delay, force=False)

    def _fail_all(self, error: BakbakError) -> None:
        for ctx in list(self.contexts.values()):
            requests = sorted([*ctx.inflight, *ctx.pending], key=lambda r: r.seq)
            ctx.inflight.clear()
            ctx.pending.clear()
            ctx.bound = None
            self._settle_cancels(ctx)
            for request in requests:
                self._finish(request, Outcome.FAILED, error)
            self._forget_if_idle(ctx)

    def close(self) -> list[Connection]:
        """The user closed the WebSocket: fail what is unfinished and return
        the connections to close."""
        if self.closed:
            return []
        self._fail_all(
            WebSocketClosedError(
                "The WebSocket was closed by the client.", code=1000, retryable=False
            )
        )
        self.closed = True
        for ctx in list(self._known.values()):
            ctx.signal.notify()
            ctx.cancel_signal.notify()
        self.capacity.notify()
        conns = list(self.connections)
        self.current = None
        return conns

    # Cancelling

    def cancel(self, ctx: Context) -> Connection | None:
        """End every unfinished request of ``ctx`` as cancelled, discard all
        of its audio not handed out yet, and stop a ``send_stream`` feeding
        it: a cancel means silence now (§6.3). Returns the connection to send
        the cancel message on, or ``None`` if nothing on the server needs
        cancelling."""
        ctx.cancel_epoch += 1
        self._discard_queued_audio(ctx)
        conn = ctx.bound
        requests = sorted([*ctx.inflight, *ctx.pending], key=lambda r: r.seq)
        had_inflight = bool(ctx.inflight)
        ctx.inflight.clear()
        ctx.pending.clear()
        ctx.bound = None
        for request in requests:
            self._finish(request, Outcome.CANCELLED)
        ctx.signal.notify()
        if not had_inflight or conn is None or conn.closed:
            self._forget_if_idle(ctx)
            return None
        ctx.cancels_pending += 1
        ctx.cancel_conn = conn
        ctx.last_activity = self.clock()
        self.contexts[ctx.id] = ctx  # busy until the acknowledgement
        return conn

    @staticmethod
    def _discard_queued_audio(ctx: Context) -> None:
        kept: deque[tuple[Request, Any]] = deque()
        for request, event in ctx.events:
            if isinstance(event, Chunk):
                request.queued_events -= 1
                request.audio_bytes -= len(event.audio)  # it was never delivered
            else:
                kept.append((request, event))
        ctx.events = kept

    def cancel_message(self, ctx: Context) -> str:
        return json.dumps({"context_id": ctx.id, "cancel": True}, separators=(",", ":"))

    def _cancel_acknowledged(self, ctx: Context, conn: Connection) -> None:
        if not ctx.cancels_pending or ctx.cancel_conn is not conn:
            return
        ctx.cancels_pending -= 1
        if not ctx.cancels_pending:
            self._settle_cancels(ctx)

    def _settle_cancels(self, ctx: Context) -> None:
        """The cancel is acknowledged, or its connection is gone. Requests the
        context queued meanwhile go out now: a new connection is opened if
        needed, so nothing waits behind a cancel that will never be answered."""
        conn = ctx.cancel_conn
        ctx.cancels_pending = 0
        ctx.cancel_conn = None
        ctx.cancel_signal.notify()
        self.capacity.notify()
        if self._waiting_for_connection(ctx):
            self.driver.schedule_flush()
        if conn is not None:
            self._close_if_drained(conn)
        self._forget_if_idle(ctx)

    # Read timeouts

    def check_timeouts(self) -> tuple[float | None, list[tuple[Connection, str]]]:
        """Fail requests that waited longer than the read timeout. Returns the
        seconds until the next deadline and cancel messages to send."""
        if self.read_timeout is None or self.closed:
            return None, []
        now = self.clock()
        soonest: float | None = None
        sends: list[tuple[Connection, str]] = []
        for ctx in list(self.contexts.values()):
            if not ctx.inflight and not ctx.cancels_pending:
                continue
            deadline = ctx.last_activity + self.read_timeout
            if deadline > now:
                soonest = deadline if soonest is None else min(soonest, deadline)
                continue
            ctx.last_activity = now
            if ctx.cancels_pending:
                # The server did not acknowledge a cancel: the connection is
                # unusable for this context. Close it; the close handling
                # resends or fails what other contexts had on it.
                if ctx.cancel_conn is not None and not ctx.cancel_conn.closing:
                    ctx.cancel_conn.closing = True
                    self.driver.abort(ctx.cancel_conn)
                continue
            conn = ctx.bound
            for request in list(ctx.inflight):
                if request.had_audio:
                    error: BakbakError = make_interrupted_error(
                        None,
                        audio_bytes=request.audio_bytes,
                        header_request_id=request.request_id,
                        default_message="No audio arrived within the read timeout.",
                    )
                else:
                    error = APITimeoutError("No answer arrived within the read timeout.")
                self._finish(request, Outcome.FAILED, error)
            ctx.inflight.clear()
            ctx.bound = None
            if conn is not None and not conn.closed:
                # Stop the server's work and drop its late messages.
                ctx.cancels_pending += 1
                ctx.cancel_conn = conn
                sends.append((conn, self.cancel_message(ctx)))
            else:
                self._forget_if_idle(ctx)
            soonest = now + self.read_timeout if soonest is None else soonest
        return (None if soonest is None else max(soonest - now, 0.0)), sends

    # Events

    def _emit(self, ctx: Context, request: Request, event: Any) -> None:
        ctx.events.append((request, event))
        request.queued_events += 1
        ctx.signal.notify()

    def _finish(self, request: Request, outcome: Outcome, error: BakbakError | None = None) -> None:
        if request.state is State.DONE:
            return
        request.state = State.DONE
        request.outcome = outcome
        request.error = error
        ctx = request.context
        ctx.unfinished -= 1
        self.unfinished -= 1
        if outcome is Outcome.DONE:
            leftover = request.aligner.leftover()
            if leftover:
                logger.debug("dropping %d bytes of an incomplete last sample", leftover)
            event: Any = Done(request.handle)
        elif outcome is Outcome.CANCELLED:
            event = Cancelled(request.handle)
        else:
            assert error is not None
            event = Failed(request.handle, error)
        self._emit(ctx, request, event)
        request.signal.notify()
        self.capacity.notify()
        self._forget_if_idle(ctx)

    def pop_event(self, ctx: Context) -> Any | None:
        if not ctx.events:
            return None
        request, event = ctx.events.popleft()
        request.queued_events -= 1
        return event

    def events_finished(self, ctx: Context) -> bool:
        return not ctx.events and not ctx.unfinished and (ctx.input_ended or self.closed)

    def pop_request_event(self, request: Request) -> tuple[Any | None, bool]:
        """The next event of ``request``, consuming any earlier request's
        events before it. Returns ``(event, finished)``; ``(None, False)``
        means wait."""
        ctx = request.context
        while ctx.events:
            owner, event = ctx.events[0]
            if owner.seq < request.seq:
                ctx.events.popleft()
                owner.queued_events -= 1
                continue
            if owner is request:
                ctx.events.popleft()
                request.queued_events -= 1
                return event, False
            return None, True  # a later request's events: this one's were read elsewhere
        return None, request.state is State.DONE and not request.queued_events
