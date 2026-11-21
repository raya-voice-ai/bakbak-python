"""Regression tests for WebSocket issues found in review: every unfinished
request gets an outcome, asyncio cancellation keeps the protocol consistent,
redirects are refused, contexts are released, and closing is bounded."""

from __future__ import annotations

import asyncio
import gc
import http
import json
import threading
import time
import tracemalloc
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Response
from ws_server import Server, SilentPeer, cancelled, chunk, done

import bakbak
from bakbak import AsyncBakbak, Bakbak, Timeout
from bakbak.websocket import Chunk

DEFAULTS: dict[str, Any] = {
    "voice_id": "v",
    "language": "en",
    "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
}


async def _answer_all(server: Server, ws: ServerConnection, index: int) -> None:
    """Answers every request with one chunk and done, and acknowledges cancels."""
    try:
        async for raw in ws:
            message = json.loads(raw)
            server.received.append((index, message))
            cid = message["context_id"]
            if message.get("cancel"):
                await ws.send(cancelled(cid))
                continue
            rid = message.get("text", "req")
            await ws.send(chunk(cid, b"\x01", rid))
            await ws.send(done(cid, rid))
    except Exception:
        return


async def _silent(server: Server, ws: ServerConnection, index: int) -> None:
    try:
        while True:
            await server.recv(ws, index, timeout=30)
    except Exception:
        return


def _gate_sends(ws: Any, marker: str, gate: asyncio.Event) -> None:
    """Make writes of messages containing ``marker`` wait for ``gate``: a
    write stuck under TCP back-pressure."""
    transport = ws._core.current.transport
    original = transport.send

    async def send(message: str) -> None:
        if marker in message:
            await gate.wait()
        await original(message)

    transport.send = send


# 1. A request queued behind an unacknowledged cancel gets an outcome


async def test_request_behind_a_timed_out_cancel_gets_an_outcome() -> None:
    with Server(_silent) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url, timeout=Timeout(read=0.3))
        ws = client.tts.websocket(**DEFAULTS)
        ctx = ws.context("c1")
        first = await ctx.send(text="A.")
        with pytest.raises(bakbak.APITimeoutError):
            await first.result(timeout=5)
        second = await ctx.send(text="B.")  # queued behind the unacknowledged cancel
        with pytest.raises(bakbak.APITimeoutError):
            await second.result(timeout=5)  # an outcome, not a hang
        await ws.close()
        await client.close()


def test_request_behind_a_timed_out_cancel_gets_an_outcome_sync() -> None:
    with Server(_silent) as server:
        client = Bakbak(api_key="k", base_url=server.base_url, timeout=Timeout(read=0.3))
        ws = client.tts.websocket(**DEFAULTS)
        ctx = ws.context("c1")
        with pytest.raises(bakbak.APITimeoutError):
            ctx.send(text="A.").result(timeout=5)
        with pytest.raises(bakbak.APITimeoutError):
            ctx.send(text="B.").result(timeout=5)
        ws.close()
        client.close()


async def test_request_behind_a_cancel_survives_the_connection_dropping() -> None:
    async def handler(server: Server, ws: ServerConnection, index: int) -> None:
        if index == 0:
            await server.recv(ws, index)  # A
            await ws.send(chunk("c1", b"\x01\x02"))
            await server.recv(ws, index)  # the cancel, never acknowledged
            await asyncio.sleep(0.2)
            await ws.close(1011, "internal error")
            return
        await _answer_all(server, ws, index)

    with Server(handler) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        ws = client.tts.websocket(**DEFAULTS)
        ctx = ws.context("c1")
        first = await ctx.send(text="A.")
        while first.audio_bytes == 0:
            await asyncio.sleep(0.01)
        cancelling = asyncio.create_task(ctx.cancel())
        await asyncio.sleep(0.05)
        second = await ctx.send(text="B.")  # barge-in: the next reply
        await cancelling
        result = await second.result(timeout=5)
        assert result.request_id == "B."
        await ws.close()
        await client.close()


# 2. asyncio cancellation keeps the core consistent with the wire


async def test_cancelled_send_mid_batch_keeps_answers_on_the_right_requests() -> None:
    with Server(_answer_all) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        ws = await client.tts.websocket(**DEFAULTS)
        gate = asyncio.Event()
        _gate_sends(ws, '"text":"X"', gate)
        a = ws.context("a")
        sending_x = asyncio.create_task(a.send(text="X"))
        sending_y = asyncio.create_task(a.send(text="Y"))
        await asyncio.sleep(0.1)  # one flush took [X, Y] and is stuck writing X
        sending_x.cancel()
        gate.set()
        y = await sending_y
        z = await a.send(text="Z")
        assert (await y.result(timeout=5)).request_id == "Y"
        assert (await z.result(timeout=5)).request_id == "Z"
        assert [t for t in server.texts() if t in "XYZ"] == ["X", "Y", "Z"]
        await ws.close()
        await client.close()


async def test_send_cancelled_while_connecting_is_withdrawn() -> None:
    async def slow_handshake(connection: Any, request: Any) -> None:
        await asyncio.sleep(0.5)

    with Server(_answer_all, process_request=slow_handshake) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        ws = client.tts.websocket(**DEFAULTS)
        ctx = ws.context("c1")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ctx.send(text="Abandoned."), 0.1)
        ctx.end_input()
        events = await asyncio.wait_for(_collect(ctx), 3)  # ends: nothing unfinished
        assert events == []
        await (await ws.context("c2").send(text="B.")).result(timeout=5)
        await asyncio.sleep(0.2)
        assert server.texts() == ["B."]  # the abandoned request was never sent
        await ws.close()
        await client.close()


async def test_cancel_still_reaches_the_wire_when_its_task_is_cancelled() -> None:
    async def handler(server: Server, ws: ServerConnection, index: int) -> None:
        await server.recv(ws, index)  # A
        await ws.send(chunk("a", b"\x00\x00"))
        await _answer_all(server, ws, index)

    with Server(handler) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url, timeout=Timeout(read=None))
        ws = client.tts.websocket(**DEFAULTS)
        a = ws.context("a")
        first = await a.send(text="A")
        while first.audio_bytes == 0:
            await asyncio.sleep(0.01)
        gate = asyncio.Event()
        _gate_sends(ws, '"cancel":true', gate)
        cancelling = asyncio.create_task(a.cancel())
        await asyncio.sleep(0.05)  # the cancel message is stuck behind back-pressure
        cancelling.cancel()  # the barge-in handler is itself cancelled
        gate.set()
        assert (await first.result(timeout=2)).cancelled
        later = await a.send(text="W")
        assert (await later.result(timeout=5)).request_id == "W"  # not wedged
        assert sum(1 for _, m in server.received if m.get("cancel")) == 1
        await ws.close()
        await client.close()


async def _collect(ctx: Any) -> list[Any]:
    return [event async for event in ctx.events()]


# 3. Redirects are never followed


def _redirect_to(port: int) -> Any:
    def redirect(connection: Any, request: Any) -> Response:
        headers = Headers()
        headers["Location"] = f"ws://localhost:{port}/v1/text-to-speech/ws"
        headers["Content-Length"] = "0"
        return Response(302, http.HTTPStatus(302).phrase, headers, b"")

    return redirect


async def test_websocket_refuses_redirects() -> None:
    key = "sk-live-0123456789"
    with Server(_answer_all) as other, Server(_answer_all, _redirect_to(other.port)) as api:
        client = AsyncBakbak(api_key=key, base_url=api.base_url)
        with pytest.raises(bakbak.APIStatusError) as info:
            await client.tts.websocket().connect()
        assert info.value.status == 302
        assert info.value.retryable is False
        await client.close()
        sync = Bakbak(api_key=key, base_url=api.base_url)
        with pytest.raises(bakbak.APIStatusError) as sync_info:
            sync.tts.websocket().connect()
        assert sync_info.value.status == 302
        sync.close()
    assert other.headers == []  # the other origin never saw a request, or the key


# 4. Contexts are released (§6.6)


async def _answer_quietly(server: Server, ws: ServerConnection, index: int) -> None:
    """Like _answer_all, without recording anything (for memory checks)."""
    try:
        async for raw in ws:
            cid = json.loads(raw)["context_id"]
            await ws.send(chunk(cid, b"\x01"))
            await ws.send(done(cid))
    except Exception:
        return


async def test_finished_contexts_are_released() -> None:
    with Server(_answer_quietly) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        async with client.tts.websocket(**DEFAULTS) as ws:

            async def utterances(n: int) -> None:
                for _ in range(n):
                    handle = await ws.context().send(text="Hi.")
                    async for _ in handle:
                        pass

            await utterances(500)
            assert len(ws._core.contexts) == 0
            gc.collect()
            tracemalloc.start()
            before = tracemalloc.take_snapshot()
            await utterances(2000)
            gc.collect()
            after = tracemalloc.take_snapshot()
            tracemalloc.stop()
            grown = sum(s.size_diff for s in after.compare_to(before, "filename"))
            assert grown < 100_000, f"{grown} bytes over 2000 finished utterances"
            assert len(ws._core.contexts) == 0
            # A context that is still held is the same object when asked for again.
            held = ws.context("turn-1")
            assert ws.context("turn-1")._ctx is held._ctx
        await client.close()


# 5. Invalid base64 in a WebSocket chunk


async def test_invalid_base64_fails_only_its_own_request() -> None:
    async def handler(server: Server, ws: ServerConnection, index: int) -> None:
        await server.recv(ws, index)
        await server.recv(ws, index)
        await ws.send(chunk("c1", b"\x01\x02", "req_a"))
        bad = json.loads(chunk("c1", b"", "req_a"))
        bad["data"] = "abc"
        await ws.send(json.dumps(bad))
        await ws.send(chunk("c1", b"\x03\x04", "req_a"))  # the rest of that stream
        await ws.send(done("c1", "req_a"))
        await ws.send(chunk("c1", b"\x05", "req_b"))
        await ws.send(done("c1", "req_b"))
        await ws.wait_closed()

    with Server(handler) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        async with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("c1")
            first = await ctx.send(text="A.")
            second = await ctx.send(text="B.")
            with pytest.raises(bakbak.StreamInterruptedError) as info:
                await first.result(timeout=5)
            assert info.value.audio_bytes == 2
            assert info.value.retryable is True
            result = await second.result(timeout=5)
            assert (result.request_id, result.audio_bytes) == ("req_b", 1)
            ctx.end_input()
            audio = [e.audio for e in await _collect(ctx) if isinstance(e, Chunk)]
            assert audio == [b"\x01\x02", b"\x05"]  # nothing of A's after the break
        await client.close()


# 7. Draining: closed by the client once nothing is bound to it


async def test_draining_connection_is_closed_once_unused() -> None:
    refusal = {
        "type": "error",
        "status_code": 503,
        "done": True,
        "message": "The service is restarting. Reconnect and send the request again.",
        "context_id": "c1",
        "request_id": "req_x",
    }

    async def handler(server: Server, ws: ServerConnection, index: int) -> None:
        if index == 0:
            await server.recv(ws, index)
            await ws.send(json.dumps(refusal))
            await ws.wait_closed()  # the server never closes it itself
            return
        await _answer_all(server, ws, index)

    with Server(handler) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        async with client.tts.websocket(**DEFAULTS) as ws:
            result = await (await ws.context("c1").send(text="A.")).result(timeout=5)
            assert result.request_id == "A."
            deadline = time.monotonic() + 3
            while not server.close_codes:
                assert time.monotonic() < deadline, "the draining connection stayed open"
                await asyncio.sleep(0.02)
            assert server.close_codes[0] == 1000
        await client.close()


# 8. cancel() stops a send_stream feeding the context


async def test_cancel_stops_an_async_send_stream() -> None:
    with Server(_answer_all) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        async with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("c1")
            stalled = asyncio.Event()

            async def tokens() -> AsyncIterator[str]:
                yield "First sentence here. "
                yield "Second"
                await stalled.wait()  # the LLM stalls; never released
                yield " sentence."

            streaming = asyncio.create_task(ctx.send_stream(tokens(), end_input=False))
            while not server.texts():
                await asyncio.sleep(0.01)
            await ctx.cancel()
            handles = await asyncio.wait_for(streaming, 2)  # returns without the LLM
            assert len(handles) == 1
            await asyncio.sleep(0.1)
            assert server.texts() == ["First sentence here."]
        await client.close()


def test_cancel_stops_a_sync_send_stream() -> None:
    with Server(_answer_all) as server:
        client = Bakbak(api_key="k", base_url=server.base_url)
        with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("c1")
            release = threading.Event()

            def tokens() -> Iterator[str]:
                yield "First sentence here. Second"  # the first sentence is final
                release.wait(5)
                yield " sentence. "
                yield "Third."

            handles: list[Any] = []
            streaming = threading.Thread(target=lambda: handles.extend(ctx.send_stream(tokens())))
            streaming.start()
            deadline = time.monotonic() + 5
            while not server.texts():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            ctx.cancel()
            release.set()
            streaming.join(5)
            assert not streaming.is_alive()
            time.sleep(0.1)
            assert server.texts() == ["First sentence here."]
            assert len(handles) == 1
        client.close()


# 10. Failures while reconnecting, and bounded closing


async def test_closing_an_unresponsive_peer_is_bounded() -> None:
    with SilentPeer() as peer:
        client = AsyncBakbak(api_key="k", base_url=peer.base_url)
        ws = await client.tts.websocket(**DEFAULTS)
        handle = await ws.context().send(text="Hello.")
        started = time.monotonic()
        await ws.close()
        assert time.monotonic() - started < 2.0
        with pytest.raises(bakbak.WebSocketClosedError):
            await handle.result(timeout=1)
        await client.close()


def test_closing_an_unresponsive_peer_is_bounded_sync() -> None:
    with SilentPeer() as peer:
        client = Bakbak(api_key="k", base_url=peer.base_url)
        ws = client.tts.websocket(**DEFAULTS).connect()
        handle = ws.context().send(text="Hello.")
        started = time.monotonic()
        ws.close()
        assert time.monotonic() - started < 2.0
        with pytest.raises(bakbak.WebSocketClosedError):
            handle.result(timeout=1)
        client.close()


async def test_any_reconnect_failure_fails_the_waiting_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bakbak._ws._async as driver

    calls = 0

    async def flaky(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("unexpected")

    with Server(_answer_all) as server:
        client = AsyncBakbak(api_key="k", base_url=server.base_url)
        ws = await client.tts.websocket(**DEFAULTS)
        monkeypatch.setattr(driver, "_Connect", flaky)
        current = ws._core.current
        assert current is not None
        await current.transport.close()  # the connection drops
        while ws.connected:
            await asyncio.sleep(0.01)
        handle = await ws.context().send(text="Hello.")
        with pytest.raises(bakbak.APIConnectionError, match="ValueError"):
            await handle.result(timeout=5)
        assert calls == 1
        await ws.close()
        await client.close()
