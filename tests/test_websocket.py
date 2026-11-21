"""WebSocket behaviour beyond the conformance scenarios: send_stream, LLM
deltas, cancel discarding queued audio, early resend while draining, read
timeouts, closes, back-pressure and per-request iteration."""

from __future__ import annotations

import asyncio
import base64
import re
import threading
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from conformance_support import MockWSServer

import bakbak
from bakbak import AsyncBakbak, Bakbak, Timeout
from bakbak._constants import WS_MAX_CONTEXTS, WS_MAX_UNFINISHED
from bakbak._ws._common import check_context_id, text_delta
from bakbak._ws._core import Core, Request
from bakbak.text import split_sentences
from bakbak.websocket import Cancelled, Chunk, Done, Failed

DEFAULTS: dict[str, Any] = {
    "voice_id": "v1",
    "language": "en",
    "model": "m1",
    "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
}


def _chunk(context_id: str, audio: bytes, request_id: str = "req_1") -> dict[str, Any]:
    return {
        "send": {
            "type": "chunk",
            "status_code": 206,
            "done": False,
            "data": base64.b64encode(audio).decode(),
            "context_id": context_id,
            "request_id": request_id,
        }
    }


def _done(context_id: str, request_id: str = "req_1") -> dict[str, Any]:
    return {
        "send": {
            "type": "done",
            "status_code": 200,
            "done": True,
            "context_id": context_id,
            "request_id": request_id,
        }
    }


def _client(server: MockWSServer, **kwargs: Any) -> AsyncBakbak:
    client = AsyncBakbak(api_key="test-key", base_url=server.base_url, **kwargs)
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    client._sleep = record
    return client


# Duck typing of LLM stream items


def test_text_delta_duck_typing() -> None:
    ns = SimpleNamespace
    openai_chunk = ns(choices=[ns(delta=ns(content="Hello", role=None))])
    openai_role = ns(choices=[ns(delta=ns(content=None, role="assistant"))])
    openai_usage = ns(choices=[], usage=ns(total_tokens=3))
    anthropic_text = ns(
        type="content_block_delta", index=0, delta=ns(type="text_delta", text=" there")
    )
    anthropic_json = ns(
        type="content_block_delta", index=1, delta=ns(type="input_json_delta", partial_json="{")
    )
    anthropic_thinking = ns(
        type="content_block_delta", delta=ns(type="thinking_delta", thinking="hmm")
    )
    anthropic_start = ns(type="message_start", message=ns(content=[]))
    responses_text = ns(type="response.output_text.delta", delta="Hi")
    responses_reasoning = ns(type="response.reasoning_text.delta", delta="secret")
    responses_done = ns(type="response.output_text.done", text="Whole reply")
    gemini = ns(text="Namaste", candidates=[])
    langchain = ns(type="AIMessageChunk", content="chunk")

    class RaisingText:
        @property
        def text(self) -> str:
            raise ValueError("no text part")

    assert text_delta("plain") == "plain"
    assert text_delta(openai_chunk) == "Hello"
    assert text_delta(openai_role) is None
    assert text_delta(openai_usage) is None
    assert text_delta({"choices": [{"delta": {"content": "dict"}}]}) == "dict"
    assert text_delta(anthropic_text) == " there"
    assert (
        text_delta({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "d"}})
        == "d"
    )
    assert text_delta(anthropic_json) is None
    assert text_delta(anthropic_thinking) is None
    assert text_delta(anthropic_start) is None
    assert text_delta(responses_text) == "Hi"
    assert text_delta(responses_reasoning) is None
    assert text_delta(responses_done) is None
    assert text_delta(gemini) == "Namaste"
    assert text_delta(langchain) == "chunk"
    assert text_delta(RaisingText()) is None
    assert text_delta(None) is None
    with pytest.raises(TypeError):
        text_delta(b"bytes")


def test_context_ids() -> None:
    generated = check_context_id(None)
    assert re.fullmatch(r"ctx_[0-9a-f]{32}", generated)
    assert check_context_id("turn-7.a_b") == "turn-7.a_b"
    for bad in ("", "x" * 129, "has space", "ü"):
        with pytest.raises(bakbak.BakbakError):
            check_context_id(bad)


# send_stream


TOKENS = ["Sure", ", I can", " help. Your order", " ships today", " at 5 p.m. Anything", " else?"]


def _stream_script(context_id: str, sentences: list[str]) -> list[dict[str, Any]]:
    script: list[dict[str, Any]] = [
        {"recv": {"text": s, "context_id": context_id, **DEFAULTS}} for s in sentences
    ]
    for n in range(len(sentences)):
        script += [_chunk(context_id, b"\x01\x02", f"req_{n}"), _done(context_id, f"req_{n}")]
    return [*script, {"expect_no_more": True}]


async def test_async_send_stream_sends_sentences() -> None:
    sentences = split_sentences("".join(TOKENS))
    # "p.m." does not end a sentence: "m" is a single letter.
    assert sentences == ["Sure, I can help.", "Your order ships today at 5 p.m. Anything else?"]

    async def tokens() -> AsyncIterator[object]:
        for token in TOKENS:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])

    with MockWSServer([{"script": _stream_script("turn", sentences)}]) as server:
        client = _client(server)
        async with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("turn")
            handles = await ctx.send_stream(tokens())
            audio = b"".join([a async for a in ctx.audio()])
        await client.close()
        server.assert_complete(1)
    assert len(handles) == 2
    assert audio == b"\x01\x02" * 2


def test_sync_send_stream_with_a_thread_reading_audio() -> None:
    sentences = split_sentences("".join(TOKENS))
    with MockWSServer([{"script": _stream_script("turn", sentences)}]) as server:
        client = Bakbak(api_key="test-key", base_url=server.base_url)
        with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("turn")
            received: list[bytes] = []
            reader = threading.Thread(target=lambda: received.extend(ctx.audio()))
            reader.start()
            handles = ctx.send_stream(iter(TOKENS))
            reader.join(5)
            assert [h.result(timeout=5).request_id for h in handles] == ["req_0", "req_1"]
        server.assert_complete(1)
    assert b"".join(received) == b"\x01\x02" * 2


async def test_send_after_end_input_raises() -> None:
    with MockWSServer([]) as server:
        client = _client(server)
        ws = client.tts.websocket()
        ctx = ws.context()
        ctx.end_input()
        with pytest.raises(bakbak.BakbakError, match="end_input"):
            await ctx.send(text="x")
        await ws.close()
        with pytest.raises(bakbak.BakbakError, match="closed"):
            ws.context()


# Cancel discards audio not yet handed out (§6.3)


async def test_cancel_discards_queued_audio() -> None:
    script = [
        {"recv": {"context_id": "c1"}},
        _chunk("c1", b"a" * 10),
        _chunk("c1", b"b" * 10),
        {"recv": {"context_id": "c1", "cancel": True}},
        {"send": {"type": "cancelled", "done": True, "context_id": "c1"}},
        {"expect_no_more": True},
    ]
    with MockWSServer([{"script": script}]) as server:
        client = _client(server)
        async with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("c1")
            handle = await ctx.send(text="Hello.")
            deadline = time.monotonic() + 5
            while handle.audio_bytes < 20:  # received, but nobody has read it
                assert time.monotonic() < deadline
                await asyncio.sleep(0.005)
            await ctx.cancel()
            ctx.end_input()
            events = [e async for e in ctx.events()]
        await client.close()
        server.assert_complete(1)
    assert [type(e) for e in events] == [Cancelled]
    result = await handle
    assert result.cancelled
    assert result.audio_bytes == 0


# Draining: resend on a new connection as soon as the context is free (§6.4)


async def test_draining_resends_early_and_moves_new_contexts() -> None:
    draining = {
        "send": {
            "type": "error",
            "status_code": 503,
            "done": True,
            "message": "The service is restarting. Reconnect and send the request again.",
            "error_code": "service_unavailable",
            "retryable": True,
            "context_id": "c1",
            "request_id": "req_closing",
        }
    }
    first = [{"recv": {"text": "One.", "context_id": "c1"}}, draining, {"expect_no_more": True}]
    second = [
        {"recv": {"text": "One.", "context_id": "c1"}},
        _chunk("c1", b"\x00\x01"),
        _done("c1"),
        {"recv": {"text": "Two.", "context_id": "c2"}},
        _chunk("c2", b"\x02\x03", "req_2"),
        _done("c2", "req_2"),
        {"expect_no_more": True},
    ]
    with MockWSServer([{"script": first}, {"script": second}]) as server:
        client = _client(server)
        async with client.tts.websocket(**DEFAULTS) as ws:
            one = await (await ws.context("c1").send(text="One.")).result(timeout=5)
            two = await (await ws.context("c2").send(text="Two.")).result(timeout=5)
        await client.close()
        server.assert_complete(2)
    assert (one.request_id, one.audio_bytes) == ("req_1", 2)
    assert (two.request_id, two.audio_bytes) == ("req_2", 2)


# Read timeout, closes


async def test_read_timeout_fails_the_request_and_cancels_it() -> None:
    script = [
        {"recv": {"context_id": "c1"}},
        {"recv": {"context_id": "c1", "cancel": True}},
        {"send": {"type": "cancelled", "done": True, "context_id": "c1"}},
        {"expect_no_more": True},
    ]
    with MockWSServer([{"script": script}]) as server:
        client = _client(server, timeout=Timeout(read=0.3))
        async with client.tts.websocket(**DEFAULTS) as ws:
            handle = await ws.context("c1").send(text="Hello.")
            with pytest.raises(bakbak.APITimeoutError):
                await handle.result(timeout=5)
            await asyncio.sleep(0.1)  # let the cancel be acknowledged
        await client.close()
        server.assert_complete(1)


async def test_close_1008_fails_everything_unretryably() -> None:
    script = [{"recv": {"context_id": "c1"}}, {"close": {"code": 1008, "reason": "too many"}}]
    with MockWSServer([{"script": script}]) as server:
        client = _client(server)
        async with client.tts.websocket(**DEFAULTS) as ws:
            handle = await ws.context("c1").send(text="Hello.")
            with pytest.raises(bakbak.WebSocketClosedError) as info:
                await handle.result(timeout=5)
        await client.close()
        server.assert_complete(1)
    assert info.value.code == 1008
    assert info.value.retryable is False


def test_user_close_gives_every_request_an_outcome() -> None:
    script = [{"recv": {"context_id": "c1"}}, {"expect_no_more": True}]
    with MockWSServer([{"script": script}]) as server:
        client = Bakbak(api_key="test-key", base_url=server.base_url)
        ws = client.tts.websocket(**DEFAULTS)
        ctx = ws.context("c1")
        handle = ctx.send(text="Hello.")
        ws.close()
        with pytest.raises(bakbak.WebSocketClosedError):
            handle.result(timeout=5)
        assert [type(e) for e in ctx.events()] == [Failed]
        server.assert_complete(1)


# Per-request iteration and back-pressure


async def test_iterating_a_handle_yields_only_its_audio() -> None:
    script = [
        {"recv": {"text": "A."}},
        {"recv": {"text": "B."}},
        _chunk("c1", b"aa", "req_a"),
        _done("c1", "req_a"),
        _chunk("c1", b"bb", "req_b"),
        _chunk("c1", b"BB", "req_b"),
        _done("c1", "req_b"),
        {"expect_no_more": True},
    ]
    with MockWSServer([{"script": script}]) as server:
        client = _client(server)
        async with client.tts.websocket(**DEFAULTS) as ws:
            ctx = ws.context("c1")
            await ctx.send(text="A.")
            second = await ctx.send(text="B.")
            assert [a async for a in second] == [b"bb", b"BB"]
            assert (await second).request_id == "req_b"
        await client.close()
        server.assert_complete(1)


class _Signal:
    def notify(self) -> None:
        return


class _Driver:
    def new_signal(self) -> _Signal:
        return _Signal()

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def test_back_pressure_limits() -> None:
    core = Core(_Driver(), max_retries=2, read_timeout=None)  # type: ignore[arg-type]

    def handle(request: Request) -> object:
        return request

    busy = core.context("busy")
    for _ in range(WS_MAX_UNFINISHED - 1):
        core.submit(busy, "{}", 1, handle)
    assert not core.at_capacity(busy)
    core.submit(busy, "{}", 1, handle)
    assert core.at_capacity(busy)

    core = Core(_Driver(), max_retries=2, read_timeout=None)  # type: ignore[arg-type]
    for n in range(WS_MAX_CONTEXTS):
        core.submit(core.context(f"c{n}"), "{}", 1, handle)
    assert core.at_capacity(core.context("new"))
    assert not core.at_capacity(core.context("c0"))


def test_events_are_typed_for_each_client() -> None:
    # Chunk/Done/Failed/Cancelled are shared and generic over the handle type.
    assert Chunk(b"x", "h").request == "h"
    assert Done("h").request == "h"
    assert "1 bytes" in repr(Chunk(b"x", "h"))
