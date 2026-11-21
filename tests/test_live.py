"""Tests against the real service. They spend credits.

They run only with ``pytest -m live`` and only when ``BAKBAK_API_KEY`` and
``BAKBAK_BASE_URL`` are set. Run them one at a time (the default): a test
deployment may serve each model one request at a time, and an overlap gets a
503 capacity refusal (which the client retries).

- ``-k smoke``: whole file, SSE stream and voices only, for a deployment
  without the WebSocket.
- ``BAKBAK_LIVE_VOICE_ID`` / ``BAKBAK_LIVE_LANGUAGE``: an m1 voice to use.
"""

from __future__ import annotations

import io
import os
import time
import wave
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

import bakbak
from bakbak import AsyncBakbak, Bakbak
from bakbak.audio import sample_size
from bakbak.types import OutputFormat
from bakbak.websocket import Cancelled, Chunk, Done

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("BAKBAK_API_KEY") and os.environ.get("BAKBAK_BASE_URL")),
        reason="needs BAKBAK_API_KEY and BAKBAK_BASE_URL",
    ),
]

VOICE = os.environ.get("BAKBAK_LIVE_VOICE_ID", "009ae8b5-4a03-492d-af1c-3055989ac734")
LANGUAGE = os.environ.get("BAKBAK_LIVE_LANGUAGE", "hi")
REQUEST: dict[str, Any] = {"voice_id": VOICE, "language": LANGUAGE, "model": "m1"}
SHORT = "नमस्ते, आप कैसे हैं?"
LONGER = (
    "आपका पैकेज आज शाम तक पहुँच जाएगा। डिलीवरी से पहले हमारा एजेंट आपको कॉल करेगा, "
    "और आप चाहें तो समय भी बदल सकते हैं। धन्यवाद।"
)

GRID: list[OutputFormat] = [
    {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
    {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 8000},
    {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
    {"container": "wav", "encoding": "pcm_mulaw", "sample_rate": 24000},
    {"container": "raw", "encoding": "pcm_alaw", "sample_rate": 24000},
    {"container": "wav", "encoding": "pcm_alaw", "sample_rate": 8000},
    {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 24000},
    {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 8000},
]


@pytest.fixture
def client() -> Iterator[Bakbak]:
    with Bakbak() as client:
        yield client


def test_smoke(client: Bakbak) -> None:
    voices = client.voices.list()
    assert any(v.id == VOICE for v in voices)
    assert all(v.model == "m1" for v in client.voices.list(model="m1"))

    audio = client.tts.generate(text=SHORT, **REQUEST)
    assert audio.request_id
    assert audio.data[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio.to_wav())) as w:
        assert w.getnframes() > 0

    started = time.monotonic()
    with client.tts.stream(
        text=SHORT,
        **REQUEST,
        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
    ) as stream:
        first: float | None = None
        total = 0
        for chunk in stream:
            first = first or time.monotonic() - started
            assert len(chunk.audio) % 2 == 0
            total += len(chunk.audio)
    assert stream.request_id
    assert stream.encoding == "pcm_s16le"
    assert total > 0
    assert first is not None


@pytest.mark.parametrize(
    "output_format",
    GRID,
    ids=[f"{f['container']}-{f.get('encoding')}-{f.get('sample_rate')}" for f in GRID],
)
def test_generate_output_format_grid(client: Bakbak, output_format: OutputFormat) -> None:
    audio = client.tts.generate(text=SHORT, **REQUEST, output_format=output_format)
    encoding = output_format.get("encoding")
    assert audio.encoding == encoding
    assert audio.sample_rate == output_format.get("sample_rate")
    if output_format["container"] == "wav":
        assert audio.content_type is not None
        assert audio.content_type.startswith("audio/wav")
        assert audio.data[:4] == b"RIFF"
    else:
        assert len(audio.data) % sample_size(encoding) == 0
        assert audio.to_wav()[:4] == b"RIFF"


def test_stream_f32_default_is_sample_aligned(client: Bakbak) -> None:
    with client.tts.stream(text=SHORT, **REQUEST) as stream:
        sizes = [len(chunk.audio) for chunk in stream]
    assert sizes
    assert all(size % 4 == 0 for size in sizes)
    # Without an output_format the response may carry no X-Audio-* headers;
    # the SDK reports only what the response (or the request) states.
    assert stream.encoding in (None, "pcm_f32le")


def test_bad_voice_is_a_not_found_error(client: Bakbak) -> None:
    with pytest.raises(bakbak.NotFoundError) as info:
        client.tts.generate(text=SHORT, voice_id="no-such-voice", language=LANGUAGE, model="m1")
    assert info.value.request_id
    assert info.value.retryable is False


MULAW: OutputFormat = {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000}


async def test_websocket_requests_cancel_and_send_stream() -> None:
    client = AsyncBakbak()
    async with client.tts.websocket(**REQUEST, output_format=MULAW) as ws:
        # Two requests on one context play in order.
        ctx = ws.context()
        first = await ctx.send(text=SHORT)
        second = await ctx.send(text="धन्यवाद।")
        ctx.end_input()
        events = [e async for e in ctx.events()]
        assert [type(e) for e in events if not isinstance(e, Chunk)] == [Done, Done]
        assert (await first).audio_bytes > 0
        assert (await second).request_id

        # Cancel mid-stream: silence at once, and the context stays usable.
        ctx = ws.context()
        long_request = await ctx.send(text=LONGER)
        async for event in ctx.events():
            if isinstance(event, Chunk):
                break
        await ctx.cancel()
        assert (await long_request).cancelled
        after = await ctx.send(text=SHORT)
        assert (await after.result(timeout=30)).audio_bytes > 0

        # An LLM-like stream of token pieces, sentence by sentence.
        async def tokens() -> AsyncIterator[str]:
            for piece in ["आपका ", "ऑर्डर ", "कल ", "आएगा।", " क्या ", "मैं ", "कुछ ", "और ", "बताऊँ?"]:
                yield piece

        ctx = ws.context()
        handles = await ctx.send_stream(tokens())
        audio = b"".join([chunk async for chunk in ctx.audio()])
        assert len(handles) == 2
        assert audio
        assert not [e async for e in ctx.events()]  # ended
    await client.close()


def test_websocket_sync(client: Bakbak) -> None:
    with client.tts.websocket(**REQUEST, output_format=MULAW) as ws:
        ctx = ws.context()
        handle = ctx.send(text=SHORT)
        ctx.end_input()
        audio = b"".join(ctx.audio())
        result = handle.result(timeout=30)
    assert len(audio) == result.audio_bytes > 0
    assert not any(isinstance(e, Cancelled) for e in ctx.events())
