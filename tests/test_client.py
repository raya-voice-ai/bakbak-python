"""Client construction, options, headers, privacy, and HTTP behaviour beyond
the conformance scenarios."""

from __future__ import annotations

import base64
import inspect
import json
import logging
import platform
import struct
import time
from typing import Any

import httpx
import pytest
from conformance_support import MockHTTPServer

import bakbak
from bakbak import AsyncBakbak, Bakbak, Timeout
from bakbak._streaming import AsyncAudioStream, AudioStream
from bakbak._utils import redact_url
from bakbak._ws._async import AsyncContext, AsyncRequestHandle, AsyncTTSWebSocket
from bakbak._ws._sync import Context, RequestHandle, TTSWebSocket
from bakbak.resources.tts import TTS, AsyncTTS, AsyncTTSWithRawResponse, TTSWithRawResponse
from bakbak.resources.voices import (
    AsyncVoices,
    AsyncVoicesWithRawResponse,
    Voices,
    VoicesWithRawResponse,
)

KEY = "sk-test-0123456789-secret"
TEXT = "यह गोपनीय पाठ है।"
PARAMS: dict[str, Any] = {"text": TEXT, "voice_id": "v1", "language": "hi"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAKBAK_API_KEY", raising=False)
    monkeypatch.delenv("BAKBAK_BASE_URL", raising=False)


def _exchange(response: dict[str, Any], path: str = "/v1/text-to-speech") -> dict[str, Any]:
    return {"expect_request": {"method": "POST", "path": path}, "response": response}


def _sse(*events: dict[str, Any]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _chunk(audio: bytes) -> dict[str, Any]:
    return {
        "type": "chunk",
        "status_code": 206,
        "done": False,
        "data": base64.b64encode(audio).decode(),
    }


DONE = {"type": "done", "status_code": 200, "done": True}


# Construction and options


def test_missing_key_names_the_argument_and_the_variable() -> None:
    with pytest.raises(bakbak.BakbakError) as info:
        Bakbak()
    assert "api_key" in str(info.value)
    assert "BAKBAK_API_KEY" in str(info.value)
    with pytest.raises(bakbak.BakbakError):
        AsyncBakbak()


def test_environment_is_read_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKBAK_API_KEY", KEY)
    monkeypatch.setenv("BAKBAK_BASE_URL", "https://staging.example.test/")
    client = Bakbak()
    assert client.api_key == KEY
    assert client.base_url == "https://staging.example.test"
    assert client.websocket_url == "wss://staging.example.test/v1/text-to-speech/ws"
    explicit = Bakbak(api_key="other", base_url="http://local.test:8080")
    assert explicit.api_key == "other"
    assert explicit.websocket_url == "ws://local.test:8080/v1/text-to-speech/ws"
    monkeypatch.delenv("BAKBAK_BASE_URL")
    assert Bakbak().base_url == "https://hub.getraya.app"


def test_empty_environment_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKBAK_API_KEY", "")
    with pytest.raises(bakbak.BakbakError):
        Bakbak()


def test_options_are_validated() -> None:
    with pytest.raises(bakbak.BakbakError):
        Bakbak(api_key=KEY, max_retries=-1)
    with pytest.raises(bakbak.BakbakError):
        Bakbak(api_key=KEY, base_url="ftp://example.test")
    with pytest.raises(ValueError):
        Timeout(read=0)


def test_timeout_forms() -> None:
    assert Bakbak(api_key=KEY).timeout == Timeout(connect=5.0, read=60.0, total=None)
    assert Bakbak(api_key=KEY, timeout=2).timeout == Timeout(connect=2.0, read=2.0)
    assert Bakbak(api_key=KEY, timeout=30).timeout == Timeout(connect=5.0, read=30.0)
    assert Bakbak(api_key=KEY, timeout=None).timeout == Timeout(None, None, None)
    custom = Timeout(connect=1, read=10, total=20)
    assert Bakbak(api_key=KEY, timeout=custom).timeout is custom


def test_with_options_copies_and_shares_the_pool() -> None:
    client = Bakbak(api_key=KEY, default_headers={"A": "1"})
    client._sleep = lambda seconds: None
    copy = client.with_options(max_retries=5, timeout=3, default_headers={"B": "2"})
    assert copy.max_retries == 5
    assert copy.timeout.read == 3
    assert copy.api_key == KEY
    assert copy._client is client._client
    assert copy._sleep is client._sleep
    assert copy._default_headers == {"A": "1", "B": "2"}
    assert client.max_retries == 2


def test_repr_never_shows_the_key() -> None:
    client = Bakbak(api_key=KEY)
    assert KEY not in repr(client)
    assert KEY not in str(client)
    assert KEY not in repr(client.tts.websocket())
    assert KEY not in repr(AsyncBakbak(api_key=KEY))


def test_redact_url() -> None:
    url = f"wss://hub.example.test/v1/text-to-speech/ws?X-API-Key={KEY}&a=1"
    assert KEY not in redact_url(url)
    assert "X-API-Key=***" in redact_url(url)
    assert "a=1" in redact_url(url)
    assert redact_url("https://hub.example.test/v1/voices") == "https://hub.example.test/v1/voices"


# Parity: the sync and async surfaces cannot drift apart


PAIRS = [
    (Bakbak, AsyncBakbak),
    (TTS, AsyncTTS),
    (TTSWithRawResponse, AsyncTTSWithRawResponse),
    (Voices, AsyncVoices),
    (VoicesWithRawResponse, AsyncVoicesWithRawResponse),
    (AudioStream, AsyncAudioStream),
    (TTSWebSocket, AsyncTTSWebSocket),
    (Context, AsyncContext),
    (RequestHandle, AsyncRequestHandle),
]


def _surface(cls: type) -> dict[str, Any]:
    names = {n for n in dir(cls) if not n.startswith("_")}
    names |= {n for n in getattr(cls, "__annotations__", {}) if not n.startswith("_")}
    surface: dict[str, Any] = {}
    for name in sorted(names):
        member = inspect.getattr_static(cls, name, None)
        if callable(member) and not isinstance(member, type):
            params = inspect.signature(member).parameters.values()
            surface[name] = [(p.name, p.kind, p.default) for p in params]
        else:
            surface[name] = type(member).__name__
    return surface


@pytest.mark.parametrize(("sync", "async_"), PAIRS, ids=[p[0].__name__ for p in PAIRS])
def test_sync_and_async_surfaces_match(sync: type, async_: type) -> None:
    assert _surface(sync) == _surface(async_)


# Headers, bodies and privacy


def test_request_headers_body_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="bakbak")
    seen: dict[str, Any] = {}
    exchange = _exchange(
        {
            "status": 200,
            "headers": {"content-type": "audio/wav", "x-request-id": "req_1"},
            "body_base64": "",
        }
    )
    exchange["expect_request"]["json"] = {**PARAMS, "speed": 1.2, "future_field": True}
    with MockHTTPServer([exchange]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url, default_headers={"X-Team": "voice"})
        original = client._client.send

        def spy(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return original(request, **kwargs)

        client._client.send = spy  # type: ignore[method-assign]
        audio = client.tts.generate(
            **PARAMS, speed=1.2, extra_body={"future_field": True}, extra_headers={"X-Trace": "t1"}
        )
        server.assert_complete()
    assert audio.request_id == "req_1"
    headers = seen["headers"]
    assert headers["x-api-key"] == KEY
    assert headers["content-type"] == "application/json"
    assert headers["x-team"] == "voice"
    assert headers["x-trace"] == "t1"
    assert headers["x-bakbak-client"] == f"bakbak-python/{bakbak.__version__}"
    python = ".".join(platform.python_version_tuple())
    assert headers["user-agent"].startswith(f"bakbak-python/{bakbak.__version__} python/{python} ")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "req_1" in logged
    assert KEY not in logged
    assert TEXT not in logged
    assert all(r.levelno == logging.DEBUG for r in caplog.records if r.name.startswith("bakbak"))


def test_errors_never_carry_the_key() -> None:
    response = {
        "status": 401,
        "headers": {"content-type": "application/json"},
        "json": {"message": "Unauthorized"},
    }
    with MockHTTPServer([_exchange(response)]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        with pytest.raises(bakbak.AuthenticationError) as info:
            client.tts.generate(**PARAMS)
    error = info.value
    for text in (str(error), repr(error), json.dumps(error.body), *map(str, vars(error).values())):
        assert KEY not in text


# HTTP behaviour


def test_stream_carries_partial_samples_into_the_next_chunk() -> None:
    samples = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    sse = _sse(_chunk(samples[:6]), _chunk(samples[6:9]), _chunk(samples[9:]), DONE)
    headers = {
        "content-type": "text/event-stream",
        "x-audio-format": "pcm_f32le",
        "x-audio-sample-rate": "24000",
    }
    exchange = _exchange(
        {"status": 200, "headers": headers, "sse": sse}, "/v1/text-to-speech/stream"
    )
    with MockHTTPServer([exchange]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        with client.tts.stream(**PARAMS) as stream:
            sizes = [len(c.audio) for c in stream]
        assert stream.encoding == "pcm_f32le"
        assert stream.sample_rate == 24000
        assert stream.audio_bytes == 16
    assert sizes == [4, 4, 8]  # 6 -> 4 (+2 carried), 2+3 -> 4 (+1), 1+7 -> 8


def test_stream_helpers_and_single_read(tmp_path: Any) -> None:
    sse = _sse(_chunk(b"ab"), _chunk(b"cd"), DONE)
    headers = {"content-type": "text/event-stream", "x-audio-format": "pcm_mulaw"}
    exchanges = [
        _exchange({"status": 200, "headers": headers, "sse": sse}, "/v1/text-to-speech/stream")
    ] * 2
    with MockHTTPServer(exchanges) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        stream = client.tts.stream(**PARAMS)
        assert stream.request_id is None  # nothing sent yet
        assert stream.read() == b"abcd"
        with pytest.raises(bakbak.BakbakError):
            stream.read()
        client.tts.stream(**PARAMS).save(tmp_path / "out.raw")
        server.assert_complete()
    assert (tmp_path / "out.raw").read_bytes() == b"abcd"


def test_generate_body_cut_short_is_interrupted_and_not_retried() -> None:
    response = {
        "status": 200,
        "headers": {"content-type": "audio/basic", "x-request-id": "req_cut"},
        "body_base64": base64.b64encode(b"x" * 100).decode(),
        "content_length": 1000,
    }
    with MockHTTPServer([_exchange(response)]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        sleeps: list[float] = []
        client._sleep = sleeps.append
        with pytest.raises(bakbak.StreamInterruptedError) as info:
            client.tts.generate(**PARAMS)
        server.assert_complete()
    assert info.value.audio_bytes == 100
    assert info.value.request_id == "req_cut"
    assert sleeps == []


def test_total_timeout_stops_retrying() -> None:
    slow = {
        "delay": 1.0,
        "status": 200,
        "headers": {"content-type": "audio/wav"},
        "body_base64": "",
    }
    with MockHTTPServer([_exchange(slow)]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url, timeout=Timeout(total=0.3))
        started = time.monotonic()
        with pytest.raises(bakbak.APITimeoutError):
            client.tts.generate(**PARAMS)
        assert time.monotonic() - started < 0.9
        time.sleep(0.8)  # let the slow answer finish before the server stops


def test_per_call_max_retries() -> None:
    busy = {
        "status": 503,
        "headers": {"content-type": "application/json"},
        "json": {"detail": "busy"},
    }
    with MockHTTPServer([_exchange(busy)]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        with pytest.raises(bakbak.InternalServerError):
            client.tts.generate(**PARAMS, max_retries=0)
        server.assert_complete()


def test_with_raw_response() -> None:
    busy = {"status": 503, "headers": {"retry-after": "0"}, "json": {"detail": "busy"}}
    ok = {
        "status": 200,
        "headers": {"content-type": "audio/wav", "x-request-id": "req_ok"},
        "body_base64": "UklGRg==",
    }
    voices = {
        "status": 200,
        "headers": {"content-type": "application/json", "x-request-id": "req_v"},
        "json": {"voices": [{"id": "a", "name": "A", "language": "hi", "model": "m1"}], "count": 1},
    }
    exchanges = [
        _exchange(busy),
        _exchange(ok),
        {"expect_request": {"method": "GET", "path": "/v1/voices"}, "response": voices},
    ]
    with MockHTTPServer(exchanges) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        client._sleep = lambda seconds: None
        raw = client.tts.with_raw_response.generate(**PARAMS)
        assert raw.retries_taken == 1
        assert raw.status_code == 200
        assert raw.request_id == "req_ok"
        assert raw.parse().data == b"RIFF"
        listed = client.voices.with_raw_response.list(language="hi")
        assert listed.request_id == "req_v"
        assert [v.name for v in listed.parse()] == ["A"]
        server.assert_complete()


def test_prewarm_lists_voices_once() -> None:
    voices = {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "json": {"voices": [], "count": 0},
    }
    with MockHTTPServer(
        [{"expect_request": {"method": "GET", "path": "/v1/voices"}, "response": voices}]
    ) as server:
        Bakbak(api_key=KEY, base_url=server.base_url).prewarm()
        server.assert_complete()


async def test_async_stream_and_prewarm() -> None:
    sse = _sse(_chunk(b"\x01\x02"), DONE)
    exchanges = [
        {
            "expect_request": {"method": "GET", "path": "/v1/voices"},
            "response": {"status": 200, "headers": {}, "json": {"voices": [], "count": 0}},
        },
        _exchange(
            {"status": 200, "headers": {"x-request-id": "r"}, "sse": sse},
            "/v1/text-to-speech/stream",
        ),
    ]
    with MockHTTPServer(exchanges) as server:
        async with AsyncBakbak(api_key=KEY, base_url=server.base_url) as client:
            await client.prewarm()
            raw = await client.tts.with_raw_response.stream(
                **PARAMS, output_format={"container": "raw", "encoding": "pcm_mulaw"}
            )
            assert await raw.parse().read() == b"\x01\x02"
        server.assert_complete()


# Review regressions: redirects, invalid base64, error chains, key checks


def test_http_never_follows_redirects() -> None:
    moved = {
        "status": 302,
        "headers": {"location": "http://localhost:1/elsewhere"},
        "body_base64": "",
    }
    with MockHTTPServer([_exchange(moved)]) as server:
        follower = httpx.Client(follow_redirects=True)  # even a caller's client that would
        client = Bakbak(api_key=KEY, base_url=server.base_url, http_client=follower)
        with pytest.raises(bakbak.APIStatusError) as info:
            client.tts.generate(**PARAMS)
        server.assert_complete()  # one attempt: not retried, not followed
        follower.close()
    assert type(info.value) is bakbak.APIStatusError
    assert info.value.status == 302
    assert info.value.retryable is False


def test_sse_invalid_base64_before_audio_is_retried() -> None:
    bad = {"type": "chunk", "data": "abc"}
    exchanges = [
        _exchange(
            {"status": 200, "headers": {"x-request-id": "r1"}, "sse": _sse(bad, DONE)},
            "/v1/text-to-speech/stream",
        ),
        _exchange(
            {"status": 200, "headers": {"x-request-id": "r2"}, "sse": _sse(_chunk(b"abcd"), DONE)},
            "/v1/text-to-speech/stream",
        ),
    ]
    with MockHTTPServer(exchanges) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        client._sleep = lambda seconds: None
        with client.tts.stream(**PARAMS) as stream:
            assert stream.read() == b"abcd"
        server.assert_complete()


def test_sse_invalid_base64_after_audio_is_interrupted() -> None:
    sse = _sse(_chunk(b"abcd"), {"type": "chunk", "data": "abc"}, DONE)
    exchange = _exchange(
        {"status": 200, "headers": {"x-request-id": "r1"}, "sse": sse}, "/v1/text-to-speech/stream"
    )
    with MockHTTPServer([exchange]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        with pytest.raises(bakbak.StreamInterruptedError) as info:
            client.tts.stream(**PARAMS).read()
    assert info.value.audio_bytes == 4
    assert info.value.request_id == "r1"


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending:
        item = pending.pop()
        if item is None or any(item is s for s in seen):
            continue
        seen.append(item)
        pending += [item.__cause__, item.__context__]
    return seen


def _assert_no_key_reachable(error: BaseException) -> None:
    for item in _chain(error):
        assert not isinstance(item, httpx.HTTPError), f"{type(item).__name__} in the chain"
        assert getattr(item, "request", None) is None
        for text in (repr(item), str(item), *map(repr, vars(item).values())):
            assert KEY not in text


def test_connection_errors_carry_no_request_headers() -> None:
    with MockHTTPServer([]) as server:
        url = server.base_url
    client = Bakbak(api_key=KEY, base_url=url, max_retries=0)  # nothing listens there now
    with pytest.raises(bakbak.APIConnectionError) as info:
        client.tts.generate(**PARAMS)
    _assert_no_key_reachable(info.value)
    with pytest.raises(bakbak.APIConnectionError) as ws_info:
        client.tts.websocket().connect()
    _assert_no_key_reachable(ws_info.value)


def test_body_cut_short_carries_no_request_headers() -> None:
    response = {
        "status": 200,
        "headers": {},
        "body_base64": base64.b64encode(b"x" * 10).decode(),
        "content_length": 100,
    }
    with MockHTTPServer([_exchange(response)]) as server:
        client = Bakbak(api_key=KEY, base_url=server.base_url)
        with pytest.raises(bakbak.StreamInterruptedError) as info:
            client.tts.generate(**PARAMS)
    _assert_no_key_reachable(info.value)


@pytest.mark.parametrize(
    ("key", "problem"),
    [
        ("sk-live 123", "whitespace"),
        ("sk-live-123\n", "whitespace"),
        ("sk-live\x00123", "control characters"),
        ("sk-live-123" + chr(0x2019), "non-ASCII"),
    ],
)
def test_unusable_keys_are_refused_without_echoing_them(key: str, problem: str) -> None:
    with pytest.raises(bakbak.BakbakError) as info:
        Bakbak(api_key=key)
    assert problem in str(info.value)
    assert "sk-live" not in str(info.value)
    assert "sk-live" not in repr(info.value)
    with pytest.raises(bakbak.BakbakError):
        AsyncBakbak(api_key=key)
