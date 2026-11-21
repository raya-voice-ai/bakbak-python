"""Every ``conformance/http`` scenario, against a real local HTTP server, for
both the sync and the async client."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from conformance_support import MockHTTPServer, check_error, check_sleeps, load_scenarios

from bakbak import AsyncBakbak, Bakbak, BakbakError

SCENARIOS = load_scenarios("http")


def _ids(scenarios: list[dict[str, Any]]) -> list[str]:
    return [s["id"] for s in scenarios]


def _check_result(expected: dict[str, Any], result: dict[str, Any]) -> None:
    if "audio_sha256" in expected:
        assert hashlib.sha256(result["audio"]).hexdigest() == expected["audio_sha256"]
    if "audio_bytes" in expected:
        assert len(result["audio"]) == expected["audio_bytes"]
    for key in ("chunks", "request_id", "content_type", "encoding", "sample_rate", "count", "ids"):
        if key in expected:
            assert result[key] == expected[key], f"{key}: {result[key]!r} != {expected[key]!r}"


def _check(
    scenario: dict[str, Any],
    result: dict[str, Any] | None,
    error: BaseException | None,
    sleeps: list[float],
) -> None:
    expect = scenario["expect"]
    if "error" in expect:
        assert result is None, f"expected {expect['error']['class']}, got a result"
        check_error(expect["error"], error)
    else:
        assert error is None, f"unexpected error: {error!r}"
        assert result is not None
        _check_result(expect.get("result", {}), result)
    check_sleeps(expect.get("sleeps", []), sleeps)


def _run_sync(client: Bakbak, call: dict[str, Any]) -> dict[str, Any]:
    method, params = call["method"], call["params"]
    if method == "generate":
        audio = client.tts.generate(**params)
        return {
            "audio": audio.data,
            "request_id": audio.request_id,
            "content_type": audio.content_type,
            "encoding": audio.encoding,
            "sample_rate": audio.sample_rate,
        }
    if method == "stream":
        with client.tts.stream(**params) as stream:
            chunks = list(stream)
        return {
            "audio": b"".join(c.audio for c in chunks),
            "chunks": len(chunks),
            "request_id": stream.request_id,
            "content_type": stream.content_type,
            "encoding": stream.encoding,
            "sample_rate": stream.sample_rate,
        }
    assert method == "voices.list"
    voices = client.voices.list(**params)
    return {"count": len(voices), "ids": [v.id for v in voices]}


async def _run_async(client: AsyncBakbak, call: dict[str, Any]) -> dict[str, Any]:
    method, params = call["method"], call["params"]
    if method == "generate":
        audio = await client.tts.generate(**params)
        return {
            "audio": audio.data,
            "request_id": audio.request_id,
            "content_type": audio.content_type,
            "encoding": audio.encoding,
            "sample_rate": audio.sample_rate,
        }
    if method == "stream":
        async with client.tts.stream(**params) as stream:
            chunks = [c async for c in stream]
        return {
            "audio": b"".join(c.audio for c in chunks),
            "chunks": len(chunks),
            "request_id": stream.request_id,
            "content_type": stream.content_type,
            "encoding": stream.encoding,
            "sample_rate": stream.sample_rate,
        }
    assert method == "voices.list"
    voices = await client.voices.list(**params)
    return {"count": len(voices), "ids": [v.id for v in voices]}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_http_sync(scenario: dict[str, Any]) -> None:
    sleeps: list[float] = []
    with MockHTTPServer(scenario["exchanges"]) as server:
        client = Bakbak(api_key="test-key", base_url=server.base_url, max_retries=2)
        client._sleep = sleeps.append
        result: dict[str, Any] | None = None
        error: BaseException | None = None
        try:
            result = _run_sync(client, scenario["call"])
        except BakbakError as exc:
            error = exc
        finally:
            client.close()
        server.assert_complete()
    _check(scenario, result, error, sleeps)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
async def test_http_async(scenario: dict[str, Any]) -> None:
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    with MockHTTPServer(scenario["exchanges"]) as server:
        client = AsyncBakbak(api_key="test-key", base_url=server.base_url, max_retries=2)
        client._sleep = record
        result: dict[str, Any] | None = None
        error: BaseException | None = None
        try:
            result = await _run_async(client, scenario["call"])
        except BakbakError as exc:
            error = exc
        finally:
            await client.close()
        server.assert_complete()
    _check(scenario, result, error, sleeps)
