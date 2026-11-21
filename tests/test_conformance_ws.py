"""Every ``conformance/ws`` scenario, against a scripted ``websockets.serve``
server, for both the async and the sync client."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from conformance_support import MockWSServer, check_error, check_sleeps, load_scenarios

from bakbak import APIStatusError, AsyncBakbak, Bakbak, BakbakError
from bakbak.websocket import AsyncContext, Cancelled, Chunk, Context, Done, Failed

SCENARIOS = load_scenarios("ws")
TIMEOUT = 10.0


def _ids(scenarios: list[dict[str, Any]]) -> list[str]:
    return [s["id"] for s in scenarios]


def _terminal(event: object) -> bool:
    return isinstance(event, (Done, Failed, Cancelled))


def _outcomes(handles: list[Any], events: list[Any]) -> list[dict[str, Any]]:
    """One outcome per request, in send order, built from the context's events."""
    records: dict[int, dict[str, Any]] = {id(h): {"audio": b"", "handle": h} for h in handles}
    for event in events:
        record = records[id(event.request)]
        if isinstance(event, Chunk):
            record["audio"] += event.audio
        elif isinstance(event, Done):
            record["outcome"] = "done"
        elif isinstance(event, Cancelled):
            record["outcome"] = "cancelled"
        elif isinstance(event, Failed):
            record["outcome"] = "error"
            record["error"] = event.error
    return [records[id(h)] for h in handles]


def _check_outcomes(expected: dict[str, Any], actual: dict[str, list[dict[str, Any]]]) -> None:
    assert set(actual) == set(expected), f"contexts {sorted(actual)} != {sorted(expected)}"
    for context_id, outcomes in expected.items():
        got = actual[context_id]
        assert len(got) == len(outcomes), f"{context_id}: {len(got)} requests"
        for n, (want, record) in enumerate(zip(outcomes, got, strict=True)):
            where = f"{context_id}[{n}]"
            assert record.get("outcome") == want["outcome"], (
                f"{where}: {record.get('outcome')} ({record.get('error')!r}), expected {want['outcome']}"
            )
            if "audio_bytes" in want:
                assert len(record["audio"]) == want["audio_bytes"], f"{where}: audio bytes"
            if "audio_sha256" in want:
                assert hashlib.sha256(record["audio"]).hexdigest() == want["audio_sha256"], where
            if "request_id" in want:
                assert record["handle"].request_id == want["request_id"], where
            if "error" in want:
                check_error(want["error"], record.get("error"))


def _check(
    scenario: dict[str, Any],
    server: MockWSServer,
    outcomes: dict[str, list[dict[str, Any]]],
    connect_error: BaseException | None,
    notices: list[APIStatusError],
    sleeps: list[float],
) -> None:
    expect = scenario["expect"]
    server.assert_complete(expect.get("connections"))
    if "connect_error" in expect:
        check_error(expect["connect_error"], connect_error)
    else:
        assert connect_error is None, f"connect raised {connect_error!r}"
    _check_outcomes(expect.get("contexts", {}), outcomes)
    if "connection_notices" in expect:
        assert len(notices) == expect["connection_notices"], notices
    check_sleeps(expect.get("sleeps", []), sleeps)


async def _until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.002)


def _until_sync(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.002)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
async def test_ws_async(scenario: dict[str, Any]) -> None:
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    notices: list[APIStatusError] = []
    with MockWSServer(scenario["connections"]) as server:
        client = AsyncBakbak(api_key="test-key", base_url=server.base_url, max_retries=2)
        client._sleep = record
        ws = client.tts.websocket(on_error=notices.append)
        contexts: dict[str, AsyncContext] = {}
        handles: dict[str, list[Any]] = {}
        events: dict[str, list[Any]] = {}
        collectors: list[asyncio.Task[None]] = []

        async def collect(ctx: AsyncContext, sink: list[Any]) -> None:
            async for event in ctx.events():
                sink.append(event)

        def context(context_id: str) -> AsyncContext:
            if context_id not in contexts:
                contexts[context_id] = ws.context(context_id)
                handles[context_id] = []
                events[context_id] = []
                collectors.append(
                    asyncio.create_task(collect(contexts[context_id], events[context_id]))
                )
            return contexts[context_id]

        connect_error: BaseException | None = None
        try:
            for op in scenario["client"]:
                kind = op["op"]
                if kind == "connect":
                    try:
                        await ws.connect()
                    except BakbakError as exc:
                        connect_error = exc
                elif kind == "send":
                    handle = await context(op["context"]).send(**op["request"])
                    handles[op["context"]].append(handle)
                elif kind == "wait_audio":
                    sink = events[op["context"]]
                    await _until(lambda: any(isinstance(e, Chunk) for e in sink), "audio")
                elif kind == "wait_done":
                    sink, sent = events[op["context"]], handles[op["context"]]
                    await _until(
                        lambda: sum(map(_terminal, sink)) >= len(sent), "requests to finish"
                    )
                elif kind == "cancel":
                    await asyncio.wait_for(context(op["context"]).cancel(), TIMEOUT)
                elif kind == "wait_closed":
                    await _until(lambda: not ws.connected, "the server to close")
                else:
                    raise AssertionError(f"unknown op {kind}")
            for ctx in contexts.values():
                ctx.end_input()
            await asyncio.wait_for(asyncio.gather(*collectors), TIMEOUT)
        finally:
            await ws.close()
            await client.close()
        outcomes = {cid: _outcomes(handles[cid], events[cid]) for cid in contexts}
    _check(scenario, server, outcomes, connect_error, notices, sleeps)


def _collect(ctx: Context, sink: list[Any]) -> None:
    for event in ctx.events():
        sink.append(event)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_ws_sync(scenario: dict[str, Any]) -> None:
    sleeps: list[float] = []
    notices: list[APIStatusError] = []
    with MockWSServer(scenario["connections"]) as server:
        client = Bakbak(api_key="test-key", base_url=server.base_url, max_retries=2)
        client._sleep = sleeps.append
        ws = client.tts.websocket(on_error=notices.append)
        contexts: dict[str, Context] = {}
        handles: dict[str, list[Any]] = {}
        events: dict[str, list[Any]] = {}
        collectors: list[threading.Thread] = []

        def context(context_id: str) -> Context:
            if context_id not in contexts:
                ctx = contexts[context_id] = ws.context(context_id)
                handles[context_id] = []
                sink = events[context_id] = []
                thread = threading.Thread(target=_collect, args=(ctx, sink), daemon=True)
                thread.start()
                collectors.append(thread)
            return contexts[context_id]

        connect_error: BaseException | None = None
        try:
            for op in scenario["client"]:
                kind = op["op"]
                if kind == "connect":
                    try:
                        ws.connect()
                    except BakbakError as exc:
                        connect_error = exc
                elif kind == "send":
                    ctx = context(op["context"])
                    handles[op["context"]].append(ctx.send(**op["request"]))
                elif kind == "wait_audio":
                    sink = events[op["context"]]
                    _until_sync(lambda: any(isinstance(e, Chunk) for e in sink), "audio")
                elif kind == "wait_done":
                    sink, sent = events[op["context"]], handles[op["context"]]
                    _until_sync(
                        lambda: sum(map(_terminal, sink)) >= len(sent), "requests to finish"
                    )
                elif kind == "cancel":
                    context(op["context"]).cancel()
                elif kind == "wait_closed":
                    _until_sync(lambda: not ws.connected, "the server to close")
                else:
                    raise AssertionError(f"unknown op {kind}")
            for ctx in contexts.values():
                ctx.end_input()
            for thread in collectors:
                thread.join(TIMEOUT)
                assert not thread.is_alive(), "a context's events did not end"
        finally:
            ws.close()
            client.close()
        outcomes = {cid: _outcomes(handles[cid], events[cid]) for cid in contexts}
    _check(scenario, server, outcomes, connect_error, notices, sleeps)
