"""Time to first audio (TTFA): SSE stream versus WebSocket.

Runs N requests per transport, one at a time, alternating SSE and WebSocket so
both see the same conditions. TTFA is measured from the moment the SDK is
asked for audio to the first audio chunk it hands back. Each transport gets an
unmeasured warm-up request first, so both measure kept-alive connections:
the HTTP pool for SSE, the open socket for the WebSocket.

    BAKBAK_API_KEY=... python examples/bench_ttfa.py --n 20
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time

from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import AsyncBakbak
from bakbak.types import OutputFormat
from bakbak.websocket import AsyncTTSWebSocket, Chunk, Done, Failed

OUTPUT: OutputFormat = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000}


async def sse_ttfa(client: AsyncBakbak, text: str) -> float:
    started = time.perf_counter()
    first: float | None = None
    async with client.tts.stream(
        text=text, voice_id=VOICE_ID, language=LANGUAGE, model=MODEL, output_format=OUTPUT
    ) as stream:
        async for _ in stream:  # read to the end so requests never overlap
            if first is None:
                first = time.perf_counter() - started
    assert first is not None, "no audio"
    return first


async def ws_ttfa(ws: AsyncTTSWebSocket, text: str) -> float:
    ctx = ws.context()
    started = time.perf_counter()
    await ctx.send(text=text)
    ctx.end_input()
    first: float | None = None
    async for event in ctx.events():
        if isinstance(event, Chunk) and first is None:
            first = time.perf_counter() - started
        elif isinstance(event, Failed):
            raise event.error
        elif isinstance(event, Done):
            break
    assert first is not None, "no audio"
    return first


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def report(name: str, values: list[float]) -> None:
    ms = [v * 1000 for v in values]
    print(
        f"{name:<10} n={len(ms):<3} p50={percentile(ms, 50):7.1f} ms  p95={percentile(ms, 95):7.1f} ms  "
        f"mean={statistics.fmean(ms):7.1f} ms  min={min(ms):7.1f} ms"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="TTFA of SSE versus WebSocket.")
    parser.add_argument("--n", type=int, default=20, help="requests per transport")
    parser.add_argument("--text", default="नमस्ते, आप कैसे हैं?")
    args = parser.parse_args()

    async with AsyncBakbak() as client:
        await client.prewarm()
        ws = client.tts.websocket(
            voice_id=VOICE_ID, language=LANGUAGE, model=MODEL, output_format=OUTPUT
        )
        opened = time.perf_counter()
        await ws.connect()
        print(f"WebSocket opened in {(time.perf_counter() - opened) * 1000:.1f} ms")
        await sse_ttfa(client, args.text)  # warm-up, not measured
        await ws_ttfa(ws, args.text)
        sse: list[float] = []
        websocket: list[float] = []
        for _ in range(args.n):
            sse.append(await sse_ttfa(client, args.text))
            websocket.append(await ws_ttfa(ws, args.text))
        await ws.close()
    print(f"{client.base_url}  model={MODEL}  sequential")
    report("SSE", sse)
    report("WebSocket", websocket)


if __name__ == "__main__":
    asyncio.run(main())
