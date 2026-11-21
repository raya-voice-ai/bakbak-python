"""Two conversations over one WebSocket.

Each context plays its requests in order; the two contexts run at the same
time and their audio interleaves on the connection, but each context only
sees its own. Opening the socket first means no request pays the handshake.

A deployment with little capacity may refuse a request (503) while the other
context is generating; the refusal ends that request only.

    BAKBAK_API_KEY=... python examples/websocket_contexts.py
"""

from __future__ import annotations

import asyncio

from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import AsyncBakbak
from bakbak.audio import to_wav
from bakbak.websocket import AsyncContext, Chunk, Failed

SAMPLE_RATE = 16000


async def speak(ctx: AsyncContext, sentences: list[str], path: str) -> None:
    for sentence in sentences:
        await ctx.send(text=sentence)
    ctx.end_input()
    audio = bytearray()
    async for event in ctx.events():
        if isinstance(event, Chunk):
            audio += event.audio
        elif isinstance(event, Failed):
            # A failure ends its own request only; the context carries on.
            print(f"{ctx.context_id}: a request failed: {event.error}")
    with open(path, "wb") as f:
        f.write(to_wav(bytes(audio), "pcm_s16le", SAMPLE_RATE))
    print(f"{ctx.context_id}: wrote {path} ({len(audio)} bytes)")


async def main() -> None:
    async with (
        AsyncBakbak() as client,
        client.tts.websocket(
            voice_id=VOICE_ID,
            language=LANGUAGE,
            model=MODEL,
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
        ) as ws,
    ):
        await asyncio.gather(
            speak(
                ws.context("caller-1"),
                ["नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?", "आपका ऑर्डर कल आ जाएगा।"],
                "caller-1.wav",
            ),
            speak(
                ws.context("caller-2"),
                ["आपका रिचार्ज सफल रहा।", "आपका नया बैलेंस पाँच सौ रुपये है।"],
                "caller-2.wav",
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
