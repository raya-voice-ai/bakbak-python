"""Speak an LLM's reply while it is still being written.

With OPENAI_API_KEY set, the reply comes from an OpenAI-compatible chat
completions stream (OPENAI_BASE_URL and OPENAI_MODEL choose the server and the
model). Without it, a fake token generator stands in for the LLM. Either way,
``ctx.send_stream`` sends each sentence the moment it is complete, and the
audio of the first sentence plays while the LLM is still writing the rest.

    BAKBAK_API_KEY=... python examples/llm_streaming.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import AsyncIterator

import httpx
from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import AsyncBakbak
from bakbak.audio import to_wav
from bakbak.text import Segmenter

PROMPT = "दो छोटे वाक्यों में बताइए कि आज मौसम कैसा रहेगा।"
FAKE_REPLY = (
    "आज आसमान साफ़ रहेगा और हल्की हवा चलेगी। शाम को तापमान थोड़ा कम हो जाएगा, तो एक हल्का स्वेटर साथ रखें।"
)


async def openai_compatible_stream(api_key: str) -> AsyncIterator[dict[str, object]]:
    """Chat completion chunks, parsed from the server's SSE. send_stream finds
    ``choices[0].delta.content`` in them by itself."""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with (
        httpx.AsyncClient(timeout=60) as http,
        http.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


async def fake_tokens() -> AsyncIterator[str]:
    """Pieces of a reply, a few characters at a time, at an LLM's pace."""
    i = 0
    while i < len(FAKE_REPLY):
        size = random.randint(2, 6)
        yield FAKE_REPLY[i : i + size]
        i += size
        await asyncio.sleep(0.02)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    tokens = openai_compatible_stream(api_key) if api_key else fake_tokens()
    print("tokens from", "an OpenAI-compatible API" if api_key else "a fake generator")
    async with AsyncBakbak() as client:
        ws = await client.tts.websocket(
            voice_id=VOICE_ID,
            language=LANGUAGE,
            model=MODEL,
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
        )
        ctx = ws.context()
        started = time.perf_counter()
        # Send sentences as they complete while the audio is read below. An
        # eager first clause starts the audio sooner on a long first sentence.
        sender = asyncio.create_task(
            ctx.send_stream(tokens, segmenter=Segmenter(eager_first_clause=True))
        )
        audio = bytearray()
        async for chunk in ctx.audio():
            if not audio:
                print(f"first audio after {(time.perf_counter() - started) * 1000:.0f} ms")
            audio += chunk  # a voice agent would play it here
        handles = await sender
        await ws.close()
    with open("llm_streaming.wav", "wb") as f:
        f.write(to_wav(bytes(audio), "pcm_s16le", 24000))
    print(f"{len(handles)} requests, wrote llm_streaming.wav ({len(audio)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
