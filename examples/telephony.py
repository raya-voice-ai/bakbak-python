"""Telephony audio: 8 kHz mu-law, cut into 20 ms frames.

Phone networks and media-stream APIs (for example Twilio Media Streams) carry
8 kHz mu-law in 20 ms frames of 160 bytes, base64-encoded in JSON. Asking the
API for exactly that format means no resampling or transcoding on your side.
This example prints the frames it would send and saves a WAV to listen to.

    BAKBAK_API_KEY=... python examples/telephony.py
"""

from __future__ import annotations

import base64
import json

from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import Bakbak
from bakbak.audio import to_wav

SAMPLE_RATE = 8000
FRAME_BYTES = SAMPLE_RATE // 50  # 20 ms of 8-bit mu-law


MULAW_SILENCE = b"\xff"


def send_frame(frame: bytes, n: int) -> None:
    """Stand-in for writing one media message to the call's WebSocket."""
    message = {"event": "media", "media": {"payload": base64.b64encode(frame).decode()}}
    if n < 2:
        print(json.dumps(message)[:80] + "...")


def main() -> None:
    frames = 0
    pending = b""
    audio = bytearray()
    with (
        Bakbak() as client,
        client.tts.stream(
            text="नमस्ते, आपकी कॉल हमारे लिए महत्वपूर्ण है। कृपया लाइन पर बने रहें।",
            voice_id=VOICE_ID,
            language=LANGUAGE,
            model=MODEL,
            output_format={"container": "raw", "encoding": "pcm_mulaw", "sample_rate": SAMPLE_RATE},
        ) as stream,
    ):
        for chunk in stream:
            audio += chunk.audio
            pending += chunk.audio
            while len(pending) >= FRAME_BYTES:
                send_frame(pending[:FRAME_BYTES], frames)
                pending = pending[FRAME_BYTES:]
                frames += 1
    if pending:  # the last frame, padded with silence
        send_frame(pending.ljust(FRAME_BYTES, MULAW_SILENCE), frames)
        frames += 1
    with open("telephony.wav", "wb") as f:
        f.write(to_wav(bytes(audio), "pcm_mulaw", SAMPLE_RATE))
    print(f"{frames} frames of 20 ms ({len(audio)} bytes); wrote telephony.wav")


if __name__ == "__main__":
    main()
