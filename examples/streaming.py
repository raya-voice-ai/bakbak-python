"""Stream speech into a WAV file as it is generated.

The audio arrives in chunks of whole 16-bit samples, so each one can be written
(or played) the moment it lands.

    BAKBAK_API_KEY=... python examples/streaming.py
"""

from __future__ import annotations

import time
import wave

from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import Bakbak

SAMPLE_RATE = 24000


def main() -> None:
    with Bakbak() as client, wave.open("streaming.wav", "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        started = time.perf_counter()
        first_audio: float | None = None
        with client.tts.stream(
            text="आपका पैकेज आज शाम तक पहुँच जाएगा। डिलीवरी से पहले हमारा एजेंट आपको कॉल करेगा।",
            voice_id=VOICE_ID,
            language=LANGUAGE,
            model=MODEL,
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
        ) as stream:
            for chunk in stream:
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                out.writeframes(chunk.audio)
    print(
        f"wrote streaming.wav; first audio after {first_audio * 1000:.0f} ms"
        if first_audio
        else "no audio"
    )
    print(f"request {stream.request_id}, {stream.audio_bytes} bytes")


if __name__ == "__main__":
    main()
