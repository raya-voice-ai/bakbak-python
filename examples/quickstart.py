"""Synthesize a sentence and save it as a WAV file.

BAKBAK_API_KEY=... python examples/quickstart.py
"""

from __future__ import annotations

from _settings import LANGUAGE, MODEL, VOICE_ID

from bakbak import Bakbak


def main() -> None:
    with Bakbak() as client:
        audio = client.tts.generate(
            text="नमस्ते! बकबक में आपका स्वागत है।",
            voice_id=VOICE_ID,
            language=LANGUAGE,
            model=MODEL,
            output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
        )
    audio.save("quickstart.wav")
    print(f"wrote quickstart.wav: {len(audio.data)} bytes (request {audio.request_id})")


if __name__ == "__main__":
    main()
