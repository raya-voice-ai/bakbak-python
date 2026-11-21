"""API resources: ``client.tts`` and ``client.voices``."""

from .tts import TTS, AsyncTTS
from .voices import AsyncVoices, Voices

__all__ = ["TTS", "AsyncTTS", "AsyncVoices", "Voices"]
