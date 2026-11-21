"""The official Python SDK for the Bakbak text-to-speech API by Raya.

::

    from bakbak import Bakbak

    client = Bakbak()  # reads BAKBAK_API_KEY
    audio = client.tts.generate(text="नमस्ते, आप कैसे हैं?", voice_id="…", language="hi")
    audio.save("hello.wav")

``AsyncBakbak`` has the same surface for asyncio. ``bakbak.text`` splits text
into sentences, ``bakbak.audio`` wraps raw audio in WAV, and
``bakbak.websocket`` holds the WebSocket types.
"""

from . import audio, text, types, websocket
from ._client import AsyncBakbak, Bakbak
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    BakbakError,
    InsufficientCreditsError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    StreamInterruptedError,
    UnprocessableEntityError,
    WebSocketClosedError,
)
from ._models import Audio, AudioChunk
from ._response import RawResponse
from ._streaming import AsyncAudioStream, AudioStream
from ._types import NOT_GIVEN, NotGiven, Timeout
from ._version import __version__
from .types import OutputFormat, Voice

__all__ = [
    "NOT_GIVEN",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "AsyncAudioStream",
    "AsyncBakbak",
    "Audio",
    "AudioChunk",
    "AudioStream",
    "AuthenticationError",
    "BadRequestError",
    "Bakbak",
    "BakbakError",
    "InsufficientCreditsError",
    "InternalServerError",
    "NotFoundError",
    "NotGiven",
    "OutputFormat",
    "PermissionDeniedError",
    "RateLimitError",
    "RawResponse",
    "RequestTooLargeError",
    "StreamInterruptedError",
    "Timeout",
    "UnprocessableEntityError",
    "Voice",
    "WebSocketClosedError",
    "__version__",
    "audio",
    "text",
    "types",
    "websocket",
]
