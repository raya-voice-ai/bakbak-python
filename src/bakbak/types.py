"""Request parameter types and response models.

Request fields keep their wire names (BEHAVIOR.md §2). Enumerations are open:
the known values are listed for editors and type checkers, and any other string
is accepted, so a language added on the server needs no SDK release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import Required, TypedDict

__all__ = [
    "Container",
    "Encoding",
    "Language",
    "Model",
    "OutputFormat",
    "RequestDefaults",
    "SampleRate",
    "SendParams",
    "SynthesisParams",
    "Voice",
]

Language: TypeAlias = (
    Literal[
        "hi", "mr", "te", "kn", "bn", "as", "gu", "ne", "ml", "ta", "pa", "en", "en-in", "en-us"
    ]
    | str
)
"""A language code. ``m1`` voices use ``en`` for English."""

Model: TypeAlias = Literal["standard", "m1"] | str
"""A model. It must match the voice's model; the server defaults to ``standard``."""

Container: TypeAlias = Literal["raw", "wav", "mp3"] | str
"""``raw`` or ``wav`` (whole file only) or ``mp3`` (whole file only)."""

Encoding: TypeAlias = Literal["pcm_f32le", "pcm_s16le", "pcm_mulaw", "pcm_alaw"] | str
"""Sample encoding of ``raw`` and ``wav`` audio. Always mono."""

SampleRate: TypeAlias = Literal[8000, 16000, 22050, 24000] | int
"""Sample rate in Hz. The server default is 24000."""


class OutputFormat(TypedDict, total=False):
    """The audio format to return.

    Without ``output_format`` the whole-file endpoint returns 24 kHz WAV and
    the streams send 24 kHz 32-bit float PCM.
    """

    container: Required[Container]
    encoding: Encoding
    """Required for ``raw`` and ``wav``; leave it out for ``mp3``."""
    sample_rate: SampleRate


class SynthesisParams(TypedDict, total=False):
    """The body of ``tts.generate`` and ``tts.stream``."""

    text: Required[str]
    """1 to 5,000 characters. Hindi must be in Devanagari."""
    voice_id: Required[str]
    """A voice id from ``voices.list()``. A voice belongs to one model."""
    language: Required[Language]
    model: Model
    speed: float
    """0.5 to 1.5; default 1.0."""
    output_format: OutputFormat


class RequestDefaults(TypedDict, total=False):
    """Fields a WebSocket or one of its contexts fills in for every request."""

    voice_id: str
    language: Language
    model: Model
    speed: float
    output_format: OutputFormat


class SendParams(RequestDefaults, total=False):
    """The body of one WebSocket request. Fields left out come from the
    defaults of the context and of the WebSocket."""

    text: Required[str]


@dataclass(slots=True)
class Voice:
    """A voice from ``GET /v1/voices``."""

    id: str
    name: str
    language: str
    model: str
