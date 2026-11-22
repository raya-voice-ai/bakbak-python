"""Response models: ``Audio`` and ``AudioChunk``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from typing_extensions import override

from .audio import MP3_TO_WAV, is_wav, to_wav

__all__ = ["Audio", "AudioChunk"]

_WAV_TYPES = frozenset({"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"})


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


@dataclass(slots=True)
class Audio:
    """A whole synthesized file, from ``client.tts.generate()``.

    Attributes:
        data: the audio bytes exactly as received.
        request_id: ``X-Request-Id``; quote it to support.
        content_type: ``audio/wav``, ``audio/mpeg``, ``audio/pcm``,
            ``audio/basic`` (raw mu-law) or ``application/octet-stream``
            (raw A-law).
        encoding: ``pcm_s16le``, ``pcm_f32le``, ``pcm_mulaw``, ``pcm_alaw`` or
            ``mp3``. It is ``X-Audio-Format``; a WAV response without that
            header (no ``output_format``) gets it from the file's own header.
        sample_rate: in Hz, from ``X-Audio-Sample-Rate`` or, like ``encoding``,
            from the WAV header. ``None`` for mp3, whose rate is in its frames.
    """

    data: bytes = field(repr=False)
    request_id: str | None
    content_type: str | None
    encoding: str | None
    sample_rate: int | None

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write exactly the bytes received to ``path``."""
        with open(path, "wb") as f:
            f.write(self.data)

    def to_wav(self) -> bytes:
        """The audio as a WAV file.

        WAV audio is returned unchanged; raw PCM, mu-law and A-law are wrapped
        in a WAV header (see :func:`bakbak.audio.to_wav`).

        Raises:
            ValueError: for mp3, which has to be decoded rather than wrapped;
                raw audio whose encoding and sample rate the response did not
                state; or raw audio that is not a whole number of samples.
        """
        if _media_type(self.content_type) in _WAV_TYPES or is_wav(self.data):
            return self.data
        if self.encoding == "mp3" or _media_type(self.content_type) == "audio/mpeg":
            raise ValueError(MP3_TO_WAV)
        if self.encoding is None or self.sample_rate is None:
            raise ValueError(
                "The response did not state the audio's encoding and sample rate, so it "
                "cannot be wrapped in WAV; if you know them, use "
                "bakbak.audio.to_wav(audio.data, encoding, sample_rate)"
            )
        return to_wav(self.data, self.encoding, self.sample_rate)

    @override
    def __repr__(self) -> str:
        return (
            f"Audio(bytes={len(self.data)}, content_type={self.content_type!r}, "
            f"encoding={self.encoding!r}, sample_rate={self.sample_rate!r}, "
            f"request_id={self.request_id!r})"
        )


@dataclass(slots=True)
class AudioChunk:
    """One piece of a streamed response.

    Attributes:
        audio: the decoded audio. It always holds whole samples.
        step_time: the model's time for this step in seconds, when sent.
        request_id: the request's id.
    """

    audio: bytes
    step_time: float | None = None
    request_id: str | None = None

    @override
    def __repr__(self) -> str:
        return (
            f"AudioChunk(audio=<{len(self.audio)} bytes>, step_time={self.step_time!r}, "
            f"request_id={self.request_id!r})"
        )
