"""Audio helpers: WAV wrapping and sample sizes (BEHAVIOR.md §5.2).

There are no playback helpers here on purpose; hand the bytes to the audio
library you already use.
"""

from __future__ import annotations

import struct

from ._utils import as_mapping

__all__ = ["sample_size", "to_wav"]

_SAMPLE_SIZES: dict[str, int] = {"pcm_f32le": 4, "pcm_s16le": 2, "pcm_mulaw": 1, "pcm_alaw": 1}
# WAVE format tags: PCM, IEEE float, A-law, mu-law.
_FORMAT_TAGS: dict[str, int] = {"pcm_s16le": 1, "pcm_f32le": 3, "pcm_alaw": 6, "pcm_mulaw": 7}


def sample_size(encoding: str | None) -> int:
    """Bytes per sample of a mono ``encoding``; 1 for anything unknown."""
    return _SAMPLE_SIZES.get(encoding or "", 1)


def stream_sample_size(output_format: object) -> int:
    """Bytes per sample a stream requested with ``output_format`` will carry.

    Without an ``output_format`` the streams send 32-bit float PCM.
    """
    fields = as_mapping(output_format)
    if fields is None:
        return _SAMPLE_SIZES["pcm_f32le"]
    encoding = fields.get("encoding")
    return sample_size(encoding) if isinstance(encoding, str) else 1


def to_wav(data: bytes, encoding: str, sample_rate: int) -> bytes:
    """Wrap raw mono audio in a WAV header.

    ``pcm_s16le`` gets the canonical 44-byte PCM header. ``pcm_f32le``
    (format 3), ``pcm_alaw`` (6) and ``pcm_mulaw`` (7) get an 18-byte ``fmt ``
    chunk (``cbSize`` 0) plus a ``fact`` chunk: 58 bytes in all, as the WAV
    specification requires for non-PCM formats.

    Raises:
        ValueError: for an encoding WAV cannot carry this way (such as mp3), or
            a sample rate that is not a positive integer.
    """
    tag = _FORMAT_TAGS.get(encoding)
    if tag is None:
        known = ", ".join(sorted(_FORMAT_TAGS))
        raise ValueError(f"Cannot wrap {encoding!r} audio in WAV; expected one of: {known}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("sample_rate must be a positive integer")
    width = _SAMPLE_SIZES[encoding]
    size = len(data)
    # RIFF chunks are word aligned: an odd-sized data chunk gets a pad byte that
    # the RIFF size counts and the data size does not.
    pad = b"\x00" if size % 2 else b""
    if tag == 1:
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + size + len(pad), b"WAVE",
            b"fmt ", 16, tag, 1, sample_rate, sample_rate * width, width, width * 8,
            b"data", size,
        )  # fmt: skip
    else:
        header = struct.pack(
            "<4sI4s4sIHHIIHHH4sII4sI",
            b"RIFF", 50 + size + len(pad), b"WAVE",
            b"fmt ", 18, tag, 1, sample_rate, sample_rate * width, width, width * 8, 0,
            b"fact", 4, size // width,
            b"data", size,
        )  # fmt: skip
    return b"".join((header, data, pad))
