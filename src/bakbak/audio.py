"""Audio helpers: WAV wrapping and sample sizes (BEHAVIOR.md §5.2).

There are no playback helpers here on purpose; hand the bytes to the audio
library you already use.
"""

from __future__ import annotations

import struct

from ._utils import as_mapping

__all__ = ["sample_size", "to_wav"]

MP3_TO_WAV = (
    "MP3 audio cannot be wrapped in WAV; decode it with an audio library, or request "
    "output_format={'container': 'wav', 'encoding': 'pcm_s16le'} instead"
)

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


# (format tag, bits per sample) of a mono WAV, by encoding name.
_WAV_ENCODINGS: dict[tuple[int, int], str] = {
    (1, 16): "pcm_s16le",
    (3, 32): "pcm_f32le",
    (6, 8): "pcm_alaw",
    (7, 8): "pcm_mulaw",
}
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def is_wav(data: bytes) -> bool:
    """Whether ``data`` starts like a WAV file."""
    return len(data) >= 12 and data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE"


def wav_format(data: bytes) -> tuple[str, int] | None:
    """The ``(encoding, sample_rate)`` a mono WAV file's ``fmt `` chunk states.

    ``None`` for anything else: not WAV, more than one channel, an encoding the
    API does not produce, or a header cut short. The RIFF and ``data`` sizes are
    not trusted, since a WAV written as a stream may give them as 0xFFFFFFFF.
    """
    if not is_wav(data):
        return None
    offset = 12
    while offset + 8 <= len(data):
        chunk, size = data[offset : offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        body = offset + 8
        if chunk == b"fmt ":
            if size < 16 or body + 16 > len(data):
                return None
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if tag == _WAVE_FORMAT_EXTENSIBLE:
                # The real tag is the first two bytes of the sub-format GUID.
                if size < 40 or body + 26 > len(data):
                    return None
                tag = struct.unpack_from("<H", data, body + 24)[0]
            encoding = _WAV_ENCODINGS.get((tag, bits))
            if encoding is None or channels != 1 or rate <= 0:
                return None
            return encoding, rate
        if chunk == b"data":
            return None  # fmt must come before the audio
        offset = body + size + (size & 1)
    return None


def to_wav(data: bytes, encoding: str, sample_rate: int) -> bytes:
    """Wrap raw mono audio in a WAV header.

    ``pcm_s16le`` gets the canonical 44-byte PCM header. ``pcm_f32le``
    (format 3), ``pcm_alaw`` (6) and ``pcm_mulaw`` (7) get an 18-byte ``fmt ``
    chunk (``cbSize`` 0) plus a ``fact`` chunk: 58 bytes in all, as the WAV
    specification requires for non-PCM formats.

    Raises:
        ValueError: for an encoding WAV cannot carry this way (such as mp3), a
            sample rate that is not a positive integer, or ``pcm_s16le`` or
            ``pcm_f32le`` data that is not a whole number of samples (an odd
            length, or one that is not a multiple of 4), whose header would
            disagree with the data.
    """
    if encoding == "mp3":
        raise ValueError(MP3_TO_WAV)
    tag = _FORMAT_TAGS.get(encoding)
    if tag is None:
        known = ", ".join(sorted(_FORMAT_TAGS))
        raise ValueError(f"Cannot wrap {encoding!r} audio in WAV; expected one of: {known}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("sample_rate must be a positive integer")
    width = _SAMPLE_SIZES[encoding]
    size = len(data)
    if size % width:
        raise ValueError(
            f"{size} bytes of {encoding} audio is not a whole number of {width}-byte "
            "samples, so it cannot be wrapped in WAV"
        )
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
