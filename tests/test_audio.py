"""WAV wrapping (§5.2), the SSE decoder (§5.1) and sample alignment."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from bakbak import Audio
from bakbak._sse import SSEDecoder
from bakbak._streaming import SampleAligner
from bakbak.audio import sample_size, to_wav


def test_s16_wav_decodes_with_the_wave_module() -> None:
    samples = struct.pack("<4h", 0, 1000, -1000, 32767)
    data = to_wav(samples, "pcm_s16le", 8000)
    assert len(data) == 44 + len(samples)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 8000
        assert w.getnframes() == 4
        assert w.readframes(4) == samples


@pytest.mark.parametrize(
    ("encoding", "tag", "width"), [("pcm_f32le", 3, 4), ("pcm_alaw", 6, 1), ("pcm_mulaw", 7, 1)]
)
def test_non_pcm_wav_has_the_58_byte_layout(encoding: str, tag: int, width: int) -> None:
    samples = bytes(range(8)) * width
    data = to_wav(samples, encoding, 24000)
    assert len(data) == 58 + len(samples)
    riff, riff_size, wave_id = struct.unpack_from("<4sI4s", data, 0)
    assert (riff, wave_id) == (b"RIFF", b"WAVE")
    assert riff_size == len(data) - 8
    fmt = struct.unpack_from("<4sIHHIIHHH", data, 12)
    assert fmt == (b"fmt ", 18, tag, 1, 24000, 24000 * width, width, width * 8, 0)
    assert struct.unpack_from("<4sII", data, 38) == (b"fact", 4, len(samples) // width)
    assert struct.unpack_from("<4sI", data, 50) == (b"data", len(samples))
    assert data[58:] == samples


def test_odd_data_gets_a_pad_byte() -> None:
    data = to_wav(b"\x01\x02\x03", "pcm_mulaw", 8000)
    assert len(data) == 58 + 4
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    assert struct.unpack_from("<I", data, 54)[0] == 3  # the data size excludes the pad


def test_to_wav_refuses_what_it_cannot_wrap() -> None:
    with pytest.raises(ValueError, match="mp3"):
        to_wav(b"x", "mp3", 24000)
    with pytest.raises(ValueError, match="sample_rate"):
        to_wav(b"x", "pcm_s16le", 0)


def test_sample_sizes() -> None:
    assert [sample_size(e) for e in ("pcm_f32le", "pcm_s16le", "pcm_mulaw", "pcm_alaw")] == [
        4,
        2,
        1,
        1,
    ]
    assert sample_size("future_codec") == 1
    assert sample_size(None) == 1


def test_audio_to_wav() -> None:
    raw = Audio(b"\x00\x01" * 4, "r", "audio/pcm", "pcm_s16le", 16000)
    assert raw.to_wav()[:4] == b"RIFF"
    wav = Audio(b"RIFF....", "r", "audio/wav", "pcm_s16le", 16000)
    assert wav.to_wav() == wav.data
    mp3 = Audio(b"ID3", "r", "audio/mpeg", "mp3", None)
    with pytest.raises(ValueError):
        mp3.to_wav()
    unknown = Audio(b"\x00", "r", "application/octet-stream", None, None)
    with pytest.raises(ValueError, match="to_wav"):
        unknown.to_wav()
    assert "bytes=8" in repr(raw)


def test_audio_save_writes_exact_bytes(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "a.raw"  # type: ignore[operator]
    Audio(b"\x00\xff\x10", None, None, None, None).save(path)
    assert path.read_bytes() == b"\x00\xff\x10"


def _feed_all(pieces: list[bytes]) -> list[tuple[str, str]]:
    decoder = SSEDecoder()
    out: list[tuple[str, str]] = []
    for piece in pieces:
        out += decoder.feed(piece)
    return out


def test_sse_lf_crlf_comments_and_multiline() -> None:
    stream = (
        b": comment\r\n\r\n"
        b'event: chunk\r\nid: 7\r\nretry: 100\r\ndata: {"a":\r\ndata: 1}\r\n\r\n'
        b"data: plain\n\n"
        b"event: lonely\n\n"
        b"data:no-space\n\n"
    )
    expected = [("chunk", '{"a":\n1}'), ("", "plain"), ("", "no-space")]
    assert _feed_all([stream]) == expected
    # Byte by byte, including a CR and LF arriving separately.
    assert _feed_all([bytes([b]) for b in stream]) == expected


def test_sse_incomplete_event_is_not_dispatched() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data: {}\n") == []
    assert decoder.feed(b"data: more") == []


def test_sse_long_line() -> None:
    payload = "x" * (3 * 1024 * 1024)
    pieces = [b"data: " + payload.encode()[:1000]]
    pieces += [payload.encode()[i : i + 65536] for i in range(1000, len(payload), 65536)]
    pieces.append(b"\n\n")
    assert _feed_all(pieces) == [("", payload)]


def test_sample_aligner_never_splits_a_sample() -> None:
    aligner = SampleAligner(4)
    assert aligner.feed(b"abcdef") == b"abcd"
    assert aligner.feed(b"gh") == b"efgh"
    assert aligner.feed(b"i") == b""
    assert aligner.leftover() == 1
    assert SampleAligner(1).feed(b"abc") == b"abc"
