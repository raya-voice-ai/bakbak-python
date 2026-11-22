"""WAV wrapping (§5.2), the SSE decoder (§5.1) and sample alignment."""

from __future__ import annotations

import io
import struct
import wave

import httpx
import pytest

from bakbak import Audio
from bakbak._base_client import _audio
from bakbak._sse import SSEDecoder
from bakbak._streaming import SampleAligner
from bakbak.audio import sample_size, to_wav, wav_format


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
    with pytest.raises(ValueError, match="MP3 audio cannot be wrapped in WAV"):
        to_wav(b"x", "mp3", 24000)
    with pytest.raises(ValueError, match="pcm_alaw"):
        to_wav(b"x", "opus", 24000)
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
    # Not the advice for raw audio: following it would not help.
    with pytest.raises(ValueError, match="MP3 audio cannot be wrapped in WAV; decode it"):
        mp3.to_wav()
    unknown = Audio(b"\x00", "r", "application/octet-stream", None, None)
    with pytest.raises(ValueError, match=r"did not state.*to_wav"):
        unknown.to_wav()
    assert "bytes=8" in repr(raw)
    # A WAV body is returned as it is, whatever it was labelled.
    labelled = Audio(to_wav(b"\x00\x00", "pcm_s16le", 24000), "r", "audio/mpeg", "mp3", None)
    assert labelled.to_wav() == labelled.data


def _fmt(tag: int, channels: int, rate: int, bits: int, extra: bytes = b"") -> bytes:
    width = bits // 8
    body = struct.pack("<HHIIHH", tag, channels, rate, rate * width, width, bits) + extra
    return b"fmt " + struct.pack("<I", len(body)) + body


def _riff(*chunks: bytes, size: int = 0xFFFFFFFF) -> bytes:
    return b"RIFF" + struct.pack("<I", size) + b"WAVE" + b"".join(chunks)


@pytest.mark.parametrize("encoding", ["pcm_s16le", "pcm_f32le", "pcm_mulaw", "pcm_alaw"])
def test_wav_format_reads_what_to_wav_writes(encoding: str) -> None:
    assert wav_format(to_wav(b"\x00" * 8, encoding, 16000)) == (encoding, 16000)


def test_wav_format_tolerates_streamed_sizes_and_other_chunks() -> None:
    data_chunk = b"data" + struct.pack("<I", 0xFFFFFFFF) + b"\x00\x01" * 10
    # The server's standard model writes 0xFFFFFFFF for both sizes.
    assert wav_format(_riff(_fmt(1, 1, 24000, 16), data_chunk)) == ("pcm_s16le", 24000)
    # A chunk before fmt, odd-sized and so followed by a pad byte.
    junk = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\x00"
    assert wav_format(_riff(junk, _fmt(3, 1, 22050, 32), data_chunk)) == ("pcm_f32le", 22050)
    # WAVE_FORMAT_EXTENSIBLE: the tag is the start of the sub-format GUID.
    guid = struct.pack("<H", 7) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    extensible = struct.pack("<HHI", 22, 8, 4) + guid
    assert wav_format(_riff(_fmt(0xFFFE, 1, 8000, 8, extensible))) == ("pcm_mulaw", 8000)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"RIFF",
        b"ID3\x04\x00" + bytes(64),  # mp3
        b"\xff\xfb\x90\x64" + bytes(64),
        b"RIFF\x00\x00\x00\x00WAVE",  # no chunks at all
        _riff(_fmt(1, 2, 24000, 16)),  # stereo
        _riff(_fmt(1, 1, 24000, 24)),  # 24-bit
        _riff(_fmt(1, 1, 0, 16)),  # no rate
        _riff(_fmt(1, 1, 24000, 16))[:30],  # fmt cut short
        _riff(b"data" + struct.pack("<I", 4) + b"\x00" * 4, _fmt(1, 1, 24000, 16)),  # data first
        _riff(b"LIST" + struct.pack("<I", 0xFFFFFFFF) + b"x" * 8),  # runs off the end
        _riff(_fmt(0xFFFE, 1, 8000, 8, struct.pack("<HHI", 22, 8, 4))),  # GUID missing
    ],
)
def test_wav_format_refuses_anything_else(data: bytes) -> None:
    assert wav_format(data) is None


def test_whole_file_metadata_falls_back_to_the_wav_header() -> None:
    wav = _riff(_fmt(1, 1, 24000, 16), b"data" + struct.pack("<I", 4) + b"\x00" * 4)
    bare = httpx.Response(200, headers={"content-type": "audio/wav", "x-request-id": "req_1"})
    audio = _audio(bare, wav)
    assert (audio.encoding, audio.sample_rate) == ("pcm_s16le", 24000)
    # The headers win when the server sends them.
    stated = httpx.Response(
        200, headers={"x-audio-format": "pcm_s16le", "x-audio-sample-rate": "16000"}
    )
    assert _audio(stated, wav).sample_rate == 16000
    # mp3 states its format and no rate; the request's rate is not guessed.
    mp3 = _audio(httpx.Response(200, headers={"x-audio-format": "mp3"}), b"ID3" + bytes(16))
    assert (mp3.encoding, mp3.sample_rate) == ("mp3", None)
    # Raw audio with no headers stays unknown.
    raw = _audio(httpx.Response(200, headers={"content-type": "audio/pcm"}), b"\x00" * 8)
    assert (raw.encoding, raw.sample_rate) == (None, None)


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
