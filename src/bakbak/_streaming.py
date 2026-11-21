"""SSE stream results: ``AudioStream`` and ``AsyncAudioStream``.

A stream is lazy: the request is sent when the stream is entered (``with``) or
first iterated. Before any audio, failures are retried by the client's retry
loop; once audio has been handed out, a failure raises
``StreamInterruptedError`` and is never retried (BEHAVIOR.md §4.1).
"""

from __future__ import annotations

import binascii
import json
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import httpx
from typing_extensions import Self

from ._exceptions import BakbakError, make_interrupted_error
from ._models import AudioChunk
from ._sse import SSEDecoder
from ._utils import as_mapping, logger

if TYPE_CHECKING:
    from .types import OutputFormat

__all__ = ["AsyncAudioStream", "AudioStream"]


def parse_events(decoder: SSEDecoder, chunk: bytes) -> list[Mapping[str, object]]:
    """Decode the JSON payloads of the events completed by ``chunk``.

    The payload's ``type`` is authoritative; the ``event:`` name only fills it
    in when the payload has none. Payloads that are not JSON objects are
    ignored, as are unknown types further up.
    """
    out: list[Mapping[str, object]] = []
    for name, data in decoder.feed(chunk):
        try:
            payload = as_mapping(json.loads(data))
        except ValueError:
            logger.debug("ignoring an SSE event whose data is not JSON (%d bytes)", len(data))
            continue
        if payload is None:
            continue
        if "type" not in payload and name:
            payload = {**payload, "type": name}
        out.append(payload)
    return out


def decode_audio(event: Mapping[str, object]) -> bytes:
    """The audio of a ``chunk`` event, base64-decoded exactly once."""
    data = event.get("data")
    if not isinstance(data, str) or not data:
        return b""
    return binascii.a2b_base64(data)


def step_time(event: Mapping[str, object]) -> float | None:
    value = event.get("step_time")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class SampleAligner:
    """Holds back a partial sample so chunks never split one (§5.2)."""

    __slots__ = ("_carry", "_width")

    def __init__(self, width: int) -> None:
        self._width = width
        self._carry = b""

    def feed(self, audio: bytes) -> bytes:
        if self._width <= 1:
            return audio
        if self._carry:
            audio = self._carry + audio
        cut = len(audio) - len(audio) % self._width
        self._carry = audio[cut:]
        return audio[:cut] if self._carry else audio

    def leftover(self) -> int:
        """Bytes of an incomplete trailing sample, which are dropped."""
        return len(self._carry)


RawT = TypeVar("RawT")


@dataclass(slots=True)
class OpenStream(Generic[RawT]):
    """A stream whose first audio (or its ``done``) has arrived.

    Attributes:
        response: the open response.
        pending: events already parsed and not yet handled, starting with the
            first audio event, each with its audio if already decoded.
        decoder: the SSE decoder, holding any partial event.
        raw: the response's byte iterator, to continue reading from.
        deadline: when the call's total timeout ends, if it has one.
    """

    response: httpx.Response
    pending: deque[tuple[Mapping[str, object], bytes | None]]
    decoder: SSEDecoder
    raw: RawT
    deadline: float | None


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class _StreamBase:
    __slots__ = (
        "_aligner",
        "_closed",
        "_consumed",
        "_deadline",
        "_format",
        "_open",
        "audio_bytes",
    )

    def __init__(self, output_format: OutputFormat | None, sample_width: int) -> None:
        self._format = output_format
        self._aligner = SampleAligner(sample_width)
        self._open: OpenStream[Any] | None = None
        self._consumed = False
        self._closed = False
        self._deadline: float | None = None
        self.audio_bytes = 0
        """Audio bytes handed out so far."""

    @property
    def response(self) -> httpx.Response | None:
        """The HTTP response, once its headers have arrived."""
        return self._open.response if self._open is not None else None

    @property
    def request_id(self) -> str | None:
        """``X-Request-Id``, once the response headers have arrived."""
        return self._open.response.headers.get("x-request-id") if self._open else None

    @property
    def content_type(self) -> str | None:
        return self._open.response.headers.get("content-type") if self._open else None

    @property
    def encoding(self) -> str | None:
        """``X-Audio-Format``, else the requested ``output_format`` encoding."""
        if self._open is not None:
            value = self._open.response.headers.get("x-audio-format")
            if value:
                return value
        return self._format.get("encoding") if self._format else None

    @property
    def sample_rate(self) -> int | None:
        """``X-Audio-Sample-Rate``, else the requested sample rate."""
        if self._open is not None:
            value = _header_int(self._open.response.headers, "x-audio-sample-rate")
            if value is not None:
                return value
        return self._format.get("sample_rate") if self._format else None

    def _begin(self) -> None:
        if self._consumed:
            raise BakbakError("This stream has already been read; streams can be read once.")
        if self._closed:
            raise BakbakError("This stream is closed.")
        self._consumed = True

    def _handle(
        self, event: Mapping[str, object], decoded: bytes | None = None
    ) -> AudioChunk | bool:
        """An ``AudioChunk`` to hand out, ``True`` at the end, ``False`` to skip."""
        kind = event.get("type")
        if kind == "chunk":
            if decoded is None:
                try:
                    decoded = decode_audio(event)
                except (binascii.Error, ValueError):
                    # The stream breaks here (§6.4 note); never a raw binascii.Error.
                    raise make_interrupted_error(
                        None,
                        audio_bytes=self.audio_bytes,
                        header_request_id=self.request_id,
                        default_message="The server sent audio that is not valid base64.",
                    ) from None
            audio = self._aligner.feed(decoded)
            if not audio:
                return False
            self.audio_bytes += len(audio)
            return AudioChunk(audio, step_time(event), self.request_id)
        if kind == "done":
            if self._aligner.leftover():
                logger.debug(
                    "dropping %d bytes of an incomplete last sample", self._aligner.leftover()
                )
            return True
        if kind == "error":
            raise make_interrupted_error(
                event, audio_bytes=self.audio_bytes, header_request_id=self.request_id
            )
        return False

    def _check_deadline(self) -> None:
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise make_interrupted_error(
                None,
                audio_bytes=self.audio_bytes,
                header_request_id=self.request_id,
                default_message="The total timeout elapsed before the stream finished.",
            )

    def _broken(self, exc: Exception) -> BakbakError:
        if isinstance(exc, httpx.TimeoutException):
            message = "No audio arrived within the read timeout."
        else:
            message = "The connection broke before the stream finished."
        return make_interrupted_error(
            None,
            audio_bytes=self.audio_bytes,
            header_request_id=self.request_id,
            default_message=message,
        )

    def _truncated(self) -> BakbakError:
        return make_interrupted_error(
            None, audio_bytes=self.audio_bytes, header_request_id=self.request_id
        )


class AudioStream(_StreamBase):
    """The audio of ``client.tts.stream()``, as it is generated.

    Iterate it for ``AudioChunk`` objects, or use ``iter_audio()``, ``read()``
    or ``save()``. It can be read once. Use it as a context manager, or read
    it to the end, so the connection is released.

    ``request_id``, ``content_type``, ``encoding`` and ``sample_rate`` are set
    once the response headers have arrived (on ``with`` or the first chunk).
    """

    __slots__ = ("_start",)

    def __init__(
        self,
        start: Callable[[], OpenStream[Iterator[bytes]]],
        *,
        output_format: OutputFormat | None,
        sample_width: int,
    ) -> None:
        super().__init__(output_format, sample_width)
        self._start = start

    def _ensure_open(self) -> OpenStream[Iterator[bytes]]:
        if self._open is None:
            if self._closed:
                raise BakbakError("This stream is closed.")
            self._open = self._start()
            self._deadline = self._open.deadline
        return self._open

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[AudioChunk]:
        return self._chunks()

    def _chunks(self) -> Iterator[AudioChunk]:
        self._begin()
        opened = self._ensure_open()
        try:
            while opened.pending:
                result = self._handle(*opened.pending.popleft())
                if result is True:
                    return
                if result is not False:
                    yield result
            failure: BakbakError | None = None
            try:
                for raw in opened.raw:
                    for event in parse_events(opened.decoder, raw):
                        result = self._handle(event)
                        if result is True:
                            return
                        if result is not False:
                            yield result
                    self._check_deadline()
            except (httpx.TimeoutException, httpx.TransportError, httpx.StreamError) as exc:
                failure = self._broken(exc)
            # Raised outside the except block, so no httpx exception (which
            # holds the request headers) is chained to it.
            raise failure or self._truncated()
        finally:
            self.close()

    def iter_audio(self) -> Iterator[bytes]:
        """Iterate over the audio bytes only."""
        for chunk in self:
            yield chunk.audio

    def read(self) -> bytes:
        """Read the whole stream and return its audio."""
        return b"".join(self.iter_audio())

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the audio to ``path`` as it arrives, exactly as received."""
        with open(path, "wb") as f:
            for audio in self.iter_audio():
                f.write(audio)

    def close(self) -> None:
        """Release the connection. Safe to call more than once."""
        self._closed = True
        if self._open is not None:
            self._open.response.close()


class AsyncAudioStream(_StreamBase):
    """The audio of ``client.tts.stream()`` on ``AsyncBakbak``.

    The async twin of :class:`AudioStream`: ``async for``, ``async with``,
    and awaitable ``read()``, ``save()`` and ``close()``.
    """

    __slots__ = ("_start",)

    def __init__(
        self,
        start: Callable[[], Awaitable[OpenStream[AsyncIterator[bytes]]]],
        *,
        output_format: OutputFormat | None,
        sample_width: int,
    ) -> None:
        super().__init__(output_format, sample_width)
        self._start = start

    async def _ensure_open(self) -> OpenStream[AsyncIterator[bytes]]:
        if self._open is None:
            if self._closed:
                raise BakbakError("This stream is closed.")
            self._open = await self._start()
            self._deadline = self._open.deadline
        return self._open

    async def __aenter__(self) -> Self:
        await self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[AudioChunk]:
        self._begin()
        opened = await self._ensure_open()
        try:
            while opened.pending:
                result = self._handle(*opened.pending.popleft())
                if result is True:
                    return
                if result is not False:
                    yield result
            failure: BakbakError | None = None
            try:
                async for raw in opened.raw:
                    for event in parse_events(opened.decoder, raw):
                        result = self._handle(event)
                        if result is True:
                            return
                        if result is not False:
                            yield result
                    self._check_deadline()
            except (httpx.TimeoutException, httpx.TransportError, httpx.StreamError) as exc:
                failure = self._broken(exc)
            # Raised outside the except block, so no httpx exception (which
            # holds the request headers) is chained to it.
            raise failure or self._truncated()
        finally:
            await self.close()

    async def iter_audio(self) -> AsyncIterator[bytes]:
        """Iterate over the audio bytes only."""
        async for chunk in self:
            yield chunk.audio

    async def read(self) -> bytes:
        """Read the whole stream and return its audio."""
        return b"".join([audio async for audio in self.iter_audio()])

    async def save(self, path: str | os.PathLike[str]) -> None:
        """Write the audio to ``path`` as it arrives, exactly as received."""
        with open(path, "wb") as f:  # noqa: ASYNC230 - small sequential writes of local files
            async for audio in self.iter_audio():
                f.write(audio)

    async def close(self) -> None:
        """Release the connection. Safe to call more than once."""
        self._closed = True
        if self._open is not None:
            await self._open.response.aclose()
