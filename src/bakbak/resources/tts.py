"""``client.tts``: whole files, SSE streams and the WebSocket."""

# The resources are the client's own internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING

from typing_extensions import Unpack

from .._constants import STREAM_PATH, TTS_PATH
from .._exceptions import APIStatusError
from .._models import Audio
from .._response import RawResponse
from .._streaming import AsyncAudioStream, AudioStream
from .._types import NOT_GIVEN, NotGiven, TimeoutLike
from .._ws._async import AsyncTTSWebSocket
from .._ws._sync import TTSWebSocket
from ..audio import stream_sample_size
from ..types import RequestDefaults, SynthesisParams

if TYPE_CHECKING:
    from .._client import AsyncBakbak, Bakbak

__all__ = ["TTS", "AsyncTTS"]


class TTS:
    """Text to speech."""

    def __init__(self, client: Bakbak) -> None:
        self._client = client

    def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> Audio:
        """Synthesize a whole file (``POST /v1/text-to-speech``).

        Args:
            text: 1 to 5,000 characters.
            voice_id: a voice from ``client.voices.list()``.
            language: e.g. ``"hi"``; ``m1`` voices use ``"en"`` for English.
            model: ``"standard"`` (the default) or ``"m1"``; it must match the voice.
            speed: 0.5 to 1.5. Changes ``standard`` voices only; ``m1`` ignores it.
            output_format: e.g. ``{"container": "wav", "encoding": "pcm_s16le",
                "sample_rate": 24000}``. Without it the result is 24 kHz WAV.
            extra_headers: headers to add to this request.
            extra_body: fields to add to the body, for anything the SDK does not
                model yet.
            timeout: overrides the client's timeout for this call.
            max_retries: overrides the client's ``max_retries`` for this call.

        Returns:
            The audio, with its request id and format.
        """
        return self.with_raw_response.generate(
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            max_retries=max_retries,
            **params,
        ).parse()

    def stream(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> AudioStream:
        """Stream one request's audio as it is generated (Server-Sent Events).

        Takes the same arguments as :meth:`generate`. The ``output_format``
        container must be ``raw``; without one, the stream is 24 kHz 32-bit
        float PCM.

        The request is sent when the stream is entered or first iterated::

            with client.tts.stream(text="नमस्ते", voice_id=..., language="hi") as stream:
                for chunk in stream:
                    play(chunk.audio)
        """
        client = self._client
        options = client._options(timeout, max_retries, extra_headers)
        body = client._body(params, extra_body)
        output_format = params.get("output_format")
        return AudioStream(
            lambda: client._request(
                "POST",
                STREAM_PATH,
                body=body,
                accept="text/event-stream",
                options=options,
                handle=client._start_stream,
            )[0],
            output_format=output_format,
            sample_width=stream_sample_size(output_format),
        )

    def websocket(
        self,
        *,
        on_error: Callable[[APIStatusError], None] | None = None,
        **defaults: Unpack[RequestDefaults],
    ) -> TTSWebSocket:
        """A WebSocket that carries many streaming requests on one connection.

        Nothing is opened until ``connect()``, ``with``, or the first send.

        Args:
            on_error: called (from the reader thread) with errors the server
                reports about no request in particular. They are logged at
                debug level when it is not set.
            **defaults: request fields every context fills in, e.g. ``voice_id``,
                ``language``, ``model`` and ``output_format``.
        """
        return TTSWebSocket(self._client, defaults, on_error)

    @cached_property
    def with_raw_response(self) -> TTSWithRawResponse:
        """The same methods, returning a ``RawResponse`` with the HTTP response."""
        return TTSWithRawResponse(self._client)


class TTSWithRawResponse:
    def __init__(self, client: Bakbak) -> None:
        self._client = client

    def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> RawResponse[Audio]:
        client = self._client
        audio, response, retries = client._request(
            "POST",
            TTS_PATH,
            body=client._body(params, extra_body),
            accept="*/*",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._read_audio,
        )
        return RawResponse(response, audio, retries)

    def stream(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> RawResponse[AudioStream]:
        """Send the request now and return once its first audio has arrived."""
        client = self._client
        opened, response, retries = client._request(
            "POST",
            STREAM_PATH,
            body=client._body(params, extra_body),
            accept="text/event-stream",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._start_stream,
        )
        output_format = params.get("output_format")
        stream = AudioStream(
            lambda: opened,
            output_format=output_format,
            sample_width=stream_sample_size(output_format),
        )
        stream.__enter__()
        return RawResponse(response, stream, retries)


class AsyncTTS:
    """Text to speech."""

    def __init__(self, client: AsyncBakbak) -> None:
        self._client = client

    async def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> Audio:
        """Synthesize a whole file (``POST /v1/text-to-speech``).

        See :meth:`bakbak.resources.tts.TTS.generate` for the arguments.
        """
        raw = await self.with_raw_response.generate(
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            max_retries=max_retries,
            **params,
        )
        return raw.parse()

    def stream(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> AsyncAudioStream:
        """Stream one request's audio as it is generated (Server-Sent Events).

        The request is sent when the stream is entered or first iterated::

            async with client.tts.stream(text="नमस्ते", voice_id=..., language="hi") as stream:
                async for chunk in stream:
                    play(chunk.audio)
        """
        client = self._client
        options = client._options(timeout, max_retries, extra_headers)
        body = client._body(params, extra_body)
        output_format = params.get("output_format")

        async def start() -> _AsyncOpen:
            opened, _, _ = await client._request(
                "POST",
                STREAM_PATH,
                body=body,
                accept="text/event-stream",
                options=options,
                handle=client._start_stream,
            )
            return opened

        return AsyncAudioStream(
            start, output_format=output_format, sample_width=stream_sample_size(output_format)
        )

    def websocket(
        self,
        *,
        on_error: Callable[[APIStatusError], None] | None = None,
        **defaults: Unpack[RequestDefaults],
    ) -> AsyncTTSWebSocket:
        """A WebSocket that carries many streaming requests on one connection.

        Nothing is opened until ``await ws.connect()`` (or ``await`` on the
        result, or ``async with``), or the first send.

        Args:
            on_error: called with errors the server reports about no request
                in particular. They are logged at debug level when it is not set.
            **defaults: request fields every context fills in, e.g. ``voice_id``,
                ``language``, ``model`` and ``output_format``.
        """
        return AsyncTTSWebSocket(self._client, defaults, on_error)

    @cached_property
    def with_raw_response(self) -> AsyncTTSWithRawResponse:
        """The same methods, returning a ``RawResponse`` with the HTTP response."""
        return AsyncTTSWithRawResponse(self._client)


class AsyncTTSWithRawResponse:
    def __init__(self, client: AsyncBakbak) -> None:
        self._client = client

    async def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> RawResponse[Audio]:
        client = self._client
        audio, response, retries = await client._request(
            "POST",
            TTS_PATH,
            body=client._body(params, extra_body),
            accept="*/*",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._read_audio,
        )
        return RawResponse(response, audio, retries)

    async def stream(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        **params: Unpack[SynthesisParams],
    ) -> RawResponse[AsyncAudioStream]:
        """Send the request now and return once its first audio has arrived."""
        client = self._client
        opened, response, retries = await client._request(
            "POST",
            STREAM_PATH,
            body=client._body(params, extra_body),
            accept="text/event-stream",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._start_stream,
        )

        async def start() -> _AsyncOpen:
            return opened

        output_format = params.get("output_format")
        stream = AsyncAudioStream(
            start, output_format=output_format, sample_width=stream_sample_size(output_format)
        )
        await stream.__aenter__()
        return RawResponse(response, stream, retries)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .._streaming import OpenStream

    _AsyncOpen = OpenStream[AsyncIterator[bytes]]
