"""``Bakbak`` and ``AsyncBakbak``: the two clients, with identical surfaces."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

import httpx
from typing_extensions import Self

from ._base_client import AsyncAPIClient, CallOptions, SyncAPIClient
from ._constants import DEFAULT_MAX_RETRIES, VOICES_PATH
from ._types import NOT_GIVEN, NotGiven, TimeoutLike
from .resources.tts import TTS, AsyncTTS
from .resources.voices import AsyncVoices, Voices

__all__ = ["AsyncBakbak", "Bakbak"]


class Bakbak(SyncAPIClient):
    """The Bakbak text-to-speech client.

    Args:
        api_key: your API key. Defaults to the ``BAKBAK_API_KEY`` environment
            variable, read here (not at import).
        base_url: defaults to ``BAKBAK_BASE_URL``, then
            ``https://hub.getraya.app``.
        timeout: a ``bakbak.Timeout``; a number of seconds, which sets the
            read timeout; or ``None`` for no timeouts. The default is 5 s to
            connect and 60 s between bytes (including the time to first audio).
        max_retries: retries of a failed request before any audio (default 2).
        default_headers: headers to send with every request.
        http_client: an ``httpx.Client`` to use, e.g. with a proxy. The client
            does not close it.

    Raises:
        BakbakError: if there is no API key.

    Example::

        from bakbak import Bakbak

        client = Bakbak()
        audio = client.tts.generate(text="नमस्ते", voice_id="…", language="hi")
        audio.save("hello.wav")
    """

    tts: TTS
    voices: Voices

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.tts = TTS(self)
        self.voices = Voices(self)

    def with_options(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> Bakbak:
        """A copy of this client with some options changed.

        The copy shares this client's connection pool unless ``http_client``
        is given; ``default_headers`` are added to this client's.
        """
        copy = Bakbak(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            max_retries=self.max_retries if isinstance(max_retries, NotGiven) else max_retries,
            default_headers={**self._default_headers, **(default_headers or {})},
            http_client=http_client or self._client,
        )
        copy._sleep = self._sleep
        return copy

    def prewarm(self) -> None:
        """Open a connection to the API now, so the next request pays no TCP
        or TLS handshake. It lists the voices once, which also checks the key.

        Raises:
            AuthenticationError: the key was refused.
            APIConnectionError: the API could not be reached.
        """
        self._request(
            "GET",
            VOICES_PATH,
            body=None,
            accept="application/json",
            options=CallOptions(timeout=self.timeout, max_retries=0, extra_headers=None),
            handle=self._read_voices,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncBakbak(AsyncAPIClient):
    """The Bakbak text-to-speech client for asyncio.

    Takes the same arguments as :class:`Bakbak` (``http_client`` is an
    ``httpx.AsyncClient``).

    Example::

        from bakbak import AsyncBakbak

        client = AsyncBakbak()
        audio = await client.tts.generate(text="नमस्ते", voice_id="…", language="hi")
    """

    tts: AsyncTTS
    voices: AsyncVoices

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.tts = AsyncTTS(self)
        self.voices = AsyncVoices(self)

    def with_options(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> AsyncBakbak:
        """A copy of this client with some options changed.

        The copy shares this client's connection pool unless ``http_client``
        is given; ``default_headers`` are added to this client's.
        """
        copy = AsyncBakbak(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            max_retries=self.max_retries if isinstance(max_retries, NotGiven) else max_retries,
            default_headers={**self._default_headers, **(default_headers or {})},
            http_client=http_client or self._client,
        )
        copy._sleep = self._sleep
        return copy

    async def prewarm(self) -> None:
        """Open a connection to the API now, so the next request pays no TCP
        or TLS handshake. It lists the voices once, which also checks the key.

        Raises:
            AuthenticationError: the key was refused.
            APIConnectionError: the API could not be reached.
        """
        await self._request(
            "GET",
            VOICES_PATH,
            body=None,
            accept="application/json",
            options=CallOptions(timeout=self.timeout, max_retries=0, extra_headers=None),
            handle=self._read_voices,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
