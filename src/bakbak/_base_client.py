"""The HTTP core: configuration, headers, and the one retry loop per client.

Every HTTP call goes through ``_request``: build the request, send it, map
failures onto the error hierarchy, and retry what BEHAVIOR.md §4.2 allows,
waiting as §4.3 says. What happens to a successful response is up to the
``handle`` callback of the call, which may itself raise a retryable error, for
example an SSE ``error`` event that arrives before any audio.
"""

from __future__ import annotations

import asyncio
import binascii
import json
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TypeVar, cast
from urllib.parse import urlsplit

import httpx

from ._constants import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    KEEPALIVE_EXPIRY,
    WEBSOCKET_PATH,
)
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    BakbakError,
    StreamInterruptedError,
    event_status,
    make_interrupted_error,
    make_status_error,
)
from ._models import Audio
from ._retry import retry_delay
from ._sse import SSEDecoder
from ._streaming import OpenStream, decode_audio, parse_events
from ._types import NotGiven, Timeout, TimeoutLike, coerce_timeout
from ._utils import as_mapping, client_header, logger, user_agent, websocket_url
from .audio import wav_format
from .types import Voice

__all__ = ["AsyncAPIClient", "SyncAPIClient"]

T = TypeVar("T")


@dataclass(slots=True)
class CallOptions:
    """The options of one call, resolved against the client's defaults."""

    timeout: Timeout
    max_retries: int
    extra_headers: Mapping[str, str] | None


def _key_problem(key: str) -> str | None:
    """What is wrong with a key that cannot be a header value, if anything."""
    if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return "is not a string"
    if any(ch.isspace() for ch in key):
        return "contains whitespace"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in key):
        return "contains control characters"
    if not key.isascii():
        return "contains non-ASCII characters"
    return None


def _check_max_retries(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise BakbakError("max_retries must be a non-negative int")
    return value


def _httpx_timeout(timeout: Timeout, deadline: float | None) -> httpx.Timeout:
    read = timeout.read
    connect = timeout.connect
    if deadline is not None:
        remaining = max(deadline - time.monotonic(), 0.001)
        read = remaining if read is None else min(read, remaining)
        connect = remaining if connect is None else min(connect, remaining)
    return httpx.Timeout(connect=connect, read=read, write=read, pool=read)


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _audio(response: httpx.Response, data: bytes) -> Audio:
    headers = response.headers
    encoding = headers.get("x-audio-format") or None
    sample_rate = _header_int(headers, "x-audio-sample-rate")
    if encoding is None and sample_rate is None:
        # Without output_format the server sends its WAV with no X-Audio-*
        # headers; the file says what it holds (BEHAVIOR.md 5.2).
        found = wav_format(data)
        if found is not None:
            encoding, sample_rate = found
    return Audio(
        data=data,
        request_id=headers.get("x-request-id"),
        content_type=headers.get("content-type"),
        encoding=encoding,
        sample_rate=sample_rate,
    )


def _voices(response: httpx.Response, content: bytes) -> list[Voice]:
    try:
        body = as_mapping(json.loads(content))
    except ValueError:
        body = None
    entries = body.get("voices") if body is not None else None
    if not isinstance(entries, list):
        raise APIError(
            "The API returned a voice list the SDK cannot read.",
            status=response.status_code,
            request_id=response.headers.get("x-request-id"),
        )
    voices: list[Voice] = []
    for item in cast("list[object]", entries):
        entry = as_mapping(item)
        if entry is None:
            continue
        voices.append(
            Voice(
                id=str(entry.get("id", "")),
                name=str(entry.get("name", "")),
                language=str(entry.get("language", "")),
                model=str(entry.get("model", "")),
            )
        )
    return voices


def filter_voices(voices: list[Voice], model: str | None, language: str | None) -> list[Voice]:
    return [
        v
        for v in voices
        if (model is None or v.model == model) and (language is None or v.language == language)
    ]


class _StreamStart:
    """Finds the first audio event of an SSE response (or its ``done``).

    An ``error`` event before any audio is an ordinary, possibly retryable,
    ``APIStatusError``; a stream that ends before any audio is a retryable
    connection error.
    """

    __slots__ = ("decoder", "pending", "response")

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.decoder = SSEDecoder()
        self.pending: deque[tuple[Mapping[str, object], bytes | None]] = deque()

    def feed(self, raw: bytes) -> bool:
        """Parse ``raw``; True once the first audio or ``done`` is pending."""
        events = parse_events(self.decoder, raw)
        for i, event in enumerate(events):
            kind = event.get("type")
            if kind == "chunk" and event.get("data"):
                try:
                    audio = decode_audio(event)  # decoded once, here
                except (binascii.Error, ValueError):
                    # The stream broke before any audio: retried (§6.4 note).
                    raise make_interrupted_error(
                        None,
                        audio_bytes=0,
                        header_request_id=self.response.headers.get("x-request-id"),
                        default_message="The server sent audio that is not valid base64.",
                    ) from None
                self.pending.append((event, audio))
                self.pending.extend((e, None) for e in events[i + 1 :])
                return True
            if kind == "done":
                self.pending.extend((e, None) for e in events[i:])
                return True
            if kind == "error":
                raise make_status_error(
                    event_status(event),
                    event,
                    header_request_id=self.response.headers.get("x-request-id"),
                )
        return False

    def ended(self) -> StreamInterruptedError:
        """The body ended with no audio and no ``done``: a connection failure,
        retried like one (§5.1)."""
        return make_interrupted_error(
            None,
            audio_bytes=0,
            header_request_id=self.response.headers.get("x-request-id"),
            default_message="The stream ended before any audio arrived.",
        )


def _transport_error(exc: Exception, phase: str) -> APIConnectionError:
    """The SDK error for an httpx failure. It is raised without its cause:
    httpx exceptions hold the request, and so the X-API-Key header (§8)."""
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(f"The request timed out {phase}.")
    detail = str(exc)[:200]
    return APIConnectionError(
        f"Connection error {phase}: {type(exc).__name__}" + (f": {detail}" if detail else ".")
    )


def _body_failure(exc: Exception, received: int, response: httpx.Response) -> BakbakError:
    """A whole-file body broke: interrupted after its first byte, else a
    connection error that can be retried."""
    if received:
        return make_interrupted_error(
            None,
            audio_bytes=received,
            header_request_id=response.headers.get("x-request-id"),
            default_message="The audio ended before it was complete.",
        )
    return _transport_error(exc, "before the audio started")


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() > deadline


class _BaseClient:
    """Configuration shared by ``Bakbak`` and ``AsyncBakbak``."""

    api_key: str
    base_url: str
    timeout: Timeout
    max_retries: int

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: TimeoutLike | NotGiven,
        max_retries: int,
        default_headers: Mapping[str, str] | None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV) or None
        if not api_key:
            raise BakbakError(
                "No API key was found: pass api_key= to the client, or set the "
                f"{API_KEY_ENV} environment variable."
            )
        problem = _key_problem(api_key)
        if problem:
            # Never echo the key: it could end up in a log.
            raise BakbakError(
                f"The API key {problem}, so it cannot be sent in a header; check for a "
                "stray space, line break or quote where it was copied."
            )
        if base_url is None:
            base_url = os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        base_url = base_url.strip().rstrip("/")
        if urlsplit(base_url).scheme not in ("http", "https"):
            raise BakbakError("base_url must start with https:// (or http://)")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = Timeout() if isinstance(timeout, NotGiven) else coerce_timeout(timeout)
        self.max_retries = _check_max_retries(max_retries)
        self._default_headers: dict[str, str] = dict(default_headers or {})

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    @property
    def websocket_url(self) -> str:
        """The URL of the synthesis WebSocket."""
        return websocket_url(self.base_url, WEBSOCKET_PATH)

    def _options(
        self,
        timeout: TimeoutLike | NotGiven,
        max_retries: int | NotGiven,
        extra_headers: Mapping[str, str] | None,
    ) -> CallOptions:
        return CallOptions(
            timeout=self.timeout if isinstance(timeout, NotGiven) else coerce_timeout(timeout),
            max_retries=(
                self.max_retries
                if isinstance(max_retries, NotGiven)
                else _check_max_retries(max_retries)
            ),
            extra_headers=extra_headers,
        )

    def _identity_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "X-Bakbak-Client": client_header()}

    def _headers(self, options: CallOptions, *, accept: str, has_body: bool) -> dict[str, str]:
        headers = {"User-Agent": user_agent(), "Accept": accept, **self._identity_headers()}
        if has_body:
            headers["Content-Type"] = "application/json"
        headers.update(self._default_headers)
        if options.extra_headers:
            headers.update(options.extra_headers)
        return headers

    def websocket_headers(self) -> dict[str, str]:
        """Headers of the WebSocket upgrade (the User-Agent is set separately)."""
        return {**self._identity_headers(), **self._default_headers}

    @staticmethod
    def _body(params: Mapping[str, object], extra_body: Mapping[str, object] | None) -> bytes:
        payload = {**params, **extra_body} if extra_body else params
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    @staticmethod
    def _deadline(options: CallOptions) -> float | None:
        total = options.timeout.total
        return time.monotonic() + total if total is not None else None

    @staticmethod
    def _status_error(response: httpx.Response, content: bytes | None) -> APIStatusError:
        body: object = None
        if content:
            try:
                body = json.loads(content)
            except ValueError:
                body = None
        return make_status_error(
            response.status_code,
            body,
            header_request_id=response.headers.get("x-request-id"),
            reason=response.reason_phrase,
        )

    def _next_wait(
        self,
        error: BakbakError,
        retries: int,
        options: CallOptions,
        deadline: float | None,
        retry_after: str | None,
    ) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        if not getattr(error, "retryable", False):
            return None
        if isinstance(error, StreamInterruptedError) and error.audio_bytes:
            return None  # never retry once audio has been delivered (§4.1)
        if retries >= options.max_retries:
            return None
        delay = retry_delay(retries + 1, retry_after)
        if deadline is not None and time.monotonic() + delay >= deadline:
            return None
        return delay


class SyncAPIClient(_BaseClient):
    """The synchronous HTTP core."""

    _client: httpx.Client
    _owns_client: bool
    _sleep: Callable[[float], None]

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: TimeoutLike | NotGiven,
        max_retries: int,
        default_headers: Mapping[str, str] | None,
        http_client: httpx.Client | None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY),
            follow_redirects=False,
        )
        self._sleep = time.sleep

    def close(self) -> None:
        """Close the connection pool, unless it was passed in as ``http_client``."""
        if self._owns_client:
            self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        accept: str,
        options: CallOptions,
        handle: Callable[[httpx.Response, float | None], T],
    ) -> tuple[T, httpx.Response, int]:
        headers = self._headers(options, accept=accept, has_body=body is not None)
        deadline = self._deadline(options)
        url = self.base_url + path
        retries = 0
        while True:
            retry_after: str | None = None
            started = time.monotonic()
            logger.debug("%s %s (attempt %d)", method, path, retries + 1)
            try:
                request = self._client.build_request(
                    method, url, headers=headers, content=body,
                    timeout=_httpx_timeout(options.timeout, deadline),
                )  # fmt: skip
                # Never follow a redirect, even with a caller's http_client:
                # it would carry X-API-Key to another origin (§8).
                response = self._client.send(request, stream=True, follow_redirects=False)
            except httpx.HTTPError as exc:
                error: BakbakError = _transport_error(exc, "before any response")
            else:
                logger.debug(
                    "%s %s -> %d (request_id %s) in %.0f ms",
                    method, path, response.status_code,
                    response.headers.get("x-request-id"), (time.monotonic() - started) * 1000,
                )  # fmt: skip
                try:
                    if not 200 <= response.status_code < 300:
                        retry_after = response.headers.get("retry-after")
                        try:
                            content = response.read()
                        except httpx.HTTPError:
                            content = None
                        raise self._status_error(response, content)
                    return handle(response, deadline), response, retries
                except (APIStatusError, APIConnectionError, StreamInterruptedError) as exc:
                    response.close()
                    error = exc
                except BaseException:
                    response.close()
                    raise
            wait = self._next_wait(error, retries, options, deadline, retry_after)
            if wait is None:
                raise error
            retries += 1
            logger.debug("retrying %s %s in %.2f s after: %s", method, path, wait, error.message)
            self._sleep(wait)

    # Handlers: what to do with a successful response.

    @staticmethod
    def _read_audio(response: httpx.Response, deadline: float | None) -> Audio:
        parts: list[bytes] = []
        received = 0
        failure: BakbakError | None = None
        try:
            for piece in response.iter_bytes():
                parts.append(piece)
                received += len(piece)
                if _past(deadline):
                    raise httpx.ReadTimeout("total timeout")
        except httpx.HTTPError as exc:
            failure = _body_failure(exc, received, response)
        finally:
            response.close()
        if failure is not None:
            raise failure  # outside the except block: no httpx exception in its chain
        return _audio(response, b"".join(parts))

    @staticmethod
    def _start_stream(
        response: httpx.Response, deadline: float | None
    ) -> OpenStream[Iterator[bytes]]:
        start = _StreamStart(response)
        raw = response.iter_bytes()
        failure: BakbakError | None = None
        try:
            for piece in raw:
                if start.feed(piece):
                    return OpenStream(response, start.pending, start.decoder, raw, deadline)
                if _past(deadline):
                    raise httpx.ReadTimeout("total timeout")
        except httpx.HTTPError as exc:
            failure = _transport_error(exc, "before any audio")
        raise failure or start.ended()

    @staticmethod
    def _read_voices(response: httpx.Response, deadline: float | None) -> list[Voice]:
        failure: BakbakError | None = None
        content = b""
        try:
            content = response.read()
        except httpx.HTTPError as exc:
            failure = _transport_error(exc, "while reading the voice list")
        if failure is not None:
            raise failure
        return _voices(response, content)


class AsyncAPIClient(_BaseClient):
    """The asynchronous HTTP core."""

    _client: httpx.AsyncClient
    _owns_client: bool
    _sleep: Callable[[float], Awaitable[None]]

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: TimeoutLike | NotGiven,
        max_retries: int,
        default_headers: Mapping[str, str] | None,
        http_client: httpx.AsyncClient | None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY),
            follow_redirects=False,
        )
        self._sleep = asyncio.sleep

    async def close(self) -> None:
        """Close the connection pool, unless it was passed in as ``http_client``."""
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        accept: str,
        options: CallOptions,
        handle: Callable[[httpx.Response, float | None], Awaitable[T]],
    ) -> tuple[T, httpx.Response, int]:
        headers = self._headers(options, accept=accept, has_body=body is not None)
        deadline = self._deadline(options)
        url = self.base_url + path
        retries = 0
        while True:
            retry_after: str | None = None
            started = time.monotonic()
            logger.debug("%s %s (attempt %d)", method, path, retries + 1)
            try:
                request = self._client.build_request(
                    method, url, headers=headers, content=body,
                    timeout=_httpx_timeout(options.timeout, deadline),
                )  # fmt: skip
                # Never follow a redirect, even with a caller's http_client:
                # it would carry X-API-Key to another origin (§8).
                response = await self._client.send(request, stream=True, follow_redirects=False)
            except httpx.HTTPError as exc:
                error: BakbakError = _transport_error(exc, "before any response")
            else:
                logger.debug(
                    "%s %s -> %d (request_id %s) in %.0f ms",
                    method, path, response.status_code,
                    response.headers.get("x-request-id"), (time.monotonic() - started) * 1000,
                )  # fmt: skip
                try:
                    if not 200 <= response.status_code < 300:
                        retry_after = response.headers.get("retry-after")
                        try:
                            content = await response.aread()
                        except httpx.HTTPError:
                            content = None
                        raise self._status_error(response, content)
                    return await handle(response, deadline), response, retries
                except (APIStatusError, APIConnectionError, StreamInterruptedError) as exc:
                    await response.aclose()
                    error = exc
                except BaseException:
                    await response.aclose()
                    raise
            wait = self._next_wait(error, retries, options, deadline, retry_after)
            if wait is None:
                raise error
            retries += 1
            logger.debug("retrying %s %s in %.2f s after: %s", method, path, wait, error.message)
            await self._sleep(wait)

    @staticmethod
    async def _read_audio(response: httpx.Response, deadline: float | None) -> Audio:
        parts: list[bytes] = []
        received = 0
        failure: BakbakError | None = None
        try:
            async for piece in response.aiter_bytes():
                parts.append(piece)
                received += len(piece)
                if _past(deadline):
                    raise httpx.ReadTimeout("total timeout")
        except httpx.HTTPError as exc:
            failure = _body_failure(exc, received, response)
        finally:
            await response.aclose()
        if failure is not None:
            raise failure  # outside the except block: no httpx exception in its chain
        return _audio(response, b"".join(parts))

    @staticmethod
    async def _start_stream(
        response: httpx.Response, deadline: float | None
    ) -> OpenStream[AsyncIterator[bytes]]:
        start = _StreamStart(response)
        raw = response.aiter_bytes()
        failure: BakbakError | None = None
        try:
            async for piece in raw:
                if start.feed(piece):
                    return OpenStream(response, start.pending, start.decoder, raw, deadline)
                if _past(deadline):
                    raise httpx.ReadTimeout("total timeout")
        except httpx.HTTPError as exc:
            failure = _transport_error(exc, "before any audio")
        raise failure or start.ended()

    @staticmethod
    async def _read_voices(response: httpx.Response, deadline: float | None) -> list[Voice]:
        failure: BakbakError | None = None
        content = b""
        try:
            content = await response.aread()
        except httpx.HTTPError as exc:
            failure = _transport_error(exc, "while reading the voice list")
        if failure is not None:
            raise failure
        return _voices(response, content)
