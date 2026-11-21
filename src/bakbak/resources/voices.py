"""``client.voices``."""

# The resources are the client's own internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import builtins
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING

from .._base_client import filter_voices
from .._constants import VOICES_PATH
from .._response import RawResponse
from .._types import NOT_GIVEN, NotGiven, TimeoutLike
from ..types import Language, Model, Voice

if TYPE_CHECKING:
    from .._client import AsyncBakbak, Bakbak

__all__ = ["AsyncVoices", "Voices"]


class Voices:
    """The voice catalog."""

    def __init__(self, client: Bakbak) -> None:
        self._client = client

    def list(
        self,
        *,
        model: Model | None = None,
        language: Language | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> builtins.list[Voice]:
        """List the voices, optionally only those of one model and language.

        The filters are applied by the SDK; the API returns every voice.
        """
        return self.with_raw_response.list(
            model=model,
            language=language,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        ).parse()

    @cached_property
    def with_raw_response(self) -> VoicesWithRawResponse:
        """The same methods, returning a ``RawResponse`` with the HTTP response."""
        return VoicesWithRawResponse(self._client)


class VoicesWithRawResponse:
    def __init__(self, client: Bakbak) -> None:
        self._client = client

    def list(
        self,
        *,
        model: Model | None = None,
        language: Language | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> RawResponse[builtins.list[Voice]]:
        client = self._client
        voices, response, retries = client._request(
            "GET",
            VOICES_PATH,
            body=None,
            accept="application/json",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._read_voices,
        )
        return RawResponse(response, filter_voices(voices, model, language), retries)


class AsyncVoices:
    """The voice catalog."""

    def __init__(self, client: AsyncBakbak) -> None:
        self._client = client

    async def list(
        self,
        *,
        model: Model | None = None,
        language: Language | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> builtins.list[Voice]:
        """List the voices, optionally only those of one model and language.

        The filters are applied by the SDK; the API returns every voice.
        """
        raw = await self.with_raw_response.list(
            model=model,
            language=language,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        )
        return raw.parse()

    @cached_property
    def with_raw_response(self) -> AsyncVoicesWithRawResponse:
        """The same methods, returning a ``RawResponse`` with the HTTP response."""
        return AsyncVoicesWithRawResponse(self._client)


class AsyncVoicesWithRawResponse:
    def __init__(self, client: AsyncBakbak) -> None:
        self._client = client

    async def list(
        self,
        *,
        model: Model | None = None,
        language: Language | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutLike | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> RawResponse[builtins.list[Voice]]:
        client = self._client
        voices, response, retries = await client._request(
            "GET",
            VOICES_PATH,
            body=None,
            accept="application/json",
            options=client._options(timeout, max_retries, extra_headers),
            handle=client._read_voices,
        )
        return RawResponse(response, filter_voices(voices, model, language), retries)
