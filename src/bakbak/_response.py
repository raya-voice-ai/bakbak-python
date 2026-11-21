"""``RawResponse``: what ``with_raw_response`` methods return."""

from __future__ import annotations

from typing import Generic, TypeVar

import httpx
from typing_extensions import override

__all__ = ["RawResponse"]

T = TypeVar("T")


class RawResponse(Generic[T]):
    """The HTTP response behind a call, plus its parsed result.

    Attributes:
        http_response: the ``httpx.Response`` of the attempt that succeeded.
        retries_taken: how many retries the call needed.
    """

    __slots__ = ("_parsed", "http_response", "retries_taken")

    def __init__(self, http_response: httpx.Response, parsed: T, retries_taken: int) -> None:
        self.http_response = http_response
        self.retries_taken = retries_taken
        self._parsed = parsed

    @property
    def headers(self) -> httpx.Headers:
        return self.http_response.headers

    @property
    def status_code(self) -> int:
        return self.http_response.status_code

    @property
    def request_id(self) -> str | None:
        return self.http_response.headers.get("x-request-id")

    def parse(self) -> T:
        """The result the plain method would have returned."""
        return self._parsed

    @override
    def __repr__(self) -> str:
        return (
            f"RawResponse(status_code={self.status_code}, request_id={self.request_id!r}, "
            f"retries_taken={self.retries_taken})"
        )
