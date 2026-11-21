"""Events a WebSocket context yields (BEHAVIOR.md §6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from typing_extensions import override

from .._exceptions import BakbakError

__all__ = ["Cancelled", "Chunk", "Done", "Failed", "RequestResult", "WebSocketEvent"]

H = TypeVar("H")


@dataclass(slots=True)
class Chunk(Generic[H]):
    """Audio of one request. It always holds whole samples."""

    audio: bytes
    request: H

    @override
    def __repr__(self) -> str:
        return f"Chunk(audio=<{len(self.audio)} bytes>, request={self.request!r})"


@dataclass(slots=True)
class Done(Generic[H]):
    """The request finished."""

    request: H


@dataclass(slots=True)
class Failed(Generic[H]):
    """The request failed; ``error`` says why. Later requests still run."""

    request: H
    error: BakbakError


@dataclass(slots=True)
class Cancelled(Generic[H]):
    """The request was cancelled by ``ctx.cancel()``."""

    request: H


WebSocketEvent: TypeAlias = Chunk[H] | Done[H] | Failed[H] | Cancelled[H]
"""What ``ctx.events()`` yields."""


@dataclass(slots=True)
class RequestResult:
    """The outcome of a request that did not fail.

    Attributes:
        request_id: the server's id for the request, if it sent one.
        audio_bytes: the audio delivered.
        cancelled: whether ``ctx.cancel()`` ended it.
    """

    request_id: str | None
    audio_bytes: int
    cancelled: bool = False
