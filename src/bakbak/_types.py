"""Client option types: the NOT_GIVEN sentinel and Timeout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from typing_extensions import override

from ._constants import DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT

__all__ = ["NOT_GIVEN", "NotGiven", "Timeout", "TimeoutLike"]


class NotGiven:
    """Marks an option the caller did not pass, as distinct from ``None``.

    ``timeout=None`` disables timeouts, while leaving ``timeout`` out keeps the
    client's default. ``NOT_GIVEN`` is the value of the latter.
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN: Final = NotGiven()


@dataclass(slots=True, frozen=True)
class Timeout:
    """Timeouts in seconds (BEHAVIOR.md §4.4). ``None`` disables one.

    Attributes:
        connect: TCP and TLS setup, plus the upgrade for a WebSocket.
        read: the longest wait for the next byte of a response, or for the next
            message of a WebSocket context that has unfinished requests. It
            includes the time to first audio.
        total: an optional limit on a whole HTTP call, retries and backoff
            included. It is checked between reads, and caps each read.
    """

    connect: float | None = DEFAULT_CONNECT_TIMEOUT
    read: float | None = DEFAULT_READ_TIMEOUT
    total: float | None = None

    def __post_init__(self) -> None:
        for name in ("connect", "read", "total"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or math.isnan(value)
                or value <= 0
            ):
                raise ValueError(f"Timeout.{name} must be a positive number of seconds or None")


TimeoutLike: TypeAlias = float | Timeout | None
"""A ``Timeout``; a number, which sets the read timeout (and caps the connect
timeout at that value); or ``None`` for no timeouts at all."""


def coerce_timeout(value: TimeoutLike) -> Timeout:
    """Normalise the accepted timeout forms into a ``Timeout``."""
    if isinstance(value, Timeout):
        return value
    if value is None:
        return Timeout(connect=None, read=None, total=None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("timeout must be a number of seconds, a bakbak.Timeout, or None")
    seconds = float(value)
    return Timeout(connect=min(DEFAULT_CONNECT_TIMEOUT, seconds), read=seconds, total=None)
