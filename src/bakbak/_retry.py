"""How long to wait before a retry (BEHAVIOR.md §4.3)."""

from __future__ import annotations

import email.utils
import math
import random
from collections.abc import Callable
from datetime import datetime, timezone

from ._constants import INITIAL_RETRY_DELAY, MAX_RETRY_AFTER, MAX_RETRY_DELAY, RETRY_JITTER

__all__ = ["backoff_delay", "parse_retry_after", "retry_delay"]


def backoff_delay(retry: int, rand: Callable[[], float] = random.random) -> float:
    """Wait before retry ``retry`` (1-based) when the server gave no Retry-After:
    ``min(0.5 * 2^(n-1), 8) * (1 - 0.25 * r)`` with ``r`` uniform in [0, 1)."""
    base = min(INITIAL_RETRY_DELAY * 2 ** (retry - 1), MAX_RETRY_DELAY)
    return base * (1 - RETRY_JITTER * rand())


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header, capped at 60 s.

    Accepts a number of seconds or an HTTP date; returns ``None`` when the
    header is absent or unreadable.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    if math.isnan(seconds):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


def retry_delay(retry: int, retry_after: str | None) -> float:
    """The wait before retry ``retry``: Retry-After if usable, else backoff."""
    delay = parse_retry_after(retry_after)
    return backoff_delay(retry) if delay is None else delay
