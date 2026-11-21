"""Identity headers, logging and URL helpers."""

from __future__ import annotations

import logging
import platform
import re
import sys
from collections.abc import Mapping
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ._constants import PACKAGE_NAME
from ._version import __version__

__all__ = ["as_mapping", "client_header", "logger", "redact_url", "user_agent", "websocket_url"]

logger = logging.getLogger("bakbak")
"""The SDK's logger. It logs at DEBUG only, and never the API key or any text."""

_SECRET_QUERY_KEYS = frozenset({"x-api-key", "api_key", "apikey", "key", "token"})


def redact_url(url: str) -> str:
    """``url`` with any credential in the query string replaced by ``***``."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [
        (k, "***" if k.lower() in _SECRET_QUERY_KEYS else v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query, safe="*")))


def user_agent() -> str:
    """``bakbak-python/<version> python/<version> <os>/<arch>`` (BEHAVIOR.md §8)."""
    python = ".".join(str(p) for p in sys.version_info[:3])
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{PACKAGE_NAME}/{__version__} python/{python} {system}/{machine}"


def client_header() -> str:
    """The ``X-Bakbak-Client`` value."""
    return f"{PACKAGE_NAME}/{__version__}"


_SCHEME = re.compile(r"^http", re.IGNORECASE)


def websocket_url(base_url: str, path: str) -> str:
    """The WebSocket URL for ``path``: the base URL with https -> wss (or
    http -> ws), plus the path."""
    parts = urlsplit(base_url)
    scheme = _SCHEME.sub("ws", parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/") + path, "", ""))


def as_mapping(value: object) -> Mapping[str, object] | None:
    """``value`` as a JSON object, or ``None`` if it is not one."""
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else None
