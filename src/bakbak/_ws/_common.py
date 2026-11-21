"""Pieces the async and sync WebSocket clients share."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from collections.abc import Mapping, Sequence
from typing import cast

from websockets.exceptions import InvalidStatus, InvalidURI, WebSocketException
from websockets.frames import Frame

from .._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BakbakError,
    make_status_error,
)
from .._utils import logger
from ..audio import stream_sample_size

__all__ = [
    "WEBSOCKET_LOGGER",
    "build_request",
    "check_context_id",
    "connect_error",
    "new_context_id",
    "open_failure",
    "text_delta",
]

_CONTEXT_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


class _Redact(logging.Filter):
    """Keeps the key and the synthesis text out of the websockets library's
    debug log: it logs every handshake header and every frame."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and args:
            first: object = args[0]
            if isinstance(first, Frame):
                record.args = (f"{first.opcode.name} frame ({len(first.data)} bytes)", *args[1:])
            elif len(args) == 2 and isinstance(first, str) and first.lower() == "x-api-key":
                record.args = (first, "***")
        return True


WEBSOCKET_LOGGER = logging.getLogger("bakbak.websockets")
"""The logger the websockets library writes to for this SDK's connections."""
WEBSOCKET_LOGGER.addFilter(_Redact())


def new_context_id() -> str:
    """``ctx_`` plus 32 hex digits from a CSPRNG."""
    return f"ctx_{secrets.token_hex(16)}"


def check_context_id(context_id: str | None) -> str:
    if context_id is None:
        return new_context_id()
    if not isinstance(context_id, str) or not _CONTEXT_ID.fullmatch(context_id):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise BakbakError("context_id must be 1 to 128 characters of A-Z a-z 0-9 . _ -")
    return context_id


def build_request(
    defaults: Mapping[str, object],
    params: Mapping[str, object],
    extra_body: Mapping[str, object] | None,
    context_id: str,
) -> tuple[str, int]:
    """The JSON message of one request and the sample width of its audio."""
    payload: dict[str, object] = {**defaults, **params}
    if extra_body:
        payload.update(extra_body)
    payload["context_id"] = context_id
    wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return wire, stream_sample_size(payload.get("output_format"))


# Typed items whose ``text`` or ``content`` is a piece of the reply: Anthropic's
# text delta, a text block, and LangChain's message chunk. Other typed items
# (tool calls, reasoning, a final "done" event with the whole text) are skipped.
_TEXT_KINDS = frozenset({"text", "text_delta", "AIMessageChunk"})


def _field(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return cast("Mapping[str, object]", obj).get(name)
    try:
        return getattr(obj, name, None)
    except Exception:  # a property that raises, e.g. a response with no text part
        return None


def text_delta(item: object) -> str | None:
    """The text a stream item carries, found by duck typing.

    Accepts ``str``; OpenAI chat chunks (``choices[0].delta.content``);
    OpenAI Responses events (``response.output_text.delta``); Anthropic
    ``content_block_delta`` events (``delta.text``); Gemini chunks (``.text``);
    and objects or dicts with a ``content`` string. Anything else, such as a
    tool call, a reasoning delta or a usage event, carries no text: ``None``.
    """
    if isinstance(item, str):
        return item
    if item is None:
        return None
    if isinstance(item, (bytes, bytearray, memoryview)):
        raise TypeError("send_stream() takes text, not bytes")
    choices = _field(item, "choices")
    if choices is not None:
        if isinstance(choices, Sequence) and choices and not isinstance(choices, str):
            content = _field(_field(cast("Sequence[object]", choices)[0], "delta"), "content")
            return content if isinstance(content, str) else None
        return None
    kind = _field(item, "type")
    delta = _field(item, "delta")
    if isinstance(delta, str):
        accepted = kind is None or (isinstance(kind, str) and kind.endswith("output_text.delta"))
        return delta if accepted else None
    if delta is not None:
        text = _field(delta, "text")
        return text if isinstance(text, str) else None
    if kind is not None and kind not in _TEXT_KINDS:
        return None
    text = _field(item, "text")
    if isinstance(text, str):
        return text
    content = _field(item, "content")
    return content if isinstance(content, str) else None


def connect_error(exc: BaseException) -> tuple[BakbakError, str | None] | None:
    """Map a failure to open the WebSocket onto the error hierarchy, with the
    refusal's Retry-After header. ``None`` for exceptions that are not about
    connecting (they propagate)."""
    if isinstance(exc, InvalidStatus):
        response = exc.response
        raw = response.body or b""
        try:
            body: object = json.loads(raw) if raw else None
        except ValueError:
            body = None
        error: APIStatusError = make_status_error(
            response.status_code,
            body,
            header_request_id=response.headers.get("X-Request-Id"),
            reason=response.reason_phrase,
        )
        return error, response.headers.get("Retry-After")
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return APITimeoutError("Timed out opening the WebSocket."), None
    if isinstance(exc, InvalidURI):
        return BakbakError("The WebSocket URL is not valid; check base_url."), None
    if isinstance(exc, (OSError, WebSocketException)):
        return APIConnectionError(f"Could not open the WebSocket: {type(exc).__name__}."), None
    return None


def open_failure(exc: BaseException) -> BakbakError:
    """What a request waiting for a connection fails with when opening one
    raised ``exc``: the SDK error itself, or a connection error that names
    an unexpected one (every waiting request must get an outcome)."""
    if isinstance(exc, BakbakError):
        return exc
    logger.debug("opening the WebSocket raised an unexpected error", exc_info=exc)
    return APIConnectionError(
        f"Could not open the WebSocket: {type(exc).__name__}: {exc}", retryable=False
    )
