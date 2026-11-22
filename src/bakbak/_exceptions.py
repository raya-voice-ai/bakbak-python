"""The error hierarchy of BEHAVIOR.md §3, and how errors are built from responses.

No error ever holds the API key or the request headers: not in its message, not
in ``repr()``/``str()``, and not in any attribute.
"""

from __future__ import annotations

import http
from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import override

from ._constants import RETRYABLE_STATUSES
from ._utils import as_mapping

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "BakbakError",
    "InsufficientCreditsError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "RequestTooLargeError",
    "StreamInterruptedError",
    "UnprocessableEntityError",
    "WebSocketClosedError",
]


def _restore(cls: type[BakbakError], message: str, state: dict[str, Any]) -> BakbakError:
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    error.__dict__.update(state)
    return error


class BakbakError(Exception):
    """Anything raised by the SDK."""

    message: str

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        # Errors cross process boundaries (multiprocessing, task queues) intact.
        return (_restore, (type(self), self.message, dict(self.__dict__)))

    @override
    def __str__(self) -> str:
        return self.message

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


class APIError(BakbakError):
    """The service answered with an error, or a stream broke.

    Attributes:
        status: the HTTP status, or the event's ``status_code``; ``None`` for a
            truncated stream.
        error_code: the body's ``error_code``, if it had one.
        message: a sentence to show a person.
        retryable: whether sending the same request again can succeed.
        request_id: ``X-Request-Id``, else the body's or the event's
            ``request_id``. Quote it to support.
        body: the parsed JSON body (or stream event), so extra fields such as
            ``required_inr`` stay readable.
    """

    status: int | None
    error_code: str | None
    retryable: bool
    request_id: str | None
    body: object | None

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
        body: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.retryable = retryable
        self.request_id = request_id
        self.body = body

    @override
    def __str__(self) -> str:
        details = [
            f"status {self.status}" if self.status is not None else None,
            f"error_code {self.error_code}" if self.error_code else None,
            f"request_id {self.request_id}" if self.request_id else None,
        ]
        shown = ", ".join(d for d in details if d)
        return f"{self.message} ({shown})" if shown else self.message

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status={self.status!r}, error_code={self.error_code!r}, "
            f"message={self.message!r}, retryable={self.retryable!r}, "
            f"request_id={self.request_id!r})"
        )


class APIStatusError(APIError):
    """An HTTP error status, or an in-band error before any audio."""

    status: int | None


class BadRequestError(APIStatusError):
    """400."""


class AuthenticationError(APIStatusError):
    """401: the API key is missing, wrong or revoked."""


class InsufficientCreditsError(APIStatusError):
    """402: the account's balance does not cover the request."""


class PermissionDeniedError(APIStatusError):
    """403."""


class NotFoundError(APIStatusError):
    """404: for example an unknown ``voice_id``."""


class RequestTooLargeError(APIStatusError):
    """413: the text or the body is too long."""


class UnprocessableEntityError(APIStatusError):
    """422: the request failed validation."""


class RateLimitError(APIStatusError):
    """429: for example the account's concurrency limit."""


class InternalServerError(APIStatusError):
    """5xx: the service or the model failed, or is out of capacity."""


class StreamInterruptedError(APIError):
    """The request failed after audio was delivered. It is never retried
    automatically, because the audio already played would play twice.

    Attributes:
        audio_bytes: how much audio was handed out before the failure.
    """

    audio_bytes: int

    def __init__(
        self,
        message: str,
        *,
        audio_bytes: int,
        status: int | None = None,
        error_code: str | None = None,
        retryable: bool = True,
        request_id: str | None = None,
        body: object | None = None,
    ) -> None:
        super().__init__(
            message,
            status=status,
            error_code=error_code,
            retryable=retryable,
            request_id=request_id,
            body=body,
        )
        self.audio_bytes = audio_bytes

    @override
    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, audio_bytes={self.audio_bytes!r})"


class APIConnectionError(BakbakError):
    """No usable answer: a network, TLS or DNS failure.

    Attributes:
        retryable: whether trying again can succeed.
    """

    retryable: bool

    def __init__(
        self, message: str = "Could not connect to the API.", *, retryable: bool = True
    ) -> None:
        super().__init__(message)
        self.retryable = retryable


class APITimeoutError(APIConnectionError):
    """A timeout before any audio arrived."""

    def __init__(self, message: str = "The request timed out.") -> None:
        super().__init__(message, retryable=True)


class WebSocketClosedError(APIConnectionError):
    """The WebSocket closed and a request could not continue.

    Attributes:
        code: the close code, if the server sent one.
        reason: the close reason, if any.
    """

    code: int | None
    reason: str

    def __init__(
        self, message: str, *, code: int | None = None, reason: str = "", retryable: bool = False
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.code = code
        self.reason = reason

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, message={self.message!r}, "
            f"retryable={self.retryable!r})"
        )


_STATUS_CLASSES: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: InsufficientCreditsError,
    403: PermissionDeniedError,
    404: NotFoundError,
    413: RequestTooLargeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def status_error_class(status: int | None) -> type[APIStatusError]:
    """The ``APIStatusError`` subclass for an HTTP status."""
    if status is None:
        return APIStatusError
    if status >= 500:
        return InternalServerError
    return _STATUS_CLASSES.get(status, APIStatusError)


def _nonempty(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


# pydantic wraps a validator's ValueError as "Value error, <text>". The server's
# own `message` quotes the text without it, so it is dropped before comparing.
_PYDANTIC_VALUE_ERROR = "Value error, "


def _render_detail(detail: list[object]) -> str | None:
    """Render a validation ``detail`` list as ``"<loc>: <msg>"`` entries."""
    parts: list[str] = []
    for item in detail:
        entry = as_mapping(item)
        if entry is None:
            text = _nonempty(item)
            if text:
                parts.append(text)
            continue
        msg = _nonempty(entry.get("msg"))
        if msg and msg.startswith(_PYDANTIC_VALUE_ERROR):
            msg = _nonempty(msg[len(_PYDANTIC_VALUE_ERROR) :])
        loc = entry.get("loc")
        path = [str(p) for p in cast("list[object]", loc)] if isinstance(loc, list) else []
        if path and path[0] == "body":
            path = path[1:]
        if msg:
            parts.append(f"{'.'.join(path)}: {msg}" if path else msg)
    return "; ".join(parts) or None


def _reason(status: int | None) -> str | None:
    if status is None:
        return None
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return None


def error_message(body: object, status: int | None, reason: str | None = None) -> str:
    """The message of BEHAVIOR.md §3.2.

    When the body has both a ``message`` and a ``detail`` list whose rendering
    the message does not already contain, the rendering is appended (after a
    space if the message ends a sentence, else after ": "), so a generic "The
    request is malformed or invalid." still says which field was wrong.
    """
    fields = as_mapping(body)
    if fields is not None:
        message = _nonempty(fields.get("message"))
        detail = fields.get("detail")
        detail_text = (
            _render_detail(cast("list[object]", detail)) if isinstance(detail, list) else None
        )
        if message:
            if detail_text and detail_text not in message:
                base = message.rstrip()  # trailing whitespace does not count
                joiner = " " if base.endswith((".", "!", "?")) else ": "
                return f"{base}{joiner}{detail_text}"
            return message
        if isinstance(detail, str) and detail.strip():
            return detail
        if detail_text:
            return detail_text
        error = _nonempty(fields.get("error"))
        if error:
            return error
    phrase = _nonempty(reason) or _reason(status)
    if phrase:
        return phrase
    return f"HTTP {status}" if status is not None else "The request failed."


def is_retryable(body: object, status: int | None) -> bool:
    """The rule of BEHAVIOR.md §3.3."""
    fields = as_mapping(body)
    flag = fields.get("retryable") if fields is not None else None
    if isinstance(flag, bool):
        return flag
    return status in RETRYABLE_STATUSES


def make_status_error(
    status: int | None,
    body: object,
    *,
    header_request_id: str | None = None,
    reason: str | None = None,
) -> APIStatusError:
    """Build the ``APIStatusError`` for an HTTP error or an in-band error event."""
    request_id = _nonempty(header_request_id)
    error_code = None
    fields = as_mapping(body)
    if fields is not None:
        if request_id is None:
            request_id = _nonempty(fields.get("request_id"))
        error_code = _nonempty(fields.get("error_code"))
    cls = status_error_class(status)
    return cls(
        error_message(body, status, reason),
        status=status,
        error_code=error_code,
        retryable=is_retryable(body, status),
        request_id=request_id,
        body=body,
    )


def make_interrupted_error(
    event: Mapping[str, object] | None,
    *,
    audio_bytes: int,
    header_request_id: str | None = None,
    default_message: str = "The audio stream ended before it was complete.",
) -> StreamInterruptedError:
    """Build the ``StreamInterruptedError`` for a failure after audio.

    ``event`` is the error event, or ``None`` when the stream simply stopped
    (then the status is ``None`` and the error is retryable).
    """
    if event is not None:
        base = make_status_error(event_status(event), event, header_request_id=header_request_id)
        return StreamInterruptedError(
            base.message,
            audio_bytes=audio_bytes,
            status=base.status,
            error_code=base.error_code,
            retryable=base.retryable,
            request_id=base.request_id,
            body=event,
        )
    return StreamInterruptedError(
        default_message,
        audio_bytes=audio_bytes,
        status=None,
        retryable=True,
        request_id=_nonempty(header_request_id),
    )


def event_status(event: Mapping[str, object]) -> int | None:
    """The ``status_code`` of a stream or WebSocket event, if it is an int."""
    value = event.get("status_code")
    return value if isinstance(value, int) and not isinstance(value, bool) else None
