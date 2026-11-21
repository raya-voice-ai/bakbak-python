"""Error classes, messages (§3.2), retryability (§3.3) and Retry-After (§4.3)."""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

import bakbak
from bakbak._exceptions import error_message, make_status_error, status_error_class
from bakbak._retry import backoff_delay, parse_retry_after


@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (400, bakbak.BadRequestError),
        (401, bakbak.AuthenticationError),
        (402, bakbak.InsufficientCreditsError),
        (403, bakbak.PermissionDeniedError),
        (404, bakbak.NotFoundError),
        (413, bakbak.RequestTooLargeError),
        (422, bakbak.UnprocessableEntityError),
        (429, bakbak.RateLimitError),
        (500, bakbak.InternalServerError),
        (503, bakbak.InternalServerError),
        (599, bakbak.InternalServerError),
        (409, bakbak.APIStatusError),
        (None, bakbak.APIStatusError),
    ],
)
def test_status_classes(status: int | None, cls: type[bakbak.APIStatusError]) -> None:
    assert status_error_class(status) is cls


def test_hierarchy() -> None:
    assert issubclass(bakbak.APIStatusError, bakbak.APIError)
    assert issubclass(bakbak.StreamInterruptedError, bakbak.APIError)
    assert not issubclass(bakbak.StreamInterruptedError, bakbak.APIStatusError)
    assert issubclass(bakbak.APIError, bakbak.BakbakError)
    assert issubclass(bakbak.APITimeoutError, bakbak.APIConnectionError)
    assert issubclass(bakbak.WebSocketClosedError, bakbak.APIConnectionError)
    assert issubclass(bakbak.APIConnectionError, bakbak.BakbakError)
    assert not issubclass(bakbak.APIConnectionError, bakbak.APIError)


DETAIL = [
    {"type": "missing", "loc": ["body", "voice_id"], "msg": "Field required"},
    {"loc": ["body", "output_format", "encoding"], "msg": "Value error, bad"},
    {"loc": ["body"], "msg": "Body problem"},
    {"msg": "No location"},
    "plain entry",
    {"loc": ["body", "x"]},
    {"loc": ["body", "items", 0], "msg": "Indexed"},
]
RENDERED = (
    "voice_id: Field required; output_format.encoding: Value error, bad; Body problem; "
    "No location; plain entry; items.0: Indexed"
)


@pytest.mark.parametrize(
    ("body", "status", "reason", "expected"),
    [
        ({"message": "Shown.", "detail": "hidden"}, 404, None, "Shown."),
        ({"detail": "A sentence."}, 400, None, "A sentence."),
        ({"detail": DETAIL}, 422, None, RENDERED),
        ({"error": "Insufficient credits"}, 402, None, "Insufficient credits"),
        ({"message": "  ", "detail": "  ", "error": ""}, 502, "Bad Gateway", "Bad Gateway"),
        (None, 503, None, "Service Unavailable"),
        ("<html>", 599, None, "HTTP 599"),
        # The detail list is appended to a message that does not contain it:
        (
            {"message": "Invalid.", "detail": DETAIL[:1]},
            422,
            None,
            "Invalid. voice_id: Field required",
        ),
        ({"message": "Why?", "detail": DETAIL[:1]}, 422, None, "Why? voice_id: Field required"),
        ({"message": "Stop!", "detail": DETAIL[:1]}, 422, None, "Stop! voice_id: Field required"),
        (
            {"message": "Invalid", "detail": DETAIL[:1]},
            422,
            None,
            "Invalid: voice_id: Field required",
        ),
        (
            {"message": "Invalid.  ", "detail": DETAIL[:1]},
            422,
            None,
            "Invalid. voice_id: Field required",
        ),
        (
            {"message": "Invalid \n", "detail": DETAIL[:1]},
            422,
            None,
            "Invalid: voice_id: Field required",
        ),
        (
            {"message": "Invalid request: voice_id: Field required.", "detail": DETAIL[:1]},
            422,
            None,
            "Invalid request: voice_id: Field required.",
        ),
        ({"message": "Shown", "detail": [{"loc": ["body", "x"]}]}, 422, None, "Shown"),
    ],
)
def test_message(body: object, status: int, reason: str | None, expected: str) -> None:
    assert error_message(body, status, reason) == expected


@pytest.mark.parametrize(
    ("body", "status", "retryable"),
    [
        ({}, 408, True),
        ({}, 429, True),
        ({}, 500, True),
        ({}, 502, True),
        ({}, 503, True),
        ({}, 504, True),
        ({}, 501, False),
        ({}, 400, False),
        ({"retryable": False}, 503, False),
        ({"retryable": True}, 400, True),
        ({"retryable": "yes"}, 400, False),
    ],
)
def test_retryable(body: dict[str, object], status: int, retryable: bool) -> None:
    assert make_status_error(status, body).retryable is retryable


def test_request_id_prefers_the_header() -> None:
    body = {"request_id": "from-body"}
    assert make_status_error(500, body, header_request_id="from-header").request_id == "from-header"
    assert make_status_error(500, body).request_id == "from-body"
    assert make_status_error(500, {}).request_id is None


def test_str_repr_and_pickle() -> None:
    error = make_status_error(
        402,
        {"message": "No balance.", "error_code": "insufficient_credits", "required_inr": 1.5},
        header_request_id="req_1",
    )
    assert (
        str(error) == "No balance. (status 402, error_code insufficient_credits, request_id req_1)"
    )
    assert repr(error).startswith("InsufficientCreditsError(status=402")
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is bakbak.InsufficientCreditsError
    assert copy.message == error.message
    assert copy.body == error.body
    interrupted = bakbak.StreamInterruptedError("Broke.", audio_bytes=10, request_id="r")
    restored = pickle.loads(pickle.dumps(interrupted))
    assert restored.audio_bytes == 10
    assert "audio_bytes=10" in repr(restored)
    closed = bakbak.WebSocketClosedError("Closed.", code=1008, reason="policy")
    assert pickle.loads(pickle.dumps(closed)).code == 1008


def test_retry_after_seconds_and_dates() -> None:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("1") == 1.0
    assert parse_retry_after(" 2.5 ") == 2.5
    assert parse_retry_after("120") == 60.0  # capped
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after(format_datetime(now + timedelta(seconds=7), usegmt=True), now) == 7.0
    assert parse_retry_after(format_datetime(now + timedelta(hours=1), usegmt=True), now) == 60.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=5), usegmt=True), now) == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("nan") is None


@pytest.mark.parametrize(
    ("retry", "low", "high"),
    [(1, 0.375, 0.5), (2, 0.75, 1.0), (3, 1.5, 2.0), (5, 6.0, 8.0), (9, 6.0, 8.0)],
)
def test_backoff_ranges(retry: int, low: float, high: float) -> None:
    assert backoff_delay(retry, lambda: 0.0) == high
    assert backoff_delay(retry, lambda: 0.999999) == pytest.approx(low, abs=1e-5)
    assert low <= backoff_delay(retry) <= high
