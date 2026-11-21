"""The sentence segmenter: the shared vectors, the streaming invariant, and a
differential fuzz against the reference implementation."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from types import ModuleType
from typing import Any

import pytest
from conformance_support import CONFORMANCE

from bakbak.text import MAX_CHARS_LIMIT, Segmenter, split_sentences

CASES: list[dict[str, Any]] = json.loads(
    (CONFORMANCE / "segmenter" / "cases.json").read_text("utf-8")
)["cases"]


def _reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bakbak_segmenter_reference", CONFORMANCE / "segmenter" / "reference.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _stream(text: str, pieces: list[str], options: dict[str, Any]) -> list[str]:
    assert "".join(pieces) == text
    segmenter = Segmenter(**options)
    out: list[str] = []
    for piece in pieces:
        out += segmenter.push(piece)
    return out + segmenter.flush()


def _random_cuts(text: str, rng: random.Random) -> list[str]:
    pieces, i = [], 0
    while i < len(text):
        n = rng.randint(1, 9)
        pieces.append(text[i : i + n])
        i += n
    return pieces


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_vector(case: dict[str, Any]) -> None:
    assert split_sentences(case["input"], **case["options"]) == case["sentences"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_streaming_invariant(case: dict[str, Any]) -> None:
    text, options, expected = case["input"], case["options"], case["sentences"]
    assert _stream(text, [text] if text else [], options) == expected
    # One code point per piece (Python strings are sequences of code points).
    assert _stream(text, list(text), options) == expected
    rng = random.Random(case["name"])
    for _ in range(20):
        assert _stream(text, _random_cuts(text, rng), options) == expected


def test_differential_fuzz_against_reference() -> None:
    """Batch and streamed output equal the reference's on random texts built
    from the characters the rules care about."""
    reference = _reference()
    alphabet = [
        *"ab cd. Dr. etc. 3.14 ! ? … । ॥ \n\t,;:—\"') कष््‍‌ A. no. Inc. 1. 22. Mr. 🙏",
        "\n1. ", "\n 12. ", " e.g. ", "क्ष", "डॉ. ", "x.y", "Rs. ", "\r\n",
    ]  # fmt: skip
    rng = random.Random(20261121)
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        options: dict[str, Any] = {
            "min_chars": rng.choice([0, 1, 2, 5, 20]),
            "first_min_chars": rng.choice([0, 1, 5, 20]),
            "max_chars": rng.choice([1, 2, 3, 7, 10, 25, 60, 400]),
            "eager_first_clause": rng.random() < 0.3,
            "first_clause_min_chars": rng.choice([0, 3, 10, 24]),
        }
        expected = reference.split_sentences(text, reference.Options(**options))
        assert split_sentences(text, **options) == expected, (text, options)
        assert _stream(text, _random_cuts(text, rng), options) == expected, (text, options)
        assert _stream(text, list(text), options) == expected, (text, options)


def test_lengths_are_code_points() -> None:
    # An astral character is one code point (two UTF-16 units): five of them
    # fit a max_chars of 5 exactly.
    bold = "".join(chr(0x1D400 + i) for i in range(5))  # MATHEMATICAL BOLD A to E
    text = f"{bold} {bold}"
    assert split_sentences(text, max_chars=5) == [bold, bold]


def test_flush_resets_for_the_next_text() -> None:
    segmenter = Segmenter()
    assert segmenter.push("Hello there. How") == ["Hello there."]
    assert segmenter.flush() == ["How"]
    # After a flush the first-sentence minimum applies again.
    assert segmenter.push("Hi. ") == []
    assert segmenter.push("Next") == ["Hi."]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"max_chars": MAX_CHARS_LIMIT + 1}, ValueError),
        ({"max_chars": 0}, ValueError),
        ({"min_chars": -1}, ValueError),
        ({"first_clause_min_chars": -1}, ValueError),
        ({"max_chars": 10.5}, TypeError),
    ],
)
def test_options_are_checked(options: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        Segmenter(**options)


def test_push_takes_text_only() -> None:
    with pytest.raises(TypeError):
        Segmenter().push(b"bytes")  # type: ignore[arg-type]
