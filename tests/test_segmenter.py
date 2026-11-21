"""The sentence segmenter: the shared vectors, the streaming invariant, and a
differential fuzz against the reference implementation."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
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
    # Never write bytecode into conformance/: it must match its manifest.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
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


def test_long_words_and_hard_cuts_match_the_reference() -> None:
    """Words longer than any rule word, ending in one (``xyzdr.``, ``abcetc.``),
    cut anywhere by small max_chars: the capped look-behind must not change a
    single split."""
    reference = _reference()
    parts = ["dr", "etc", "no", "a", "1", "22", "prof", "approx", "x" * 7, "क्ष" * 4, "डॉ"]
    gaps = ["", ".", ". ", " ", "\n", ". A", ".\n", ". " + " " * 12, ".\n\t \t  ", "\t" * 9]
    rng = random.Random(7)
    for _ in range(1500):
        words = []
        for _ in range(rng.randint(1, 12)):
            word = "".join(rng.choice(parts) for _ in range(rng.randint(1, 4)))
            words.append(word + rng.choice(gaps))
        text = rng.choice([" ", ""]).join(words)
        options: dict[str, Any] = {
            "min_chars": rng.choice([0, 1, 20]),
            "max_chars": rng.choice([3, 5, 9, 17, 400]),
        }
        expected = reference.split_sentences(text, reference.Options(**options))
        assert split_sentences(text, **options) == expected, (text, options)
        assert _stream(text, list(text), options) == expected, (text, options)
        assert _stream(text, _random_cuts(text, rng), options) == expected, (text, options)


@pytest.mark.parametrize(
    "text",
    [
        ("यह एक वाक्य है। आपका पैकेज आज आएगा! " * 6000)[:200_000],
        ("शब्द बिना विराम के " * 12000)[:200_000],
        "क" * 200_000,
    ],
    ids=["punctuated", "unpunctuated", "no-spaces"],
)
def test_segmentation_is_linear(text: str) -> None:
    """200,000 characters, with or without punctuation, well under a second;
    streamed in LLM-sized pieces, still linear."""
    started = time.perf_counter()
    whole = split_sentences(text)
    assert time.perf_counter() - started < 1.0
    started = time.perf_counter()
    assert _stream(text, [text[i : i + 6] for i in range(0, len(text), 6)], {}) == whole
    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize(
    "text",
    [
        "This is the end." + " " * 10 + "1. x and then more words.",
        "This is the end." + " \t" * 40 + "12. x and then more words.",
        "This is the end.\n" + " " * 10 + "1. x and then more words.",
        "Item" + " " * 30 + "3. x is next.",
    ],
    ids=["spaces", "spaces-and-tabs", "newline-then-spaces", "word-then-spaces"],
)
@pytest.mark.parametrize("min_chars", [0, 20])
def test_space_run_before_a_list_marker(text: str, min_chars: int) -> None:
    """Whether "1." starts a line depends on the character before a run of
    spaces, however long: trimming the buffer inside the run must not turn a
    boundary into a list marker (or the reverse)."""
    reference = _reference()
    options: dict[str, Any] = {"min_chars": min_chars}
    expected = reference.split_sentences(text, reference.Options(**options))
    assert split_sentences(text, **options) == expected
    assert _stream(text, list(text), options) == expected
    for cut in range(1, len(text)):
        assert _stream(text, [text[:cut], text[cut:]], options) == expected, cut
