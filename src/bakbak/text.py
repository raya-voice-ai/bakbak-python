"""Sentence segmentation for streaming text into speech (BEHAVIOR.md §7).

The models voice whole sentences best, and an LLM sends its reply a few
characters at a time. :class:`Segmenter` turns that stream into sentence-sized
requests as early as it safely can; :func:`split_sentences` does the same for a
whole text. Both follow ``conformance/segmenter/reference.py`` exactly, so the
Python, JavaScript and LiveKit clients split every text the same way.

The guiding rule is "when in doubt, don't split": a missed boundary only makes
one request a little longer, while a false split puts a falling intonation in
the middle of a sentence. ``3.14``, ``getraya.app``, ``Dr. Rao`` and
``है।राहुल`` never split.

Lengths are counted in Unicode code points.
"""

from __future__ import annotations

import unicodedata
from typing import NamedTuple

__all__ = ["MAX_CHARS_LIMIT", "Segmenter", "split_sentences"]

MAX_CHARS_LIMIT = 5000
"""The API's text limit; ``max_chars`` can never exceed it."""


def _cps(*codepoints: int) -> frozenset[str]:
    return frozenset(chr(c) for c in codepoints)


# Every character class is listed by code point, as in the reference.
_WHITESPACE = _cps(
    0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B, 0x85, 0xA0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
    0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
)  # fmt: skip
# . ! ? … । ॥ 。 ！ ？  # noqa: RUF003
_TERMINATORS = _cps(0x2E, 0x21, 0x3F, 0x2026, 0x0964, 0x0965, 0x3002, 0xFF01, 0xFF1F)
# " ' ) ] } ’ ” » › 》 」 』  # noqa: RUF003
_CLOSERS = _cps(0x22, 0x27, 0x29, 0x5D, 0x7D, 0x2019, 0x201D, 0xBB, 0x203A, 0x300B, 0x300D, 0x300F)
# , ; : — – ، (Arabic comma, used in Urdu)  # noqa: RUF003
_SOFT_DELIMITERS = _cps(0x2C, 0x3B, 0x3A, 0x2014, 0x2013, 0x060C)
# ZWNJ, ZWJ
_JOINERS = _cps(0x200C, 0x200D)
# The viramas of Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
# Kannada, Malayalam and Sinhala.
_VIRAMAS = _cps(0x094D, 0x09CD, 0x0A4D, 0x0ACD, 0x0B4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D4D, 0x0DCA)

_TITLES = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "rs", "smt",
        "shri", "sh", "lt", "col", "gen", "capt", "sgt", "hon", "rev", "mt",
        # Devanagari: Dr. (डॉ), Prof. (प्रो), Kumari (कु), late (स्व), Mr. (मि), Rs. (रु)
        "".join(map(chr, (0x0921, 0x0949))),
        "".join(map(chr, (0x092A, 0x094D, 0x0930, 0x094B))),
        "".join(map(chr, (0x0915, 0x0941))),
        "".join(map(chr, (0x0938, 0x094D, 0x0935))),
        "".join(map(chr, (0x092E, 0x093F))),
        "".join(map(chr, (0x0930, 0x0941))),
    }
)  # fmt: skip
_AMBIGUOUS = frozenset(
    {
        "etc", "inc", "ltd", "co", "corp", "approx", "dept", "govt", "est",
        "fig", "no", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec",
    }
)  # fmt: skip


def _speakable(s: str) -> bool:
    """True if ``s`` has a letter or a number (Unicode categories L* and N*)."""
    return any(unicodedata.category(ch)[0] in "LN" for ch in s)


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "M"


def _is_word_char(ch: str) -> bool:
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"):
        return True
    cp = ord(ch)
    return 0x0900 <= cp <= 0x0DFF and cp not in (0x0964, 0x0965)


def _word_before(text: str, dot: int) -> tuple[str, int]:
    """The word ending right before index ``dot``, and where it starts."""
    i = dot
    while i > 0 and _is_word_char(text[i - 1]):
        i -= 1
    return text[i:dot], i


def _at_line_start(text: str, start: int) -> bool:
    i = start
    while i > 0 and text[i - 1] in (" ", "\t"):
        i -= 1
    return i == 0 or text[i - 1] == "\n"


def _is_ascii_digits(word: str) -> bool:
    return 1 <= len(word) <= 3 and all("0" <= c <= "9" for c in word)


def _single_latin_letter(word: str) -> bool:
    return len(word) == 1 and (("a" <= word <= "z") or ("A" <= word <= "Z"))


def _strip(s: str) -> str:
    a, b = 0, len(s)
    while a < b and s[a] in _WHITESPACE:
        a += 1
    while b > a and s[b - 1] in _WHITESPACE:
        b -= 1
    return s[a:b]


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WHITESPACE:
        i += 1
    return i


class _Candidate(NamedTuple):
    end: int  # where the sentence ends (exclusive)
    decided: bool
    boundary: bool  # meaningful only when decided


def _candidates(
    text: str, start: int, final: bool, clause: tuple[int, int] | None
) -> list[_Candidate]:
    """Boundary candidates in ``text[start:]``, in order.

    ``clause``, when given, is ``(group_start, first_clause_min_chars)``: soft
    delimiters followed by whitespace also count, if the stripped text from the
    group start through the delimiter is long enough.
    """
    out: list[_Candidate] = []
    n = len(text)
    i = start
    while i < n:
        ch = text[i]
        if ch == "\n":
            j = i
            while j < n and text[j] in _WHITESPACE:
                j += 1
            out.append(_Candidate(i, final or j < n, True))
            i = j
            continue
        is_clause = (
            clause is not None
            and ch in _SOFT_DELIMITERS
            and len(_strip(text[clause[0] : i + 1])) >= clause[1]
        )
        if ch not in _TERMINATORS and not is_clause:
            i += 1
            continue
        run_start = i
        if is_clause:
            j = i + 1
        else:
            j = i
            while j < n and text[j] in _TERMINATORS:
                j += 1
        run_end = j
        while j < n and text[j] in _CLOSERS:
            j += 1
        end = j  # the sentence ends after the closers
        if j >= n:
            # Nothing after the run yet: undecided unless final (and the end of
            # the text is a boundary anyway).
            if not final:
                out.append(_Candidate(end, False, False))
            break
        if text[j] not in _WHITESPACE:
            i = j  # "3.14", "a.com", "है।राहुल": not a candidate
            continue
        k = j
        while k < n and text[k] in _WHITESPACE:
            k += 1
        if k >= n and not final:
            out.append(_Candidate(end, False, False))
            break
        next_char = text[k] if k < n else ""
        boundary = True
        if not is_clause and run_end - run_start == 1 and text[run_start] == ".":
            word, word_start = _word_before(text, run_start)
            lowered = word.lower()
            if (
                _single_latin_letter(word)
                or lowered in _TITLES
                or (_is_ascii_digits(word) and _at_line_start(text, word_start))
                or (lowered in _AMBIGUOUS and not ("A" <= next_char <= "Z"))
            ):
                boundary = False
        out.append(_Candidate(end, True, boundary))
        i = k
    return out


def _wrap_cut(text: str, start: int, max_chars: int) -> int:
    """Where to cut the window ``text[start:start + max_chars]``; > ``start``."""
    window_end = start + max_chars
    half = start + max_chars // 2
    for j in range(window_end, half, -1):
        if text[j - 1] in _SOFT_DELIMITERS:
            return j
    for j in range(window_end, start, -1):
        # Cut before the whitespace, and never produce an empty piece.
        if text[j - 1] in _WHITESPACE and _strip(text[start : j - 1]):
            return j - 1
    # Hard cut. Never before a combining mark or a joiner, or right after a
    # virama or a joiner, which would split an Indic conjunct or a cluster.
    cut = window_end
    while cut > start + 1 and (
        _is_mark(text[cut])
        or text[cut] in _JOINERS
        or text[cut - 1] in _VIRAMAS
        or text[cut - 1] in _JOINERS
    ):
        cut -= 1
    return cut if cut > start + 1 else window_end


def _keep_from(text: str, gs: int) -> int:
    """Where the buffer can be trimmed when the pending text starts at ``gs``.

    The boundary rules look behind a full stop: at the word before it, and at
    whether that word starts a line. The reference trims the buffer at ``gs``,
    which loses that context and breaks the streaming invariant in rare cases
    (a list marker such as "1." right after an emitted sentence with a small
    ``min_chars``, or a full stop right after a hard cut). Keeping the word
    before ``gs``, the spaces before it, and one more character makes each
    look-behind stop exactly where it stops in the whole text.
    """
    t = gs
    while t > 0 and _is_word_char(text[t - 1]):
        t -= 1
    while t > 0 and text[t - 1] in (" ", "\t"):
        t -= 1
    return t - 1 if t > 0 else 0


def _check_option(name: str, value: int, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{name} must be an int")
    if value < low or (high is not None and value > high):
        bound = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


class Segmenter:
    """Splits streamed text into sentences as soon as each one is final.

    ``push()`` returns the sentences the new text completed; ``flush()``
    returns the rest at the end of the input and resets the segmenter. Any way
    of cutting a text into pushed pieces gives the same sentences as
    :func:`split_sentences` on the whole text.

    Args:
        min_chars: sentences shorter than this are merged with the next one.
        first_min_chars: the same minimum for the first sentence. The default
            of 1 sends a short first sentence ("Sure.") at once.
        max_chars: longer runs are wrapped at a soft delimiter or a space. At
            most 5,000, the API's limit.
        eager_first_clause: until the first sentence is out, also split after
            a comma, semicolon, colon or dash, which starts audio sooner when
            the first sentence is long.
        first_clause_min_chars: the shortest first clause ``eager_first_clause``
            will send on its own.
    """

    __slots__ = (
        "_buf",
        "_first",
        "_start",
        "_wrapping",
        "eager_first_clause",
        "first_clause_min_chars",
        "first_min_chars",
        "max_chars",
        "min_chars",
    )

    def __init__(
        self,
        *,
        min_chars: int = 20,
        first_min_chars: int = 1,
        max_chars: int = 400,
        eager_first_clause: bool = False,
        first_clause_min_chars: int = 24,
    ) -> None:
        self.min_chars = _check_option("min_chars", min_chars, 0)
        self.first_min_chars = _check_option("first_min_chars", first_min_chars, 0)
        self.max_chars = _check_option("max_chars", max_chars, 1, MAX_CHARS_LIMIT)
        self.eager_first_clause = bool(eager_first_clause)
        self.first_clause_min_chars = _check_option(
            "first_clause_min_chars", first_clause_min_chars, 0
        )
        self._buf = ""
        self._start = 0  # where the pending text starts in _buf (see _keep_from)
        self._first = True
        self._wrapping = False  # inside a wrapped group: no merging until it ends

    def push(self, text: str) -> list[str]:
        """Add text; return the sentences that are now final."""
        if not isinstance(text, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"push() takes str, not {type(text).__name__}")
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """End the input: return the remaining sentences and reset."""
        out = self._drain(final=True)
        self._buf = ""
        self._start = 0
        self._first = True
        self._wrapping = False
        return out

    def _min(self) -> int:
        return self.first_min_chars if self._first else self.min_chars

    def _emit(self, out: list[str], piece: str) -> None:
        # A piece with nothing to say (only punctuation or symbols) is dropped:
        # the API refuses text that has no sentence in it.
        if piece and _speakable(piece):
            out.append(piece)
            self._first = False

    def _drain(self, final: bool) -> list[str]:
        max_chars = self.max_chars
        out: list[str] = []
        text = self._buf
        gs = _skip_ws(text, self._start)  # group start
        while True:
            clause = (
                (gs, self.first_clause_min_chars)
                if (self.eager_first_clause and self._first)
                else None
            )
            cands = _candidates(text, gs, final, clause)
            emitted = False
            for c in cands:
                if not c.decided:
                    break
                if not c.boundary:
                    continue
                group = _strip(text[gs : c.end])
                if len(group) > max_chars:
                    break  # handled by wrapping below
                if not _speakable(group):
                    continue  # punctuation only: nothing to say yet, extend
                if self._wrapping or len(group) >= self._min():
                    self._emit(out, group)
                    self._wrapping = False
                    gs = _skip_ws(text, c.end)
                    emitted = True
                    break
                # too short: extend the group to the next boundary
            if emitted:
                continue
            # No boundary closes a group of acceptable size. Wrap if the group
            # is certainly longer than max_chars.
            limit = gs + max_chars
            first_open = next((c.end for c in cands if not c.decided or c.boundary), None)
            certainly_long = (first_open is None or first_open > limit) and len(text) > limit
            if final and len(_strip(text[gs:])) > max_chars:
                certainly_long = True
            if certainly_long and len(_strip(text[gs:])) > max_chars:
                cut = _wrap_cut(text, gs, max_chars)
                self._emit(out, _strip(text[gs:cut]))
                self._wrapping = True
                gs = _skip_ws(text, cut)
                continue
            if final:
                self._emit(out, _strip(text[gs:]))
                gs = len(text)
            break
        keep = _keep_from(text, gs)
        self._buf = text[keep:]
        self._start = gs - keep
        return out


def split_sentences(
    text: str,
    *,
    min_chars: int = 20,
    first_min_chars: int = 1,
    max_chars: int = 400,
    eager_first_clause: bool = False,
    first_clause_min_chars: int = 24,
) -> list[str]:
    """Split a whole text into sentence-sized requests.

    The options are those of :class:`Segmenter`.

    >>> split_sentences("आपका पैकेज आज शाम तक पहुँच जाएगा। राहुल आपको कॉल करेगा।")
    ['आपका पैकेज आज शाम तक पहुँच जाएगा।', 'राहुल आपको कॉल करेगा।']
    """
    segmenter = Segmenter(
        min_chars=min_chars,
        first_min_chars=first_min_chars,
        max_chars=max_chars,
        eager_first_clause=eager_first_clause,
        first_clause_min_chars=first_clause_min_chars,
    )
    return segmenter.push(text) + segmenter.flush()
