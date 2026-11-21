"""A Server-Sent Events decoder (BEHAVIOR.md §5.1).

It accepts LF and CRLF line ends, ignores comments and the ``id:`` and
``retry:`` fields, joins multi-line ``data:`` with ``\\n`` and dispatches an
event on a blank line. It has no line-length limit: one chunk's base64 can be
large.
"""

from __future__ import annotations

__all__ = ["SSEDecoder"]


class SSEDecoder:
    """Turns response bytes into ``(event, data)`` pairs."""

    __slots__ = ("_buf", "_data", "_event", "_scanned")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._scanned = 0  # bytes of _buf known to hold no line end
        self._data: list[str] = []
        self._event = ""

    def feed(self, chunk: bytes) -> list[tuple[str, str]]:
        """Add bytes; return the events they completed."""
        buf = self._buf
        buf += chunk
        events: list[tuple[str, str]] = []
        pos = 0
        search = self._scanned
        while True:
            nl = buf.find(b"\n", search)
            if nl < 0:
                break
            end = nl - 1 if nl > pos and buf[nl - 1] == 0x0D else nl
            event = self._line(bytes(buf[pos:end]))
            if event is not None:
                events.append(event)
            pos = search = nl + 1
        if pos:
            del buf[:pos]
        self._scanned = len(buf)
        return events

    def _line(self, line: bytes) -> tuple[str, str] | None:
        if not line:
            if not self._data:
                self._event = ""
                return None
            event = (self._event, "\n".join(self._data))
            self._data = []
            self._event = ""
            return event
        if line[0] == 0x3A:  # ":" starts a comment
            return None
        colon = line.find(b":")
        if colon < 0:
            name, value = line, b""
        else:
            name, value = line[:colon], line[colon + 1 :]
            if value[:1] == b" ":
                value = value[1:]
        if name == b"data":
            self._data.append(value.decode("utf-8", errors="replace"))
        elif name == b"event":
            self._event = value.decode("utf-8", errors="replace")
        return None
