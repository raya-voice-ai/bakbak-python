"""The WebSocket client: many streaming requests on one connection.

Get one from ``client.tts.websocket()``; see BEHAVIOR.md §6 for the protocol.
"""

from ._ws._async import AsyncContext, AsyncRequestHandle, AsyncTTSWebSocket
from ._ws._sync import Context, RequestHandle, TTSWebSocket
from ._ws.events import Cancelled, Chunk, Done, Failed, RequestResult, WebSocketEvent

__all__ = [
    "AsyncContext",
    "AsyncRequestHandle",
    "AsyncTTSWebSocket",
    "Cancelled",
    "Chunk",
    "Context",
    "Done",
    "Failed",
    "RequestHandle",
    "RequestResult",
    "TTSWebSocket",
    "WebSocketEvent",
]
