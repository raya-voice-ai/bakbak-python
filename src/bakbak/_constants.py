"""Defaults and limits shared by the HTTP and WebSocket clients."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://hub.getraya.app"
API_KEY_ENV = "BAKBAK_API_KEY"
BASE_URL_ENV = "BAKBAK_BASE_URL"

PACKAGE_NAME = "bakbak-python"

TTS_PATH = "/v1/text-to-speech"
STREAM_PATH = "/v1/text-to-speech/stream"
WEBSOCKET_PATH = "/v1/text-to-speech/ws"
VOICES_PATH = "/v1/voices"

DEFAULT_MAX_RETRIES = 2
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 60.0

# Backoff (BEHAVIOR.md §4.3): retry n waits min(0.5 * 2^(n-1), 8) * (1 - 0.25 * r).
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
RETRY_JITTER = 0.25
MAX_RETRY_AFTER = 60.0

# Statuses that are worth retrying when the body does not say otherwise (§3.3).
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Idle HTTP connections are kept this long, so a request a few seconds after the
# previous one (or after prewarm()) skips the TCP and TLS handshakes.
KEEPALIVE_EXPIRY = 30.0

# WebSocket limits of one connection (§1). The client never exceeds them.
WS_MAX_UNFINISHED = 64
WS_MAX_CONTEXTS = 16
# Large enough for any server message; one chunk's base64 can be big.
WS_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
# How long the client waits for its own close handshake to complete.
WS_CLOSE_TIMEOUT = 2.0
# A connection that stayed open this long counts as healthy even if it never
# received a message, so dropping it does not count as a failed reconnect.
WS_HEALTHY_AFTER = 10.0
