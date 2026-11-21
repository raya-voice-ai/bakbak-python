# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/), and entries after 1.0.0rc1 are
written by release-please from the commit history.

## 1.0.0rc1 (unreleased)

The first release candidate of the official Python SDK.

### Features

- `Bakbak` and `AsyncBakbak` clients with identical surfaces: `tts.generate`,
  `tts.stream` (Server-Sent Events), `tts.websocket` and `voices.list`, plus
  `with_raw_response` and `with_options()`.
- WebSocket client with contexts, request handles, cancel, `send_stream` for LLM
  token streams (OpenAI-, Anthropic- and Gemini-style events by duck typing),
  back-pressure, and automatic resends across server restarts and dropped
  connections.
- Safe retries with Retry-After and jittered backoff, never after audio has been
  delivered; the error hierarchy of the shared client specification.
- `bakbak.text`: the sentence segmenter shared by every official Bakbak client.
- `bakbak.audio.to_wav` for raw PCM, mu-law and A-law.
- Conformance with the shared client specification (`conformance/`), tested
  against local HTTP and WebSocket servers.
