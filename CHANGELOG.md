# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/), and entries after 1.0.0rc1 are
written by release-please from the commit history.

## [1.0.0-rc.3](https://github.com/raya-voice-ai/bakbak-python/compare/v1.0.0-rc.2...v1.0.0-rc.3) (2026-11-22)


### Bug Fixes

* keep the WebSocket's internal callbacks off its public surface ([b62de6d](https://github.com/raya-voice-ai/bakbak-python/commit/b62de6d60b2f47b4324d704bb03bd2268cf37495))
* read a headerless WAV's encoding and sample rate from its fmt chunk ([b62de6d](https://github.com/raya-voice-ai/bakbak-python/commit/b62de6d60b2f47b4324d704bb03bd2268cf37495))
* say plainly that MP3 audio can't be wrapped in WAV ([b62de6d](https://github.com/raya-voice-ai/bakbak-python/commit/b62de6d60b2f47b4324d704bb03bd2268cf37495))
* stop repeating a validation error's text in the 422 message ([b62de6d](https://github.com/raya-voice-ai/bakbak-python/commit/b62de6d60b2f47b4324d704bb03bd2268cf37495))


### Documentation

* fix the README's PyPI links, the m1 speed example and the WebSocket example ([b62de6d](https://github.com/raya-voice-ai/bakbak-python/commit/b62de6d60b2f47b4324d704bb03bd2268cf37495))

## [1.0.0-rc.2](https://github.com/raya-voice-ai/bakbak-python/compare/v1.0.0-rc.1...v1.0.0-rc.2) (2026-11-21)


### Features

* add Bakbak, AsyncBakbak and the WebSocket client ([286bd3b](https://github.com/raya-voice-ai/bakbak-python/commit/286bd3b651aa0ca0cbeac993c6797ed9f048e4c1))
* add the HTTP core, errors and audio helpers ([22c25c1](https://github.com/raya-voice-ai/bakbak-python/commit/22c25c1f8e62345983e12feec6b5ce74175d9800))
* add the sentence segmenter ([7d68a3f](https://github.com/raya-voice-ai/bakbak-python/commit/7d68a3f66990381adf12269a3be3039c93cb9723))


### Bug Fixes

* give every WebSocket request an outcome, and keep the protocol consistent ([26c580e](https://github.com/raya-voice-ai/bakbak-python/commit/26c580e7156fad92184d09aca2bb145f348b6144))
* give waiting WebSocket requests an outcome on any connect failure ([6caef8a](https://github.com/raya-voice-ai/bakbak-python/commit/6caef8a4c8485baf887ca3a67b4f14cf441edefa))
* ignore trailing whitespace when appending a detail list to a message ([f1ef504](https://github.com/raya-voice-ai/bakbak-python/commit/f1ef504ea102470b98e8aa232af78751e30bf996))
* keep exactly the reference's segmenter look-behind between pushes ([4a2e6b5](https://github.com/raya-voice-ai/bakbak-python/commit/4a2e6b5080d16798938bb80b6f064680687f1843))
* keep the API key and text out of the websockets library's debug log ([7bf5791](https://github.com/raya-voice-ai/bakbak-python/commit/7bf579184c4535ed99ae9e45fd9d2240124e67ca))
* keep the version in the form release-please writes ([b71fde0](https://github.com/raya-voice-ai/bakbak-python/commit/b71fde0db4d6f8014bdac0787e3a848ab0ade2ec))


### Performance Improvements

* make sentence segmentation linear in the text's length ([43d5dec](https://github.com/raya-voice-ai/bakbak-python/commit/43d5dec2824fd28e1b8dcc585f80a43f26035bc8))


### Documentation

* add runnable examples and a TTFA benchmark ([c61bf03](https://github.com/raya-voice-ai/bakbak-python/commit/c61bf0373622cf0bd8a4a281f1fcee2eb335314c))
* add the README, changelog, contributing and security guides ([e426eb2](https://github.com/raya-voice-ai/bakbak-python/commit/e426eb2cb8bda47847f8d1109a4b6db38a43cf48))
* link the documentation site and describe the new guarantees ([e569f8a](https://github.com/raya-voice-ai/bakbak-python/commit/e569f8a0901188b2c9e74bdc4d1d955851a84eef))

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
