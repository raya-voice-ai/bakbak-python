# Bakbak Python SDK

The official Python client for the **Bakbak text-to-speech API** by Raya: natural
speech in Hindi, English and other Indian languages, as whole files, as a stream,
or over a WebSocket built for voice agents.

- Sync (`Bakbak`) and asyncio (`AsyncBakbak`) clients with identical surfaces
- Whole files, Server-Sent Events streams, and a multiplexed WebSocket
- Sentence segmentation for LLM token streams, built in
- Automatic, safe retries: never after audio has been delivered
- Typed throughout (`py.typed`), with three runtime dependencies: `httpx`, `websockets`, `typing-extensions`

Documentation: [Python SDK guide](https://docs.litwizlabs.com/documentation/sdk-python) ·
[WebSocket API reference](https://docs.litwizlabs.com/api-reference/tts-websocket) ·
[Errors](https://docs.litwizlabs.com/documentation/tts-errors)

## Install

```sh
pip install --pre bakbak-python     # 1.0 is in release candidates; drop --pre after 1.0.0
```

The package is `bakbak-python` on PyPI; you import it as `bakbak`.

Python 3.10 or newer.

## Quickstart

```python
from bakbak import Bakbak

client = Bakbak()  # reads BAKBAK_API_KEY

audio = client.tts.generate(
    text="नमस्ते, आप कैसे हैं?",
    voice_id="009ae8b5-4a03-492d-af1c-3055989ac734",  # Anika M1
    language="hi",
    model="m1",
)
audio.save("hello.wav")
```

The same with asyncio:

```python
import asyncio
from bakbak import AsyncBakbak


async def main() -> None:
    async with AsyncBakbak() as client:
        audio = await client.tts.generate(text="नमस्ते!", voice_id="…", language="hi", model="m1")
        audio.save("hello.wav")


asyncio.run(main())
```

Every example in [`examples/`](https://github.com/raya-voice-ai/bakbak-python/tree/main/examples) runs as is with `BAKBAK_API_KEY` set.

## Configuration

```python
from bakbak import Bakbak, Timeout

client = Bakbak(
    api_key="…",  # default: BAKBAK_API_KEY
    base_url="https://hub.getraya.app",  # default: BAKBAK_BASE_URL, then this
    timeout=Timeout(connect=5, read=60, total=None),
    max_retries=2,
    default_headers={"X-Team": "voice"},
    http_client=None,  # your own httpx.Client, e.g. with a proxy
)
```

| Option | Default | Meaning |
|---|---|---|
| `api_key` | `BAKBAK_API_KEY` | Sent as `X-API-Key`. Environment variables are read when the client is constructed, never at import. |
| `base_url` | `BAKBAK_BASE_URL`, then `https://hub.getraya.app` | The WebSocket URL is derived from it (`https` → `wss`). |
| `timeout` | 5 s connect, 60 s read | A `Timeout(connect, read, total)`; a number, which sets the read timeout (and caps connect at it); or `None` for none. The read timeout includes the time to first audio. `total` limits a whole call, retries included. |
| `max_retries` | `2` | Retries of a failed request before any audio, so at most 3 attempts. |
| `default_headers` | none | Added to every request. |
| `http_client` | a pooled `httpx.Client` | Bring your own (`httpx.AsyncClient` for `AsyncBakbak`). The SDK does not close it. |

Every HTTP method also takes `timeout`, `max_retries`, `extra_headers` and (for
synthesis) `extra_body`. `client.with_options(...)` returns a copy with some
options changed that shares the connection pool:

```python
patient = client.with_options(timeout=120, max_retries=4)
```

## Whole files

```python
audio = client.tts.generate(
    text="आपका ऑर्डर कल पहुँचेगा।",
    voice_id=voice_id,  # a standard voice
    language="hi",
    model="standard",
    speed=1.1,  # 0.5 to 1.5; standard voices only
    output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
)
audio.data  # bytes, exactly as received
audio.request_id  # quote it to support
audio.content_type  # audio/wav, audio/mpeg, audio/pcm, audio/basic, application/octet-stream
audio.encoding, audio.sample_rate  # e.g. "pcm_s16le", 24000; for mp3, "mp3", None
audio.save("out.wav")
audio.to_wav()  # WAV bytes; raw PCM, mu-law and A-law get a header
```

`speed` changes the speaking rate of `standard` voices. `m1` voices accept it
and speak at their own pace.

`audio.encoding` and `audio.sample_rate` come from the `X-Audio-Format` and
`X-Audio-Sample-Rate` headers. Without `output_format` the server sends neither,
and the SDK reads them from the WAV file's own header. An mp3 states no sample
rate (it is in the MP3 frames), and `to_wav()` raises `ValueError` for it: decode
it with an audio library, or ask for `{"container": "wav", ...}` instead.

`output_format`:

| Field | Values | Notes |
|---|---|---|
| `container` | `raw`, `wav`, `mp3` | Streams (SSE and WebSocket) accept only `raw`. |
| `encoding` | `pcm_f32le`, `pcm_s16le`, `pcm_mulaw`, `pcm_alaw` | Required for `raw` and `wav`; leave it out for `mp3`. Mono. |
| `sample_rate` | `8000`, `16000`, `22050`, `24000` | Default 24000. |

Without `output_format`, a whole file is 24 kHz WAV and a stream is 24 kHz
32-bit float PCM. Languages, models and encodings are typed as the known values
plus any string, so a value the server adds later needs no SDK release.

## Streaming (SSE)

```python
with client.tts.stream(
    text="आपका पैकेज आज शाम तक पहुँच जाएगा।",
    voice_id=voice_id,
    language="hi",
    model="m1",
    output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
) as stream:
    for chunk in stream:  # AudioChunk(audio, step_time, request_id)
        player.write(chunk.audio)
print(stream.request_id, stream.audio_bytes)
```

The request is sent when the stream is entered or first iterated. A chunk always
holds whole samples: if the server splits a sample across two events, the SDK
carries the remainder over. A stream also has `iter_audio()` (bytes only),
`read()` (all of it) and `save(path)`. It can be read once. On `AsyncBakbak` it is
`async with` / `async for`, and `read()`, `save()` and `close()` are awaitable.

## WebSocket: many requests, one connection

The WebSocket carries any number of streaming requests over one connection, and
saves the per-request connection and gateway cost. It is the fastest path to
first audio and the one to use for voice agents.

```python
from bakbak import AsyncBakbak

async with AsyncBakbak() as client:  # in an async function
    async with client.tts.websocket(
        voice_id=voice_id,
        language="hi",
        model="m1",
        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    ) as ws:  # opens the socket now (prewarm)
        ctx = ws.context()  # requests on a context play in order
        await ctx.send(text="पहला वाक्य।")
        await ctx.send(text="दूसरा वाक्य।")
        ctx.end_input()
        async for audio in ctx.audio():  # bytes, in order; raises the first failure
            player.write(audio)
```

- **Contexts.** `ws.context(context_id=None, **defaults)` returns a context; the
  id defaults to a random `ctx_…`. Requests on one context run one after another;
  different contexts run at the same time. Fields left out of `send()` come from
  the context's defaults, then the WebSocket's.
- **Events.** `ctx.events()` yields `Chunk(audio, request)`, `Done(request)`,
  `Failed(request, error)` and `Cancelled(request)` in order, and ends after
  `end_input()` once every request has an outcome. An error ends its own request
  only. `ctx.audio()` yields just the bytes.
- **Request handles.** `send()` returns a handle: `await handle` gives
  `RequestResult(request_id, audio_bytes, cancelled)` or raises the request's
  error, and `async for audio in handle` yields that request's audio.
- **Cancel.** `await ctx.cancel()` stops the context at once: unfinished requests
  end as `Cancelled`, all of its audio not yet handed to you is dropped, and a
  `send_stream` feeding it stops sending and returns. The cancel still reaches the
  server if the task calling it is itself cancelled. The context stays usable.
- **Connection management.** The first send opens the socket if you did not.
  Server restarts (close 1001, or a draining refusal) are handled by resending on
  a new connection; a dropped connection fails only the requests that already
  had audio (`StreamInterruptedError`) and resends the rest. Every unfinished
  request always gets an outcome, and cancelling the task awaiting `send()` never
  leaves a half-sent request behind. `send()` waits only if the connection's
  limits (64 unfinished requests, 16 busy contexts) are reached; open more
  WebSockets for more parallelism. A WebSocket can live for days: finished
  contexts are released, so its memory does not grow with the requests it serves.
- **Connection-level errors.** Errors about no request in particular go to
  `client.tts.websocket(on_error=callback)`.

The sync client (`Bakbak`) has the same API without `await`; it reads the socket
on a background thread, so read `ctx.audio()` on one thread while another sends:

```python
client = Bakbak()
with client.tts.websocket(voice_id=voice_id, language="hi", model="m1") as ws:
    ctx = ws.context()
    handle = ctx.send(text="नमस्ते।")
    ctx.end_input()
    audio = b"".join(ctx.audio())
    print(handle.result().request_id)
```

### Speaking an LLM's reply as it is written

`ctx.send_stream(tokens)` splits text that arrives in pieces into sentences and
sends each one the moment it is complete, then calls `end_input()`. It takes
strings, or the stream events of OpenAI-, Anthropic- and Gemini-style SDKs
directly: the text delta is found by duck typing (`choices[0].delta.content`,
`response.output_text.delta`, `content_block_delta` → `delta.text`,
`candidates[0].content.parts[*].text`, or a `text` or `content` string), and
tool calls, reasoning deltas and Gemini's thought parts are skipped. Gemini's raw
REST/SSE JSON, parsed into dicts, works the same as its SDK's objects. No LLM SDK
is imported.

```python
from openai import AsyncOpenAI

llm = AsyncOpenAI()
tokens = await llm.chat.completions.create(model="…", messages=messages, stream=True)

ctx = ws.context()
sender = asyncio.create_task(ctx.send_stream(tokens))
async for audio in ctx.audio():
    player.write(audio)
await sender
```

If the tokens end with no text, and some of them had none of the shapes above
(a framework's own event objects, say, or tuples), `send_stream` ends input as
usual and then raises `TypeError` naming the shapes it reads, rather than
return having sent nothing: map such events to `str` first. A stream it can
read that simply has no text, such as one that only calls tools, is not an
error, and neither is a `send_stream` stopped by `cancel()`.

See [`examples/llm_streaming.py`](https://github.com/raya-voice-ai/bakbak-python/blob/main/examples/llm_streaming.py).

## Voices

```python
voices = client.voices.list(model="m1", language="hi")  # filtered by the SDK
for voice in voices:
    print(voice.id, voice.name, voice.language, voice.model)
```

A voice belongs to one model, and the voice decides which languages work.
Today's catalog:

| Model | Languages with voices |
|---|---|
| `m1` | `hi`, `en`, `kn` |
| `standard` | `hi`, `en-in`, `en-us`, `as`, `bn`, `gu`, `kn`, `ml`, `mr`, `ne`, `ta`, `te` |

`m1` voices use `en` for English; `standard` voices use `en-in` or `en-us`. The
server also accepts `pa` (Punjabi), which has no voice yet. `voices.list()` is
the source of truth: voices are added without an SDK release.

## Sentence segmentation

The splitter `send_stream` uses is public:

```python
from bakbak.text import Segmenter, split_sentences

split_sentences("Dr. Rao will call at 5 p.m. today. Please stay on the line.")
# ['Dr. Rao will call at 5 p.m. today.', 'Please stay on the line.']

segmenter = Segmenter(eager_first_clause=True)  # first audio sooner on a long first sentence
for token in tokens:
    for sentence in segmenter.push(token):
        ...
remaining = segmenter.flush()
```

It understands the danda (।, ॥), titles and initials (`Dr.`, `डॉ.`, `A. P. J.`),
decimals and domains (`3.14`, `getraya.app`), list markers and abbreviations, and
never splits inside an Indic conjunct. After the first sentence, one shorter
than `min_chars` is held and sent with the next, so
`split_sentences("आपका ऑर्डर तैयार है। कल डिलीवरी होगी॥ धन्यवाद।")` gives two
pieces, not three; the last piece goes out whatever its length. Options: `min_chars` (20),
`first_min_chars` (1), `max_chars` (400, at most 5,000), `eager_first_clause`
(False), `first_clause_min_chars` (24). Lengths are counted in code points. The
JavaScript SDK and the LiveKit plugin split every text exactly the same way.

## Audio helpers

```python
from bakbak.audio import to_wav

wav = to_wav(raw_bytes, "pcm_mulaw", 8000)
```

`pcm_s16le` gets the canonical 44-byte header; `pcm_f32le`, `pcm_alaw` and
`pcm_mulaw` get the 58-byte header (with a `fact` chunk) that their WAV formats
require. Audio that is not a whole number of samples (`pcm_s16le` of odd
length, or `pcm_f32le` whose length is not a multiple of 4) raises `ValueError`
rather than get a header that disagrees with it; so does `audio.to_wav()`.
There are no playback helpers: hand the bytes to your audio library.

## Errors

Every error derives from `bakbak.BakbakError`:

```
BakbakError
├── APIError                        the service answered, or a stream broke
│   ├── APIStatusError              an HTTP status, or an in-band error before audio
│   │   ├── BadRequestError (400), AuthenticationError (401),
│   │   ├── InsufficientCreditsError (402), PermissionDeniedError (403),
│   │   ├── NotFoundError (404), RequestTooLargeError (413),
│   │   ├── UnprocessableEntityError (422), RateLimitError (429),
│   │   └── InternalServerError (≥ 500)
│   └── StreamInterruptedError      failed after audio was delivered
└── APIConnectionError              no usable answer: network, TLS, DNS
    ├── APITimeoutError
    └── WebSocketClosedError
```

`APIError` has `status`, `error_code`, `message`, `retryable`, `request_id` and
`body` (the parsed JSON, so fields such as `required_inr` stay readable).
`StreamInterruptedError` adds `audio_bytes`: how much audio you received.
Requests the API gateway refuses before they reach the service (an invalid key,
an unknown path) have no `error_code`; their class says what happened.

```python
import bakbak

try:
    audio = client.tts.generate(text=text, voice_id=voice_id, language="hi")
except bakbak.InsufficientCreditsError as e:
    print("Top up:", e.body["required_inr"])
except bakbak.APIStatusError as e:
    print(e.status, e.error_code, e.message, e.request_id)
except bakbak.APIConnectionError:
    print("Could not reach the API")
```

Errors never contain your API key or request headers, in any attribute, `str()`,
`repr()` or chained cause. A key with whitespace, control or non-ASCII characters
(a copy-paste slip) is refused when the client is created, without echoing it.
The SDK never follows redirects, on HTTP or the WebSocket: a 3xx is an
`APIStatusError`, so the key is never sent to another origin. See the
[error reference](https://docs.litwizlabs.com/documentation/tts-errors) for every
`error_code`.

## Retries and timeouts

Before any audio arrives, the SDK retries connection errors, timeouts, and errors
whose `retryable` is true (by default 408, 429, 500, 502, 503 and 504): up to
`max_retries` times, waiting `Retry-After` when the server sends it (at most
60 s), otherwise 0.5 s, 1 s, 2 s … with jitter. **Once audio has been delivered
it never retries**: the start would play twice, and the request has already been
charged. You get `StreamInterruptedError` with the `request_id` and `audio_bytes`
instead.

On the WebSocket, a retryable refusal is resent on its context only if no later
request of that context was already sent; otherwise resending would break the
order, and the error is delivered.

## Raw responses

```python
raw = client.tts.with_raw_response.generate(text=text, voice_id=voice_id, language="hi")
raw.headers, raw.status_code, raw.request_id, raw.retries_taken
audio = raw.parse()
```

## Forward compatibility

Fields the SDK does not model yet go in `extra_body` (merged into the request
body) and `extra_headers`. The server refuses fields it does not know with a 422,
so `extra_body` is for fields the server already supports. Unknown response
fields and stream events are ignored. The SDK sends exactly the fields you give
it; the server validates them.

## Latency tips

- **Reuse one client.** It keeps connections alive; `client.prewarm()` opens one
  ahead of the first request.
- **Use the WebSocket for agents.** Open it early (`await ws.connect()`, or
  `async with`), so the first request pays no handshake. Measured from a machine
  in the same region as the API (m1, 20 sequential requests each, kept-alive
  connections, `examples/bench_ttfa.py`): time to first audio p50 **20.7 ms over
  the WebSocket** versus 33.3 ms over SSE. From farther away both include your
  network's round trips (one test from a laptop saw 58 ms versus 158 ms), and the
  WebSocket stays ahead.
- **Stream sentence by sentence** with `send_stream`, and consider
  `Segmenter(eager_first_clause=True)` when first sentences are long.
- **Ask for the format you play.** For phone calls use
  `{"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000}`, and skip
  resampling and transcoding (see [`examples/telephony.py`](https://github.com/raya-voice-ai/bakbak-python/blob/main/examples/telephony.py)).
- **Keep the event loop free.** `send_stream` reads a sync iterable on the event
  loop; pass an async iterable from an async LLM client.

## Logging

The SDK logs to `logging.getLogger("bakbak")` at DEBUG level only: request ids,
statuses, timings and sizes. It never logs the API key or any text.

```python
import logging

logging.basicConfig()
logging.getLogger("bakbak").setLevel(logging.DEBUG)
```

## More

- [Python SDK guide](https://docs.litwizlabs.com/documentation/sdk-python)
- [WebSocket API reference](https://docs.litwizlabs.com/api-reference/tts-websocket)
- [Errors and error codes](https://docs.litwizlabs.com/documentation/tts-errors)
- [`examples/`](https://github.com/raya-voice-ai/bakbak-python/tree/main/examples): runnable scripts for each feature
- [`conformance/BEHAVIOR.md`](https://github.com/raya-voice-ai/bakbak-python/blob/main/conformance/BEHAVIOR.md): the behaviour
  every official Bakbak client follows, and the API in one page
- [CHANGELOG](https://github.com/raya-voice-ai/bakbak-python/blob/main/CHANGELOG.md) · [CONTRIBUTING](https://github.com/raya-voice-ai/bakbak-python/blob/main/CONTRIBUTING.md) ·
  [SECURITY](https://github.com/raya-voice-ai/bakbak-python/blob/main/SECURITY.md)

Licensed under the [Apache License 2.0](https://github.com/raya-voice-ai/bakbak-python/blob/main/LICENSE).
