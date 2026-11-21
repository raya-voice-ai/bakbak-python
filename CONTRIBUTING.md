# Contributing

Thank you for helping. This SDK is one of four official Bakbak clients (Python,
JavaScript, the MCP server and the LiveKit plugin) that must behave identically,
so changes to behaviour start in the shared specification, not here.

## Development setup

```sh
uv sync                      # creates .venv with the dev tools
uv run pytest                # unit and conformance tests (no network)
uv run ruff check . && uv run ruff format --check .
uv run pyright               # strict for src/ and examples/
```

Python 3.10 to 3.14 are supported. To test the oldest dependencies we support:

```sh
uv venv -p 3.10 .venv-floor
VIRTUAL_ENV=.venv-floor uv pip install -e . "websockets==14.2" "httpx==0.27.0" "typing-extensions==4.10.0" pytest pytest-asyncio
.venv-floor/bin/python -m pytest
```

## Tests

- `tests/test_conformance_*.py` run every scenario in `conformance/http` and
  `conformance/ws` against a real local HTTP server and a scripted WebSocket
  server, for both the sync and the async client.
- `tests/test_segmenter.py` runs the segmenter vectors, the streaming invariant
  and a differential fuzz against `conformance/segmenter/reference.py`.
- `tests/test_live.py` calls a real deployment and spends credits. It runs only
  with `-m live`, and only when `BAKBAK_API_KEY` and `BAKBAK_BASE_URL` are set:

  ```sh
  BAKBAK_API_KEY=… BAKBAK_BASE_URL=… uv run pytest -m live
  BAKBAK_API_KEY=… BAKBAK_BASE_URL=… uv run pytest -m live -k smoke   # no WebSocket
  ```

  Run them one at a time (the default). A test deployment may serve each model
  one request at a time, so overlapping requests get 503 capacity refusals.

Never commit API keys, `.env` files, internal hostnames or IP addresses,
machine paths, or customer data.

## The conformance directory

`conformance/` is a copy of the shared client specification (`BEHAVIOR.md`),
its fixtures, and the segmenter reference. **Never edit it here.** It is synced
byte for byte from the canonical copy, and CI checks it two ways:

- `cd conformance && sha256sum -c MANIFEST.sha256` must pass on every build;
- a weekly job compares `MANIFEST.sha256` with the other official clients'
  copies and warns when they drift.

To change the specification, change the canonical copy, sync it into every
client repository, and update each client in the same round. If a fixture
looks wrong, raise it there; a scenario this SDK cannot express may be skipped
in its test with a written reason, never deleted.

## Commits and releases

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`), which drive the releases:

1. Every push to `main` updates a release pull request opened by
   release-please, with the next version and the changelog.
2. Merging that pull request tags the release (`v…`) and creates the GitHub
   release.
3. The same workflow then builds the package and publishes it to PyPI with
   trusted publishing (no stored token) and build attestations.

Until 1.0.0 the versions are release candidates (`1.0.0rc1`, `1.0.0rc2`, …;
release-please writes them as `1.0.0-rc.N`, which PyPI normalises). To release
1.0.0 itself, remove `prerelease`, `prerelease-type` and `versioning` from
`release-please-config.json` and add `"release-as": "1.0.0"` for that one
release.

PyPI must list this repository's `release-please.yml` workflow and its `pypi`
environment as a trusted publisher of the `bakbak-python` project before the first
publish. (The distribution is `bakbak-python`; the import package is `bakbak`.)
