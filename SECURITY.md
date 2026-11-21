# Security

## Reporting a vulnerability

Please report security issues privately to **support@getraya.app**, not in a
public GitHub issue. Include the SDK version (`bakbak.__version__`), the Python
version, and steps to reproduce. Never include your API key: if a key may have
been exposed, revoke it and create a new one.

We acknowledge reports within three working days and keep you informed until
the issue is resolved.

## Supported versions

Security fixes go into the latest release. Until 1.0.0, that is the latest
release candidate.

## How the SDK handles your key

- The key is sent only in the `X-API-Key` header, only to the `base_url` you
  configure. The SDK makes no other network calls: no telemetry and no update
  checks.
- Errors, `repr()`/`str()` of clients and errors, and the SDK's logs never
  contain the key or the request headers. Logs never contain synthesis text.
