"""Settings the examples share, from the environment.

BAKBAK_API_KEY is required (the client reads it). BAKBAK_BASE_URL, and the
voice settings below, are optional.
"""

from __future__ import annotations

import os

VOICE_ID = os.environ.get("BAKBAK_VOICE_ID", "009ae8b5-4a03-492d-af1c-3055989ac734")  # Anika M1
LANGUAGE = os.environ.get("BAKBAK_LANGUAGE", "hi")
MODEL = os.environ.get("BAKBAK_MODEL", "m1")
