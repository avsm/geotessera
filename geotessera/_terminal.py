"""Shared terminal helpers for the geotessera command-line tools.

Both CLI entry points (``cli`` and ``registry_cli``) render Rich output and
need the same emoji-suppression logic for dumb/piped/legacy terminals, so it
lives here in one place rather than being copied into each module.
"""

import os
import sys

from rich.console import Console

# A single shared console so terminal capability detection is consistent across
# both CLIs.
console = Console()


def emoji(text: str) -> str:
    """Return ``text`` on capable terminals, ``""`` for dumb/piped/legacy ones.

    Uses Rich's terminal detection plus explicit checks for ``TERM=dumb`` and
    Windows legacy consoles whose cp1252/ascii encodings cannot render emoji.
    """
    # Dumb terminal (e.g. ``TERM=dumb``, as set in the test harness).
    if os.environ.get("TERM", "").lower() == "dumb":
        return ""

    # Windows legacy console with a non-UTF encoding.
    if sys.platform == "win32":
        try:
            encoding = sys.stdout.encoding or ""
            if encoding.lower() in ("cp1252", "ascii", ""):
                return ""
        except Exception:
            return ""

    return text if console.is_terminal else ""
