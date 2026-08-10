"""ANSI styling, defined once.

Seven modules had their own copy of these escape codes. Beyond the duplication,
none of them honoured NO_COLOR or checked whether stdout was a terminal, so
piping any script into a file produced escape sequences.
"""

from __future__ import annotations

import os
import sys

__all__ = ["GREEN", "RED", "YELLOW", "CYAN", "DIM", "BOLD", "RESET", "enabled"]


def enabled() -> bool:
    """Colour unless NO_COLOR is set or stdout is not a terminal.

    Jupyter's stdout is not a tty but does render ANSI, so treat an active
    IPython kernel as colour-capable.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if "ipykernel" in sys.modules:
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_ON = enabled()


def _code(seq: str) -> str:
    return seq if _ON else ""


GREEN = _code("\033[92m")
RED = _code("\033[91m")
YELLOW = _code("\033[93m")
CYAN = _code("\033[96m")
DIM = _code("\033[90m")
BOLD = _code("\033[1m")
RESET = _code("\033[0m")
