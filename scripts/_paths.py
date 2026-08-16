"""Where the PyTorch original lives — one definition for all three gates.

check_alignment.py, check_signatures.py and build_study.py all need the
TorchCode original next to this checkout, and each used to hardcode the same
sibling path. Two layouts are in the wild:

    <parent>/JAXCode/  +  <parent>/TorchCode-master-original/   (zip names)
    <parent>/jaxcode/  +  <parent>/upstream/torchcode/          (by role)

Both are probed, and TORCHCODE_ORIGINAL overrides either. A missing original is
not an error — every caller degrades to skipping its check, because CI clones
this repo alone.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CANDIDATES = (
    ROOT.parent / "upstream" / "torchcode",
    ROOT.parent / "TorchCode-master-original",
)


def original_repo() -> Path:
    """Path to the PyTorch original. May not exist — callers check."""
    env = os.environ.get("TORCHCODE_ORIGINAL")
    if env:
        return Path(env)
    for candidate in _CANDIDATES:
        if candidate.exists():
            return candidate
    return _CANDIDATES[0]
