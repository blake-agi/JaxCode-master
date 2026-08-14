#!/usr/bin/env python3
"""List notebooks whose content differs from their notebooks/_pristine/ baseline.

This is how log-mistake auto-detects which problem(s) you've been practicing,
so you don't have to name the task yourself — anything that differs from the
blank template you were handed is something you wrote code in. Most recently
modified first, so "whatever I was just doing" naturally sorts to the top.

Same diff-against-_pristine logic as refresh_notebooks.py uses to decide what
is safe to overwrite; a notebook with no pristine baseline yet is skipped
(nothing to compare against) rather than guessed at.

    python scripts/list_edited_notebooks.py            # task ids, newest first
    python scripts/list_edited_notebooks.py --paths    # full notebook paths
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Optional override so your working notebooks can live outside this checkout
# (e.g. in a private practice repo). Unset — as in any fresh clone — this is
# exactly ROOT/notebooks, so nothing changes for anyone else.
WORK = Path(os.environ.get("JAXCODE_NOTEBOOKS_DIR") or ROOT / "notebooks")
PRISTINE = WORK / "_pristine"

# "05_attention" -> "attention", but "b_01_grad_basics" -> "grad_basics" — the
# added (non-ported) problems carry a b_NN prefix, one underscore longer than
# the ported problems' plain NN prefix.
_STEM_RE = re.compile(r"^(?:b_)?\d+_(.+)$")


def task_id_of(stem: str) -> str:
    m = _STEM_RE.match(stem)
    return m.group(1) if m else stem


def edited_notebooks() -> list[Path]:
    out = []
    for nb in sorted(WORK.glob("*.ipynb")):
        ref = PRISTINE / nb.name
        if not ref.exists():
            continue  # no baseline to diff against — can't tell, skip
        if not filecmp.cmp(nb, ref, shallow=False):
            out.append(nb)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", action="store_true", help="print full paths instead of task ids")
    args = ap.parse_args()

    for nb in edited_notebooks():
        print(nb if args.paths else task_id_of(nb.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
