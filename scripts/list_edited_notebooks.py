#!/usr/bin/env python3
"""List notebooks whose content differs from their notebooks/_pristine/ baseline.

This is how log-mistake auto-detects which problem(s) you've been practicing,
so you don't have to name the task yourself — anything that differs from the
blank template you were handed is something you wrote code in. Most recently
modified first, so "whatever I was just doing" naturally sorts to the top.

Same diff-against-_pristine logic as refresh_notebooks.py uses to decide what
is safe to overwrite; a notebook with no pristine baseline yet is skipped
(nothing to compare against) rather than guessed at.

Empty output is meaningful — it means nothing has been edited — so a missing
directory must NOT produce it. See _require_dirs().

    python scripts/list_edited_notebooks.py            # task ids, newest first
    python scripts/list_edited_notebooks.py --paths    # full notebook paths
"""

from __future__ import annotations

import argparse
import filecmp
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _paths import notebooks_dir, notebooks_dir_problem  # noqa: E402
from jax_judge._term import DIM, RED, RESET  # noqa: E402

WORK = notebooks_dir()
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

    # Fatal, not a warning: printing nothing is how this script says "no
    # notebook has been edited", and log-it stops on that. A wrong directory
    # must never be able to say the same thing.
    problem = notebooks_dir_problem(require_pristine=True)
    if problem:
        print(f"{RED}{problem}{RESET}", file=sys.stderr)
        print(f"  {DIM}(refusing to print an empty list that would read as "
              f"'nothing to log'){RESET}", file=sys.stderr)
        return 2

    for nb in edited_notebooks():
        print(nb if args.paths else task_id_of(nb.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
