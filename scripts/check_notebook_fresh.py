#!/usr/bin/env python3
"""Warn when a notebook has been run more recently than it has been saved.

log-it reads the implementation cell OFF DISK. If you ran check() and then said
"log it" without saving, log-it summarises the *previous* version of your code —
and the try you actually want recorded is still sitting in an unsaved buffer.
That has happened.

The signal needs no cooperation from the editor: `study/attempts.csv` is written
by the kernel when check() runs, and the .ipynb is written when you save. So

    last attempt timestamp  >  notebook mtime   =>  you ran code that is not on disk

False positives are possible and harmless: saving before the run and never after
leaves the CODE current while the outputs are stale. False negatives cannot
happen — a file saved after its last run is genuinely up to date.

    python scripts/check_notebook_fresh.py           # only the tasks at risk
    python scripts/check_notebook_fresh.py --all     # every task with attempts
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _paths import notebooks_dir, notebooks_dir_problem  # noqa: E402
from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402

_STEM_RE = re.compile(r"^(?:b_)?\d+_(.+)$")


def _study_dir() -> Path:
    from jax_judge.progress import _default_study_dir
    return _default_study_dir()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every task, not just stale ones")
    args = ap.parse_args()

    problem = notebooks_dir_problem()
    if problem:
        print(f"{RED}{problem}{RESET}", file=sys.stderr)
        return 2

    attempts = _study_dir() / "attempts.csv"
    if not attempts.exists():
        print(f"{DIM}No attempts.csv yet — nothing to check.{RESET}")
        return 0

    last: dict[str, str] = {}
    with attempts.open(newline="") as fh:
        for row in csv.DictReader(fh):
            last[row["task"]] = row["timestamp"]      # file is chronological

    work = notebooks_dir()
    stale, fresh = [], []
    for nb in sorted(work.glob("*.ipynb")):
        m = _STEM_RE.match(nb.stem)
        tid = m.group(1) if m else nb.stem
        if tid not in last:
            continue
        ran = datetime.fromisoformat(last[tid])
        saved = datetime.fromtimestamp(nb.stat().st_mtime)
        (stale if ran > saved else fresh).append((tid, ran, saved, nb))

    if args.all:
        for tid, ran, saved, _ in sorted(fresh, key=lambda r: r[1]):
            print(f"  {GREEN}✓{RESET} {tid:<26} {DIM}saved {(saved - ran).total_seconds():+.0f}s "
                  f"after its last run{RESET}")

    if not stale:
        print(f"{GREEN}✓ every notebook is saved at least as recently as it was last run{RESET}")
        return 0

    print(f"\n{YELLOW}{BOLD}⚠️  Run more recently than saved — the buffer may hold "
          f"work that is not on disk:{RESET}\n")
    for tid, ran, saved, nb in sorted(stale, key=lambda r: r[1], reverse=True):
        mins = (ran - saved).total_seconds() / 60
        print(f"  {tid:<26} last run {ran:%Y-%m-%d %H:%M}  ·  saved {mins:.0f} min earlier")
        print(f"  {DIM}{nb}{RESET}")
    print(f"\n  {DIM}Save the notebook (Cmd+S) and re-run, or continue knowing the log "
          f"will describe the version on disk.{RESET}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
