#!/usr/bin/env python3
"""Copy notebooks into ./notebooks/ WITHOUT clobbering your work.

A plain `cp templates/*.ipynb notebooks/` destroys any problem you have already
started. This compares each destination against notebooks/_pristine/ — the copy
as it was handed to you — and skips anything you have since modified.

    python scripts/refresh_notebooks.py           # safe: skip edited files
    python scripts/refresh_notebooks.py --force   # overwrite everything
    python scripts/refresh_notebooks.py --list    # show what would happen
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402

WORK = ROOT / "notebooks"
PRISTINE = WORK / "_pristine"
SOLUTIONS = WORK / "_solutions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite edited notebooks too")
    ap.add_argument("--list", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    PRISTINE.mkdir(exist_ok=True)
    SOLUTIONS.mkdir(exist_ok=True)

    new, updated, protected = [], [], []

    for src in sorted((ROOT / "templates").glob("*.ipynb")):
        dst = WORK / src.name
        ref = PRISTINE / src.name

        if not dst.exists():
            new.append(src.name)
        elif ref.exists() and not filecmp.cmp(dst, ref, shallow=False):
            # You changed it since it was handed to you.
            protected.append(src.name)
            if not args.force:
                continue
        elif filecmp.cmp(dst, src, shallow=False):
            continue                      # already identical, nothing to do
        else:
            updated.append(src.name)

        if not args.list:
            shutil.copy2(src, dst)
            shutil.copy2(src, ref)

    if not args.list:
        for s in (ROOT / "solutions").glob("*.ipynb"):
            shutil.copy2(s, SOLUTIONS / s.name)

    print()
    print(f"  {GREEN}{len(new)} added{RESET}, {len(updated)} refreshed, "
          f"{YELLOW}{len(protected)} protected{RESET}")
    if protected:
        verb = "OVERWRITTEN (--force)" if args.force else "left alone"
        colour = RED if args.force else DIM
        print(f"\n  {BOLD}Notebooks you have edited — {verb}:{RESET}")
        for n in protected:
            print(f"    {colour}{n}{RESET}")
        if not args.force:
            print(f"\n  {DIM}To take the newer version of one, copy it yourself:{RESET}")
            print(f"    {DIM}cp templates/{protected[0]} notebooks/{RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
