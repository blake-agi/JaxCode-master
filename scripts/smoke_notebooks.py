#!/usr/bin/env python3
"""Execute solution notebooks end-to-end and assert the judge reports all-pass.

This is the integration test the unit-level verifier cannot give you: it proves
the generated notebook actually runs in a real kernel — imports, stub cell,
demo cell, and the submit cell that calls check().

    python scripts/smoke_notebooks.py              # all solution notebooks
    python scripts/smoke_notebooks.py 01 05 12     # just these numbers
    python scripts/smoke_notebooks.py --templates  # blank templates must not crash
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[1m", "\033[0m",
)

ANSI = re.compile(r"\033\[[0-9;]*m")


def _outputs_text(nb) -> str:
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append(out.get("text", ""))
            elif out.get("output_type") == "error":
                chunks.append("\n".join(out.get("traceback", [])))
            elif "data" in out:
                chunks.append(str(out["data"].get("text/plain", "")))
    return ANSI.sub("", "\n".join(chunks))


def run_notebook(path: Path, timeout: int = 600) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    # Progress writes go to a throwaway file so smoke runs never touch real data.
    with tempfile.TemporaryDirectory() as tmp:
        setup = nbformat.v4.new_code_cell(
            f'import os; os.environ["PROGRESS_PATH"] = r"{Path(tmp) / "progress.json"}"'
        )
        nb.cells.insert(0, setup)
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        )
        try:
            client.execute()
        except CellExecutionError as e:
            return False, f"cell raised: {str(e)[:400]}"
        return True, _outputs_text(nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers", nargs="*", help="notebook number prefixes, e.g. 01 07")
    ap.add_argument("--templates", action="store_true",
                    help="run blank templates instead (they must not crash)")
    args = ap.parse_args()

    d = ROOT / ("templates" if args.templates else "solutions")
    paths = sorted(p for p in d.glob("*.ipynb") if not p.name.startswith("00_"))
    if args.numbers:
        paths = [p for p in paths if p.name.split("_")[0] in args.numbers]

    if not paths:
        print(f"{RED}No notebooks matched{RESET}")
        return 2

    print(f"\n{BOLD}Executing {len(paths)} notebook(s) from {d.name}/{RESET}\n")
    failures = 0

    for p in paths:
        ok, text = run_notebook(p)

        if not ok:
            failures += 1
            print(f"  {RED}❌ {p.name}{RESET}")
            print(f"     {RED}{text}{RESET}")
            continue

        if args.templates:
            # A blank stub SHOULD fail its tests — we only assert it ran and the
            # judge produced a verdict rather than exploding.
            if "tests passed" in text or "❌" in text or "💥" in text:
                print(f"  {GREEN}✅ {p.name}{RESET} {DIM}(ran; judge reported a verdict){RESET}")
            else:
                failures += 1
                print(f"  {RED}❌ {p.name} — judge produced no verdict{RESET}")
            continue

        if "All" in text and "tests passed" in text:
            print(f"  {GREEN}✅ {p.name}{RESET}")
        else:
            failures += 1
            print(f"  {RED}❌ {p.name} — judge did not report all-pass{RESET}")
            tail = "\n".join(text.strip().split("\n")[-12:])
            print(f"{DIM}{tail}{RESET}")

    print()
    if failures:
        print(f"{RED}{BOLD}{failures}/{len(paths)} notebook(s) failed{RESET}\n")
        return 1
    print(f"{GREEN}{BOLD}All {len(paths)} notebooks executed cleanly{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
