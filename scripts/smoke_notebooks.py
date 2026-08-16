#!/usr/bin/env python3
"""Execute solution notebooks end-to-end and assert the judge reports all-pass.

This is the integration test the unit-level verifier cannot give you: it proves
the generated notebook actually runs in a real kernel — imports, stub cell,
demo cell, and the submit cell that calls check().

    python scripts/smoke_notebooks.py              # all solution notebooks
    python scripts/smoke_notebooks.py 01 05 12     # just these numbers
    python scripts/smoke_notebooks.py --templates  # blank templates must not crash
    python scripts/smoke_notebooks.py --banned     # banned_examples must be REJECTED

--banned exists because a handful of problems are defined by what you may not
call, and the judge enforces that by reading the submitted function's source.
probe_tests.py cannot cover it: it execs attacks from strings, where
inspect.getsource has nothing to read. Only a real kernel does.
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


def run_notebook(path: Path, timeout: int = 600, allow_errors: bool = False) -> tuple[bool, str]:
    """Execute a notebook. Returns (ran_to_completion, all output text).

    allow_errors keeps going past a raising cell. Blank templates need it: the
    stub returns None, so the scratch cell legitimately blows up — but the
    submit cell after it still has to reach the judge and print a verdict.
    """
    nb = nbformat.read(path, as_version=4)
    # Every solution notebook calls check(), so a smoke run is 50+ fake solves.
    # Both sinks have to be redirected: PROGRESS_PATH was covered from the
    # start, but log_attempt()/log_aid() arrived later and wrote 54 bogus rows
    # into the real attempts.csv the first time this ran afterwards.
    with tempfile.TemporaryDirectory() as tmp:
        setup = nbformat.v4.new_code_cell(
            "import os\n"
            f'os.environ["PROGRESS_PATH"] = r"{Path(tmp) / "progress.json"}"\n'
            f'os.environ["JAXCODE_STUDY_DIR"] = r"{tmp}"'
        )
        nb.cells.insert(0, setup)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=allow_errors,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        except CellExecutionError as e:
            return False, f"cell raised: {str(e)[:400]}"
        return True, _outputs_text(nb)


IMPL_MARKER = "# ✏️ YOUR IMPLEMENTATION HERE"


def _all_passed(text: str) -> bool:
    """The judge's all-pass verdict. A partial pass prints "5/6 tests passed",
    so matching "tests passed" alone reads a failure as a success."""
    return "All" in text and "tests passed" in text


def run_banned() -> int:
    """Every `banned_examples` entry must make its notebook's check() FAIL.

    The example is dropped into the template's implementation cell, so this
    exercises exactly what a learner's submission does: the judge reads the
    function's source out of the executed cell and rejects the banned call.
    """
    sys.path.insert(0, str(ROOT))
    from jax_judge.tasks import TASKS  # noqa: E402  (needs ROOT on the path)

    cases = [(f"{tid} [{i + 1}]", tid, t, src) for tid, t in sorted(TASKS.items())
             for i, src in enumerate(t.get("banned_examples", []))]
    if not cases:
        print(f"{YELLOW}No task declares banned_examples{RESET}")
        return 0

    print(f"\n{BOLD}Checking {len(cases)} banned implementation(s){RESET}\n")
    failures = 0

    for label, tid, task, src in cases:
        path = ROOT / "templates" / f"{task['number']}_{tid}.ipynb"
        if not path.exists():
            failures += 1
            print(f"  {RED}❌ {label} — no template at {path.name}{RESET}")
            continue

        nb = nbformat.read(path, as_version=4)
        patched = 0
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.source.startswith(IMPL_MARKER):
                cell.source = f"{IMPL_MARKER}\n\n{src.strip()}"
                patched += 1
        if patched != 1:
            failures += 1
            print(f"  {RED}❌ {label} — patched {patched} implementation cells, expected 1{RESET}")
            continue

        with tempfile.NamedTemporaryFile(suffix=".ipynb", dir=path.parent) as tmp:
            nbformat.write(nb, tmp.name)
            ok, text = run_notebook(Path(tmp.name), allow_errors=True)

        lines = src.strip().splitlines()
        gist = next((ln.strip() for ln in lines if "return" in ln), lines[-1].strip())
        if ok and not _all_passed(text):
            print(f"  {GREEN}✅ {label}{RESET} {DIM}rejected: {gist[:64]}{RESET}")
        else:
            failures += 1
            print(f"  {RED}❌ {label} — banned implementation was ACCEPTED{RESET}")
            print(f"     {DIM}{gist[:64]}{RESET}")

    print()
    if failures:
        print(f"{RED}{BOLD}{failures}/{len(cases)} banned implementation(s) slipped through{RESET}\n")
        return 1
    print(f"{GREEN}{BOLD}All {len(cases)} banned implementations were rejected{RESET}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers", nargs="*", help="notebook number prefixes, e.g. 01 07")
    ap.add_argument("--templates", action="store_true",
                    help="run blank templates instead (they must not crash)")
    ap.add_argument("--banned", action="store_true",
                    help="assert every task's banned_examples are REJECTED")
    args = ap.parse_args()

    if args.banned:
        return run_banned()

    d = ROOT / ("templates" if args.templates else "solutions")
    paths = sorted(p for p in d.glob("*.ipynb") if not p.name.startswith("00_"))
    if args.numbers:
        # Notebook labels are "01".."41" and "b_01".."b_11", so the label is not
        # simply everything before the first underscore.
        label = re.compile(r"^((?:b_)?\d+)_")
        def _label(p: Path) -> str:
            m = label.match(p.name)
            return m.group(1) if m else ""
        paths = [p for p in paths if _label(p) in args.numbers]

    if not paths:
        print(f"{RED}No notebooks matched{RESET}")
        return 2

    print(f"\n{BOLD}Executing {len(paths)} notebook(s) from {d.name}/{RESET}\n")
    failures = 0

    for p in paths:
        ok, text = run_notebook(p, allow_errors=args.templates)

        if not ok:
            failures += 1
            print(f"  {RED}❌ {p.name}{RESET}")
            print(f"     {RED}{text}{RESET}")
            continue

        if args.templates:
            # A blank stub SHOULD fail its tests. What must hold is that the
            # submit cell still reaches the judge and prints a real verdict,
            # rather than the learner seeing only an opaque traceback.
            # A blank template should hit the "still the blank starter stub"
            # path. Running the tests is also an acceptable verdict, but the
            # stub message is the one we actually want the learner to see.
            stub_notice = "still the blank starter stub" in text
            ran_tests = "tests passed" in text or "Testing:" in text
            if stub_notice:
                print(f"  {GREEN}✅ {p.name}{RESET} {DIM}(stub detected){RESET}")
            elif ran_tests:
                # Not merely cosmetic: the learner gets a wall of NoneType
                # tracebacks instead of "you have not implemented this yet".
                failures += 1
                print(f"  {RED}❌ {p.name} — blank stub NOT detected; "
                      f"the judge ran the tests instead{RESET}")
            else:
                failures += 1
                print(f"  {RED}❌ {p.name} — submit cell never produced a judge verdict{RESET}")
                tail = "\n".join(text.strip().split("\n")[-10:])
                print(f"{DIM}{tail}{RESET}")
            continue

        if _all_passed(text):
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
