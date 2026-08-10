#!/usr/bin/env python3
"""Run every task's reference solution against its own test suite.

This is the repo's safety net: a task is only correct if its published
solution passes every test it ships. Run it after editing any task file.

    python scripts/verify_tasks.py            # all tasks
    python scripts/verify_tasks.py relu rope  # just these
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# The sharding task needs more than one device. Fake 8 on CPU — this has to be
# set before anything imports jax, so it stays at the very top of the file.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast  # noqa: E402
import textwrap  # noqa: E402

from jax_judge._contract import MissingSymbols, build_namespace, render_test  # noqa: E402
from jax_judge.engine import _body_is_placeholder, _methods_are_placeholders  # noqa: E402
from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402,F401
from jax_judge.tasks import TASKS, list_tasks  # noqa: E402

REQUIRED_KEYS = {
    "title", "category", "number", "difficulty", "function_name",
    "hint", "description", "stub", "solution", "tests",
}


def _stub_verdict(source: str, fn_name: str) -> bool | None:
    """Would the judge call this source an unimplemented stub?

    Mirrors engine._looks_unimplemented's AST branch. We cannot call that
    directly here because it needs inspect.getsource, which fails on objects
    exec'd from a string (it works in a notebook via IPython's linecache).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    node = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
         and n.name == fn_name),
        None,
    )
    if node is None:
        return None
    if isinstance(node, ast.ClassDef):
        methods = [n for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return _methods_are_placeholders(methods, _body_is_placeholder)
    return _body_is_placeholder(node)


def verify(task_id: str, task: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []

    missing = REQUIRED_KEYS - set(task)
    if missing:
        problems.append(f"missing keys: {', '.join(sorted(missing))}")
        return False, problems

    fn_name = task["function_name"]

    # The stub must at least define the expected symbol, or the notebook's
    # submit cell fails with a confusing NameError before the user starts.
    if fn_name not in task["stub"]:
        problems.append(f"stub does not mention '{fn_name}'")

    # And the judge must RECOGNISE it as blank, otherwise a learner who just
    # opened the notebook gets a wall of NoneType tracebacks instead of the
    # "you have not implemented this yet" message. bpe regressed exactly here:
    # its stub hands you a written __init__, so "every method is a placeholder"
    # was false.
    if _stub_verdict(task["stub"], fn_name) is not True:
        problems.append(
            f"the judge would NOT recognise this stub as blank — check "
            f"_looks_unimplemented against '{fn_name}'"
        )
    if _stub_verdict(task["solution"], fn_name) is True:
        problems.append("the judge would mistake the reference solution for a blank stub")

    namespace: dict = {}
    try:
        exec(compile(task["solution"], f"<solution:{task_id}>", "exec"), namespace)
    except Exception as e:
        problems.append(f"solution failed to exec: {type(e).__name__}: {e}")
        return False, problems

    if fn_name not in namespace:
        problems.append(f"solution does not define '{fn_name}'")
        return False, problems

    # Build the namespace through the SAME contract the notebook judge uses,
    # so a task cannot pass here and then fail with NameError in a real notebook.
    try:
        judge_ns = build_namespace(task, namespace)
    except MissingSymbols as missing:
        for name in missing.missing:
            problems.append(
                f"tests will need '{name}' but the solution never defines it "
                "(declare it in extra_names, or define it in the solution)"
            )
        return False, problems

    for test in task["tests"]:
        code = render_test(task, test)
        ns = dict(judge_ns)
        try:
            exec(compile(code, f"<test:{task_id}:{test['name']}>", "exec"), ns)
        except AssertionError as e:
            problems.append(f"test '{test['name']}' FAILED: {e or 'assertion failed'}")
        except Exception as e:
            tb = traceback.format_exc().strip().split("\n")[-1]
            problems.append(f"test '{test['name']}' ERRORED: {type(e).__name__}: {e} | {tb}")

    return not problems, problems


def main() -> int:
    wanted = sys.argv[1:]
    items = [(tid, t) for tid, t in list_tasks() if not wanted or tid in wanted]

    if wanted:
        unknown = set(wanted) - set(TASKS)
        if unknown:
            print(f"{RED}Unknown task(s): {', '.join(sorted(unknown))}{RESET}")
            return 2

    print(f"\n{BOLD}Verifying {len(items)} task(s){RESET}\n")
    failures = 0
    t_start = time.perf_counter()

    for task_id, task in items:
        t0 = time.perf_counter()
        ok, problems = verify(task_id, task)
        dt = (time.perf_counter() - t0) * 1000
        n_tests = len(task.get("tests", []))
        if ok:
            print(f"  {GREEN}✅{RESET} {task_id:<26s} {DIM}{n_tests} tests  {dt:6.0f}ms{RESET}")
        else:
            failures += 1
            print(f"  {RED}❌ {task_id:<26s} {n_tests} tests  {dt:6.0f}ms{RESET}")
            for p in problems:
                print(f"       {RED}{p}{RESET}")

    total = time.perf_counter() - t_start
    print()
    if failures:
        print(f"{RED}{BOLD}{failures}/{len(items)} task(s) failed{RESET} {DIM}({total:.1f}s){RESET}\n")
        return 1
    print(f"{GREEN}{BOLD}All {len(items)} tasks verified{RESET} {DIM}({total:.1f}s){RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
