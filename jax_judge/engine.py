"""Core judge engine — grabs user functions from the Jupyter namespace and tests them."""

from __future__ import annotations

import ast
import inspect
import textwrap
import time
import traceback
from typing import Any

from jax_judge.tasks import get_task, TASKS
from jax_judge.progress import mark_solved, mark_attempted

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[90m"
_BOLD = "\033[1m"


def _get_user_namespace() -> dict[str, Any]:
    """Get the calling notebook's global namespace via IPython."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except ImportError:
        pass
    # Fallback: try to grab caller's globals
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        return frame.f_back.f_back.f_globals
    return {}


def _body_is_placeholder(node: ast.AST) -> bool:
    """True if a def's body is only a docstring plus pass / ... / raise."""
    body = list(getattr(node, "body", []))
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop the docstring
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, (ast.Pass, ast.Raise)):
        return True
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _looks_unimplemented(obj: Any) -> bool:
    """Detect the untouched starter stub, so we can say so instead of
    dumping five identical NoneType tracebacks at the learner."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    if not tree.body:
        return False
    node = tree.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _body_is_placeholder(node)
    if isinstance(node, ast.ClassDef):
        methods = [
            n for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return bool(methods) and all(_body_is_placeholder(m) for m in methods)
    return False


def _test_frame_hint(exc: BaseException) -> str | None:
    """The last frame inside the test snippet itself, ignoring library internals."""
    frames = [
        f for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.startswith("<test:")
    ]
    if not frames:
        return None
    f = frames[-1]
    line = (f.line or "").strip()
    return f"line {f.lineno}: {line}" if line else f"line {f.lineno}"


def check(task_id: str) -> None:
    """Run all tests for a task against the user's implementation.

    Usage in a notebook cell:
        from jax_judge import check
        check("relu")
    """
    task = get_task(task_id)
    if task is None:
        available = ", ".join(f"'{k}'" for k in sorted(TASKS))
        print(f"{_RED}Unknown task '{task_id}'. Available: {available}{_RESET}")
        return

    fn_name = task["function_name"]
    user_ns = _get_user_namespace()

    if fn_name not in user_ns:
        print(f"\n{_RED}❌ Function/class '{fn_name}' not found in your notebook.{_RESET}")
        print(f"{_DIM}   Make sure you defined it in a cell above and ran that cell.{_RESET}\n")
        return

    user_fn = user_ns[fn_name]

    if _looks_unimplemented(user_fn):
        kind = "class" if inspect.isclass(user_fn) else "function"
        print(f"\n{_YELLOW}✏️  '{fn_name}' is still the blank starter stub.{_RESET}")
        print(f"{_DIM}   Replace the `pass` in the {kind} body with your "
              f"implementation,{_RESET}")
        print(f"{_DIM}   re-run that cell, then run check(\"{task_id}\") again.{_RESET}")
        print(f"{_DIM}   Stuck already? hint(\"{task_id}\"){_RESET}\n")
        return

    tests = task["tests"]
    passed = 0
    total = len(tests)

    print(f"\n{_BOLD}🧪 Testing: {task['title']} ({task['difficulty']}){_RESET}")
    print(f"{'─' * 56}")

    total_time = 0.0

    for i, test in enumerate(tests, 1):
        test_code = test["code"].replace("{fn}", fn_name)
        namespace: dict[str, Any] = {fn_name: user_fn}

        t0 = time.perf_counter()
        try:
            exec(compile(test_code, f"<test:{test['name']}>", "exec"), namespace)  # noqa: S102
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            passed += 1
            print(f"  {_GREEN}✅ [{i}/{total}] {test['name']}{_RESET} {_DIM}({elapsed*1000:.1f}ms){_RESET}")
        except AssertionError as e:
            msg = str(e) or "Assertion failed"
            print(f"  {_RED}❌ [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{msg}{_RESET}")
        except Exception as e:
            print(f"  {_RED}💥 [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{type(e).__name__}: {e}{_RESET}")
            where = _test_frame_hint(e)
            if where:
                print(f"     {_DIM}in test {where}{_RESET}")
            if "NoneType" in str(e):
                print(f"     {_DIM}Looks like {fn_name} returned None — "
                      f"is a `return` missing?{_RESET}")

    print(f"{'─' * 56}")

    if passed == total:
        print(f"  {_GREEN}{_BOLD}🎉 All {total} tests passed! ({total_time*1000:.1f}ms total){_RESET}")
        print(f"  {_DIM}Timings include XLA tracing/compilation, so they are not a{_RESET}")
        print(f"  {_DIM}clean benchmark — use %timeit on a warmed-up jit for that.{_RESET}")
        mark_solved(task_id, total_time)
        print(f"  {_DIM}Progress saved. Run status() to see your dashboard.{_RESET}\n")
    else:
        print(f"  {_YELLOW}📊 {passed}/{total} tests passed.{_RESET}")
        mark_attempted(task_id)
        print(f"  {_DIM}Keep going! Use hint(\"{task_id}\") if you're stuck,{_RESET}")
        print(f"  {_DIM}or solution(\"{task_id}\") to see a reference implementation.{_RESET}\n")


def hint(task_id: str) -> None:
    """Show a hint for the given task."""
    task = get_task(task_id)
    if task is None:
        print(f"{_RED}Unknown task '{task_id}'.{_RESET}")
        return
    print(f"\n{_YELLOW}💡 Hint for {task['title']}:{_RESET}")
    print(f"   {task['hint']}\n")


def solution(task_id: str) -> None:
    """Print the reference implementation for a task.

    Spoiler warning — try `hint(task_id)` first.
    """
    task = get_task(task_id)
    if task is None:
        print(f"{_RED}Unknown task '{task_id}'.{_RESET}")
        return
    print(f"\n{_CYAN}📖 Reference solution — {task['title']}:{_RESET}")
    print(f"{_DIM}{'─' * 56}{_RESET}")
    print(task["solution"].strip())
    print(f"{_DIM}{'─' * 56}{_RESET}\n")
