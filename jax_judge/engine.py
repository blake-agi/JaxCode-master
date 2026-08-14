"""Core judge engine — grabs user functions from the Jupyter namespace and tests them."""

from __future__ import annotations

import ast
import inspect
import textwrap
import time
import traceback
from typing import Any

from jax_judge._contract import MissingSymbols, build_namespace, render_test
from jax_judge._term import BOLD as _BOLD
from jax_judge._term import CYAN as _CYAN
from jax_judge._term import DIM as _DIM
from jax_judge._term import GREEN as _GREEN
from jax_judge._term import RED as _RED
from jax_judge._term import RESET as _RESET
from jax_judge._term import YELLOW as _YELLOW
from jax_judge.tasks import get_task, TASKS
from jax_judge.progress import log_attempt, mark_solved, mark_attempted


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


def _parse_source(obj: Any) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None


def _func_is_placeholder(fn: Any) -> bool:
    tree = _parse_source(fn)
    if tree is None or not tree.body:
        return False
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return _body_is_placeholder(node)


def _unwrap(obj: Any) -> Any:
    """Peel JAX/functools decorators off to reach the underlying function.

    @jax.jit and @partial(jax.jit, ...) set __wrapped__; @jax.custom_vjp and
    friends keep the primal on .fun. Without this, a decorated stub looks like
    an opaque object rather than an unimplemented function.
    """
    seen: set[int] = set()
    for _ in range(10):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        nxt = None
        for attr in ("__wrapped__", "fun", "func"):
            cand = getattr(obj, attr, None)
            if callable(cand) and cand is not obj:
                nxt = cand
                break
        if nxt is None:
            break
        obj = nxt
    return obj


def _methods_are_placeholders(methods: list, is_placeholder) -> bool:
    """True when every method the learner is meant to fill is still a stub.

    __init__ is judged separately: some stubs hand it to you already written
    (SimpleBPE's sets self.merges = []), so requiring EVERY method to be a
    placeholder would miss those and dump raw test failures on the learner
    instead of "you have not implemented this yet".
    """
    if not methods:
        return False
    name = (lambda m: m.name if hasattr(m, "name") else m.__name__)
    fillable = [m for m in methods if name(m) != "__init__"]
    return all(is_placeholder(m) for m in (fillable or methods))


def _looks_unimplemented(obj: Any) -> bool:
    """Detect the untouched starter stub, so we can say so instead of
    dumping five identical NoneType tracebacks at the learner."""
    obj = _unwrap(obj)

    if inspect.isclass(obj):
        # In a Jupyter kernel inspect.getsource(cls) raises TypeError, because
        # __main__ has no __file__ for findsource to search. Individual methods
        # DO resolve, via IPython's linecache entry for the cell — so inspect
        # those instead, keeping only the ones defined on this class.
        tree = _parse_source(obj)
        if tree is not None and tree.body and isinstance(tree.body[0], ast.ClassDef):
            methods = [
                n for n in tree.body[0].body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            return _methods_are_placeholders(methods, _body_is_placeholder)

        own = [
            m for _, m in inspect.getmembers(obj, inspect.isfunction)
            if getattr(m, "__qualname__", "").split(".")[0] == obj.__name__
        ]
        return _methods_are_placeholders(own, _func_is_placeholder)

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return _func_is_placeholder(obj)
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

    # Some tasks ask for more than one symbol (e.g. bpe wants train_bpe AND
    # apply_merges). build_namespace resolves them all, or reports what is absent.
    try:
        base_ns = build_namespace(task, user_ns)
    except MissingSymbols as missing:
        names = missing.missing
        print(f"\n{_RED}❌ Also need: {', '.join(names)}{_RESET}")
        print(f"{_DIM}   This task asks for more than one function. Define "
              f"{'them' if len(names) > 1 else 'it'} and re-run that cell.{_RESET}\n")
        return

    for i, test in enumerate(tests, 1):
        test_code = render_test(task, test)
        namespace: dict[str, Any] = dict(base_ns)

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
        log_attempt(task_id, passed, total, total_time, solved=True)
        print(f"  {_DIM}Progress saved. Run status() to see your dashboard.{_RESET}\n")
    else:
        print(f"  {_YELLOW}📊 {passed}/{total} tests passed.{_RESET}")
        mark_attempted(task_id)
        log_attempt(task_id, passed, total, total_time, solved=False)
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
