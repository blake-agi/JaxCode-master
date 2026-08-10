"""The single definition of what a test snippet sees when the judge runs it.

Four callers execute task tests — the notebook judge (`engine`), the web API
(`web_engine`), the reference-solution verifier (`scripts/verify_tasks.py`) and
the adversarial prober (`scripts/probe_tests.py`). If any of them builds the
namespace slightly differently, a task can pass one and fail another. That has
already happened once: the verifier handed tests the solution's whole module
namespace, so tasks referencing an undeclared helper passed CI and then raised
NameError in a real notebook.

So the contract lives here, once:

  * a test's source is its `code` with the literal ``{fn}`` replaced by the
    task's ``function_name``
  * a test runs with ONLY ``function_name`` bound, plus anything the task
    declares in ``extra_names`` — nothing else leaks in
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = ["required_names", "render_test", "build_namespace", "MissingSymbols"]


class MissingSymbols(LookupError):
    """Raised when a required symbol is absent from the supplied lookup."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(", ".join(missing))


def required_names(task: dict[str, Any]) -> list[str]:
    """Every symbol a task's tests are entitled to see, in a stable order."""
    return [task["function_name"], *task.get("extra_names", [])]


def render_test(task: dict[str, Any], test: dict[str, Any]) -> str:
    """A test's executable source, with ``{fn}`` resolved."""
    return test["code"].replace("{fn}", task["function_name"])


def build_namespace(
    task: dict[str, Any],
    lookup: Callable[[str], Any] | dict[str, Any],
    *,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """The exact globals a test snippet executes with.

    `lookup` is either a mapping or a callable resolving a name to an object —
    the notebook's user namespace, or a dict built from exec'ing a solution.

    Raises MissingSymbols listing every name that could not be resolved, so
    callers can report all of them at once rather than one per run.
    """
    get = lookup.get if hasattr(lookup, "get") else None
    wanted = list(names) if names is not None else required_names(task)

    resolved: dict[str, Any] = {}
    missing: list[str] = []
    for name in wanted:
        value = get(name, _MISSING) if get else lookup(name)
        if value is _MISSING:
            missing.append(name)
        else:
            resolved[name] = value

    if missing:
        raise MissingSymbols(missing)
    return resolved


class _Missing:
    __slots__ = ()


_MISSING = _Missing()
