"""Task definitions for JAXCode.

Each module in this package exposes a ``TASK`` dict:

    TASK = {
        "title":         str,   # human-readable problem title
        "category":      str,   # one of _registry.CATEGORIES
        "order":         int,   # position within the category
        "difficulty":    str,   # "Easy" | "Medium" | "Hard"
        "function_name": str,   # the symbol the judge looks for in your namespace
        "hint":          str,   # nudge shown by hint()
        "description":   str,   # markdown problem statement (drives the notebook)
        "stub":          str,   # starter code for the blank template notebook
        "solution":      str,   # reference implementation
        "demo":          str,   # optional scratch cell for poking at your impl
        "tests":         list,  # [{"name": str, "code": str}] — "{fn}" is substituted
    }
"""

from jax_judge.tasks._registry import (
    CATEGORIES,
    DIFFICULTY_ORDER,
    TASKS,
    get_task,
    list_by_difficulty,
    list_tasks,
)

__all__ = [
    "CATEGORIES",
    "DIFFICULTY_ORDER",
    "TASKS",
    "get_task",
    "list_by_difficulty",
    "list_tasks",
]
