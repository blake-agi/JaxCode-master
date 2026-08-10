"""Task definitions for JAXCode.

Each module in this package exposes a ``TASK`` dict:

    TASK = {
        "title":         str,   # human-readable problem title
        "category":      str,   # one of _registry.CATEGORIES
        "difficulty":    str,   # "Easy" | "Medium" | "Hard"
        "function_name": str,   # the symbol the judge looks for in your namespace
        "hint":          str,   # nudge shown by hint()
        "description":   str,   # markdown problem statement (drives the notebook)
        "stub":          str,   # starter code for the blank template notebook
        "solution":      str,   # reference implementation
        "demo":          str,   # optional scratch cell for poking at your impl
        "extra_names":   list,  # optional extra symbols the tests need from the
                                # notebook namespace, when one function is not enough
        "tests":         list,  # [{"name": str, "code": str}] — "{fn}" is substituted
    }

Each test runs in a namespace containing ONLY ``function_name`` plus anything
listed in ``extra_names`` — nothing else from the notebook leaks in. If a test
references a second helper, declare it or it will raise NameError for the user.
"""

from jax_judge.tasks._registry import (
    CATEGORIES,
    TASKS,
    get_task,
    list_tasks,
)

__all__ = [
    "CATEGORIES",
    "TASKS",
    "get_task",
    "list_tasks",
]
