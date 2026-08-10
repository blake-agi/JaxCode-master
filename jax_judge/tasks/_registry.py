"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

# Display grouping for status(), the welcome notebook and the README table.
CATEGORIES = [
    "JAX Fundamentals",
    "Core Ops & Layers",
    "Attention & Transformers",
    "Training",
    "Inference & Decoding",
    "RLHF & Preference Losses",
]

TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    """Category first, then the authored notebook number.

    The number IS the curriculum order: "01".."41" carried over from the
    PyTorch original, and "b_01".."b_11" for the JAX-only additions. There used
    to be a separate `order` field as well, which let the two disagree — a
    category then listed as 01, 19, 02, 03 ...
    """
    task_id, task = item
    cat = task.get("category", "")
    cat_idx = CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES)
    return (cat_idx, task.get("number", task_id))


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    """Every task in curriculum order: by category, then by notebook number."""
    return sorted(TASKS.items(), key=_sort_key)
