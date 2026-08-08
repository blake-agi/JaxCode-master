"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

# Curriculum order — notebooks are numbered from this list, and `status()`
# groups the dashboard by these headings.
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


def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
    task_id, task = item
    cat = task.get("category", "")
    cat_idx = CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES)
    return (cat_idx, task.get("order", 999), task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    """All tasks in curriculum order (category, then order within category)."""
    return sorted(TASKS.items(), key=_sort_key)


def list_by_difficulty() -> list[tuple[str, dict[str, Any]]]:
    """All tasks sorted Easy -> Hard."""
    return sorted(
        TASKS.items(),
        key=lambda t: (DIFFICULTY_ORDER.get(t[1]["difficulty"], 9), _sort_key(t)),
    )
