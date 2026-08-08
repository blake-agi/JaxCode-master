"""Track solved/attempted/todo progress in a local JSON file."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from jax_judge.tasks import list_tasks

PROGRESS_PATH = os.environ.get("PROGRESS_PATH", "data/progress.json")

_COLORS = {
    "solved": "\033[92m✅",     # green
    "attempted": "\033[93m🔧",  # yellow
    "todo": "\033[90m⏳",       # gray
}
_DIFF_COLORS = {
    "Easy": "\033[92m",
    "Medium": "\033[93m",
    "Hard": "\033[91m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"


def _load() -> dict[str, Any]:
    path = Path(PROGRESS_PATH)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(data: dict[str, Any]) -> None:
    path = Path(PROGRESS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def mark_solved(task_id: str, exec_time: float | None = None) -> None:
    data = _load()
    entry = data.get(task_id, {})
    entry["status"] = "solved"
    entry["solved_at"] = datetime.now().isoformat()
    if exec_time is not None:
        best = entry.get("best_time")
        entry["best_time"] = min(best, exec_time) if best else exec_time
    entry["attempts"] = entry.get("attempts", 0) + 1
    data[task_id] = entry
    _save(data)


def mark_attempted(task_id: str) -> None:
    data = _load()
    entry = data.get(task_id, {})
    if entry.get("status") != "solved":
        entry["status"] = "attempted"
    entry["attempts"] = entry.get("attempts", 0) + 1
    data[task_id] = entry
    _save(data)


def status() -> None:
    """Print a dashboard of all problems and their status, grouped by category."""
    data = _load()
    tasks = list_tasks()

    solved_count = sum(1 for t_id, _ in tasks if data.get(t_id, {}).get("status") == "solved")
    total = len(tasks)
    filled = round(28 * solved_count / total) if total else 0
    bar = "█" * filled + "░" * (28 - filled)

    print(f"\n{'─' * 60}")
    print(f"  ⚡ {_BOLD}JAXCode Progress: {solved_count}/{total} solved{_RESET}")
    print(f"  {bar}")
    print(f"{'─' * 60}")

    current_cat = None
    for task_id, task in tasks:
        cat = task.get("category", "Other")
        if cat != current_cat:
            current_cat = cat
            done = sum(
                1 for tid, t in tasks
                if t.get("category") == cat and data.get(tid, {}).get("status") == "solved"
            )
            n = sum(1 for _, t in tasks if t.get("category") == cat)
            print(f"\n  {_BOLD}{cat}{_RESET} {_DIM}({done}/{n}){_RESET}")

        entry = data.get(task_id, {})
        s = entry.get("status", "todo")
        icon = _COLORS.get(s, _COLORS["todo"])
        diff = task["difficulty"]
        diff_c = _DIFF_COLORS.get(diff, "")
        attempts = entry.get("attempts", 0)
        att_str = f"  {_DIM}({attempts} attempts){_RESET}" if attempts else ""

        print(f"    {icon} {task_id:<24s}{_RESET} {diff_c}[{diff}]{_RESET}{att_str}")

    print(f"\n{'─' * 60}")
    print(f"  {_DIM}check(\"task_id\") to submit · hint(\"task_id\") · solution(\"task_id\"){_RESET}")
    print(f"{'─' * 60}\n")


def reset_progress() -> None:
    """Clear all progress."""
    path = Path(PROGRESS_PATH)
    if path.exists():
        path.unlink()
    print("Progress reset.")
