"""Track solved/attempted/todo progress in a local JSON file."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from jax_judge._term import BOLD as _BOLD
from jax_judge._term import DIM as _DIM
from jax_judge._term import RESET as _RESET
from jax_judge.tasks import list_tasks

def _default_progress_path() -> str:
    """One progress file per checkout, regardless of where you launch from.

    This used to be the relative "data/progress.json", which resolved against
    the CURRENT WORKING DIRECTORY — so running a notebook from notebooks/ wrote
    a different file than running one from the repo root, and your dashboard
    silently forgot half your solves. Anchor it to the package instead.
    """
    return str(Path(__file__).resolve().parent.parent / "data" / "progress.json")


PROGRESS_PATH = os.environ.get("PROGRESS_PATH") or _default_progress_path()

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


def _default_study_dir() -> Path:
    """study/ lives next to the repo checkout, not inside it — it's your personal
    log, not something the JAX port ships. Same env-override pattern as
    PROGRESS_PATH so both can be redirected together in tests."""
    override = os.environ.get("JAXCODE_STUDY_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "study"


_ATTEMPTS_FIELDS = ["timestamp", "task", "number", "attempt", "passed", "total", "elapsed_s", "solved"]

_warned_no_study = False


def _study_dir_or_none() -> Path | None:
    """The study dir, or None — saying so once if it is not there.

    Skipping silently is right when there simply is no study log. It is wrong
    when the path is misconfigured, because then a solve looks recorded and
    isn't: that happened when a notebook ran without the env file loaded, so
    JAXCODE_STUDY_DIR fell back to a directory that no longer existed and a
    whole problem's attempts vanished without a word. One dim line, once per
    process, is enough to catch that while staying quiet for anyone who has
    no study/ on purpose.
    """
    global _warned_no_study
    study_dir = _default_study_dir()
    if study_dir.is_dir():
        return study_dir
    if not _warned_no_study:
        _warned_no_study = True
        print(f"  {_DIM}(no study log at {study_dir} — attempts are not being "
              f"recorded){_RESET}")
    return None


def log_attempt(task_id: str, passed: int, total: int, elapsed: float | None, solved: bool) -> None:
    """Append one row to study/attempts.csv. Best-effort: a broken or absent
    study/ directory must never break `check()` — this is a personal learning
    log, not part of the judge contract, so any failure here is swallowed."""
    try:
        study_dir = _study_dir_or_none()
        if study_dir is None:
            return

        csv_path = study_dir / "attempts.csv"
        prior_attempts = 0
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                prior_attempts = sum(1 for row in csv.DictReader(f) if row.get("task") == task_id)

        from jax_judge.tasks import TASKS
        number = TASKS.get(task_id, {}).get("number", "")

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_ATTEMPTS_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task": task_id,
                "number": number,
                "attempt": prior_attempts + 1,
                "passed": passed,
                "total": total,
                "elapsed_s": f"{elapsed:.3f}" if elapsed is not None else "",
                "solved": int(solved),
            })
    except Exception:
        pass


_AID_FIELDS = ["timestamp", "task", "kind"]


def log_aid(task_id: str, kind: str) -> None:
    """Append one row to study/aid.csv when you reach for hint() or solution().

    Without this, needing help leaves no trace: a task solved on the first
    check() after reading the reference looks identical to one solved cold,
    which inverts the very signal worth acting on. Same best-effort contract as
    log_attempt() — never let the study log break the judge.
    """
    try:
        study_dir = _study_dir_or_none()
        if study_dir is None:
            return

        csv_path = study_dir / "aid.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_AID_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task": task_id,
                "kind": kind,
            })
    except Exception:
        pass


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
