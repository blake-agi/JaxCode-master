#!/usr/bin/env python3
"""Build and refresh the personal study log in ../study/.

Four CSVs, four owners:
    attempts.csv   written by jax_judge.progress.log_attempt() on every check() —
                    a machine count of check() calls, NOT how many times you
                    actually rewrote the solution (re-running check() without
                    changing code still counts; editing without checking doesn't)
    problems.csv   written once by --init-problems, parsed off the original
                    TorchCode README (frequency/difficulty/key concepts that
                    the JAX port has no reason to duplicate elsewhere)
    mistakes.csv   written by the log-mistake skill, one row per distinct
                    error pattern you've hit
    tries.csv      written by the log-mistake skill, one row per task holding
                    the CURRENT try count read off your own try1/try2/... notes
                    in the notebook — the number that actually reflects how
                    many times you rewrote it, since you're the one who wrote
                    those notes. Overwritten each run, not appended to.

MISTAKES.md is always a generated view over the four — never edit it by hand,
re-run this script instead. It also cross-checks data/progress.json directly
for the Solved column, since that file has been tracking solves since before
attempts.csv existed and remains the single source of truth for "did I ever
solve this," independent of the per-attempt timeline.

    python scripts/build_study.py --init-problems         # (re)build problems.csv
    python scripts/build_study.py --backfill-from-progress  # seed attempts.csv
                                                              # from pre-existing
                                                              # progress.json solves
    python scripts/build_study.py                          # regenerate MISTAKES.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402
from jax_judge.progress import PROGRESS_PATH  # noqa: E402
from jax_judge.tasks import TASKS  # noqa: E402

ORIGINAL = Path(os.environ.get("TORCHCODE_ORIGINAL", ROOT.parent / "TorchCode-master-original"))
STUDY_DIR = Path(os.environ.get("JAXCODE_STUDY_DIR", ROOT.parent / "study"))
# Optional override so your working notebooks can live outside this checkout
# (e.g. in a private practice repo). Unset — as in any fresh clone — this is
# exactly ROOT/notebooks, so nothing changes for anyone else.
NOTEBOOKS_DIR = Path(os.environ.get("JAXCODE_NOTEBOOKS_DIR") or ROOT / "notebooks")

PROBLEMS_CSV = STUDY_DIR / "problems.csv"
ATTEMPTS_CSV = STUDY_DIR / "attempts.csv"
MISTAKES_CSV = STUDY_DIR / "mistakes.csv"
TRIES_CSV = STUDY_DIR / "tries.csv"
AID_CSV = STUDY_DIR / "aid.csv"
MISTAKES_MD = STUDY_DIR / "MISTAKES.md"

_PROBLEMS_FIELDS = ["number", "task", "function_name", "difficulty", "frequency", "key_concepts"]
_ATTEMPTS_FIELDS = ["timestamp", "task", "number", "attempt", "passed", "total", "elapsed_s", "solved"]
_ROW_RE = re.compile(r"^\|\s*\d+\s*\|")
_BADGE_RE = re.compile(r"badge/(\w+)-")


def _load_progress() -> dict[str, dict]:
    path = Path(PROGRESS_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _original_numbers() -> dict[str, str]:
    """task_id -> notebook number, off the original's template filenames.

    Same source of truth as check_alignment.py / check_signatures.py, so all
    three gates + this log agree on what a task is called.
    """
    nums: dict[str, str] = {}
    for p in sorted((ORIGINAL / "templates").glob("*.ipynb")):
        if p.stem.startswith("00_"):
            continue
        n, stem = p.stem.split("_", 1)
        nums[stem] = n
    if "multihead_attention" in nums:
        nums["mha"] = nums.pop("multihead_attention")
    return nums


def init_problems() -> int:
    if not ORIGINAL.exists():
        print(f"\n{YELLOW}⊘ Can't find the original repo at {ORIGINAL}{RESET}")
        print(f"  {DIM}Set TORCHCODE_ORIGINAL to point at it.{RESET}\n")
        return 1

    by_number = {n: tid for tid, n in _original_numbers().items()}
    readme = (ORIGINAL / "README.md").read_text().splitlines()

    rows: list[dict[str, str]] = []
    for line in readme:
        if not _ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().split("|")][1:-1]
        if len(cells) < 6:
            continue
        number, _link, _sig, diff_badge, freq, concepts = cells[:6]
        number = number.zfill(2)
        tid = by_number.get(number)
        if tid is None:
            continue
        m = _BADGE_RE.search(diff_badge)
        rows.append({
            "number": number,
            "task": tid,
            "function_name": TASKS.get(tid, {}).get("function_name", ""),
            "difficulty": m.group(1) if m else "",
            "frequency": freq,
            "key_concepts": concepts,
        })

    if not rows:
        print(f"{RED}Parsed 0 rows out of the original README's problem table — "
              f"the table format may have changed.{RESET}")
        return 1

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["number"])
    with open(PROBLEMS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PROBLEMS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    freq_counts = defaultdict(int)
    for r in rows:
        freq_counts[r["frequency"]] += 1
    print(f"\n{GREEN}✓ wrote {len(rows)} rows to {PROBLEMS_CSV}{RESET}")
    print(f"  {DIM}frequency: " + ", ".join(f"{k} {v}" for k, v in sorted(freq_counts.items())) + f"{RESET}\n")
    return 0


def backfill_from_progress() -> int:
    """One-time import: seed attempts.csv with tasks solved/attempted before
    log_attempt() existed. data/progress.json only ever kept the aggregate
    (attempts count, best_time, solved_at) — not a per-attempt timeline — so
    each backfilled row is that aggregate, not real per-attempt history.
    Idempotent: skips any task that already has a row in attempts.csv.
    """
    progress = _load_progress()
    if not progress:
        print(f"{YELLOW}No {PROGRESS_PATH} to backfill from.{RESET}")
        return 0

    existing = {r["task"] for r in _read_csv(ATTEMPTS_CSV)}
    nums = {tid: n for tid, n in _original_numbers().items()}

    new_rows = []
    for tid, entry in progress.items():
        if tid in existing:
            continue
        solved = entry.get("status") == "solved"
        total = len(TASKS.get(tid, {}).get("tests", []))
        new_rows.append({
            "timestamp": entry.get("solved_at", ""),
            "task": tid,
            "number": nums.get(tid, TASKS.get(tid, {}).get("number", "")),
            "attempt": entry.get("attempts", 1),
            "passed": total if solved else "",
            "total": total,
            "elapsed_s": f"{entry['best_time']:.3f}" if entry.get("best_time") is not None else "",
            "solved": int(solved),
        })

    if not new_rows:
        print(f"{DIM}Nothing to backfill — every task in progress.json already has attempts.csv rows.{RESET}")
        return 0

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ATTEMPTS_CSV.exists()
    with open(ATTEMPTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ATTEMPTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\n{GREEN}✓ backfilled {len(new_rows)} task(s) into {ATTEMPTS_CSV}{RESET}")
    print(f"  {DIM}each row is the aggregate progress.json knew (attempts count, best "
          f"time), not real per-attempt history — that only starts now.{RESET}\n")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _number_for(tid: str, p: dict[str, str]) -> str:
    """problems.csv has numbers only for the 41 ported problems (parsed off the
    original README). Added (b_*) problems have no original, so fall back to
    the number the JAX task dict itself carries."""
    return p.get("number") or TASKS.get(tid, {}).get("number", "")


def _notebook_link(tid: str, number: str) -> str | None:
    """Relative link (from STUDY_DIR) to the user's own notebook, if it exists
    on disk under the number_task naming convention every notebook uses."""
    if not number:
        return None
    nb = NOTEBOOKS_DIR / f"{number}_{tid}.ipynb"
    if not nb.exists():
        return None
    return os.path.relpath(nb, STUDY_DIR)


def _aid_by_task(aid: list[dict[str, str]], attempts: list[dict[str, str]]) -> dict[str, str]:
    """task -> aid marker, counting ONLY help taken before the first solve.

    Reading the reference after you already solved a problem is study, not a
    crutch, and must not brand the row as needing help — so anything dated
    after the earliest solved attempt is ignored.
    """
    first_solve: dict[str, str] = {}
    for a in attempts:
        if a.get("solved") == "1" and a.get("timestamp"):
            ts = a["timestamp"]
            cur = first_solve.get(a["task"])
            if cur is None or ts < cur:
                first_solve[a["task"]] = ts

    out: dict[str, str] = {}
    for row in aid:
        task, ts = row.get("task", ""), row.get("timestamp", "")
        solved_at = first_solve.get(task)
        if solved_at and ts and ts > solved_at:
            continue                      # help taken after solving — not a crutch
        # 📖 outranks 💡: reading the whole answer is the stronger signal.
        if row.get("kind") == "solution":
            out[task] = "📖"
        elif out.get(task) != "📖":
            out[task] = "💡"
    return out


def build_dashboard() -> int:
    problems = {r["task"]: r for r in _read_csv(PROBLEMS_CSV)}
    attempts = _read_csv(ATTEMPTS_CSV)
    mistakes = _read_csv(MISTAKES_CSV)
    tries_rows = {r["task"]: r for r in _read_csv(TRIES_CSV)}
    aid = _aid_by_task(_read_csv(AID_CSV), attempts)
    progress = _load_progress()

    # problems.csv only has the 41 ported problems (parsed off the original
    # README, which obviously doesn't know about JAX-only additions). Pull
    # the 11 added (b_*) problems straight from the task dicts so the board
    # always covers all 52, not just whichever ones happen to have activity.
    added = {tid for tid, t in TASKS.items() if str(t.get("number", "")).startswith("b_")}

    attempts_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for a in attempts:
        attempts_by_task[a["task"]].append(a)

    mistakes_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for m in mistakes:
        mistakes_by_task[m["task"]].append(m)

    tasks = sorted(
        set(problems) | added | set(attempts_by_task) | set(mistakes_by_task) | set(progress),
        # "01".."41" (ported) sort before "b_01".."b_11" (added) since digits
        # precede letters in ASCII — no special-casing needed.
        key=lambda t: _number_for(t, problems.get(t, {})) or "zz",
    )

    # A task gets a Details entry once there's anything to show about it —
    # a mistake, an attempt, or a progress.json record. A never-touched task
    # (still the blank template) gets no entry, and so no link in the table.
    detailed = [
        tid for tid in tasks
        if mistakes_by_task.get(tid) or attempts_by_task.get(tid) or progress.get(tid)
    ]
    detailed_set = set(detailed)

    lines: list[str] = []
    lines.append("# Mistakes")
    lines.append("")
    lines.append("Generated by `scripts/build_study.py` from `attempts.csv` / `problems.csv` / "
                  "`mistakes.csv` / `tries.csv` / `aid.csv`. Do not edit by hand — re-run the "
                  "script instead.")
    lines.append("")
    lines.append("Attempts = `check()` calls · Try = your own try1..tryN notes · "
                  "Aid = 💡 hint or 📖 solution read **before** first solving.")
    lines.append("")
    lines.append("| # | Task | Tag | Difficulty | Freq | Attempts | Try | Aid | Solved | Mistakes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tid in tasks:
        p = problems.get(tid, {})
        number = _number_for(tid, p)
        tag = TASKS.get(tid, {}).get("category", "")
        difficulty = p.get("difficulty") or TASKS.get(tid, {}).get("difficulty", "")
        # blank means "no original to compare to" (an added problem), not
        # "unknown" — mark it explicitly rather than leaving a silent gap.
        frequency = p.get("frequency") if tid in problems else "—"
        attempt_rows = attempts_by_task.get(tid, [])
        # progress.json is the authoritative "did I ever solve this" — it has
        # been tracking since before attempts.csv existed, so trust it first.
        is_solved = progress.get(tid, {}).get("status") == "solved" or any(
            a.get("solved") == "1" for a in attempt_rows
        )
        solved = "✅" if is_solved else ("🔧" if attempt_rows or progress.get(tid) else "⏳")
        n_attempts = max(len(attempt_rows), progress.get(tid, {}).get("attempts", 0))
        n_mistakes = len(mistakes_by_task.get(tid, []))
        # "Try" is your own try1/try2/... note count from the notebook, kept
        # by the log-mistake skill — separate from "Attempts" (check() calls)
        # since the two count genuinely different things. Blank until logged.
        try_count = tries_rows.get(tid, {}).get("tries", "")
        # Task name jumps to its Details entry, when it has one — the
        # notebook link itself only lives in Details, not here.
        task_cell = f"[`{tid}`](#mistake-{tid})" if tid in detailed_set else f"`{tid}`"
        lines.append(
            f"| {number or '??'} | {task_cell} | {tag} | {difficulty} | "
            f"{frequency} | {n_attempts} | {try_count} | {aid.get(tid, '')} | "
            f"{solved} | {n_mistakes} |"
        )

    if detailed:
        lines.append("")
        lines.append("## Details")
        for tid in detailed:
            lines.append("")
            number = _number_for(tid, problems.get(tid, {}))
            nb_link = _notebook_link(tid, number)
            lines.append(f'<a id="mistake-{tid}"></a>')
            heading = f"### `{tid}`"
            if nb_link:
                heading += f" — [my notebook]({nb_link})"
            lines.append(heading)
            task_mistakes = mistakes_by_task.get(tid, [])
            if not task_mistakes:
                lines.append("_No mistakes logged yet._")
                continue
            for m in sorted(task_mistakes, key=lambda r: -int(r.get("times_seen", 1))):
                seen = m.get("times_seen", "1")
                lines.append(f"- **{m.get('tag', '')}** ({seen}x) — {m.get('what_went_wrong', '')}")
                if m.get("fix"):
                    lines.append(f"  - fix: {m['fix']}")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    MISTAKES_MD.write_text("\n".join(lines) + "\n")
    print(f"\n{GREEN}✓ wrote {MISTAKES_MD}{RESET}")
    print(f"  {DIM}{len(tasks)} tasks, {len(mistakes)} mistake rows, {len(attempts)} attempt rows{RESET}\n")

    _update_readme(tasks, problems, progress, attempts_by_task)
    return 0


README_BEGIN = "<!-- BEGIN PROGRESS -->"
README_END = "<!-- END PROGRESS -->"

# Interview frequency, most-asked first — this is the order worth practising in.
_FREQ_ORDER = ["🔥", "⭐", "💡", "—", ""]
_FREQ_LABEL = {
    "🔥": "🔥 Very likely in interviews",
    "⭐": "⭐ Commonly asked",
    "💡": "💡 Emerging / differentiator",
    "—": "— Added (no PyTorch counterpart, so no frequency rating)",
}


def _update_readme(tasks, problems, progress, attempts_by_task) -> None:
    """Refresh the progress block in the practice repo's README, in place.

    Only touches a README that already carries the BEGIN/END markers, so this
    silently does nothing in a plain checkout — same spirit as every other
    path here being an optional override.
    """
    readme = STUDY_DIR.parent / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    if README_BEGIN not in text or README_END not in text:
        return

    def solved(tid: str) -> bool:
        return progress.get(tid, {}).get("status") == "solved" or any(
            a.get("solved") == "1" for a in attempts_by_task.get(tid, [])
        )

    by_freq: dict[str, list[str]] = defaultdict(list)
    for tid in tasks:
        p = problems.get(tid, {})
        by_freq[p.get("frequency") if tid in problems else "—"].append(tid)

    done, total = sum(1 for t in tasks if solved(t)), len(tasks)
    out = [README_BEGIN, ""]
    out.append(f"**{done} / {total} solved.** Grouped by how often the problem comes up in "
               "interviews — work down from the top. Full write-ups of what I got wrong are "
               "in [study/MISTAKES.md](study/MISTAKES.md).")
    out.append("")
    for freq in _FREQ_ORDER:
        group = by_freq.get(freq)
        if not group:
            continue
        g_done = sum(1 for t in group if solved(t))
        out.append(f"### {_FREQ_LABEL.get(freq, freq)} — {g_done}/{len(group)}")
        out.append("")
        for tid in group:
            p = problems.get(tid, {})
            num = _number_for(tid, p)
            diff = p.get("difficulty") or TASKS.get(tid, {}).get("difficulty", "")
            mark = "x" if solved(tid) else " "
            nb = NOTEBOOKS_DIR / f"{num}_{tid}.ipynb"
            name = (f"[`{tid}`]({os.path.relpath(nb, readme.parent)})"
                    if nb.exists() else f"`{tid}`")
            out.append(f"- [{mark}] `{num}` {name} — {diff}")
        out.append("")
    out.append(README_END)

    head, _, rest = text.partition(README_BEGIN)
    _, _, tail = rest.partition(README_END)
    readme.write_text(head + "\n".join(out) + tail)
    print(f"{GREEN}✓ refreshed the progress block in {readme}{RESET}")
    print(f"  {DIM}{done}/{total} solved{RESET}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-problems", action="store_true",
                     help="(re)build problems.csv from the original README")
    ap.add_argument("--backfill-from-progress", action="store_true",
                     help="seed attempts.csv with tasks solved before log_attempt() existed")
    args = ap.parse_args()
    if args.init_problems:
        return init_problems()
    if args.backfill_from_progress:
        return backfill_from_progress()
    return build_dashboard()


if __name__ == "__main__":
    raise SystemExit(main())
