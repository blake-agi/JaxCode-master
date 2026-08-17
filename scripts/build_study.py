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
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _paths import notebooks_dir, notebooks_dir_problem, original_repo  # noqa: E402
from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402
from jax_judge.progress import PROGRESS_PATH  # noqa: E402
from jax_judge.tasks import TASKS  # noqa: E402

ORIGINAL = original_repo()
STUDY_DIR = Path(os.environ.get("JAXCODE_STUDY_DIR", ROOT.parent / "study"))
_STUDY_FROM_ENV = bool(os.environ.get("JAXCODE_STUDY_DIR"))
NOTEBOOKS_DIR = notebooks_dir()

PROBLEMS_CSV = STUDY_DIR / "problems.csv"
ATTEMPTS_CSV = STUDY_DIR / "attempts.csv"
MISTAKES_CSV = STUDY_DIR / "mistakes.csv"
TRIES_CSV = STUDY_DIR / "tries.csv"
AID_CSV = STUDY_DIR / "aid.csv"
MISTAKES_MD = STUDY_DIR / "MISTAKES.md"

_PROBLEMS_FIELDS = ["order", "number", "task", "function_name", "difficulty", "frequency",
                    "section", "key_concepts"]
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
    section = ""
    for line in readme:
        # The table is grouped into sections and ordered by difficulty inside
        # each, NOT by problem number — 16 sits third, right after 01 and 02.
        # That sequence is the author's teaching order, so record the position
        # rather than re-deriving an order from difficulty later.
        if line.startswith("### "):
            section = line[4:].split("—")[0].strip()
            continue
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
            "order": f"{len(rows) + 1:02d}",
            "number": number,
            "task": tid,
            "function_name": TASKS.get(tid, {}).get("function_name", ""),
            "difficulty": m.group(1) if m else "",
            "frequency": freq,
            "section": section,
            "key_concepts": concepts,
        })

    if not rows:
        print(f"{RED}Parsed 0 rows out of the original README's problem table — "
              f"the table format may have changed.{RESET}")
        return 1

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
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


ROUND_GAP = timedelta(hours=24)


def _ts(row: dict[str, str]):
    try:
        return datetime.fromisoformat(row["timestamp"])
    except (KeyError, ValueError):
        return None


def rounds_by_task(attempts: list[dict[str, str]]) -> dict[str, list[list[datetime]]]:
    """task -> [[attempt times in round 1], [round 2], ...].

    A round is one sitting. Practising a problem five times in an evening is
    one round, not five — the mistakes made are all the same pass at it. A gap
    of ROUND_GAP or more starts a new one, which is what makes "5 mistakes last
    round, 2 this round" mean improvement rather than just more submissions.

    Derived from timestamps rather than stored, so it cannot drift out of sync
    with what actually happened.
    """
    times: dict[str, list[datetime]] = defaultdict(list)
    for a in attempts:
        t = _ts(a)
        if t is not None:
            times[a["task"]].append(t)

    out: dict[str, list[list[datetime]]] = {}
    for task, ts in times.items():
        ts.sort()
        groups = [[ts[0]]]
        for prev, cur in zip(ts, ts[1:]):
            (groups.append([cur]) if cur - prev >= ROUND_GAP else groups[-1].append(cur))
        out[task] = groups
    return out


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


# Tags that describe a STATE rather than a root cause. "I had no approach" can
# be true of two unrelated problems without there being anything to drill — so
# it stays in the per-task details and the counts, but grouping it as a habit
# would be noise at the top of the file.
NOT_A_PATTERN = {"no-approach"}

_FREQ_RANK = {"🔥": 0, "⭐": 1, "💡": 2}

# Solving after reading the answer doesn't mean you own it, so it pushes a
# problem up the redo queue rather than counting as a clean finish.
_AID_WEIGHT = {"📖": 2, "💡": 1}
_FREQ_WEIGHT = {"🔥": 1.5, "⭐": 1.2}
# Knowledge decays, so an old shaky problem eventually outranks a recent one.
# Saturating rather than unbounded: age should reorder the queue, never let a
# one-mistake problem from last month outrank a five-mistake one from Tuesday.
_AGE_HALF_LIFE_DAYS = 14.0
_AGE_MAX = 2.0

REDO_LEGEND = (
    "**Redo?** 🔴 last round had mistakes and it's high-frequency · 🟡 had "
    "mistakes · ✅ last round was clean · 📝 practised but not logged yet · "
    "⏸ practised within 24h, a new round can't start yet · ⏳ not started."
)


def _redo_signal(*, n_attempt_rounds: int, last_logged_round: int,
                 last_round_mistakes: int, frequency: str,
                 hours_since_last: float | None) -> str:
    """Whether this problem is worth another sitting.

    A clean round means you own it — relu solved with no mistakes needs no
    repetition, however often it comes up. Mistakes last time plus a high
    interview frequency is the combination that earns a redo.
    """
    if n_attempt_rounds == 0:
        return "⏳"
    if n_attempt_rounds > last_logged_round:
        return "📝"                       # the latest sitting was never logged
    if last_round_mistakes == 0:
        return "✅"
    if hours_since_last is not None and hours_since_last < ROUND_GAP.total_seconds() / 3600:
        return "⏸"                        # too soon for the next round to count
    return "🔴" if frequency == "🔥" else "🟡"


def build_dashboard() -> int:
    problems = {r["task"]: r for r in _read_csv(PROBLEMS_CSV)}
    attempts = _read_csv(ATTEMPTS_CSV)
    mistakes = _read_csv(MISTAKES_CSV)
    tries_rows = _read_csv(TRIES_CSV)
    aid = _aid_by_task(_read_csv(AID_CSV), attempts)
    progress = _load_progress()
    rounds = rounds_by_task(attempts)
    now = datetime.now()

    # (task, round) -> tries, and task -> highest round log-it has recorded.
    # tries.csv doubles as the receipt that a sitting was actually logged, so a
    # clean round is distinguishable from one nobody has written up yet.
    tries_by_round = {(r["task"], int(r["round"] or 1)): r.get("tries", "")
                      for r in tries_rows}
    last_logged = defaultdict(int)
    for r in tries_rows:
        last_logged[r["task"]] = max(last_logged[r["task"]], int(r["round"] or 1))

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

    # (task, round) -> mistakes, so the table can show the trend across rounds
    # and the details can lead with the most recent sitting.
    mistakes_by_round: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for m in mistakes:
        mistakes_by_round[(m["task"], int(m.get("round") or 1))].append(m)

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
    lines.append("Attempts = `check()` calls · Try = your own try1..tryN notes in the "
                  "latest round · Aid = 💡 hint or 📖 solution read **before** first "
                  "solving · Mistakes = per round, so R1:5 R2:2 means you improved. "
                  "A round is one sitting; 24h apart starts a new one.")
    lines.append("")
    lines.append(REDO_LEGEND)
    lines.append("")
    lines.append("| # | Task | Tag | Difficulty | Freq | Attempts | Try | Aid | Solved | Mistakes | Redo? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
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

        task_rounds = rounds.get(tid, [])
        n_rounds = len(task_rounds)
        # "R1:5 R2:2" — the whole trajectory, so a drop is visible at a glance.
        per_round = " ".join(
            f"R{i}:{len(mistakes_by_round.get((tid, i), []))}"
            for i in range(1, n_rounds + 1)
        ) or ""
        mistakes_cell = f"[{per_round}](#mistake-{tid})" if per_round and tid in detailed_set else per_round

        last_round_mistakes = len(mistakes_by_round.get((tid, n_rounds), [])) if n_rounds else 0
        hours_since = ((now - task_rounds[-1][-1]).total_seconds() / 3600) if n_rounds else None
        redo = _redo_signal(
            n_attempt_rounds=n_rounds,
            last_logged_round=last_logged.get(tid, 0),
            last_round_mistakes=last_round_mistakes,
            frequency=frequency or "",
            hours_since_last=hours_since,
        )

        # "Try" is your own try1/try2/... note count, for the latest round —
        # separate from "Attempts" (check() calls) since the two count
        # genuinely different things. Blank until logged.
        try_count = tries_by_round.get((tid, n_rounds), "") if n_rounds else ""
        # Task name jumps to its Details entry, when it has one — the
        # notebook link itself only lives in Details, not here.
        task_cell = f"[`{tid}`](#mistake-{tid})" if tid in detailed_set else f"`{tid}`"
        lines.append(
            f"| {number or '??'} | {task_cell} | {tag} | {difficulty} | "
            f"{frequency} | {n_attempts} | {try_count} | {aid.get(tid, '')} | "
            f"{solved} | {mistakes_cell} | {redo} |"
        )

    # --- What to do next -------------------------------------------------
    # Round 1 is a sweep of everything, alternating a JAX fundamental with an
    # algorithm problem. Algorithms go hottest-first, and easiest-first inside a
    # frequency tier, so the cheap high-frequency wins land early.
    done = {t for t in tasks
            if progress.get(t, {}).get("status") == "solved"
            or any(a.get("solved") == "1" for a in attempts_by_task.get(t, []))}

    # Hottest tier first — every 🔥 gets done regardless — and within a tier,
    # the original README's own row order, which already sequences foundations
    # before specialisations.
    algo_todo = sorted(
        (t for t in problems if t not in done),
        key=lambda t: (_FREQ_RANK.get(problems[t].get("frequency", ""), 3),
                       problems[t].get("order", "99")),
    )
    added_todo = sorted(
        (t for t in added if t not in done),
        key=lambda t: TASKS[t].get("number", ""),
    )
    sweep: list[str] = []
    for i in range(max(len(added_todo), len(algo_todo))):
        if i < len(added_todo):
            sweep.append(added_todo[i])
        if i < len(algo_todo):
            sweep.append(algo_todo[i])

    # Round 2+ is a different question: not "what haven't I done" but "what did
    # I not actually learn". Mistakes last round, weighted up by how often the
    # problem is asked and by whether the answer had to be read.
    queue = []
    for tid in done:
        n = len(rounds.get(tid, []))
        if not n:
            continue
        hrs = (now - rounds[tid][-1][-1]).total_seconds() / 3600
        if hrs < ROUND_GAP.total_seconds() / 3600:
            continue                       # still inside the same sitting
        miss = len(mistakes_by_round.get((tid, n), []))
        freq = problems.get(tid, {}).get("frequency", "")
        # Age lifts a stale problem up the queue without letting it overtake a
        # genuinely shaky recent one: saturating at _AGE_MAX means the oldest
        # possible problem is worth twice a same-day one, no more.
        age = min(1.0 + (hrs / 24) / _AGE_HALF_LIFE_DAYS, _AGE_MAX)
        score = ((miss + _AID_WEIGHT.get(aid.get(tid, ""), 0))
                 * _FREQ_WEIGHT.get(freq, 1.0) * age)
        if score > 0:
            queue.append((score, tid, miss, freq, aid.get(tid, ""), int(hrs // 24)))
    queue.sort(key=lambda r: (-r[0], r[1]))

    if sweep or queue:
        lines.append("")
        lines.append("## Next up")
        if sweep:
            lines.append("")
            lines.append("**Round 1 sweep** — one JAX fundamental, then one algorithm "
                          "problem; hottest first, and within a tier the original "
                          "README's own order, which puts foundations before "
                          "specialisations.")
            lines.append("")
            lines.append(" · ".join(
                f"`{_number_for(t, problems.get(t, {}))} {t}`" for t in sweep[:8]))
        if queue:
            lines.append("")
            lines.append("**Ready for another round** — 24h+ since the last sitting, "
                          "ranked by what stuck least: mistakes, weighted up by how "
                          "often it's asked, whether the answer had to be read, and "
                          "how long ago it was (doubling at "
                          f"{int(_AGE_HALF_LIFE_DAYS)}d, then flat).")
            lines.append("")
            for score, tid, miss, freq, a, days in queue[:6]:
                bits = [f"{miss} mistake" + ("s" if miss != 1 else "")]
                if a == "📖":
                    bits.append("📖 read the solution")
                elif a == "💡":
                    bits.append("💡 needed a hint")
                lines.append(f"- [`{tid}`](#mistake-{tid}) {freq} — {', '.join(bits)}, "
                             f"{days}d ago")

    # Same root cause hit more than once — across rounds of one problem, or
    # across different problems entirely. The per-task Details can't show this:
    # each section only knows its own history, so a pattern spanning softmax and
    # layernorm is invisible there even though it is the most actionable thing
    # in the log. Worth drilling deliberately instead of waiting to meet again.
    by_tag: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for m in mistakes:
        if m.get("tag", "") in NOT_A_PATTERN:
            continue
        by_tag[m.get("tag", "")].append(
            (m["task"], int(m.get("round") or 1), int(m.get("times_seen") or 1))
        )
    recurring = []
    for tag, occ in by_tag.items():
        total = sum(t for _, _, t in occ)
        if total >= 2:
            recurring.append((total, tag, occ))
    recurring.sort(key=lambda r: (-r[0], r[1]))

    if recurring:
        lines.append("")
        lines.append("## Recurring patterns")
        lines.append("")
        lines.append("Same root cause, hit more than once — most-repeated first. A pattern "
                      "spanning two different problems is the strongest signal here: it is "
                      "not that problem's quirk, it's a habit.")
        lines.append("")
        lines.append("| Times | Pattern | Where |")
        lines.append("|---|---|---|")
        for total, tag, occ in recurring:
            per_task: dict[str, list[int]] = defaultdict(list)
            for task, rnd, times in occ:
                per_task[task].extend([rnd] * times)
            where = " · ".join(
                f"[`{t}`](#mistake-{t}) " + ", ".join(f"R{r}" for r in sorted(set(rs)))
                + (f" ×{len(rs)}" if len(rs) > len(set(rs)) else "")
                for t, rs in sorted(per_task.items())
            )
            across = " 🔁" if len(per_task) > 1 else ""
            lines.append(f"| {total}{across} | `{tag}` | {where} |")

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

            n_rounds = max(len(rounds.get(tid, [])), last_logged.get(tid, 0))
            # Newest round first: what you got wrong most recently is what you
            # need, and earlier rounds are context underneath it.
            for rnd in range(n_rounds, 0, -1):
                group = mistakes_by_round.get((tid, rnd), [])
                when = ""
                if rnd <= len(rounds.get(tid, [])):
                    when = f" · {rounds[tid][rnd - 1][0]:%Y-%m-%d}"
                lines.append("")
                lines.append(f"**Round {rnd}**{when} — "
                             + (f"{len(group)} mistake(s)" if group else "clean"))
                for m in sorted(group, key=lambda r: -int(r.get("times_seen", 1) or 1)):
                    tag = m.get("tag", "")
                    seen = m.get("times_seen", "1")
                    # Same tag in an earlier round means it did NOT stick —
                    # worse than a fresh mistake, so say so on the line itself.
                    earlier = [r for r in range(1, rnd)
                               if any(x.get("tag") == tag for x in mistakes_by_round.get((tid, r), []))]
                    again = f" ⚠️ **again** (also R{', R'.join(map(str, earlier))})" if earlier else ""
                    times = f" ({seen}x this round)" if int(seen or 1) > 1 else ""
                    lines.append(f"- **{tag}**{times}{again} — {m.get('what_went_wrong', '')}")
                    if m.get("fix"):
                        # A fix may carry fenced code and a summary, so it can
                        # be several lines. Continuation lines are indented to
                        # column 4 — the content column of the "  - fix:"
                        # bullet — or markdown ends the list and dumps the code
                        # at top level. Blank lines stay truly blank; trailing
                        # spaces would reopen a paragraph in some renderers.
                        fix_lines = m["fix"].split("\n")
                        lines.append(f"  - fix: {fix_lines[0]}")
                        lines.extend(f"    {ln}" if ln.strip() else ""
                                     for ln in fix_lines[1:])

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


def show_rounds(task: str | None) -> int:
    """What round each task is in, and which tags it has already used.

    The log-it skill calls this so the 24h rule lives in exactly one place:
    it needs the round to stamp on new rows, and the earlier tags to decide
    whether today's mistake is a repeat or something new.
    """
    attempts = _read_csv(ATTEMPTS_CSV)
    rounds = rounds_by_task(attempts)
    mistakes = _read_csv(MISTAKES_CSV)
    logged = defaultdict(int)
    for r in _read_csv(TRIES_CSV):
        logged[r["task"]] = max(logged[r["task"]], int(r["round"] or 1))

    tags: dict[tuple[str, int], list[str]] = defaultdict(list)
    for m in mistakes:
        tags[(m["task"], int(m.get("round") or 1))].append(m.get("tag", ""))

    now = datetime.now()
    names = [task] if task else sorted(rounds, key=lambda t: rounds[t][-1][-1], reverse=True)
    for t in names:
        rs = rounds.get(t, [])
        if not rs:
            print(f"{t}: no attempts recorded — round would be 1")
            continue
        cur, last = len(rs), rs[-1][-1]
        hrs = (now - last).total_seconds() / 3600
        state = ("this round is still open — log into it"
                 if hrs < ROUND_GAP.total_seconds() / 3600
                 else f"a new sitting now would be round {cur + 1}")
        print(f"\n{BOLD}{t}{RESET}: current round {cur} "
              f"({len(rs[-1])} attempts, last {hrs:.1f}h ago)")
        print(f"  logged up to round {logged.get(t, 0)} · {state}")
        for i in range(1, cur + 1):
            got = tags.get((t, i), [])
            print(f"  {DIM}R{i}: {', '.join(got) if got else '(clean)'}{RESET}")
    return 0


def _study_data_exists() -> bool:
    """Refuse to run against a directory that holds no study log.

    The failure this catches is silent and convincing. With JAXCODE_STUDY_DIR
    unloaded — a plain terminal that never sourced the env file — STUDY_DIR
    falls back inside the checkout, every CSV reads as absent, and the script
    writes a perfectly formatted board reporting that you have solved nothing.
    The real log is untouched, which is exactly why it goes unnoticed; the only
    trace is a stray study/ directory somewhere you did not mean to write.

    --init-problems is exempt: creating the directory is its whole job.
    """
    if any(p.exists() for p in (PROBLEMS_CSV, ATTEMPTS_CSV, MISTAKES_CSV,
                                TRIES_CSV, AID_CSV)):
        return True

    print(f"{RED}No study log in {STUDY_DIR}{RESET}", file=sys.stderr)
    if not _STUDY_FROM_ENV:
        print(f"  {YELLOW}JAXCODE_STUDY_DIR is unset, so this fell back to the "
              f"in-repo default.{RESET}\n"
              f"  If your log lives elsewhere, the env file was not loaded. "
              f"Point at it and re-run:\n"
              f"    {DIM}export JAXCODE_STUDY_DIR=/path/to/study{RESET}",
              file=sys.stderr)
    print(f"  {DIM}Starting from scratch? "
          f"python scripts/build_study.py --init-problems{RESET}",
          file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-problems", action="store_true",
                     help="(re)build problems.csv from the original README")
    ap.add_argument("--backfill-from-progress", action="store_true",
                     help="seed attempts.csv with tasks solved before log_attempt() existed")
    ap.add_argument("--rounds", nargs="?", const="", metavar="TASK",
                     help="show the current round and past tags (all tasks, or one)")
    args = ap.parse_args()

    # Warn, never fail: NOTEBOOKS_DIR is only consulted for "does this notebook
    # exist" links, so a wrong path costs links in MISTAKES.md and the README
    # progress block — the dashboard itself still builds from the CSVs.
    # require_exists stays off so CI, which legitimately has no notebooks, is
    # not nagged; an unloaded .env is a real mistake and still warns.
    nb_problem = notebooks_dir_problem(require_exists=False)
    if nb_problem:
        print(f"{YELLOW}{nb_problem}{RESET}", file=sys.stderr)
        print(f"  {DIM}(continuing — notebook links will be omitted){RESET}",
              file=sys.stderr)

    if args.init_problems:
        return init_problems()
    if not _study_data_exists():
        return 2
    if args.backfill_from_progress:
        return backfill_from_progress()
    if args.rounds is not None:
        return show_rounds(args.rounds or None)
    return build_dashboard()


if __name__ == "__main__":
    raise SystemExit(main())
