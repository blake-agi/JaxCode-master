---
name: log-it
description: End-of-problem routine — summarize the commented-out try1..tryN attempts in whatever notebook(s) you've been editing into study/mistakes.csv and study/tries.csv, regenerate study/MISTAKES.md, then commit and push the practice repo. Auto-detects which notebook(s) changed — no task name needed. Use after finishing a problem, when the user says things like "log it", "log my mistakes", "log mistakes for attention", or "把 attention 记进错题本".
---

# log-it

Turns the debugging trail you leave behind in a notebook — commented-out
attempts, `# !!!` markers, scratch-cell experiments — into durable rows in
`study/mistakes.csv` and a try count in `study/tries.csv`, then regenerates
`study/MISTAKES.md`.

The try count is deliberately separate from `study/attempts.csv`'s "Attempts"
column. Attempts counts `check()` calls — a machine fact that can over- or
under-count (re-running check() without changing code inflates it; editing
without checking doesn't move it at all). Try count instead comes from your
own `# try1` / `# try2` / ... notes, which is the number that actually means
"how many times I rewrote this" — more valuable precisely because you wrote
it, not a program.

This is the ONLY place that reads notebook comments and writes an interpreted
summary. Nothing else in the repo does this automatically — `jax_judge.check()`
only logs the bare fact of an attempt (pass/fail/time) to `study/attempts.csv`,
by design. This skill is where the *insight* gets extracted, and only when you
explicitly trigger it.

## Input

None required. The user should not need to name a task — figure out what
they were practicing yourself. If they DID name one explicitly (e.g. "log
mistakes for attention"), use that instead of auto-detecting and skip step 1.

## Steps

1. **Auto-detect the task(s).** Run:
   ```bash
   cd JaxCode-master && python scripts/list_edited_notebooks.py
   ```
   This prints task ids for every notebook whose content differs from its
   `notebooks/_pristine/` baseline — i.e. every problem with real code in
   it — most recently modified first. It correctly maps both `NN_name.ipynb`
   (ported problems) and `b_NN_name.ipynb` (added problems) to their task id.
   - **Nothing printed** → nothing has been edited since the last refresh;
     say so and stop, there's nothing to log.
   - **One task id** → use it, no need to ask.
   - **Several task ids** → process ALL of them in one pass. Don't ask the
     user to pick one — summarize each and report which ones you covered at
     the end. (If a notebook was already fully captured in a prior run,
     re-processing it is harmless: mistake rows dedupe by tag, so you'll
     just find nothing new for it — say so rather than padding the log.)
   - Each task id maps to `jax_judge/tasks/<task_id>.py` and
     `notebooks/<number>_<task_id>.ipynb` for the steps below.

2. **Read the notebook.** Find the implementation cell (marked
   `# ✏️ YOUR IMPLEMENTATION HERE` or similar). Extract:
   - Every commented-out attempt block (`# try1`, `# try2`, ... or any
     earlier version of the function/class that's been commented out).
   - Every `# !!!` (or similar self-flagged) annotation and the line it's on.
   - The final, live (uncommented) implementation.
   - Any scratch cells that show debugging output relevant to an attempt.
   - **Notes admitting help** — `no idea`, `looked at the answer`, `gave up`,
     `copied`, or a try block whose whole body is a note rather than code.
     Treat these as an *aid* signal (step 6b), not only as mistake material.

3. **Read the reference.** Open `jax_judge/tasks/<task_id>.py` and look at
   `TASK["solution"]` for how the canonical implementation differs from the
   user's final version, in case a real gap survived even in the passing
   submission (e.g. missing dtype-safety, a less general einsum pattern).

3b. **Find the round, and what was already tried.** Run:
   ```bash
   python scripts/build_study.py --rounds <task_id>
   ```
   It reports the current round, how long ago the last attempt was, and every
   tag used in each previous round. A *round* is one sitting: several `check()`
   calls in an evening are one round, and a gap of 24h or more starts the next.
   The rule lives in that script — never re-derive it by eyeballing timestamps.

   This is what makes the log worth keeping: "R1:5 R2:2" says the second pass
   went better, which a running total cannot. So the round matters as much as
   the mistake.

4. **Distill each attempt into a mistake row.** One row per *distinct kind*
   of error — collapse near-duplicates within the same sitting (e.g. two
   `# !!!` markers about the same root cause are one row, not two). For each:
   - `tag`: short kebab-case slug, unique per root cause, e.g.
     `rank-hardcode`, `truthy-array-mask`, `mask-after-softmax`,
     `einsum-fixed-rank`. **Check the earlier rounds' tags from step 3b
     first.** If this is the same root cause as a previous round, reuse that
     exact tag — that is what surfaces it as a repeat, and a repeat is the
     single most important thing this log can tell you. Only mint a new tag
     for a genuinely different cause. Tags are also worth reusing across
     tasks when the pattern is the same.
   - `what_went_wrong`: one sentence, concrete, in the user's own terms
     where possible (their comment is often already the best phrasing).
   - `fix`: one sentence, what the corrected code actually does differently.

5. **Merge into `study/mistakes.csv`.** Fields:
   `task,round,tag,what_went_wrong,fix,first_seen,times_seen`.
   Read the existing CSV (via the `csv` module logic, or by hand — it's
   small). For each distilled mistake, using the round from step 3b:
   - If a row with the same `(task, round, tag)` already exists — i.e. you
     are adding to a sitting you already logged — increment its `times_seen`
     and leave `first_seen` untouched.
   - Otherwise append a new row with that `round`, `times_seen = 1`, and
     `first_seen` = the date the mistake was FIRST made in any round (carry
     it over from the earlier row when the tag is a repeat; today's date when
     it is new).
   **A repeated tag gets its own row in the new round — never edit the old
   round's row.** The history of which round each mistake belongs to is the
   whole point; collapsing them back into one row destroys it.
   Write the file back preserving the header and all untouched rows.
   If `study/` does not exist yet, create it — unlike the automatic
   `log_attempt()` path, this is a user-triggered action, so it's fine to
   create the directory here.

6. **Count tries and record them.** In the same implementation cell, count:
   (number of commented-out attempt blocks) + 1 for the final live
   implementation. A commented final attempt still labelled e.g. `# try4`
   that then also appears live is the SAME attempt — don't count it twice.
   A notebook with zero commented attempts (solved clean) is `tries = 1`.
   Note that on a later round the notebook usually holds only THAT round's
   attempts, since you start from a fresh cell — count what is there.

   Write this to `study/tries.csv` (fields: `task,round,tries,updated_at`,
   `updated_at` = today, `YYYY-MM-DD`), one row per `(task, round)`: replace
   the row if that pair already exists, otherwise append.

   **Always write this row, even when there were no mistakes at all.** The
   dashboard uses tries.csv as the receipt that a sitting was written up, so
   a genuinely clean round shows as ✅ rather than 📝 "not logged yet". A
   clean round is a result worth recording, not an absence of one.

6b. **Backfill aid, only if it's missing.** `study/aid.csv`
   (`timestamp,task,kind` where kind is `hint` or `solution`) is normally
   written automatically by `log_aid()` whenever you call `hint()` or
   `solution()`. The skill only fills gaps: help taken outside the notebook,
   or before that instrumentation existed, survives *only* as a comment.
   So if step 2 found a note admitting help and `aid.csv` has no row for that
   task, append one — `kind` = `solution` if the note mentions the answer or
   reference, else `hint`; timestamp from the attempt/solve evidence, dated
   **before** the first solve or the dashboard will (correctly) ignore it.
   If `aid.csv` already covers the task, change nothing — the machine record
   wins.

7. **Regenerate the dashboard.** Run:
   ```bash
   cd JaxCode-master && python scripts/build_study.py
   ```
   (adjust the `cd` if already in that directory). This reads all four
   CSVs and rewrites `study/MISTAKES.md` — never edit that file directly.

8. **Commit and push the practice repo.** This is the end of the routine, so
   the work lands without a separate ask. One commit covers both the notebook
   and the study log when they share a repo.

   Find the repo rather than assuming a path — it is wherever the study log
   lives, which the judge already knows:
   ```bash
   python -c "from jax_judge.progress import _default_study_dir as d; print(d())"
   cd "$(git -C "<that path>" rev-parse --show-toplevel)" && git add -A && git status --short
   ```
   If that directory is not inside a git repository, skip this step and say so
   — do not run `git init` uninvited.

   Review what's staged before committing — confirm nothing unexpected crept
   in (no `.venv`, no `_pristine/`, no `_solutions/`, no stray large files).
   Then commit with a message naming the task(s) and what was learned, and
   push to `origin`'s current branch. If the push fails (offline, auth), say
   so plainly and leave the commit in place — do not retry in a loop.

   Only the practice repo is auto-committed. Changes to the JaxCode repo
   (tooling, this skill) are a separate concern — mention them and let the
   user decide.

9. **Report back.** Lead with the round and how it compares: "round 2, 2
   mistakes, down from 5". Then the mistakes themselves. Keep it short — a few
   bullet points, not the whole CSV.

   **Call out repeats explicitly.** A tag that also appeared in an earlier
   round means the fix did not stick, which is worse than a fresh mistake and
   is the thing most worth saying out loud. Do not bury it in a list.

   Finish with the try count and the commit.

## What NOT to do

- Don't run this automatically on every `check()` call — the user explicitly
  does not want per-submit summarization. It only runs when triggered.
- Don't invent a mistake that isn't actually evidenced in the notebook's
  comments/history — if the user solved it clean on try 1, say so and log
  nothing.
- Don't touch `study/attempts.csv` or `study/problems.csv` — those belong to
  `jax_judge.progress.log_attempt()` and `scripts/build_study.py
  --init-problems` respectively. This skill owns `study/tries.csv` and
  `study/mistakes.csv` outright, and may only *backfill gaps* in
  `study/aid.csv`, which `log_aid()` owns.
- Don't delete or rewrite the commented-out attempts in the notebook itself
  — they're the source material and the user's own record; this skill only
  reads them.
