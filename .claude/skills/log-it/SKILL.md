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
   cd jaxcode && python scripts/list_edited_notebooks.py
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

2b. **Run anything you are about to assert.** Before a claim about API
   behaviour goes into a row, execute it — the venv is right there:

   ```bash
   .venv/bin/python -c "import jax, jax.numpy as jnp; from flax import nnx; ..."
   ```

   This is not optional care, it is the point. A wrong line in a mistakes log
   is worse than a missing one, because the user will *believe* it — this log
   is what they revise from. It has already happened: a row once asserted
   "`.T` / `@` don't pass through an `nnx.Param`", which three lines of Python
   would have refuted in ten seconds. It shipped, and had to be corrected
   later.

   Two things this catches that reading cannot:
   - The user's own `# !!!` note may be **wrong**. It is evidence of what they
     believed, not of what is true. When it is wrong, saying so *is* the
     finding — that is the most valuable row you can write.
   - Library versions drift. Check the installed one (`flax.__version__`)
     rather than what you remember; deprecations (`.value` → `[...]`) are
     exactly the kind of thing a study log should be right about.

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

4. **Distill each attempt into a mistake row.** One row per *distinct root
   cause*. **Every `# !!!` note earns a row** — those notes are the whole
   reason this log exists, and whether one was strictly a *bug* is not yours
   to judge. Two things follow that are easy to get wrong:
   - A note on code that **passed** still counts. "This version works but is
     locked to 2 dims", "`[..., 0]` is silent, `.squeeze(-1)` is not" — the
     row records what the earlier version got wrong and what the later one
     does better. A refactor the user made *after* going green is a finding,
     not an absence of one.
   - The only reason to merge two notes is the **same root cause in the same
     sitting** (e.g. two `# !!!` markers about one bad index). Different
     causes stay separate however small they look.

   For each row:
   - `tag`: short kebab-case slug, unique per root cause, e.g.
     `rank-hardcode`, `truthy-array-mask`, `mask-after-softmax`,
     `einsum-fixed-rank`. **Check the earlier rounds' tags from step 3b
     first.** If this is the same root cause as a previous round, reuse that
     exact tag — that is what surfaces it as a repeat, and a repeat is the
     single most important thing this log can tell you. Only mint a new tag
     for a genuinely different cause. Tags are also worth reusing across
     tasks when the pattern is the same.
     **Split by what you need to KNOW, not by how badly it went.** Two rows
     about the same API belong apart when one is about constructing it and
     the other about using it (`nnx-param-api` vs `nnx-param-unwrap`); they
     belong together when they are the same gap seen twice. Whether one was
     a crash and the other only a wrong belief in a comment is *not* the
     dividing line.
   **What matters, in order — this is the user's own ranking:**
   1. `fix` — **the correct code.** This is what gets re-read. Everything
      else is context for it.
   2. `what_went_wrong` — the wrong *code* or the wrong *idea*. Concrete.
   3. Nothing else. **The process of trying is noise: cut it.** "写了 try1
      就卡住了，没提交过任何一版就去看 hint（3 次）和参考解" is a row saying
      nothing usable. No try-numbers, no "第一次 check() 就 6/6", no counting
      hints, no "自己批注" attributions. Those facts already live in
      `attempts.csv` / `aid.csv` / `tries.csv`, and a dashboard reads them
      from there. A mistake row is for the *content* of the mistake.

   - `what_went_wrong`: one sentence, and it must name the actual wrong code
     or wrong belief — quote the offending expression where you can
     (`` `logits[targets]` —— 以为是每行取目标那一列 ``). If you cannot say
     what was wrong without narrating how the sitting went, you have not
     found the mistake yet. For a `no-approach` row this means naming the
     **knowledge that was missing**, not the fact of being stuck: "缺的是 jit
     的核心约束：shape 必须在 trace 期就确定" — that is a finding; "写了 try1
     就卡住了" is not.
   - `fix`: **the most important field in the row — not one line.** The user
     re-reads this to answer "对的代码怎么写", so it must be copy-pasteable.
     Show the wrong form and the right form together, marked ❌ / ✅, with the
     real error name or the real wrong output on the ❌ line — a fix is much
     easier to trust when you can see what the failure looked like. Then one
     sentence that generalises past this problem:

     ~~~
     <one line of lead-in prose, ending in a colon>

     ```python
     # the code, with the wrong form and the right form side by side
     # inline comments carry the ✅ / ⚠️ marks
     ```

     **总结**：the one sentence you would want in your head at the
     keyboard next time.
     ~~~

     Newlines are fine — `build_study.py` indents continuation lines into the
     bullet, so fenced blocks render. Keep the code SHORT and real: paste
     what you actually ran, not a sketch. A fix that is one line of prose is
     acceptable only when the mistake genuinely has no code to show.

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
     it is new). Carry it across TASKS too: a tag reused on a different
     problem is the same habit, and its `first_seen` should still say when
     that habit first bit — that is what the Recurring patterns table reads.
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

6c. **Write the takeaway for any tag that just became recurring.**
   `study/patterns.csv` (`tag,summary,updated_at`) holds one takeaway per tag,
   and the Recurring patterns section renders it. A tag with no summary shows
   a 📝 prompt instead — that prompt is addressed to you, so clear it.

   After step 5, check which tags now total ≥ 2 across all rows (that is the
   threshold `build_study.py` uses; `no-approach` is excluded as a state
   rather than a cause). If such a tag has no `patterns.csv` row, write one.
   Update the existing summary instead when a new occurrence sharpens it.

   **The summary is a THIRD piece of writing** — not a copy of either row's
   `fix`. Each `fix` says what to do in that problem; the summary says what
   the two occurrences *together* mean, in a form that applies to a problem
   not yet seen. Same shape as a `fix` (prose, optional code, `**总结**`), but
   aimed one level up:

   > `reduce-drops-axis` — not "softmax needs keepdims" and not "layernorm
   > needs keepdims", but: *a reduce whose result gets broadcast back against
   > x always needs keepdims — ask whether it is about to be subtracted from
   > or divided into x.*

   A summary that only restates the tag name in a full sentence is worse than
   the 📝 prompt, because it looks done.

7. **Regenerate the dashboard.** Run:
   ```bash
   cd jaxcode && python scripts/build_study.py
   ```
   (adjust the `cd` if already in that directory). This reads every study CSV
   and rewrites `study/MISTAKES.md` — never edit that file directly.

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
  comments/history — if the user solved it clean on try 1 with no `# !!!`
  notes, say so and log nothing.
- But don't *drop* one either. A `# !!!` note is evidence by definition, and
  "that wasn't really a bug, it already passed" is not a reason to skip it —
  that call has been made wrongly before, and the notes it discarded were the
  best material of the sitting. When in doubt, log the row; a duplicate root
  cause merges by tag anyway, so the cost of over-logging is near zero and the
  cost of under-logging is a lost lesson.
- Don't touch `study/attempts.csv` or `study/problems.csv` — those belong to
  `jax_judge.progress.log_attempt()` and `scripts/build_study.py
  --init-problems` respectively. This skill owns `study/tries.csv`,
  `study/mistakes.csv` and `study/patterns.csv` outright, and may only
  *backfill gaps* in `study/aid.csv`, which `log_aid()` owns.
- Don't rewrite rows from earlier sittings on your own initiative just to
  match a format introduced later. Improve a row when you are already there
  for a real reason — a new occurrence, a claim you just found to be wrong.
  **If the user asks for a sweep, do it** (they did on 2026-08-16); the point
  is not to touch old rows uninvited, and a sweep is still bound by step 2b:
  every code sample gets run before it lands.
- Don't delete or rewrite the commented-out attempts in the notebook itself
  — they're the source material and the user's own record; this skill only
  reads them.
