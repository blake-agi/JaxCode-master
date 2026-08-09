#!/usr/bin/env python3
"""Generate every template and solution notebook from the TASK definitions.

The task files under jax_judge/tasks/ are the single source of truth: they carry
the problem statement, the starter stub, the reference solution and the tests.
This script renders them into notebooks, so nothing is maintained twice.

    python scripts/generate_notebooks.py            # regenerate everything
    python scripts/generate_notebooks.py --check    # fail if out of date (CI)

Notebooks are numbered by curriculum order (category, then order within it), so
adding a task renumbers the ones after it. That is intentional — the numbering
is derived, never hand-assigned.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_judge.tasks import list_tasks  # noqa: E402

TEMPLATES_DIR = ROOT / "templates"
SOLUTIONS_DIR = ROOT / "solutions"

# Used only to build the "Open in Colab" links. Point this at your own fork.
REPO = os.environ.get("JAXCODE_REPO", "YOUR-GITHUB-USERNAME/JAXCode")
BRANCH = os.environ.get("JAXCODE_BRANCH", "master")

DIFF_EMOJI = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}

INSTALL_CELL = """\
# Install jax-judge in Colab (no-op in JupyterLab/Docker)
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q jax-judge flax')
except ImportError:
    pass
"""


def _colab_url(rel_path: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel_path}"


def _badge(rel_path: str) -> str:
    return (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({_colab_url(rel_path)})"
    )


def _lines(source: str) -> list[str]:
    """nbformat wants every line but the last to keep its trailing newline."""
    parts = source.rstrip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + parts[-1:]


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def _notebook(cells: list[dict]) -> dict:
    # nbformat >= 4.5 requires a cell id. Derive it from the position so
    # regenerating an unchanged task produces a byte-identical file.
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _setup_cell(task: dict) -> str:
    """Imports plus a one-line environment check."""
    lines = []
    if task.get("notebook_setup"):
        lines.append(task["notebook_setup"].rstrip("\n"))
        lines.append("")
    lines.append("import jax")
    lines.append("import jax.numpy as jnp")
    if "nnx" in task["stub"] or "nnx" in task["solution"]:
        lines.append("from flax import nnx")
    lines.append("")
    lines.append('print("JAX", jax.__version__, "|", jax.devices())')
    return "\n".join(lines)


def build_template(task_id: str, task: dict, num: int) -> dict:
    rel = f"templates/{num:02d}_{task_id}.ipynb"
    emoji = DIFF_EMOJI.get(task["difficulty"], "⚪")

    header = (
        f"{_badge(rel)}\n\n"
        f"# {emoji} {task['difficulty']}: {task['title']}\n\n"
        f"*{task['category']}*\n"
        f"{task['description'].strip()}\n"
    )

    cells = [
        _markdown(header),
        _code(INSTALL_CELL),
        _code(_setup_cell(task)),
        _code("# ✏️ YOUR IMPLEMENTATION HERE\n\n" + task["stub"].strip()),
    ]

    if task.get("demo"):
        cells.append(
            _code(
                "# 🔍 Scratch cell — poke at your implementation\n"
                + task["demo"].strip()
            )
        )

    cells.append(
        _code(
            "# ✅ SUBMIT — run this cell to check your solution\n"
            "from jax_judge import check, hint, solution\n\n"
            f'check("{task_id}")\n\n'
            f'# hint("{task_id}")      # stuck? nudge without the answer\n'
            f'# solution("{task_id}")  # spoiler: the reference implementation'
        )
    )
    return _notebook(cells)


def build_solution(task_id: str, task: dict, num: int) -> dict:
    rel = f"solutions/{num:02d}_{task_id}_solution.ipynb"
    emoji = DIFF_EMOJI.get(task["difficulty"], "⚪")

    header = (
        f"{_badge(rel)}\n\n"
        f"# {emoji} Solution: {task['title']}\n\n"
        f"*{task['category']} · {task['difficulty']}*\n\n"
        f"Reference implementation. Try it yourself in "
        f"`{num:02d}_{task_id}.ipynb` first.\n\n"
        "---\n"
        f"{task['description'].strip()}\n"
    )

    cells = [
        _markdown(header),
        _code(INSTALL_CELL),
        _code(_setup_cell(task)),
        _code("# ✅ REFERENCE SOLUTION\n\n" + task["solution"].strip()),
    ]

    if task.get("demo"):
        cells.append(_code("# 🔍 Verify\n" + task["demo"].strip()))

    cells.append(
        _code(
            "# Run the judge against the reference solution\n"
            "from jax_judge import check\n\n"
            f'check("{task_id}")'
        )
    )
    return _notebook(cells)


def build_welcome(numbered: list[tuple[int, str, dict]]) -> dict:
    total = len(numbered)
    by_diff: dict[str, int] = {}
    for _, _, task in numbered:
        by_diff[task["difficulty"]] = by_diff.get(task["difficulty"], 0) + 1
    counts = " · ".join(
        f"{DIFF_EMOJI.get(d, '')} {by_diff.get(d, 0)} {d}"
        for d in ["Easy", "Medium", "Hard"]
        if by_diff.get(d)
    )

    lines = [
        _badge("templates/00_welcome.ipynb"),
        "",
        "# ⚡ Welcome to JAXCode",
        "",
        "**Crack the JAX interview.** Implement operators, layers and training",
        "machinery from scratch — in JAX and Flax NNX.",
        "",
        f"**{total} problems** — {counts}",
        "",
        "---",
        "",
        "## How it works",
        "",
        "1. Open a numbered notebook",
        "2. Fill in the function or `nnx.Module` in the ✏️ cell",
        "3. Run the ✅ submit cell — the judge runs the real test suite",
        "",
        "```python",
        "from jax_judge import check, hint, solution, status, reset_progress",
        "",
        'status()             # dashboard of every problem and your progress',
        'check("relu")        # grade your implementation',
        'hint("relu")         # a nudge, not the answer',
        'solution("relu")     # spoiler: the reference implementation',
        "```",
        "",
        "---",
        "",
        "## The problems",
        "",
    ]

    current_cat = None
    for num, task_id, task in numbered:
        if task["category"] != current_cat:
            current_cat = task["category"]
            lines += ["", f"### {current_cat}", ""]
            lines.append("| # | Problem | Difficulty | Task id |")
            lines.append("|---|---|---|---|")
        emoji = DIFF_EMOJI.get(task["difficulty"], "⚪")
        nb = f"{num:02d}_{task_id}.ipynb"
        lines.append(
            f"| {num:02d} | [{task['title']}]({nb}) | {emoji} {task['difficulty']} "
            f"| `{task_id}` |"
        )

    lines += [
        "",
        "---",
        "",
        "## Suggested route",
        "",
        "Work **JAX Fundamentals** first even if you know the ML content cold —",
        "the rest of the problems assume you are fluent with `vmap`, `scan`,",
        "pytrees and explicit PRNG keys. After that, follow the categories in",
        "order or jump to whatever your interview is targeting.",
    ]

    return _notebook(
        [
            _markdown("\n".join(lines)),
            _code(
                "# Your progress dashboard\n"
                "from jax_judge import status\n\n"
                "status()"
            ),
        ]
    )


README = ROOT / "README.md"
README_START = "<!-- PROBLEMS:START -->"
README_END = "<!-- PROBLEMS:END -->"


def build_readme_table(numbered: list[tuple[int, str, dict]]) -> str:
    by_diff: dict[str, int] = {}
    for _, _, task in numbered:
        by_diff[task["difficulty"]] = by_diff.get(task["difficulty"], 0) + 1
    counts = " · ".join(
        f"{DIFF_EMOJI.get(d, '')} {by_diff.get(d, 0)} {d}"
        for d in ["Easy", "Medium", "Hard"]
        if by_diff.get(d)
    )

    lines = [f"**{len(numbered)} problems** — {counts}", ""]

    current_cat = None
    for num, task_id, task in numbered:
        if task["category"] != current_cat:
            current_cat = task["category"]
            n = sum(1 for _, _, t in numbered if t["category"] == current_cat)
            lines += [
                "",
                f"### {current_cat} ({n})",
                "",
                "| # | Problem | Difficulty | `task_id` |",
                "|---|---|---|---|",
            ]
        emoji = DIFF_EMOJI.get(task["difficulty"], "⚪")
        nb = f"templates/{num:02d}_{task_id}.ipynb"
        lines.append(
            f"| {num:02d} | [{task['title']}]({nb}) | {emoji} {task['difficulty']} "
            f"| `{task_id}` |"
        )

    return "\n".join(lines).strip()


def render_readme(numbered: list[tuple[int, str, dict]]) -> str | None:
    """README text with the problem table refreshed, or None if markers are absent."""
    if not README.exists():
        return None
    text = README.read_text()
    if README_START not in text or README_END not in text:
        return None
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    table = build_readme_table(numbered)
    return f"{head}{README_START}\n{table}\n{README_END}{tail}"


def numbered_tasks() -> list[tuple[int, str, dict]]:
    return [
        (i, task_id, task)
        for i, (task_id, task) in enumerate(list_tasks(), start=1)
    ]


def render_all() -> dict[Path, str]:
    """Path -> notebook JSON text, for everything we generate."""
    out: dict[Path, str] = {}
    numbered = numbered_tasks()

    out[TEMPLATES_DIR / "00_welcome.ipynb"] = json.dumps(
        build_welcome(numbered), indent=1, ensure_ascii=False
    ) + "\n"

    for num, task_id, task in numbered:
        t_path = TEMPLATES_DIR / f"{num:02d}_{task_id}.ipynb"
        s_path = SOLUTIONS_DIR / f"{num:02d}_{task_id}_solution.ipynb"
        out[t_path] = json.dumps(
            build_template(task_id, task, num), indent=1, ensure_ascii=False
        ) + "\n"
        out[s_path] = json.dumps(
            build_solution(task_id, task, num), indent=1, ensure_ascii=False
        ) + "\n"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the notebooks on disk differ from the task definitions",
    )
    args = ap.parse_args()

    rendered = render_all()
    numbered = numbered_tasks()
    readme_text = render_readme(numbered)

    if args.check:
        stale = [p for p, text in rendered.items() if not p.exists() or p.read_text() != text]
        if readme_text is not None and README.read_text() != readme_text:
            stale.append(README)
        existing = set(TEMPLATES_DIR.glob("*.ipynb")) | set(SOLUTIONS_DIR.glob("*.ipynb"))
        orphans = existing - set(rendered)
        if stale or orphans:
            for p in sorted(stale):
                print(f"stale:  {p.relative_to(ROOT)}")
            for p in sorted(orphans):
                print(f"orphan: {p.relative_to(ROOT)}")
            print("\nRun: python scripts/generate_notebooks.py")
            return 1
        print(f"✅ {len(rendered)} notebooks are up to date")
        return 0

    # Regenerate from scratch so renames never leave orphans behind.
    for d in (TEMPLATES_DIR, SOLUTIONS_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for path, text in rendered.items():
        path.write_text(text)

    if readme_text is not None:
        README.write_text(readme_text)
        print("✅ Refreshed the README problem table")

    n_tasks = len(numbered)
    print(f"✅ Generated {len(rendered)} notebooks for {n_tasks} tasks")
    print(f"   templates/  {n_tasks + 1} files (including 00_welcome)")
    print(f"   solutions/  {n_tasks} files")
    if REPO.startswith("YOUR-GITHUB"):
        print()
        print("⚠️  Colab badges point at a placeholder repo. Set JAXCODE_REPO to fix:")
        print('    JAXCODE_REPO="you/JAXCode" python scripts/generate_notebooks.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
