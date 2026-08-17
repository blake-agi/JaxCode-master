#!/usr/bin/env python3
"""Guard the JAX port against drifting away from its PyTorch original.

The 41 ported problems must stay recognisably the SAME problems as in the
TorchCode original (located by _paths.original_repo): same `function_name`,
same `difficulty`, same notebook number. Deviations are allowed only when JAX
genuinely forces one, and only when recorded in JAX_FORCED below with a reason.

This gate exists because the port silently drifted on 21 function names and
22 difficulty labels before anyone noticed.

    python scripts/check_alignment.py           # fail on unrecorded drift
    python scripts/check_alignment.py --table   # emit the alignment table
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _paths import original_repo  # noqa: E402

ORIGINAL = original_repo()

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[1m", "\033[0m",
)

# Deviations from the original that JAX genuinely forces. Anything not listed
# here is drift and fails the gate. Keep the reason specific — "more idiomatic"
# is not a reason, "PyTorch mutates in place and JAX cannot" is.
JAX_FORCED: dict[str, dict[str, str]] = {
    # task_id: {"field": "reason"}
}

# Problems I added that have no PyTorch counterpart. They carry b_* numbers.
ADDED = {
    "grad_basics", "vmap_batching", "jit_static", "pytree_ops", "prng_keys",
    "lax_scan", "lax_control_flow", "custom_vjp", "higher_order_grad",
    "remat_checkpoint", "sharding_basics",
    # Not a JAX-specific idea, but it has no counterpart in the PyTorch original
    # and the axis/keepdims discipline it drills is worth its own problem.
    "logsumexp",
    # Cross-entropy used to be one "Easy" problem carrying three ideas. It is
    # now a ladder: 16 is the original's problem (logsumexp allowed, as the
    # original's own rules say), b_14 takes logsumexp away, b_15 adds label
    # smoothing and the padding mask. Neither addition has a counterpart in the
    # original.
    "cross_entropy_fused",
    "cross_entropy_full",
    # Diffusion language models postdate the PyTorch original entirely. The
    # loss is the masked-CE the 16 -> b_14 -> b_15 chain builds up to; the
    # sampler is where the absorbing posterior earns its keep.
    "masked_diffusion",
    "diffusion_sampling",
    # Whole-model assembly. The 41 stop at one GPT-2 block; the wiring bugs
    # (RoPE placement, missing final norm, untied head) only appear once the
    # pieces are put together.
    "mini_gpt",
    # Problem 40 with the training loop as a lax.scan. The original has no
    # counterpart because a Python loop is the only way torch writes it — the
    # whole point here is that jit makes that unaffordable (2000 unrolled steps
    # take ~31s to compile; scan does 20000 in ~0.04s).
    "linear_regression_scan",
    # Nested scan with the PRNG key in the carry. No torch counterpart because
    # torch shuffles with a stateful global RNG and loops in Python; the whole
    # difficulty here — split the key, carry the parent, keep the carry
    # structure identical — only exists in the functional formulation.
    "minibatch_sgd_scan",
}


def _task_dicts(tasks_dir: Path) -> dict[str, dict]:
    """Pull the literal string fields out of each TASK dict without importing."""
    out: dict[str, dict] = {}
    for p in sorted(tasks_dir.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        fields: dict[str, str] = {}
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                        if isinstance(v.value, str):
                            fields[k.value] = v.value
                if "function_name" in fields:
                    break
        if fields:
            out[p.stem] = fields
    return out


def _original_numbers() -> dict[str, str]:
    """task_id -> notebook number, read off the original's template filenames."""
    nums: dict[str, str] = {}
    for p in sorted((ORIGINAL / "templates").glob("*.ipynb")):
        if p.stem.startswith("00_"):
            continue
        n, stem = p.stem.split("_", 1)
        nums[stem] = n
    # The single filename/module mismatch in the original.
    if "multihead_attention" in nums:
        nums["mha"] = nums.pop("multihead_attention")
    return nums


def load() -> tuple[dict, dict, dict]:
    if not ORIGINAL.exists():
        # CI checks out this repo alone, so the sibling is usually absent there.
        # Skipping is correct: the gate protects local edits, and a missing
        # reference is not a failure of the code under test.
        print(f"\n{YELLOW}⊘ Skipping alignment check — original repo not found at{RESET}")
        print(f"  {DIM}{ORIGINAL}{RESET}")
        print(f"  {DIM}Set TORCHCODE_ORIGINAL to enable it.{RESET}\n")
        raise SystemExit(0)
    orig = _task_dicts(ORIGINAL / "torch_judge" / "tasks")
    mine = _task_dicts(ROOT / "jax_judge" / "tasks")
    return orig, mine, _original_numbers()


def check() -> int:
    orig, mine, nums = load()
    problems: list[str] = []

    missing = set(orig) - set(mine)
    if missing:
        problems.append(f"ported problems missing from the JAX port: {sorted(missing)}")

    unexpected = set(mine) - set(orig) - ADDED
    if unexpected:
        problems.append(f"JAX tasks with no original and not declared in ADDED: {sorted(unexpected)}")

    for tid in sorted(set(orig) & set(mine)):
        forced = JAX_FORCED.get(tid, {})
        for field in ("function_name", "difficulty"):
            o, m = orig[tid].get(field), mine[tid].get(field)
            if o == m:
                continue
            if field in forced:
                continue
            problems.append(
                f"{tid}: {field} drifted  {o!r} (original) -> {m!r} (JAX). "
                f"Either match it, or record it in JAX_FORCED with a reason."
            )
        want = nums.get(tid)
        got = mine[tid].get("number")
        if want and got and want != got:
            problems.append(f"{tid}: number {got!r} should be {want!r} to match the original")

    for tid in sorted(ADDED & set(mine)):
        got = mine[tid].get("number")
        if got and not got.startswith("b_"):
            problems.append(f"{tid}: added problems need a b_* number, got {got!r}")

    print(f"\n{BOLD}Alignment with {ORIGINAL.name}{RESET}\n")
    if problems:
        for p in problems:
            print(f"  {RED}✗ {p}{RESET}")
        print(f"\n{RED}{BOLD}{len(problems)} alignment problem(s){RESET}\n")
        return 1

    n_forced = sum(len(v) for v in JAX_FORCED.values())
    print(f"  {GREEN}✓ all {len(set(orig) & set(mine))} ported problems match the original{RESET}")
    print(f"  {GREEN}✓ {len(ADDED & set(mine))} added problems carry b_* numbers{RESET}")
    if n_forced:
        print(f"  {YELLOW}! {n_forced} recorded JAX-forced deviation(s){RESET}")
        for tid, fields in sorted(JAX_FORCED.items()):
            for f, why in fields.items():
                print(f"      {DIM}{tid}.{f}: {why}{RESET}")
    print()
    return 0


def table() -> int:
    orig, mine, nums = load()
    print("| # | task_id | original fn | JAX fn | original diff | JAX diff | status |")
    print("|---|---|---|---|---|---|---|")
    for tid in sorted(set(orig) & set(mine), key=lambda t: nums.get(t, "99")):
        o, m = orig[tid], mine[tid]
        fn_ok = o["function_name"] == m["function_name"]
        d_ok = o["difficulty"] == m["difficulty"]
        status = "✅" if (fn_ok and d_ok) else ("⚠️ " + ", ".join(
            x for x, ok in (("name", fn_ok), ("difficulty", d_ok)) if not ok))
        print(f"| {nums.get(tid,'??')} | `{tid}` | `{o['function_name']}` | "
              f"`{m['function_name']}` | {o['difficulty']} | {m['difficulty']} | {status} |")
    print()
    for tid in sorted(ADDED & set(mine)):
        m = mine[tid]
        print(f"| {m.get('number','b_??')} | `{tid}` | — | `{m['function_name']}` | "
              f"— | {m['difficulty']} | added |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="emit the alignment table as markdown")
    args = ap.parse_args()
    return table() if args.table else check()


if __name__ == "__main__":
    raise SystemExit(main())
