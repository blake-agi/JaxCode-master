#!/usr/bin/env python3
"""Assert the ported problems still have the ORIGINAL's shape, not just its name.

check_alignment.py compares function_name, difficulty and notebook number. That
is not enough. `mha` kept the name MultiHeadAttention while its signature drifted
from the original's forward(Q, K, V) to __call__(x, mask) — which silently
removed the problem's whole point, that one class does self- AND cross-attention.
`grpo_loss` kept its name while becoming a different algorithm.

So this compares the callable SHAPE: kind (function vs class), method names, and
positional parameter names and order. Renaming a parameter or reordering two of
them breaks anyone cross-referencing the two repos, and reordering silently
breaks positional calls.

    python scripts/check_signatures.py            # fail on undeclared drift
    python scripts/check_signatures.py --list     # show every difference

Deviations JAX genuinely forces are declared in ALLOWED below, with a reason.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_judge._term import BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402
from jax_judge.tasks import TASKS  # noqa: E402

ORIGINAL = Path(os.environ.get("TORCHCODE_ORIGINAL", ROOT.parent / "TorchCode-master-original"))

# Parameters JAX requires that PyTorch does not: explicit PRNG and nnx plumbing.
JAX_ONLY_PARAMS = {"rngs", "key"}

# torch spells the forward pass `forward`; nnx spells it `__call__`.
METHOD_ALIASES = {"forward": "__call__"}

# Declared, reasoned deviations. Anything not listed here is drift.
# task_id -> {"what": "why"}
ALLOWED: dict[str, dict[str, str]] = {
    "adam": {
        "shape": "PyTorch's MyAdam holds the params and mutates them in step(); "
                 "JAX arrays are immutable and there is no .grad, so the class "
                 "keeps its name with optax-shaped init()/update().",
    },
    "gradient_clipping": {
        "parameters": "PyTorch reads p.grad off a parameter list and mutates it "
                      "in place; JAX takes the gradient pytree and returns the "
                      "clipped one.",
    },
    "linear_regression": {
        "shape": "the nn_linear method builds an nnx.Linear and runs a manual "
                 "SGD loop, since there is no torch.optim.",
    },
    # --- lowercase q/k/v: PEP8, and it is what every JAX example uses. The
    #     positional meaning is identical, so nothing is lost.
    "attention": {
        "mask": "optional mask added so the same function serves the masked "
                "cases; the original's 3-arg call still works unchanged",
    },
    "causal_attention": {"Q": "lowercase q/k/v per PEP8 and JAX convention"},
    "linear_attention": {"Q": "lowercase q/k/v per PEP8 and JAX convention"},
    "flash_attention": {"Q": "lowercase q/k/v per PEP8 and JAX convention"},
    "sliding_window": {"Q": "lowercase q/k/v per PEP8 and JAX convention"},
    # --- JAX naming and JAX-forced arguments
    "softmax": {
        "dim": "jnp reductions take `axis`, not `dim`; using dim would be "
               "actively misleading in JAX",
    },
    # --- optional additions that leave the original's call working
    "cross_entropy": {
        "label_smoothing": "optional; the original's 2-arg call is unchanged",
        "ignore_index": "optional padding mask, the standard companion to "
                        "cross-entropy and cross-checked against torch",
    },
    "gelu": {
        "approximate": "optional; exists because jax.nn.gelu defaults to the "
                       "TANH form while torch defaults to erf — a real JAX trap",
    },
    "rope": {"base": "optional; the original hardcodes 10000, which is the default"},
    "dropout": {"deterministic": "eval-mode flag; PyTorch reads it off global "
                                 "module state, which JAX has no equivalent of"},
    "embedding": {"attend": "extra method for weight tying; does not change "
                            "the original's constructor or __call__"},
    "linear": {"use_bias": "optional; defaults to True, matching nn.Linear"},
    "ppo_loss": {"mask": "optional padding mask; the original's 4-arg call is unchanged"},
    "beam_search": {"length_penalty": "optional; defaults to 1.0, which is the "
                                      "original's implicit behaviour"},
    "gpt2_block": {"_attn": "private helper; the public __call__ is identical"},
    "topk_sampling": {"key": "JAX has no global RNG, so sampling takes a key"},
    "speculative_decoding": {"key": "JAX has no global RNG, so sampling takes a key"},
    "weight_init": {"key": "JAX has no global RNG and no in-place normal_()"},
}


def _orig_numbers() -> dict[str, str]:
    nums: dict[str, str] = {}
    for p in sorted((ORIGINAL / "templates").glob("*.ipynb")):
        if p.stem.startswith("00_"):
            continue
        n, stem = p.stem.split("_", 1)
        nums[stem] = n
    if "multihead_attention" in nums:
        nums["mha"] = nums.pop("multihead_attention")
    return nums


def _orig_solution(tid: str, nums: dict[str, str]) -> str:
    hits = list((ORIGINAL / "solutions").glob(f"{nums[tid]}_*_solution.ipynb"))
    if not hits:
        return ""
    for cell in json.load(open(hits[0]))["cells"]:
        src = "".join(cell["source"])
        if "SOLUTION" in src:
            return src
    return ""


def shape_of(source: str, symbol: str) -> tuple[str, dict[str, list[str]]] | None:
    """(kind, {method: [positional param names]}) for `symbol` in `source`."""
    try:
        tree = ast.parse(re.sub(r"^#\s*✅\s*SOLUTION\s*$", "", source, flags=re.M))
    except SyntaxError:
        return None
    node = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
         and n.name == symbol),
        None,
    )
    if node is None:
        return None

    def params(fn: ast.FunctionDef) -> list[str]:
        names = [a.arg for a in fn.args.args if a.arg != "self"]
        names += [a.arg for a in fn.args.kwonlyargs]
        return [p for p in names if p not in JAX_ONLY_PARAMS]

    if isinstance(node, ast.ClassDef):
        methods = {
            METHOD_ALIASES.get(m.name, m.name): params(m)
            for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return ("class", methods)
    return ("function", {METHOD_ALIASES.get(node.name, node.name): params(node)})


def compare(tid: str, nums: dict[str, str]) -> list[str]:
    task = TASKS[tid]
    o = shape_of(_orig_solution(tid, nums), task["function_name"])
    m = shape_of(task["solution"], task["function_name"])
    if o is None:
        return [f"could not parse the ORIGINAL solution for '{task['function_name']}'"]
    if m is None:
        return [f"could not parse the JAX solution for '{task['function_name']}'"]

    out: list[str] = []
    if o[0] != m[0]:
        out.append(f"kind changed: {o[0]} -> {m[0]}")
    for meth, o_params in o[1].items():
        if meth not in m[1]:
            out.append(f"missing method {meth}()")
            continue
        if o_params != m[1][meth]:
            out.append(
                f"{meth}({', '.join(o_params)})  ->  ({', '.join(m[1][meth])})"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show every difference, allowed or not")
    args = ap.parse_args()

    if not ORIGINAL.exists():
        print(f"\n{YELLOW}⊘ Skipping signature check — original repo not found at{RESET}")
        print(f"  {DIM}{ORIGINAL}{RESET}\n")
        return 0

    nums = _orig_numbers()
    drift: dict[str, list[str]] = {}
    allowed_hits = 0

    print(f"\n{BOLD}Signature alignment with {ORIGINAL.name}{RESET}\n")
    for tid in sorted(nums):
        if tid not in TASKS:
            continue
        diffs = compare(tid, nums)
        if not diffs:
            continue
        declared = ALLOWED.get(tid, {})
        undeclared = [
            d for d in diffs
            if not any(k == "shape" or k in d for k in declared)
        ]
        if args.list:
            for d in diffs:
                mark = f"{DIM}(allowed){RESET}" if d not in undeclared else f"{RED}DRIFT{RESET}"
                print(f"  {nums[tid]} {tid:<22s} {d}  {mark}")
        if undeclared:
            drift[tid] = undeclared
        else:
            allowed_hits += 1

    if drift:
        if not args.list:
            for tid, ds in sorted(drift.items()):
                print(f"  {RED}✗ {nums[tid]} {tid}{RESET}")
                for d in ds:
                    print(f"      {d}")
        print(f"\n{RED}{BOLD}{len(drift)} task(s) drifted from the original's signature{RESET}")
        print(f"{DIM}Match the original, or declare the deviation in ALLOWED with a reason.{RESET}\n")
        return 1

    n = sum(1 for t in nums if t in TASKS)
    print(f"  {GREEN}✓ all {n} ported problems keep the original's callable shape{RESET}")
    if allowed_hits:
        print(f"  {YELLOW}! {allowed_hits} declared JAX-forced deviation(s){RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
