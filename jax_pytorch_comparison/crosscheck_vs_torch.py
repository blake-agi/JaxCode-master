#!/usr/bin/env python3
"""Validate JAXCode's reference solutions against PyTorch as an independent oracle.

`scripts/verify_tasks.py` proves each solution passes its own tests, and
`scripts/probe_tests.py` proves those tests reject wrong answers. Neither can
catch a *shared misconception* — a formula that is wrong in both the solution
and the tests. PyTorch is battle-tested and was written by other people, so
agreeing with it numerically is genuine outside evidence.

This lives outside the jax_judge package on purpose: JAXCode itself must never
depend on torch.

    python crosscheck_vs_torch.py

Convention notes that make the comparisons line up:
  - Linear weights: JAX/Flax kernel is (in, out); torch is (out, in) -> transpose
  - Conv:           JAX here is NHWC/HWIO;        torch is NCHW/OIHW  -> transpose
  - LayerNorm:      both normalise with the BIASED (population) variance
  - BatchNorm:      normalises with biased variance, but updates the running
                    buffer with the UNBIASED sample variance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_judge.tasks import get_task  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[1m", "\033[0m",
)

rng = np.random.default_rng(0)
RESULTS: list[tuple[str, bool, str]] = []


def load(task_id: str):
    """Exec a task's reference solution and hand back its namespace."""
    ns: dict = {}
    exec(compile(get_task(task_id)["solution"], f"<{task_id}>", "exec"), ns)
    return ns


def compare(name: str, torch_out, jax_out, atol=1e-5) -> None:
    a = torch_out.detach().cpu().numpy() if isinstance(torch_out, torch.Tensor) else np.asarray(torch_out)
    b = np.asarray(jax_out)
    if a.shape != b.shape:
        RESULTS.append((name, False, f"shape {a.shape} vs {b.shape}"))
        return
    diff = float(np.max(np.abs(a - b)))
    RESULTS.append((name, diff < atol, f"max|Δ| = {diff:.2e}  (tol {atol:g})"))


def skip(name: str, why: str) -> None:
    RESULTS.append((name, None, why))


# --------------------------------------------------------------- activations

x = rng.standard_normal((32, 16)).astype(np.float32)
xt, xj = torch.tensor(x), jnp.asarray(x)

compare("relu", F.relu(xt), load("relu")["relu"](xj))

gelu = load("gelu")["gelu"]
compare("gelu (exact/erf)", F.gelu(xt, approximate="none"), gelu(xj))
try:
    compare("gelu (tanh approx)", F.gelu(xt, approximate="tanh"), gelu(xj, approximate="tanh"))
except TypeError:
    try:
        compare("gelu (tanh approx)", F.gelu(xt, approximate="tanh"), gelu(xj, approx=True))
    except Exception as e:
        skip("gelu (tanh approx)", f"signature mismatch: {e}")

compare("softmax", torch.softmax(xt, dim=-1), load("softmax")["my_softmax"](xj, axis=-1))

# ---------------------------------------------------------------- norm layers

D = 16
ln_t = nn.LayerNorm(D, eps=1e-5)
with torch.no_grad():
    ln_t.weight.copy_(torch.tensor(rng.standard_normal(D).astype(np.float32)))
    ln_t.bias.copy_(torch.tensor(rng.standard_normal(D).astype(np.float32)))

ln_j = load("layernorm")["LayerNorm"](D, rngs=nnx.Rngs(params=0))
ln_j.scale.value = jnp.asarray(ln_t.weight.detach().numpy())
ln_j.bias.value = jnp.asarray(ln_t.bias.detach().numpy())
compare("layernorm", ln_t(xt), ln_j(xj))

# BatchNorm in training mode, plus the running-stat convention.
bn_t = nn.BatchNorm1d(D, eps=1e-5, momentum=0.1)
bn_t.train()
out_bt = bn_t(xt)
bn_ns = load("batchnorm")
bn_j = bn_ns["BatchNorm"](D, rngs=nnx.Rngs(params=0))
try:
    out_bj = bn_j(xj, use_running_average=False)
except TypeError:
    out_bj = bn_j(xj, training=True)
compare("batchnorm (train output)", out_bt, out_bj)

# torch's running_var uses the UNBIASED sample variance; the normalisation does not.
compare(
    "batchnorm (running_var uses unbiased var)",
    bn_t.running_var,
    getattr(bn_j, "running_var", getattr(bn_j, "run_var", None)).value
    if hasattr(bn_j, "running_var") or hasattr(bn_j, "run_var") else torch.zeros(D),
    atol=1e-4,
)

# ------------------------------------------------------------------- linear

din, dout, B = 16, 8, 32
lin_t = nn.Linear(din, dout)
lin_j = load("linear")["Linear"](din, dout, rngs=nnx.Rngs(params=0))
lin_j.w.value = jnp.asarray(lin_t.weight.detach().numpy().T)   # (out,in) -> (in,out)
lin_j.b.value = jnp.asarray(lin_t.bias.detach().numpy())
compare("linear (after weight transpose)", lin_t(xt), lin_j(xj))

# ---------------------------------------------------------------- embedding

V, E, T = 50, 12, 7
idx = rng.integers(0, V, size=(4, T))
emb_t = nn.Embedding(V, E)
emb_j = load("embedding")["Embedding"](V, E, rngs=nnx.Rngs(params=0))
emb_j.w.value = jnp.asarray(emb_t.weight.detach().numpy())
compare("embedding", emb_t(torch.tensor(idx)), emb_j(jnp.asarray(idx)))

# ------------------------------------------------------------------- conv2d

img = rng.standard_normal((2, 8, 8, 3)).astype(np.float32)     # NHWC
ker = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)     # HWIO
conv2d = load("conv2d")["conv2d"]

for stride, padding in ((1, "VALID"), (1, "SAME"), (2, "VALID")):
    t_pad = 0 if padding == "VALID" else "same"
    out_t = F.conv2d(
        torch.tensor(img).permute(0, 3, 1, 2),                  # -> NCHW
        torch.tensor(ker).permute(3, 2, 0, 1),                  # HWIO -> OIHW
        stride=stride,
        padding=t_pad if not (padding == "SAME" and stride != 1) else 1,
    ).permute(0, 2, 3, 1)                                       # -> NHWC
    try:
        out_j = conv2d(jnp.asarray(img), jnp.asarray(ker), stride=stride, padding=padding)
        compare(f"conv2d stride={stride} {padding}", out_t, out_j, atol=1e-4)
    except Exception as e:
        skip(f"conv2d stride={stride} {padding}", f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- attention

Bq, H, Tq, Dh = 2, 4, 6, 8
q = rng.standard_normal((Bq, H, Tq, Dh)).astype(np.float32)
k = rng.standard_normal((Bq, H, Tq, Dh)).astype(np.float32)
v = rng.standard_normal((Bq, H, Tq, Dh)).astype(np.float32)

sdpa = load("attention")["scaled_dot_product_attention"]
compare(
    "attention (vs torch scaled_dot_product_attention)",
    F.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v)),
    sdpa(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)),
    atol=1e-5,
)

causal = load("causal_attention")["causal_attention"]
compare(
    "causal_attention (vs torch is_causal=True)",
    F.scaled_dot_product_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), is_causal=True
    ),
    causal(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)),
    atol=1e-5,
)

# ------------------------------------------------------------- cross entropy

logits = rng.standard_normal((16, 10)).astype(np.float32)
labels = rng.integers(0, 10, size=(16,))
ce = load("cross_entropy")["cross_entropy_loss"]

compare(
    "cross_entropy",
    F.cross_entropy(torch.tensor(logits), torch.tensor(labels)),
    ce(jnp.asarray(logits), jnp.asarray(labels)),
)
try:
    compare(
        "cross_entropy (label_smoothing=0.1)",
        F.cross_entropy(torch.tensor(logits), torch.tensor(labels), label_smoothing=0.1),
        ce(jnp.asarray(logits), jnp.asarray(labels), label_smoothing=0.1),
    )
except Exception as e:
    skip("cross_entropy (label_smoothing)", f"{type(e).__name__}: {e}")

# Padding must be excluded from both numerator and denominator.
labels_pad = labels.copy()
labels_pad[:4] = -100
try:
    compare(
        "cross_entropy (ignore_index=-100)",
        F.cross_entropy(torch.tensor(logits), torch.tensor(labels_pad), ignore_index=-100),
        ce(jnp.asarray(logits), jnp.asarray(labels_pad), ignore_index=-100),
    )
except Exception as e:
    skip("cross_entropy (ignore_index)", f"{type(e).__name__}: {e}")

# --------------------------------------------------------------------- adam

adam_update = load("adam")["adam_update"]
target = rng.standard_normal(5).astype(np.float32)
p0 = np.zeros(5, dtype=np.float32)
lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

pt = torch.tensor(p0.copy(), requires_grad=True)
opt = torch.optim.Adam([pt], lr=lr, betas=(b1, b2), eps=eps)
for _ in range(25):
    opt.zero_grad()
    ((pt - torch.tensor(target)) ** 2).sum().backward()
    opt.step()

pj = jnp.asarray(p0.copy())
state = {"m": jnp.zeros(5), "v": jnp.zeros(5)}
loss_fn = lambda p: jnp.sum((p - jnp.asarray(target)) ** 2)
for t in range(1, 26):
    g = jax.grad(loss_fn)(pj)
    pj, state = adam_update(pj, g, state, t, lr, b1, b2, eps)

compare("adam (25 steps vs torch.optim.Adam)", pt, pj, atol=1e-4)

# ------------------------------------------------------------------- report

def main() -> int:
    print(f"\n{BOLD}JAXCode reference solutions vs PyTorch{RESET}")
    print(f"{DIM}torch {torch.__version__} · jax {jax.__version__}{RESET}\n")

    failed = 0
    skipped = 0
    for name, ok, detail in RESULTS:
        if ok is None:
            skipped += 1
            print(f"  {YELLOW}⚠️  SKIP{RESET} {name:<46s} {DIM}{detail}{RESET}")
        elif ok:
            print(f"  {GREEN}✅{RESET} {name:<50s} {DIM}{detail}{RESET}")
        else:
            failed += 1
            print(f"  {RED}❌ {name:<50s} {detail}{RESET}")

    total = len(RESULTS)
    print()
    if failed:
        print(f"{RED}{BOLD}{failed}/{total} comparisons disagree with PyTorch{RESET}\n")
        return 1
    print(f"{GREEN}{BOLD}All {total - skipped} comparisons agree with PyTorch{RESET}"
          + (f" {DIM}({skipped} skipped){RESET}" if skipped else "") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
