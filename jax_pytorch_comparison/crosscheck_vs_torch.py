#!/usr/bin/env python3
"""Validate JAXCode's reference solutions against PyTorch as an independent oracle.

`scripts/verify_tasks.py` proves each solution passes its own tests, and
`scripts/probe_tests.py` proves those tests reject wrong answers. Neither can
catch a *shared misconception* — a formula that is wrong in both the solution
and the tests, because the same author wrote both. PyTorch was written by other
people and is battle-tested, so agreeing with it numerically is outside evidence.

This lives outside the jax_judge package on purpose: JAXCode itself must never
depend on torch.

    python crosscheck_vs_torch.py

Convention notes that make the comparisons line up:
  - Linear weights: JAX/Flax kernel is (in, out); torch is (out, in) -> transpose
  - Conv:           JAXCode uses NHWC/HWIO;       torch is NCHW/OIHW  -> transpose
  - LayerNorm:      both normalise with the BIASED (population) variance
  - BatchNorm:      normalises with the biased variance, but updates the running
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
RESULTS: list[tuple[str, bool | None, str]] = []


def load(task_id: str) -> dict:
    ns: dict = {}
    exec(compile(get_task(task_id)["solution"], f"<{task_id}>", "exec"), ns)
    return ns


def compare(name: str, torch_out, jax_out, atol: float = 1e-5) -> None:
    a = torch_out.detach().cpu().numpy() if isinstance(torch_out, torch.Tensor) else np.asarray(torch_out)
    b = np.asarray(jax_out)
    if a.shape != b.shape:
        RESULTS.append((name, False, f"shape {a.shape} vs {b.shape}"))
        return
    diff = float(np.max(np.abs(a - b)))
    RESULTS.append((name, diff < atol, f"max|Δ| = {diff:.2e}  (tol {atol:g})"))


def skip(name: str, why: str) -> None:
    RESULTS.append((name, None, why))


def guarded(name: str, fn) -> None:
    """Run one comparison, recording an error instead of aborting the sweep."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        skip(name, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------- activations

x = rng.standard_normal((32, 16)).astype(np.float32)
xt, xj = torch.tensor(x), jnp.asarray(x)

guarded("relu", lambda: compare("relu", F.relu(xt), load("relu")["relu"](xj)))

_gelu = load("gelu")["my_gelu"]
guarded("gelu (exact/erf)", lambda: compare(
    "gelu (exact/erf)", F.gelu(xt, approximate="none"), _gelu(xj, approximate=False)))
guarded("gelu (tanh approx)", lambda: compare(
    "gelu (tanh approx)", F.gelu(xt, approximate="tanh"), _gelu(xj, approximate=True)))

guarded("softmax", lambda: compare(
    "softmax", torch.softmax(xt, dim=-1), load("softmax")["my_softmax"](xj, axis=-1)))

# ---------------------------------------------------------------- norm layers

D = 16


def _layernorm() -> None:
    ln_t = nn.LayerNorm(D, eps=1e-5)
    with torch.no_grad():
        ln_t.weight.copy_(torch.tensor(rng.standard_normal(D).astype(np.float32)))
        ln_t.bias.copy_(torch.tensor(rng.standard_normal(D).astype(np.float32)))
    my_layer_norm = load("layernorm")["my_layer_norm"]
    gamma = jnp.asarray(ln_t.weight.detach().numpy())
    beta = jnp.asarray(ln_t.bias.detach().numpy())
    compare("layernorm", ln_t(xt), my_layer_norm(xj, gamma, beta))


guarded("layernorm", _layernorm)


def _batchnorm() -> None:
    bn_t = nn.BatchNorm1d(D, eps=1e-5, momentum=0.1)
    bn_t.train()
    out_t = bn_t(xt)

    my_batch_norm = load("batchnorm")["my_batch_norm"]
    g, b = jnp.ones(D), jnp.zeros(D)
    rm, rv = jnp.zeros(D), jnp.ones(D)
    out_j, rm_new, rv_new = my_batch_norm(xj, g, b, rm, rv, momentum=0.1, training=True)

    compare("batchnorm (training output)", out_t, out_j)
    compare("batchnorm (running_mean)", bn_t.running_mean, rm_new, atol=1e-5)

    # This task follows the ORIGINAL TorchCode convention (and flax's): the
    # running buffer uses the BIASED variance. torch.nn.BatchNorm1d uses the
    # unbiased one, so the gap is exactly the Bessel factor n/(n-1).
    n = x.shape[0]
    biased = x.var(axis=0, ddof=0)
    compare("batchnorm (biased running_var, flax/TorchCode convention)",
            torch.tensor(np.asarray(0.9 + 0.1 * biased)), rv_new, atol=1e-5)
    compare("batchnorm (torch gap is exactly n/(n-1))",
            bn_t.running_var,
            jnp.asarray(0.9 + 0.1 * biased * (n / (n - 1))), atol=1e-5)

    # Eval mode normalises with the buffers and changes nothing.
    eval_out, rm2, rv2 = my_batch_norm(xj, g, b, rm_new, rv_new, training=False)
    expected = (x - np.asarray(rm_new)) / np.sqrt(np.asarray(rv_new) + 1e-5)
    compare("batchnorm (eval uses the buffers)", torch.tensor(expected), eval_out, atol=1e-4)


guarded("batchnorm", _batchnorm)

# ------------------------------------------------------------------- linear


def _linear() -> None:
    lin_t = nn.Linear(16, 8)
    lin_j = load("linear")["SimpleLinear"](16, 8, rngs=nnx.Rngs(params=0))
    lin_j.w[...] = jnp.asarray(lin_t.weight.detach().numpy().T)   # (out,in) -> (in,out)
    lin_j.b[...] = jnp.asarray(lin_t.bias.detach().numpy())
    compare("linear (after weight transpose)", lin_t(xt), lin_j(xj))


guarded("linear", _linear)

# ---------------------------------------------------------------- embedding


def _embedding() -> None:
    V, E, T = 50, 12, 7
    idx = rng.integers(0, V, size=(4, T))
    emb_t = nn.Embedding(V, E)
    emb_j = load("embedding")["MyEmbedding"](V, E, rngs=nnx.Rngs(params=0))
    emb_j.table[...] = jnp.asarray(emb_t.weight.detach().numpy())
    compare("embedding", emb_t(torch.tensor(idx)), emb_j(jnp.asarray(idx)))


guarded("embedding", _embedding)

# ------------------------------------------------------------------- conv2d


def _conv2d() -> None:
    # The task now uses torch's own NCHW/OIHW layout, so nothing is transposed.
    img = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)   # NCHW
    ker = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)   # OIHW
    my_conv2d = load("conv2d")["my_conv2d"]

    for stride, pad in ((1, 0), (2, 0), (1, 1), (2, 1)):
        out_t = F.conv2d(torch.tensor(img), torch.tensor(ker), stride=stride, padding=pad)
        out_j = my_conv2d(jnp.asarray(img), jnp.asarray(ker), stride=stride, padding=pad)
        compare(f"conv2d stride={stride} pad={pad}", out_t, out_j, atol=1e-4)


guarded("conv2d", _conv2d)

# ---------------------------------------------------------------- attention

B, H, T, Dh = 2, 4, 6, 8
q = rng.standard_normal((B, H, T, Dh)).astype(np.float32)
k = rng.standard_normal((B, H, T, Dh)).astype(np.float32)
v = rng.standard_normal((B, H, T, Dh)).astype(np.float32)

guarded("attention", lambda: compare(
    "attention (vs torch SDPA)",
    F.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v)),
    load("attention")["scaled_dot_product_attention"](
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)),
))

guarded("causal_attention", lambda: compare(
    "causal_attention (vs torch is_causal=True)",
    F.scaled_dot_product_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), is_causal=True),
    load("causal_attention")["causal_attention"](
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)),
))

# ------------------------------------------------------------- cross entropy

logits = rng.standard_normal((16, 10)).astype(np.float32)
labels = rng.integers(0, 10, size=(16,))
_ce = load("cross_entropy")["cross_entropy_loss"]

guarded("cross_entropy", lambda: compare(
    "cross_entropy",
    F.cross_entropy(torch.tensor(logits), torch.tensor(labels)),
    _ce(jnp.asarray(logits), jnp.asarray(labels)),
))
guarded("cross_entropy (label_smoothing)", lambda: compare(
    "cross_entropy (label_smoothing=0.1)",
    F.cross_entropy(torch.tensor(logits), torch.tensor(labels), label_smoothing=0.1),
    _ce(jnp.asarray(logits), jnp.asarray(labels), label_smoothing=0.1),
))


def _ce_ignore() -> None:
    lab = labels.copy()
    lab[:4] = -1                       # JAXCode's default ignore_index is -1
    compare(
        "cross_entropy (ignore_index, padding excluded)",
        F.cross_entropy(torch.tensor(logits), torch.tensor(lab), ignore_index=-1),
        _ce(jnp.asarray(logits), jnp.asarray(lab), ignore_index=-1),
    )


guarded("cross_entropy (ignore_index)", _ce_ignore)

# --------------------------------------------------------------------- adam


def _adam() -> None:
    MyAdam = load("adam")["MyAdam"]
    target = rng.standard_normal(5).astype(np.float32)
    p0 = np.zeros(5, dtype=np.float32)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

    pt = torch.tensor(p0.copy(), requires_grad=True)
    opt = torch.optim.Adam([pt], lr=lr, betas=(b1, b2), eps=eps)
    for _ in range(25):
        opt.zero_grad()
        ((pt - torch.tensor(target)) ** 2).sum().backward()
        opt.step()

    opt = MyAdam(lr=lr, betas=(b1, b2), eps=eps)
    pj = jnp.asarray(p0.copy())
    state = opt.init(pj)
    loss_fn = lambda p: jnp.sum((p - jnp.asarray(target)) ** 2)  # noqa: E731
    for _ in range(25):
        pj, state = opt.update(pj, jax.grad(loss_fn)(pj), state)

    compare("adam (25 steps vs torch.optim.Adam)", pt, pj, atol=1e-4)


guarded("adam", _adam)

# ------------------------------------------------------------------- report


def main() -> int:
    print(f"\n{BOLD}JAXCode reference solutions vs PyTorch{RESET}")
    print(f"{DIM}torch {torch.__version__} · jax {jax.__version__}{RESET}\n")

    failed = skipped = 0
    for name, ok, detail in RESULTS:
        if ok is None:
            skipped += 1
            print(f"  {YELLOW}⚠️  SKIP{RESET} {name:<44s} {DIM}{detail}{RESET}")
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
    tail = f" {DIM}({skipped} skipped){RESET}" if skipped else ""
    print(f"{GREEN}{BOLD}All {total - skipped} comparisons agree with PyTorch{RESET}{tail}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
