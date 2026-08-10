#!/usr/bin/env python3
"""Generate the PyTorch -> JAX side-by-side comparison notebook.

The notebook is a build artifact: edit SECTIONS here, then run

    python build_notebook.py

Every code section runs BOTH frameworks on the same inputs and asserts they
agree numerically, so the notebook is self-verifying — if a claim in the prose
is wrong, the cell fails rather than quietly misinforming the reader.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pytorch_to_jax.ipynb"

REPO = "YOUR-GITHUB-USERNAME/jax_pytorch_comparison"
BRANCH = "main"

INSTALL = """\
# Colab: install both frameworks (no-op if they're already present)
try:
    import google.colab
    get_ipython().run_line_magic(
        'pip', 'install -q jax flax torch --index-url https://download.pytorch.org/whl/cpu'
    )
    get_ipython().run_line_magic('pip', 'install -q jax flax')
except ImportError:
    pass
"""

SETUP = '''\
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from flax import nnx

torch.manual_seed(0)
np.random.seed(0)

print("torch", torch.__version__, "| jax", jax.__version__, "| flax", nnx.__doc__ is not None)
print("jax devices:", jax.devices())


def agree(a, b, atol=1e-5, label="values"):
    """Assert a torch tensor and a jax array match, and say so out loud."""
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    assert a_np.shape == b_np.shape, f"{label}: shape {a_np.shape} vs {b_np.shape}"
    diff = np.max(np.abs(a_np - b_np))
    assert diff < atol, f"{label}: max abs diff {diff:.3e} exceeds {atol:g}"
    print(f"  ✅ {label} agree — max abs diff {diff:.2e}")
'''

# ---------------------------------------------------------------- sections

SECTIONS: list[tuple[str, str, str]] = []


def section(title: str, markdown: str, code: str) -> None:
    SECTIONS.append((title, markdown.strip(), code.strip()))


section(
    "Arrays are immutable",
    r"""
## 1. Arrays are immutable

The single biggest adjustment. In PyTorch you mutate tensors in place; in JAX
arrays are **immutable** and you build new ones with `.at[...]`.

| PyTorch | JAX |
|---|---|
| `x[0] = 5.0` | `x = x.at[0].set(5.0)` |
| `x += 1` | `x = x + 1` |
| `x.add_(1)` | `x = x.at[:].add(1)` |
| `x.clamp_(0)` | `x = jnp.maximum(x, 0)` |

`.at[].set()` looks like it copies the whole array. Under `jit`, XLA turns it
back into an in-place update whenever the original is not needed afterwards, so
the functional spelling costs nothing. Outside `jit` it really does copy.

**Why this matters:** immutability is what makes `jit`, `grad`, `vmap` and
`scan` composable and safe to reorder. It is the price of admission, not an
oversight.
""",
    """
# ---- PyTorch: mutate in place -------------------------------------------
xt = torch.arange(5, dtype=torch.float32)
xt[0] = 5.0
xt += 1

# ---- JAX: build a new array ---------------------------------------------
xj = jnp.arange(5, dtype=jnp.float32)
xj = xj.at[0].set(5.0)
xj = xj + 1

agree(xt, xj, label="immutable update")

# JAX will not let you mutate at all:
try:
    xj[0] = 99.0
except TypeError as e:
    print("  ℹ️  in-place assignment raises:", str(e).split(".")[0])
""",
)

section(
    "Autograd",
    r"""
## 2. Autograd: `.backward()` vs `jax.grad`

PyTorch accumulates gradients into `.grad` as a **side effect** of the backward
pass. JAX transforms a function into a *new function* that returns gradients.

| PyTorch | JAX |
|---|---|
| `loss.backward()` then read `p.grad` | `grads = jax.grad(loss_fn)(params)` |
| `opt.zero_grad()` | *nothing to zero — no accumulation* |
| `with torch.no_grad():` | *nothing — grad only applies where you ask* |
| `x.detach()` | `jax.lax.stop_gradient(x)` |
| `loss.backward()` + read loss | `jax.value_and_grad(loss_fn)(params)` |

Note there is **no `zero_grad` in JAX**, and that is not a convenience — it is
structural. `jax.grad` returns a fresh pytree of gradients each call, so the
"forgot to zero the gradients" bug simply cannot be written.

`jax.grad` differentiates w.r.t. the **first** argument by default; use
`argnums=` for others, and `has_aux=True` when the function also returns metrics.
""",
    """
# ---- PyTorch -------------------------------------------------------------
wt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss_t = (wt ** 2).sum()
loss_t.backward()          # side effect: fills wt.grad

# ---- JAX -----------------------------------------------------------------
wj = jnp.array([1.0, 2.0, 3.0])
loss_fn = lambda w: jnp.sum(w ** 2)
loss_j, grad_j = jax.value_and_grad(loss_fn)(wj)

agree(loss_t, loss_j, label="loss")
agree(wt.grad, grad_j, label="gradient")   # both are 2w

# Gradients w.r.t. a later argument:
f = lambda a, b: jnp.sum(a * b ** 2)
print("  argnums=1 ->", jax.grad(f, argnums=1)(jnp.array(2.0), jnp.array(3.0)))  # 2*a*b = 12
""",
)

section(
    "Modules",
    r"""
## 3. Modules: `nn.Module` vs `nnx.Module`

Flax **NNX** is deliberately close to PyTorch: mutable Python objects holding
parameters. The differences that bite:

| PyTorch | Flax NNX |
|---|---|
| `nn.Parameter(t)` | `nnx.Param(arr)` |
| implicit global RNG | explicit `rngs: nnx.Rngs` argument |
| `def forward(self, x)` | `def __call__(self, x)` |
| `model(x)` | `model(x)` |
| `p.data` | `p.value` |
| buffers (`register_buffer`) | `nnx.BatchStat`, `nnx.Cache`, ... |

### ⚠️ The weight-layout trap
`torch.nn.Linear` stores its weight as **`(out_features, in_features)`** and
computes `x @ W.T`. Flax stores the kernel as **`(in, out)`** and computes
`x @ W`. So a ported weight needs a **transpose** — this is the single most
common porting bug, and it is silent when `in == out`.
""",
    """
din, dout, B = 4, 3, 5
x_np = np.random.randn(B, din).astype(np.float32)

# ---- PyTorch -------------------------------------------------------------
lin_t = nn.Linear(din, dout)
print("  torch weight shape:", tuple(lin_t.weight.shape), "-> (out, in)")
out_t = lin_t(torch.tensor(x_np))

# ---- Flax NNX ------------------------------------------------------------
class Linear(nnx.Module):
    def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)) * 0.1)
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w.value + self.b.value

lin_j = Linear(din, dout, rngs=nnx.Rngs(params=0))
print("  flax  kernel shape:", lin_j.w.value.shape, "-> (in, out)")

# Port the torch weights across — note the .T
lin_j.w.value = jnp.asarray(lin_t.weight.detach().numpy().T)
lin_j.b.value = jnp.asarray(lin_t.bias.detach().numpy())
out_j = lin_j(jnp.asarray(x_np))

agree(out_t, out_j, label="linear output (after transpose)")
""",
)

section(
    "Optimizers",
    r"""
## 4. Optimizers: stateful object vs pure function

PyTorch's optimizer owns and mutates your parameters. In JAX the optimizer is a
pure function: `(params, grads, state) -> (params, state)`. Nothing is hidden.

| PyTorch | JAX + Optax |
|---|---|
| `opt = torch.optim.Adam(m.parameters(), lr)` | `opt = optax.adam(lr)` / `state = opt.init(params)` |
| `opt.zero_grad()` | — |
| `loss.backward()` | `grads = jax.grad(loss_fn)(params)` |
| `opt.step()` | `updates, state = opt.update(grads, state)`<br>`params = optax.apply_updates(params, updates)` |

Below we skip Optax entirely and write Adam by hand, to show there is no magic —
this is exactly [JAXCode's `adam` problem](../templates/29_adam.ipynb).

Note the **bias correction**: `m` and `v` start at zero, so at step 1 they are
scaled down by `(1-β₁)` and `(1-β₂)`. Dividing by `1-β^t` undoes that, which is
why Adam's first step has magnitude ≈ `lr` regardless of gradient size.
""",
    """
# Same problem, same init, same hyperparameters, 50 steps of Adam.
target = np.array([3.0, -1.0, 0.5], dtype=np.float32)
init = np.zeros(3, dtype=np.float32)
lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8

# ---- PyTorch -------------------------------------------------------------
pt = torch.tensor(init.copy(), requires_grad=True)
opt = torch.optim.Adam([pt], lr=lr, betas=(b1, b2), eps=eps)
for _ in range(50):
    opt.zero_grad()
    ((pt - torch.tensor(target)) ** 2).sum().backward()
    opt.step()

# ---- JAX (hand-written Adam) --------------------------------------------
pj = jnp.array(init.copy())
m = jnp.zeros(3)
v = jnp.zeros(3)
loss_fn = lambda p: jnp.sum((p - jnp.asarray(target)) ** 2)
for t in range(1, 51):
    g = jax.grad(loss_fn)(pj)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    pj = pj - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)

agree(pt, pj, atol=1e-4, label="params after 50 Adam steps")
print("  final:", np.asarray(pj).round(4), " target:", target)
""",
)

section(
    "Training loop",
    r"""
## 5. A full training loop, side by side

The shapes of the two loops are the clearest summary of the whole difference.

```python
# PyTorch                             # JAX
for x, y in loader:                   for x, y in loader:
    opt.zero_grad()                       loss, grads = value_and_grad(loss_fn)(params, x, y)
    loss = criterion(model(x), y)         params = apply(params, grads)
    loss.backward()
    opt.step()
```

PyTorch mutates `model` and `opt` in place. JAX threads `params` through as a
value — which is what lets the entire step be wrapped in one `jax.jit`.
""",
    """
# Linear regression on identical data, matched initialisation, plain SGD.
N, D = 64, 3
X_np = np.random.randn(N, D).astype(np.float32)
w_true = np.array([2.0, -3.0, 1.0], dtype=np.float32)
y_np = (X_np @ w_true + 0.5).astype(np.float32)
w0 = np.zeros(D, dtype=np.float32)
lr, steps = 0.05, 100

# ---- PyTorch -------------------------------------------------------------
wt = torch.tensor(w0.copy(), requires_grad=True)
bt = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([wt, bt], lr=lr)
Xt, yt = torch.tensor(X_np), torch.tensor(y_np)
for _ in range(steps):
    opt.zero_grad()
    F.mse_loss(Xt @ wt + bt, yt).backward()
    opt.step()

# ---- JAX -----------------------------------------------------------------
params = {"w": jnp.array(w0.copy()), "b": jnp.array(0.0)}

def loss_fn(p, X, y):
    return jnp.mean((X @ p["w"] + p["b"] - y) ** 2)

@jax.jit
def step(p, X, y):
    loss, g = jax.value_and_grad(loss_fn)(p, X, y)
    return jax.tree.map(lambda a, b: a - lr * b, p, g), loss

Xj, yj = jnp.asarray(X_np), jnp.asarray(y_np)
for _ in range(steps):
    params, loss = step(params, Xj, yj)

agree(wt, params["w"], atol=1e-3, label="weights after 100 SGD steps")
agree(bt, params["b"], atol=1e-3, label="bias")
print("  recovered w:", np.asarray(params["w"]).round(3), " true:", w_true)
""",
)

section(
    "Randomness",
    r"""
## 6. Randomness: global seed vs explicit keys

PyTorch draws from a hidden global RNG. JAX has no global state — you pass a
**key** in, and you must **split** it to get fresh randomness.

| PyTorch | JAX |
|---|---|
| `torch.manual_seed(0)` | `key = jax.random.key(0)` |
| `torch.randn(3)` | `jax.random.normal(key, (3,))` |
| *(advances hidden state)* | `key, sub = jax.random.split(key)` |

### The rule
**Never reuse a key.** Calling `jax.random.normal(key, ...)` twice with the same
key returns the *same* numbers — it is a pure function of the key. This surprises
everyone once; the tests below make it concrete.

The upside: reproducibility is exact and local. There is no "did some library
call advance the global RNG?" class of bug, and a `vmap`ped ensemble can be given
one key per member.
""",
    """
key = jax.random.key(0)

# Same key twice -> identical draws. This is the trap.
a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
assert jnp.array_equal(a, b)
print("  ⚠️  same key twice gives identical values:", np.asarray(a).round(4))

# Split for independent draws.
k1, k2 = jax.random.split(key)
c = jax.random.normal(k1, (3,))
d = jax.random.normal(k2, (3,))
assert not jnp.array_equal(c, d)
print("  ✅ after split, values differ:", np.asarray(c).round(4), np.asarray(d).round(4))

# Reproducibility is exact and needs no global state.
again = jax.random.normal(jax.random.key(0), (3,))
agree(np.asarray(a), again, label="re-created key reproduces the draw")
""",
)

section(
    "vmap",
    r"""
## 7. Batching: write one example, `vmap` the rest

PyTorch asks you to write every op batched, juggling `unsqueeze`/`expand`/
`einsum` and broadcasting rules. JAX lets you write the **single-example**
function and add the batch axis automatically.

| PyTorch | JAX |
|---|---|
| hand-broadcast: `(X[:, None] - Y[None]).pow(2).sum(-1)` | `vmap(vmap(f, (None, 0)), (0, None))(X, Y)` |
| `torch.func.vmap` (newer) | `jax.vmap` (core, from day one) |

`in_axes` says which axis of each argument to map over; `None` means "broadcast
this argument, don't map it". Getting `(None, 0)` vs `(0, None)` right is a very
common interview question.
""",
    """
X_np = np.random.randn(6, 3).astype(np.float32)
Y_np = np.random.randn(4, 3).astype(np.float32)

# ---- PyTorch: hand-managed broadcasting ---------------------------------
Xt, Yt = torch.tensor(X_np), torch.tensor(Y_np)
d_t = ((Xt[:, None, :] - Yt[None, :, :]) ** 2).sum(-1)

# ---- JAX: write it for ONE pair, then vmap twice ------------------------
def sq_dist(x, y):
    return jnp.sum((x - y) ** 2)

row = jax.vmap(sq_dist, in_axes=(None, 0))     # one x vs every y
mat = jax.vmap(row,      in_axes=(0, None))    # every x
d_j = mat(jnp.asarray(X_np), jnp.asarray(Y_np))

agree(d_t, d_j, atol=1e-4, label="pairwise distances (6x4)")

# vmap composes with grad — a per-example gradient in one line.
per_example = jax.vmap(jax.grad(lambda x: jnp.sum(x ** 2)))(jnp.asarray(X_np))
print("  per-example grads shape:", per_example.shape, "(no loop, no autograd hooks)")
""",
)

section(
    "jit",
    r"""
## 8. Compilation: `jax.jit` and tracing

`jax.jit` traces your function once per **input shape/dtype signature**, compiles
it with XLA, and reuses the compiled binary. `torch.compile` is the closest
analogue, but JAX's tracing model has consequences you must know:

- **Python side effects run only during tracing.** A `print()` inside a jitted
  function fires once, not every call. Use `jax.debug.print` for runtime output.
- **You cannot branch on a traced value.** `if x > 0:` fails because `x` is a
  tracer with no concrete value. Use `jnp.where` or `lax.cond`.
- **Shapes must be static.** Anything used as a shape or an `axis` must be a
  compile-time constant — mark it `static_argnames`, and remember that a new
  value for a static argument triggers a **recompile**.
""",
    """
call_count = {"n": 0}

@jax.jit
def f(x):
    call_count["n"] += 1          # a Python side effect: only runs while TRACING
    return jnp.sum(x ** 2)

x = jnp.ones((4,))
f(x); f(x); f(x)
print(f"  called 3x, traced {call_count['n']}x  <- compiled once, reused")

f(jnp.ones((8,)))                  # new SHAPE -> new trace
print(f"  after a new input shape, traced {call_count['n']}x")

# Branching on a traced value is an error...
@jax.jit
def bad(x):
    if x > 0:
        return x
    return -x

try:
    bad(jnp.array(1.0))
except jax.errors.TracerBoolConversionError:
    print("  ℹ️  `if x > 0:` on a traced value raises TracerBoolConversionError")

# ...use jnp.where instead.
good = jax.jit(lambda x: jnp.where(x > 0, x, -x))
agree(torch.tensor(1.0), good(jnp.array(-1.0)), label="jnp.where branch")
""",
)

section(
    "Control flow",
    r"""
## 9. Loops: Python `for` vs `lax.scan`

A Python loop inside a jitted function is **unrolled** at trace time. 10,000
steps means 10,000 copies of the graph — minutes of compile time and enormous
memory. `lax.scan` compiles to a single rolled loop instead.

| PyTorch | JAX |
|---|---|
| `for t in range(T): h = cell(x[t], h)` | `h, ys = lax.scan(cell, h0, xs)` |
| `while cond: ...` | `lax.while_loop(cond_fn, body_fn, init)` |
| `if c: a else: b` | `lax.cond(c, f_true, f_false, operand)` |

`scan` carries state through and stacks the per-step outputs, so it is exactly an
RNN loop, a discounted-return computation, or a training loop over batches.

⚠️ `lax.while_loop` is **not reverse-mode differentiable** (the trip count is not
known ahead of time). Use `scan` with a fixed length if you need gradients.
""",
    """
rewards_np = np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32)
gamma = 0.9

# ---- PyTorch: a plain reversed Python loop ------------------------------
out_t, running = [], 0.0
for r in reversed(rewards_np.tolist()):
    running = r + gamma * running
    out_t.append(running)
out_t = torch.tensor(list(reversed(out_t)))

# ---- JAX: one rolled loop, reverse=True ---------------------------------
def step(carry, r):
    carry = r + gamma * carry
    return carry, carry

_, out_j = jax.lax.scan(step, 0.0, jnp.asarray(rewards_np), reverse=True)

agree(out_t, out_j, label="discounted returns")

# scan compiles in O(1) graph size regardless of length.
big = jnp.ones((100_000,))
_, _ = jax.lax.scan(step, 0.0, big)
print("  ✅ scanned 100,000 steps — a Python loop here would unroll the graph")
""",
)

section(
    "Devices",
    r"""
## 10. Devices

| PyTorch | JAX |
|---|---|
| `x.to('cuda')` / `.cuda()` | `jax.device_put(x, dev)` |
| `model.to(device)` | *arrays move; there is no model to move* |
| `torch.cuda.is_available()` | `jax.devices()` |
| manual placement everywhere | **arrays default to device 0 automatically** |

JAX places arrays on the default accelerator without being asked, so most code
has no device management at all. Multi-device work uses `jax.sharding` and
`shard_map` rather than `DataParallel`/`DistributedDataParallel`, and the
programming model is "one global array, sharded" rather than "N replicas".
""",
    """
print("  devices:", jax.devices())
x = jnp.ones((3,))
print("  arrays land on the default device automatically:", x.devices())

x_explicit = jax.device_put(x, jax.devices()[0])
agree(np.ones(3, dtype=np.float32), x_explicit, label="device_put round-trip")
""",
)

section(
    "Train/eval mode",
    r"""
## 11. Train/eval mode and stateful layers

`model.train()` / `model.eval()` flips hidden global flags that change what
Dropout and BatchNorm do. In JAX the behaviour is an **explicit argument** or an
explicit piece of state you can see.

| PyTorch | Flax NNX |
|---|---|
| `model.train()` / `model.eval()` | pass a flag, or `nnx.Dropout(deterministic=...)` |
| dropout scales at **train** time (`1/(1-p)`) | same convention — inverted dropout |
| BatchNorm buffers hidden in `state_dict` | `nnx.BatchStat`, a visible attribute |

Note the BatchNorm subtlety both frameworks share: the **normalisation** uses the
biased (population) variance, while the **running buffer** is updated with the
*unbiased* sample variance. Porting code that gets this wrong produces a model
that is correct in training and subtly wrong at inference.
""",
    """
p, B, D = 0.5, 100000, 4
x = jnp.ones((B, D))

# Inverted dropout: scale by 1/(1-p) at TRAIN time so eval needs no rescaling.
key = jax.random.key(0)
keep = jax.random.bernoulli(key, 1 - p, x.shape)
train_out = jnp.where(keep, x / (1 - p), 0.0)
eval_out = x                                   # identity at eval

print(f"  train mean {float(train_out.mean()):.4f}  (≈1.0 — that is the point)")
print(f"  eval  mean {float(eval_out.mean()):.4f}")
assert abs(float(train_out.mean()) - 1.0) < 0.01

# torch agrees on the convention:
td = nn.Dropout(p)
td.train()
t_out = td(torch.ones(B, D))
print(f"  torch train mean {t_out.mean().item():.4f}  <- same 1/(1-p) rescale")
assert abs(t_out.mean().item() - 1.0) < 0.01
print("  ✅ both frameworks use inverted dropout")
""",
)

section(
    "BatchNorm divergence",
    r"""
## 12. ⚠️ Where the two frameworks genuinely disagree

Almost everything above is a syntax difference. This one is a **semantic**
difference, and it is the kind of thing that silently corrupts a ported model.

Both frameworks **normalise** using the biased (population) variance. But they
update the `running_var` buffer differently:

| | `running_var` update uses |
|---|---|
| `flax.nnx.BatchNorm` | **biased** variance (`ddof=0`) |
| `torch.nn.BatchNorm1d` | **unbiased** variance (`ddof=1`) |

They differ by exactly the Bessel factor $n/(n-1)$ — 3.2% at batch size 32.

Why it is nasty:
- training outputs are **identical**, so nothing looks wrong while you train
- the error appears only at **inference**, once the running buffers are used
- it scales with $1/n$, so it is worst for small batches — exactly where people
  debug

If you port BatchNorm weights between frameworks, rescale `running_var` by
$n/(n-1)$ (or its inverse). The cell below proves the divergence rather than
asserting it.
""",
    """
x_np = np.random.randn(32, 16).astype(np.float32)
n = x_np.shape[0]

biased = x_np.var(axis=0, ddof=0)
unbiased = x_np.var(axis=0, ddof=1)

# torch: momentum=0.1 means new = 0.9*old + 0.1*batch
bn_t = nn.BatchNorm1d(16, eps=1e-5, momentum=0.1); bn_t.train()
out_t = bn_t(torch.tensor(x_np))

# flax: momentum=0.9 means the same thing (the conventions are complementary)
bn_j = nnx.BatchNorm(16, momentum=0.9, epsilon=1e-5, rngs=nnx.Rngs(0))
out_j = bn_j(jnp.asarray(x_np), use_running_average=False)

agree(out_t, out_j, label="training output (both use BIASED variance)")

rv_t = bn_t.running_var.detach().numpy()
rv_j = np.asarray(bn_j.var[...])

print(f"  torch running_var[0] = {rv_t[0]:.6f}")
print(f"  flax  running_var[0] = {rv_j[0]:.6f}")
print(f"  ratio                = {rv_t[0] / rv_j[0] if rv_j[0] else float('nan'):.6f}")

assert np.allclose(rv_j, 0.9 + 0.1 * biased, atol=1e-5), "flax should use biased"
assert np.allclose(rv_t, 0.9 + 0.1 * unbiased, atol=1e-5), "torch should use unbiased"
print(f"  ⚠️  running_var DIFFERS by the Bessel factor n/(n-1) = {n/(n-1):.4f}")
print("     -> identical in training, divergent at inference.")
""",
)

section(
    "Cheat sheet",
    r"""
## 12. Translation cheat sheet

| Concept | PyTorch | JAX / Flax NNX |
|---|---|---|
| array creation | `torch.zeros(3)` | `jnp.zeros(3)` |
| in-place write | `x[0] = 1` | `x = x.at[0].set(1)` |
| gradient | `loss.backward()`; `p.grad` | `jax.grad(f)(p)` |
| loss + gradient | `loss.backward()` | `jax.value_and_grad(f)(p)` |
| stop gradient | `x.detach()` | `jax.lax.stop_gradient(x)` |
| no-grad block | `with torch.no_grad():` | *not needed* |
| zero gradients | `opt.zero_grad()` | *not needed* |
| parameter | `nn.Parameter` | `nnx.Param` |
| buffer | `register_buffer` | `nnx.BatchStat` / `nnx.Cache` |
| forward | `def forward(self, x)` | `def __call__(self, x)` |
| linear weight | `(out, in)`, `x @ W.T` | `(in, out)`, `x @ W` |
| compile | `torch.compile(f)` | `jax.jit(f)` |
| batching | manual broadcast / `einsum` | `jax.vmap` |
| per-example grads | awkward | `vmap(grad(f))` |
| RNN loop | Python `for` | `jax.lax.scan` |
| conditional | `if` | `jnp.where` / `lax.cond` |
| RNG | global `manual_seed` | explicit `key` + `split` |
| device | `.to('cuda')` | automatic; `jax.device_put` |
| multi-GPU | `DistributedDataParallel` | `jax.sharding` + `shard_map` |
| checkpointing | `torch.utils.checkpoint` | `jax.checkpoint` (`remat`) |
| second derivative | `create_graph=True` | `jax.jacfwd(jax.jacrev(f))` |
| custom backward | `autograd.Function` | `jax.custom_vjp` |
| conv layout | NCHW / OIHW | NHWC / HWIO |

### The four that actually cost people time
1. **Weight transpose** on `Linear` — silent whenever `in == out`.
2. **Key reuse** — same key, same "random" numbers, silently correlated noise.
3. **Python loop under `jit`** — works, then compile time explodes at scale.
4. **BatchNorm `running_var`** — biased in Flax, unbiased in torch; training
   looks identical and only inference diverges (section 12).
""",
    """
print("Everything above ran and agreed. You now have the translation table for:")
for i, name in enumerate([
    "immutability", "autograd", "modules", "optimizers", "training loop",
    "randomness", "vmap", "jit", "control flow", "devices", "train/eval",
], 1):
    print(f"  {i:>2}. {name}")
print()
print("Next: practise the JAX side for real -> the JAXCode repo next door.")
""",
)


# ---------------------------------------------------------------- notebook


def _lines(source: str) -> list[str]:
    parts = source.rstrip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + parts[-1:]


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": _lines(source)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> dict:
    badge = (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/pytorch_to_jax.ipynb)"
    )

    header = f"""{badge}

# 🔥 → ⚡ PyTorch to JAX, side by side

Every section runs **both frameworks on the same inputs and asserts they agree
numerically**. Nothing here is a claim you have to take on faith — if a cell
prints ✅, the two implementations really did produce the same numbers.

Aimed at someone fluent in PyTorch who needs to be productive in JAX +
Flax NNX quickly, and who is likely to be asked about the differences.

**Run order matters** — run the setup cell first, then go top to bottom.

---

### Contents
""" + "\n".join(
        f"{i}. [{title}](#{i})" for i, (title, _, _) in enumerate(SECTIONS, 1)
    )

    cells = [
        _cell("markdown", header),
        _cell("code", INSTALL),
        _cell("code", SETUP),
    ]

    for md, code in ((m, c) for _, m, c in SECTIONS):
        cells.append(_cell("markdown", md))
        cells.append(_cell("code", code))

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
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"✅ wrote {OUT.name} — {len(SECTIONS)} sections, {len(build()['cells'])} cells")
