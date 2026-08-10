"""Two-layer MLP as an nnx.Module — raw weight matrices, GELU, expansion ratio."""

TASK = {
    "title": "SwiGLU MLP",
    "category": "Core Ops & Layers",
    "order": 10,
    "difficulty": "Medium",
    "function_name": "MLP",
    "hint": (
        "Store four nnx.Param attributes: w1 (din, hidden), b1 (hidden,), "
        "w2 (hidden, dout), b2 (dout,). Draw each matrix from its OWN key — call "
        "rngs.params() twice, once per matrix, or the two layers end up correlated. "
        "Scale each matrix by 1/sqrt(fan_in): w1 by 1/sqrt(din), w2 by 1/sqrt(hidden). "
        "When hidden is None, default it to 4 * din. Forward is "
        "gelu(x @ w1 + b1) @ w2 + b2 — the @ broadcasts over any leading axes, so "
        "you never need to reshape."
    ),
    "description": r"""
Implement the **position-wise feed-forward network** — the other half of every
transformer block — as an `nnx.Module`, built from your own weight matrices.

$$\text{MLP}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2$$

with $W_1 \in \mathbb{R}^{d_{in} \times d_{hidden}}$ and
$W_2 \in \mathbb{R}^{d_{hidden} \times d_{out}}$.

### Rules
- Subclass `nnx.Module`; do **not** use `nnx.Linear` or `nnx.MLP` — write the
  matmuls yourself
- Signature: `MLP(din, dout, *, hidden=None, rngs)`
- `hidden=None` must default to **`4 * din`** (the transformer convention)
- Four `nnx.Param` attributes named exactly `w1`, `b1`, `w2`, `b2`
- Each matrix initialised as `normal / sqrt(fan_in)`; both biases zero
- Each matrix drawn from a **separate** key (`rngs.params()` twice)
- `__call__(x)` maps `(..., din) -> (..., dout)` for any number of leading axes
- Either GELU form is accepted (exact erf or the tanh approximation)

### The 4x expansion ratio
The hidden width is 4x the model width in GPT-2, GPT-3, BERT, and ViT. That is a
capacity choice, not a law — but it decides where your parameters live:

| Block component | Parameters (d = model width) |
|---|---|
| Attention (Q, K, V, O) | $4d^2$ |
| MLP ($W_1$ + $W_2$ at 4x) | $8d^2$ |

So **two thirds of a transformer's non-embedding parameters sit in the MLPs**,
not in attention. It is also where the activation memory peaks: the intermediate
tensor is `(B, T, 4d)`, four times the size of the residual stream, which is why
FFN blocks are the first thing people rematerialise (`jax.checkpoint`).

### Where the params live in NNX
Assigning `self.w1 = nnx.Param(...)` in `__init__` is the whole registration
story — NNX walks your attributes, so there is no `register_parameter` step and
no separate params dict threaded through calls. `nnx.split(module)` later pulls
those arrays out into a pytree for `jit`/`grad`, and `nnx.merge` puts them back.

One JAX-specific trap: `jax.nn.gelu` defaults to `approximate=True` (the tanh
form), the opposite of most other frameworks' default. If you are porting
published weights, that mismatch is a silent ~1e-3 error in every activation.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """gelu(x @ w1 + b1) @ w2 + b2"""

    def __init__(self, din: int, dout: int, *, hidden: int | None = None, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(..., din) -> (..., dout)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, din: int, dout: int, *, hidden: int | None = None, rngs: nnx.Rngs):
        if hidden is None:
            hidden = 4 * din                      # transformer expansion ratio

        # Two separate keys: reusing one would make w1 and w2 correlated.
        k1 = rngs.params()
        k2 = rngs.params()

        self.w1 = nnx.Param(jax.random.normal(k1, (din, hidden)) / jnp.sqrt(din))
        self.b1 = nnx.Param(jnp.zeros((hidden,)))
        self.w2 = nnx.Param(jax.random.normal(k2, (hidden, dout)) / jnp.sqrt(hidden))
        self.b2 = nnx.Param(jnp.zeros((dout,)))

        self.din, self.dout, self.hidden = din, dout, hidden

    def __call__(self, x):
        h = x @ self.w1 + self.b1                 # (..., hidden)
        h = jax.nn.gelu(h)
        return h @ self.w2 + self.b2              # (..., dout)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

mlp = MLP(64, 64, rngs=nnx.Rngs(params=0))
print("hidden width:", mlp.w1.shape[1], "= 4 x", mlp.w1.shape[0])

x = jnp.ones((2, 5, 64))
print("(B, T, D):", x.shape, "->", mlp(x).shape)
print("intermediate is 4x wider:", (x @ mlp.w1).shape)

n = sum(int(p.size) for p in jax.tree.leaves(nnx.state(mlp, nnx.Param)))
print("params:", n, " ~8*d^2 =", 8 * 64 ** 2)
''',
    "tests": [
        {
            "name": "Parameter shapes and the 4x default",
            "code": """
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 32, rngs=nnx.Rngs(params=0))
assert m.w1.shape == (32, 128), (
    f'w1 should be (din, 4*din) = (32, 128), got {m.w1.shape}. '
    'hidden must default to 4*din, and JAX stores weights as (fan_in, fan_out).'
)
assert m.b1.shape == (128,), f'b1 shape {m.b1.shape} vs (128,)'
assert m.w2.shape == (128, 32), f'w2 should be (hidden, dout) = (128, 32), got {m.w2.shape}'
assert m.b2.shape == (32,), f'b2 shape {m.b2.shape} vs (32,)'

# Explicit hidden must override the default, and din != dout must work.
m2 = {fn}(8, 5, hidden=3, rngs=nnx.Rngs(params=0))
assert m2.w1.shape == (8, 3), f'w1 {m2.w1.shape} vs (8, 3)'
assert m2.w2.shape == (3, 5), f'w2 {m2.w2.shape} vs (3, 5)'
assert m2(jnp.ones((4, 8))).shape == (4, 5), 'din != dout is broken'
""",
        },
        {
            "name": "Params are nnx.Param and discoverable",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 16, rngs=nnx.Rngs(params=0))
for name in ('w1', 'b1', 'w2', 'b2'):
    p = getattr(m, name)
    assert isinstance(p, nnx.Param), f'{name} must be an nnx.Param, got {type(p)}'

leaves = jax.tree.leaves(nnx.state(m, nnx.Param))
assert len(leaves) == 4, f'Expected exactly 4 param leaves, got {len(leaves)}'
total = sum(int(l.size) for l in leaves)
expected = 16 * 64 + 64 + 64 * 16 + 16
assert total == expected, f'Param count {total} vs {expected} (= 8*d^2 + biases)'

assert jnp.allclose(m.b1[...], 0.0), 'b1 must be initialised to zeros'
assert jnp.allclose(m.b2[...], 0.0), 'b2 must be initialised to zeros'
""",
        },
        {
            "name": "Forward matches gelu(x @ w1 + b1) @ w2 + b2",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 16, hidden=32, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 4, 16))
out = m(x)

h = x @ m.w1[...] + m.b1[...]
ref_exact = 0.5 * h * (1 + jax.scipy.special.erf(h / jnp.sqrt(2.0)))
expected = ref_exact @ m.w2[...] + m.b2[...]

assert out.shape == (2, 4, 16), f'Output shape {out.shape} vs (2, 4, 16)'
# atol is loose enough for either GELU form (exact erf or tanh approximation).
assert jnp.allclose(out, expected, atol=5e-3), (
    'Output != gelu(x @ w1 + b1) @ w2 + b2 — check the order of matmul, bias and activation'
)

# Biases must actually be used.
m.b2[...] = m.b2[...] + 100.0
assert jnp.allclose(m(x) - out, 100.0, atol=1e-3), 'b2 is not being added'
""",
        },
        {
            "name": "Activation is GELU, not ReLU or identity",
            "code": """
import jax.numpy as jnp
from flax import nnx

m = {fn}(3, 3, hidden=3, rngs=nnx.Rngs(params=0))
# Make both matrices the identity so the module computes exactly gelu(x).
m.w1[...] = jnp.eye(3)
m.w2[...] = jnp.eye(3)

x = jnp.array([[-1.0, 0.0, 2.0]])
out = m(x)
expected = jnp.array([[-0.15866, 0.0, 1.9545]])

assert jnp.allclose(out, expected, atol=2e-3), (
    f'With identity weights the output should be gelu(x) = {expected}, got {out}'
)
assert float(out[0, 0]) < -0.05, (
    f'gelu(-1) should be about -0.159, got {out[0, 0]} — a value of 0.0 means you '
    'used ReLU instead of GELU'
)

# A purely linear network could not do this: the composition must be non-linear.
lin = m(2.0 * x)
assert not jnp.allclose(lin, 2.0 * out, atol=1e-3), 'f(2x) == 2f(x): no activation applied'
""",
        },
        {
            "name": "Arbitrary leading axes and init scale",
            "code": """
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 4, hidden=16, rngs=nnx.Rngs(params=0))
for shape in [(8,), (3, 8), (2, 5, 8), (2, 3, 4, 8)]:
    out = m(jnp.ones(shape))
    assert out.shape == shape[:-1] + (4,), (
        f'Input {shape} should give {shape[:-1] + (4,)}, got {out.shape}'
    )

big = {fn}(256, 256, rngs=nnx.Rngs(params=0))
s1, s2 = float(jnp.std(big.w1[...])), float(jnp.std(big.w2[...]))
e1, e2 = 1.0 / 16.0, 1.0 / jnp.sqrt(1024.0)   # 1/sqrt(din), 1/sqrt(hidden)
assert abs(s1 - e1) < 0.3 * e1, f'w1 std {s1:.5f}, expected ~{e1:.5f} (normal / sqrt(din))'
assert abs(s2 - e2) < 0.3 * e2, (
    f'w2 std {s2:.5f}, expected ~{e2:.5f} — w2 fan_in is hidden, not din'
)

# Same-shaped matrices drawn from the same key would be identical.
sq = {fn}(64, 64, hidden=64, rngs=nnx.Rngs(params=0))
assert not jnp.allclose(sq.w1[...], sq.w2[...]), (
    'w1 and w2 are identical — call rngs.params() once per matrix instead of reusing one key'
)
""",
        },
        {
            "name": "Gradients reach all four params",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 8, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(1), (4, 8))
y = jax.random.normal(jax.random.key(2), (4, 8))

grads = nnx.grad(lambda mod: jnp.mean((mod(x) - y) ** 2))(m)
names = sorted(grads.keys())
assert names == ['b1', 'b2', 'w1', 'w2'], (
    f'Expected gradients for w1, b1, w2, b2 — got {names}'
)
for name in names:
    g = grads[name]
    # nnx.grad hands back Variables; unwrap without touching the deprecated
    # .value property (even hasattr(g, 'value') would trigger its warning).
    v = g[...] if isinstance(g, nnx.Variable) else g
    assert jnp.isfinite(v).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(v).sum()) > 0, (
        f'Gradient for {name} is all zeros — that param is not in the forward path'
    )
""",
        },
        {
            "name": "jit and vmap via split/merge",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(6, 6, hidden=12, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(3), (5, 6))
eager = m(x)

# nnx.jit handles the split/merge for you.
fast = nnx.jit(lambda mod, inp: mod(inp))(m, x)
assert jnp.allclose(fast, eager, atol=1e-5), 'nnx.jit result differs from the eager result'

# The explicit functional form: pull the params out, close over the graphdef.
graphdef, state = nnx.split(m)

def apply(state, inp):
    return nnx.merge(graphdef, state)(inp)

batched = jax.vmap(apply, in_axes=(None, 0))(state, x)
assert batched.shape == (5, 6), f'vmap output {batched.shape} vs (5, 6)'
assert jnp.allclose(batched, eager, atol=1e-5), (
    'vmapping over the batch axis must agree with calling the module on the whole batch'
)
""",
        },
    ],
}
