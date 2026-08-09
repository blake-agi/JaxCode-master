"""Causal self-attention — additive masking before the softmax."""

TASK = {
    "title": "Causal Self-Attention",
    "category": "Attention & Transformers",
    "order": 3,
    "difficulty": "Medium",
    "function_name": "causal_attention",
    "hint": (
        "Build the allowed pattern with jnp.tril(jnp.ones((T, T), dtype=bool)) — "
        "row i keeps columns 0..i. Turn it into an additive bias, "
        "jnp.where(mask, 0.0, -1e9), and ADD it to the scaled scores before "
        "jax.nn.softmax(..., axis=-1). The mask is (T, T) with no batch or head "
        "axes; broadcasting handles the rest, so the same code covers (B, T, D) "
        "and (B, H, T, Dh)."
    ),
    "description": r"""
Implement **causal (autoregressive) self-attention**: position $i$ may attend to
positions $0 \dots i$ and to nothing later.

$$\text{out} = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

### Signature
`causal_attention(q, k, v)` with `q`, `k`, `v` all `(..., T, d)` and the same `T`
(this is self-attention). Returns `(..., T, d)`. Leading axes are free, so
`(B, T, D)` and `(B, H, T, D_h)` must both work.

### Rules
- No `jax.nn.dot_product_attention`, no `nnx.MultiHeadAttention`, no
  `is_causal=` shortcut from a library
- Build the mask with `jnp.tril` / `jnp.triu` — no Python loop over positions
- The mask is applied to the **scores**, before `jax.nn.softmax`
- Must be jittable and differentiable

### Add before, do not multiply after
The tempting wrong version is:

```python
w = jax.nn.softmax(scores, axis=-1)
w = w * causal          # WRONG
out = w @ v
```

This does not merely zero the future — it corrupts the past. The softmax
denominator was computed over **all** $T$ positions, including the ones you then
deleted, so row $i$ now sums to $\sum_{j\le i} p_{ij} < 1$ instead of $1$. Two
consequences:

1. **Magnitude collapse.** Row $0$ keeps only $p_{00}$, typically $\approx 1/T$,
   so the first token's output is shrunk by ~$T\times$ while the last token's is
   untouched. The layer applies a position-dependent gain that LayerNorm then
   has to undo.
2. **Information leak.** The denominator is a function of the future keys. Even
   with the weights zeroed, $\partial\,\text{out}_0 / \partial k_5 \ne 0$, so a
   language model trained this way is reading its own labels. It will show an
   impossibly low training loss and generate garbage at inference, because at
   decode time the future keys do not exist.

Adding $-\infty$ (or a large negative number) *before* the softmax makes the
blocked logits contribute exactly $0$ to the denominator, so each row is a proper
distribution over its visible prefix and the gradient w.r.t. future keys is
exactly zero.

### Which large negative number
`-1e9` is the usual choice and is fine in `float32`/`bfloat16`. In `float16` it
overflows to `-inf` — harmless here, but if you ever add two such biases (causal
+ padding) you get `-inf + -inf`, and a fully-masked row then produces `nan`
because `softmax` computes `x - max(x) = -inf - (-inf)`. `jnp.finfo(dtype).min`
or `jnp.where(mask, scores, -1e9)` (replace rather than add) are the defensive
forms. Padding-only rows in a batch are the real-world case that hits this.

### At decode time
With a KV cache, $Q$ has length $1$ while $K$ has length $t+1$ — the mask becomes
all-ones and disappears. If you ever see a `tril` applied to a non-square score
matrix during generation, the alignment is wrong: causality is about absolute
positions, not about the shape of the block you happen to be computing.
""",
    "stub": '''import jax
import jax.numpy as jnp


def causal_attention(q, k, v):
    """Scaled dot-product self-attention with future positions masked out.

    Args:
        q: (..., T, d)
        k: (..., T, d)
        v: (..., T, d)

    Returns:
        (..., T, d) — position i attends only to positions 0..i.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def causal_attention(q, k, v):
    T, d_k = q.shape[-2], q.shape[-1]

    scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, q.dtype))

    # allowed[i, j] is True iff j <= i. Shape (T, T) broadcasts against any
    # number of leading batch/head axes.
    allowed = jnp.tril(jnp.ones((T, T), dtype=bool))

    # Additive bias applied BEFORE the softmax, so blocked logits contribute
    # exactly zero to the denominator.
    bias = jnp.where(allowed, 0.0, -1e9).astype(scores.dtype)
    scores = scores + bias

    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v
''',
    "demo": '''import jax
import jax.numpy as jnp

T, D = 5, 4
q = jax.random.normal(jax.random.key(0), (1, T, D))
k = jax.random.normal(jax.random.key(1), (1, T, D))
v = jnp.arange(T * D, dtype=jnp.float32).reshape(1, T, D)

# Look at the weight matrix the mask produces.
scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(float(D))
allowed = jnp.tril(jnp.ones((T, T), dtype=bool))
w_right = jax.nn.softmax(scores + jnp.where(allowed, 0.0, -1e9), axis=-1)
w_wrong = jax.nn.softmax(scores, axis=-1) * allowed

print("row sums, mask BEFORE softmax:", w_right.sum(-1)[0])
print("row sums, mask AFTER  softmax:", w_wrong.sum(-1)[0])
print("-> the 'after' version shrinks early positions toward zero")

out = causal_attention(q, k, v)
print("position 0 output:", out[0, 0], " == v[0]:", v[0, 0])
''',
    "tests": [
        {
            "name": "Shapes, including a head axis",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (2, 6, 16))
k = jax.random.normal(jax.random.key(1), (2, 6, 16))
v = jax.random.normal(jax.random.key(2), (2, 6, 16))
out = {fn}(q, k, v)
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape} vs (2, 6, 16)'
assert jnp.isfinite(out).all(), 'Non-finite output'

q4 = jax.random.normal(jax.random.key(3), (2, 4, 5, 8))
k4 = jax.random.normal(jax.random.key(4), (2, 4, 5, 8))
v4 = jax.random.normal(jax.random.key(5), (2, 4, 5, 8))
o4 = {fn}(q4, k4, v4)
assert o4.shape == (2, 4, 5, 8), (
    f'{o4.shape} vs (2, 4, 5, 8) — the (T, T) mask must broadcast over any '
    'number of leading axes'
)
assert jnp.allclose(o4[:, 1], {fn}(q4[:, 1], k4[:, 1], v4[:, 1]), atol=1e-5), (
    'Head 1 of the 4-D call disagrees with computing it alone'
)
""",
        },
        {
            "name": "Position 0 copies v[0] exactly",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (1, 4, 8))
k = jax.random.normal(jax.random.key(1), (1, 4, 8))
v = jax.random.normal(jax.random.key(2), (1, 4, 8))
out = {fn}(q, k, v)

assert jnp.allclose(out[:, 0], v[:, 0], atol=1e-5), (
    f'Position 0 sees exactly one key, so its softmax row is [1, 0, 0, 0] and '
    f'the output must equal v[0]. Got {out[:, 0]} vs {v[:, 0]}. If yours is a '
    'shrunken version of v[0], you multiplied the mask in after the softmax '
    'instead of adding it before.'
)
""",
        },
        {
            "name": "Each row equals attention over its own prefix",
            "code": """
import jax
import jax.numpy as jnp

B, T, D = 1, 6, 8
q = jax.random.normal(jax.random.key(0), (B, T, D))
k = jax.random.normal(jax.random.key(1), (B, T, D))
v = jax.random.normal(jax.random.key(2), (B, T, D))
out = {fn}(q, k, v)

for t in range(T):
    # Unmasked attention of query t over keys/values 0..t.
    s = (q[:, t:t + 1] @ jnp.swapaxes(k[:, :t + 1], -1, -2)) / jnp.sqrt(float(D))
    ref = (jax.nn.softmax(s, axis=-1) @ v[:, :t + 1])[:, 0]
    assert jnp.allclose(out[:, t], ref, atol=1e-5), (
        f'Row {t} does not equal renormalised attention over positions 0..{t}. '
        'Each row must be a proper probability distribution over its visible '
        'prefix, which only happens if the mask is added before the softmax.'
    )

# Rewriting the future must leave earlier outputs bit-identical.
k2 = k.at[:, 3:].set(jax.random.normal(jax.random.key(3), (B, T - 3, D)))
v2 = v.at[:, 3:].set(jax.random.normal(jax.random.key(4), (B, T - 3, D)))
assert jnp.allclose(out[:, :3], {fn}(q, k2, v2)[:, :3], atol=1e-5), (
    'Changing K/V at positions 3.. changed the outputs at positions 0..2'
)
""",
        },
        {
            "name": "Gradient w.r.t. the future is exactly zero",
            "code": """
import jax
import jax.numpy as jnp

B, T, D = 1, 6, 4
q = jax.random.normal(jax.random.key(0), (B, T, D))
k = jax.random.normal(jax.random.key(1), (B, T, D))
v = jax.random.normal(jax.random.key(2), (B, T, D))

# A loss that only touches the first two output positions.
def loss(kk, vv):
    return jnp.sum({fn}(q, kk, vv)[:, :2])

gk, gv = jax.grad(loss, argnums=(0, 1))(k, v)

assert jnp.isfinite(gk).all() and jnp.isfinite(gv).all(), 'Non-finite gradients'
assert float(jnp.abs(gv[:, :2]).sum()) > 0.0, 'No gradient reaches the visible values'
assert float(jnp.abs(gv[:, 2:]).sum()) == 0.0, (
    f'd(out[:2])/d(v[2:]) should be exactly 0, got {jnp.abs(gv[:, 2:]).sum()}'
)
assert float(jnp.abs(gk[:, 2:]).sum()) == 0.0, (
    f'd(out[:2])/d(k[2:]) should be exactly 0, got {jnp.abs(gk[:, 2:]).sum()}. '
    'A nonzero value means future keys are still inside the softmax '
    'denominator — that is the information leak a multiply-after mask causes.'
)
""",
        },
        {
            "name": "Attention weights form a valid distribution",
            "code": """
import jax
import jax.numpy as jnp

# Probe the implied weight matrix: with v = I, out[i] IS the weight row i.
T = 7
q = jax.random.normal(jax.random.key(0), (1, T, T)) * 2.0
k = jax.random.normal(jax.random.key(1), (1, T, T)) * 2.0
w = {fn}(q, k, jnp.eye(T)[None])[0]

assert jnp.all(w >= -1e-6), f'Negative attention weights: {w.min()}'
assert jnp.allclose(w.sum(axis=-1), 1.0, atol=1e-4), (
    f'Row sums must all be 1.0, got {w.sum(axis=-1)} — a multiply-after mask '
    'gives sums well below 1 for the early rows'
)
upper = w * (1.0 - jnp.tril(jnp.ones((T, T))))
assert float(jnp.abs(upper).max()) < 1e-6, (
    f'Weights above the diagonal must be 0, largest is {jnp.abs(upper).max()}'
)
assert float(w[0, 0]) > 0.999, f'w[0, 0] must be 1.0, got {w[0, 0]}'
""",
        },
        {
            "name": "Stable at large logits and T=1",
            "code": """
import jax
import jax.numpy as jnp

# Huge scores: a hand-rolled exp/sum overflows; jax.nn.softmax does not.
q = jax.random.normal(jax.random.key(0), (1, 5, 16)) * 50.0
k = jax.random.normal(jax.random.key(1), (1, 5, 16)) * 50.0
v = jax.random.normal(jax.random.key(2), (1, 5, 16))
out = {fn}(q, k, v)
assert jnp.isfinite(out).all(), (
    'Non-finite output at large logits. Either you are doing exp/sum by hand '
    'without subtracting the row max, or a -inf mask hit a row where it is the '
    'maximum. jax.nn.softmax plus a finite -1e9 bias handles both.'
)
lo, hi = jnp.min(v, axis=1), jnp.max(v, axis=1)
assert jnp.all(out >= lo - 1e-3) and jnp.all(out <= hi + 1e-3), (
    'Outputs escaped the range of v, so the weights are not a distribution'
)

# T = 1: the mask is a single True and the output is just v.
one = {fn}(jnp.ones((1, 1, 4)), jnp.ones((1, 1, 4)), jnp.array([[[9.0, 8.0, 7.0, 6.0]]]))
assert one.shape == (1, 1, 4), f'T=1 shape {one.shape}'
assert jnp.allclose(one, jnp.array([[[9.0, 8.0, 7.0, 6.0]]]), atol=1e-6), f'T=1 gave {one}'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (3, 5, 8))
k = jax.random.normal(jax.random.key(1), (3, 5, 8))
v = jax.random.normal(jax.random.key(2), (3, 5, 8))
eager = {fn}(q, k, v)

assert jnp.allclose(jax.jit({fn})(q, k, v), eager, atol=1e-5), 'jit changed the result'

vm = jax.vmap({fn})(q, k, v)
assert vm.shape == (3, 5, 8), f'{vm.shape}'
assert jnp.allclose(vm, eager, atol=1e-5), (
    'vmap over the batch disagrees with the batched call — the mask must be '
    'built from q.shape[-2], not from a fixed axis index'
)

# Differentiating through jit must work too.
g = jax.grad(lambda a: jnp.sum(jax.jit({fn})(a, k, v) ** 2))(q)
assert g.shape == q.shape and jnp.isfinite(g).all(), 'Gradient through jit failed'
""",
        },
    ],
}
