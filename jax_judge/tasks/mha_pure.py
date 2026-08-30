"""Problem 06 without Flax — small enough to type from memory."""

_LINEAR = '''class Linear:
    """Given to you, exactly as nnx.Linear is given to you in problem 06."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

_WHY = r"""
### Why this exists alongside problem 06
Interview sandboxes often ship `jax` alone, so every `nnx.Module` problem here
is unrunnable there. The API is kept as close to the `nnx` version as it can
be — same class name, same constructor arguments, same `W_q`/`W_k`/`W_v`/`W_o`
attributes — so that practising this reinforces problem 06 rather than
competing with it. Only the key changes hands:

```python
MultiHeadAttention(8, 2, rngs=nnx.Rngs(params=0))   # nnx
MultiHeadAttention(8, 2, key=jax.random.key(0))     # here
```

`Linear` is handed to you in the starter cell for the same reason `nnx.Linear`
is: the exercise is the attention, not a dense layer typed four times. It is
six lines, and `b_23` is where you write it yourself.

A plain class is not a pytree, so `jax.grad(loss)(layer)` does not work —
differentiate with respect to the input instead.
"""

TASK = {
    "title": "Multi-Head Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_26",
    "difficulty": "Medium",
    "function_name": "MultiHeadAttention",
    "hint": (
        "__init__: self.h and self.d_k = d_model // num_heads, then four "
        "Linear(d_model, d_model) from jax.random.split(key, 4). __call__: a "
        "one-line split helper, "
        "t.reshape(*t.shape[:-1], self.h, self.d_k).swapaxes(-3, -2), applied "
        "to each projection; einsum for the scores; softmax; einsum for the "
        "output; then swapaxes BACK before the final reshape. Scale by "
        "sqrt(d_k), not sqrt(d_model)."
    ),
    "description": r"""
Problem 06's multi-head attention with no Flax — and short enough to write
from memory under interview conditions.

### Signature
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, *, key): ...
    def __call__(self, Q, K, V): ...     # (B, seq, d_model) x3 -> (B, seq, d_model)
```

Four projections named `W_q`, `W_k`, `W_v`, `W_o`, each `Linear(d_model,
d_model)`, built from **four independent keys**:

```python
kq, kk, kv, ko = jax.random.split(key, 4)
```

One key used four times gives four identical matrices and raises nothing.

Scale by `1/sqrt(d_k)` where `d_k = d_model // num_heads` — not
`1/sqrt(d_model)`.

### Splitting and merging heads
The one place this reliably goes wrong. `reshape` re-divides memory, it never
reorders it, so `H` and `d_k` must be adjacent **and in that order**:

```python
split = lambda t: t.reshape(*t.shape[:-1], self.h, self.d_k).swapaxes(-3, -2)
...
o = o.swapaxes(-3, -2)                       # put H next to d_k again
o = o.reshape(*o.shape[:-2], self.h * self.d_k)   # only NOW may you collapse
```

Naming the batch (`B, S, D = Q.shape`) is fine here — the contract is 3-D and
the `nnx` original does exactly that. Negative axes (`*t.shape[:-1]`) are worth
the habit anyway, because they keep the same code working under `vmap`, but
nothing here requires it.

### A `Linear` is provided
The starter cell gives you this, the same way problem 06 gives you
`nnx.Linear`:

```python
class Linear:
    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
```
""" + _WHY,
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class MultiHeadAttention:
    """Q, K, V -> attention over num_heads heads."""

    def __init__(self, d_model, num_heads, *, key):
        pass  # Replace this

    def __call__(self, Q, K, V):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, *, key):
        self.h = num_heads
        self.d_k = d_model // num_heads
        # Four independent keys: one reused four times gives four identical
        # projections, silently.
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.W_q = Linear(d_model, d_model, key=kq)
        self.W_k = Linear(d_model, d_model, key=kk)
        self.W_v = Linear(d_model, d_model, key=kv)
        self.W_o = Linear(d_model, d_model, key=ko)

    def __call__(self, Q, K, V):
        # Negative axes throughout, so vmap can strip the batch away.
        split = lambda t: t.reshape(*t.shape[:-1], self.h, self.d_k).swapaxes(-3, -2)
        q, k, v = split(self.W_q(Q)), split(self.W_k(K)), split(self.W_v(V))

        s = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, q.dtype)
        )
        o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(s, axis=-1), v)

        # Swap FIRST so h and d_k are adjacent, then collapse them.
        o = o.swapaxes(-3, -2)
        return self.W_o(o.reshape(*o.shape[:-2], self.h * self.d_k))
''',
    "demo": '''import jax
import jax.numpy as jnp

mha = MultiHeadAttention(8, 2, key=jax.random.key(0))
print("W_q kernel:", mha.W_q.kernel.shape, " d_k:", mha.d_k)

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
print("self-attention:", mha(x, x, x).shape)

xq = jax.random.normal(jax.random.key(2), (2, 3, 8))
print("different seq_q:", mha(xq, x, x).shape)

g = jax.grad(lambda v: jnp.sum(mha(v, v, v)))(x)
print("\\nd/dx shape:", g.shape)
''',
    "tests": [
        {
            "name": "Sub-layers are named and shaped like the nnx version",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert hasattr(m, name), f"missing sub-layer {name} — keep problem 06's names"
    sub = getattr(m, name)
    assert hasattr(sub, 'kernel'), f'{name} should be a Linear with a .kernel'
    assert sub.kernel.shape == (8, 8), f'{name}.kernel {sub.kernel.shape} vs (8, 8)'

ks = [getattr(m, n).kernel for n in ('W_q', 'W_k', 'W_v', 'W_o')]
for i in range(4):
    for j in range(i + 1, 4):
        assert not jnp.allclose(ks[i], ks[j]), (
            'two projections got identical kernels — split the key four ways '
            'with jax.random.split(key, 4)'
        )
""",
        },
        {
            "name": "Matches attention computed the long way",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 5, 8, 2
m = {fn}(D, H, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (B, S, D))
out = m(x, x, x)
assert out.shape == (B, S, D), f'{out.shape} vs {(B, S, D)}'

d_k = D // H
sp = lambda t: t.reshape(B, S, H, d_k).swapaxes(-3, -2)
q, k, v = sp(m.W_q(x)), sp(m.W_k(x)), sp(m.W_v(x))
s = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(s, axis=-1), v)
ref = m.W_o(o.swapaxes(-3, -2).reshape(B, S, D))
assert jnp.allclose(out, ref, atol=1e-5), (
    'disagrees with the reference. Common causes: scaling by sqrt(d_model) '
    'instead of sqrt(d_k), or reshaping to merge heads before swapping them '
    'next to d_k.'
)
""",
        },
        {
            "name": "Heads land in the right channels after the merge",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 1, 4, 8, 2
m = {fn}(D, H, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (B, S, D))

# Make W_o the identity so the output IS the merged head block.
m.W_o.kernel = jnp.eye(D)
m.W_o.bias = jnp.zeros((D,))
merged = m(x, x, x)

d_k = D // H
sp = lambda t: t.reshape(B, S, H, d_k).swapaxes(-3, -2)
q, k, v = sp(m.W_q(x)), sp(m.W_k(x)), sp(m.W_v(x))
s = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(s, axis=-1), v)

assert jnp.allclose(merged[..., :d_k], o[:, 0], atol=1e-5), (
    'the first d_k channels should be head 0 — swap to (..., S, H, d_k) BEFORE '
    'the reshape, not after'
)
assert jnp.allclose(merged[..., d_k:], o[:, 1], atol=1e-5), 'head 1 is misplaced'
""",
        },
        {
            "name": "Q, K and V are used in the right places",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
q = jax.random.normal(jax.random.key(1), (1, 3, 8))
kv = jax.random.normal(jax.random.key(2), (1, 6, 8))

out = m(q, kv, kv)
assert out.shape == (1, 3, 8), (
    f'{out.shape} — the output length comes from Q, so seq_q != seq_k must work'
)

# Perturbing V must move the output; perturbing K must too (it changes the
# weights); swapping K and V must not be a no-op.
assert not jnp.allclose(m(q, kv, kv.at[:, 0].add(9.0)), out, atol=1e-4), 'V is ignored'
assert not jnp.allclose(m(q, kv.at[:, 0].add(9.0), kv), out, atol=1e-4), 'K is ignored'
assert not jnp.allclose(m(kv[:, :3], kv, kv), out, atol=1e-4), 'Q is ignored'
""",
        },
        {
            "name": "Gradient w.r.t. the input, and jit",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 4, 8, 2
m = {fn}(D, H, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (B, S, D))
out = m(x, x, x)

g = jax.grad(lambda v: jnp.sum(m(v, v, v)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. the input'

assert jnp.allclose(jax.jit(lambda v: m(v, v, v))(x), out, atol=1e-5), 'jit disagrees'
""",
        },
    ],
}
