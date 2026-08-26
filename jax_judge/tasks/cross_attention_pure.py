"""Problem 23 without Flax."""

_LINEAR = '''class Linear:
    """Given to you, exactly as nnx.Linear is given to you in problem 23."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

TASK = {
    "title": "Cross-Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_27",
    "difficulty": "Medium",
    "function_name": "MultiHeadCrossAttention",
    "hint": (
        "Identical to b_26 apart from one line: W_q reads x_q while W_k and "
        "W_v both read x_kv. Do not assume seq_q == seq_kv — the scores are "
        "(..., H, seq_q, seq_kv) and the output length comes from x_q. If your "
        "b_26 __call__ avoided naming the batch axis, this is nearly a rename."
    ),
    "description": r"""
Problem 23 with no Flax — and, if you have done `b_26`, almost no new code.

### Signature
```python
class MultiHeadCrossAttention:
    def __init__(self, d_model, num_heads, *, key): ...
    def __call__(self, x_q, x_kv): ...
    # (B, seq_q, d_model), (B, seq_kv, d_model) -> (B, seq_q, d_model)
```

Same four projections as `b_26`: `W_q`, `W_k`, `W_v`, `W_o`, each
`Linear(d_model, d_model)`, from `jax.random.split(key, 4)`.

### The one line that differs
```python
q = self.W_q(x_q)      # queries from one sequence
k = self.W_k(x_kv)     # keys and values from the other
v = self.W_v(x_kv)
```

That is genuinely all of it — which is why this sits right after `b_26`.

### The trap it adds
`seq_q` and `seq_kv` **differ**. The scores are `(..., H, seq_q, seq_kv)`, the
output length comes from `x_q`, and softmax runs over the last axis (the keys).
Anything that assumed a square score matrix breaks here, and a square test case
would not notice.

### A property worth checking yourself
Feed the same array as both inputs and you must get exactly self-attention
back. That single assertion catches most wiring mistakes — a swapped
`x_q`/`x_kv`, or `W_k` reading the wrong sequence.

### Why this exists alongside problem 23
Interview sandboxes often ship `jax` alone. The API is kept as close to the
`nnx` version as it can be — same class name, same arguments, same attribute
names — so practising it reinforces problem 23 instead of competing with it.
Only `rngs=nnx.Rngs(params=0)` becomes `key=jax.random.key(0)`, and `Linear` is
handed to you the way `nnx.Linear` is.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class MultiHeadCrossAttention:
    """Queries from x_q attend over keys/values from x_kv."""

    def __init__(self, d_model, num_heads, *, key):
        pass  # Replace this

    def __call__(self, x_q, x_kv):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class MultiHeadCrossAttention:
    def __init__(self, d_model, num_heads, *, key):
        self.h = num_heads
        self.d_k = d_model // num_heads
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.W_q = Linear(d_model, d_model, key=kq)
        self.W_k = Linear(d_model, d_model, key=kk)
        self.W_v = Linear(d_model, d_model, key=kv)
        self.W_o = Linear(d_model, d_model, key=ko)

    def __call__(self, x_q, x_kv):
        split = lambda t: t.reshape(*t.shape[:-1], self.h, self.d_k).swapaxes(-3, -2)
        # The entire difference from self-attention is these three lines.
        q = split(self.W_q(x_q))
        k = split(self.W_k(x_kv))
        v = split(self.W_v(x_kv))

        # (..., h, seq_q, seq_kv) — not square, so nothing may assume it is.
        s = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, q.dtype)
        )
        o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(s, axis=-1), v)

        o = o.swapaxes(-3, -2)
        return self.W_o(o.reshape(*o.shape[:-2], self.h * self.d_k))
''',
    "demo": '''import jax
import jax.numpy as jnp

ca = MultiHeadCrossAttention(8, 2, key=jax.random.key(0))

x_q = jax.random.normal(jax.random.key(1), (2, 3, 8))    # 3 queries
x_kv = jax.random.normal(jax.random.key(2), (2, 7, 8))   # 7 keys/values
print("seq_q=3, seq_kv=7 ->", ca(x_q, x_kv).shape)

x = jax.random.normal(jax.random.key(3), (2, 5, 8))
print("\\ncross(x, x) is self-attention — the best single check of the wiring")
print("  shape:", ca(x, x).shape)
''',
    "tests": [
        {
            "name": "Sub-layers named and shaped like the nnx version",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert hasattr(m, name), f'missing sub-layer {name}'
    assert getattr(m, name).kernel.shape == (8, 8), f'{name}.kernel wrong shape'

ks = [getattr(m, n).kernel for n in ('W_q', 'W_k', 'W_v', 'W_o')]
for i in range(4):
    for j in range(i + 1, 4):
        assert not jnp.allclose(ks[i], ks[j]), 'two projections share a kernel — split the key 4 ways'
""",
        },
        {
            "name": "seq_q and seq_kv may differ",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
for seq_q, seq_kv in [(3, 7), (7, 3), (1, 5), (4, 4)]:
    xq = jax.random.normal(jax.random.key(1), (2, seq_q, 8))
    xkv = jax.random.normal(jax.random.key(2), (2, seq_kv, 8))
    out = m(xq, xkv)
    assert out.shape == (2, seq_q, 8), (
        f'seq_q={seq_q}, seq_kv={seq_kv} gave {out.shape}, expected {(2, seq_q, 8)} '
        '— the output length comes from x_q'
    )
    assert jnp.isfinite(out).all(), f'non-finite output at {seq_q}x{seq_kv}'
""",
        },
        {
            "name": "cross(x, x) is exactly self-attention",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 5, 8, 2
m = {fn}(D, H, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (B, S, D))

d_k = D // H
sp = lambda t: t.reshape(B, S, H, d_k).swapaxes(-3, -2)
q, k, v = sp(m.W_q(x)), sp(m.W_k(x)), sp(m.W_v(x))
s = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(s, axis=-1), v)
ref = m.W_o(o.swapaxes(-3, -2).reshape(B, S, D))

assert jnp.allclose(m(x, x), ref, atol=1e-5), (
    'cross(x, x) must equal self-attention. Check that W_k and W_v both read '
    'x_kv while W_q reads x_q.'
)
""",
        },
        {
            "name": "Queries and keys are not interchangeable",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
xq = jax.random.normal(jax.random.key(1), (1, 4, 8))
xkv = jax.random.normal(jax.random.key(2), (1, 4, 8))
a = m(xq, xkv)

assert not jnp.allclose(a, m(xkv, xq), atol=1e-4), (
    'swapping x_q and x_kv changed nothing — one of them is being ignored'
)
assert not jnp.allclose(m(xq, xkv.at[:, 0].add(5.0)), a, atol=1e-4), (
    'changing x_kv did not change the output — W_k / W_v are reading x_q'
)
assert not jnp.allclose(m(xq.at[:, 0].add(5.0), xkv), a, atol=1e-4), (
    'changing x_q did not change the output'
)
""",
        },
        {
            "name": "Gradient w.r.t. the input, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
xq = jax.random.normal(jax.random.key(1), (2, 3, 8))
xkv = jax.random.normal(jax.random.key(2), (2, 7, 8))
out = m(xq, xkv)

g = jax.grad(lambda v: jnp.sum(m(v, xkv)))(xq)
assert g.shape == xq.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. x_q'

assert jnp.allclose(jax.jit(lambda a, b: m(a, b))(xq, xkv), out, atol=1e-5), 'jit disagrees'

vm = jax.vmap(lambda a, b: m(a, b))(xq, xkv)
assert jnp.allclose(vm, out, atol=1e-5), (
    'vmap over the batch disagrees — use negative axes so vmap can strip it'
)
""",
        },
    ],
}
