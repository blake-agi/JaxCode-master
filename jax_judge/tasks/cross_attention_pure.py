"""Problem 23 without Flax."""

TASK = {
    "title": "Cross-Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_27",
    "difficulty": "Medium",
    "function_name": "init_cross_attention",
    "extra_names": ["apply_cross_attention"],
    "hint": (
        "The pytree is identical to b_26's — four projections, each a kernel "
        "and a bias. The only change is where each one is fed: W_q sees x_q, "
        "while W_k and W_v both see x_kv. Everything downstream is the same "
        "attention. Do not assume seq_q == seq_kv anywhere: the scores are "
        "(..., H, seq_q, seq_kv) and the output length comes from Q."
    ),
    "description": r"""
Problem 23 with an explicit parameter pytree.

### Signature
```python
def init_cross_attention(key, d_model, num_heads):
    ...   # -> {"W_q": {...}, "W_k": {...}, "W_v": {...}, "W_o": {...}}

def apply_cross_attention(params, x_q, x_kv, num_heads):
    ...   # (B, seq_q, d_model), (B, seq_kv, d_model) -> (B, seq_q, d_model)
```

Same pytree as `b_26`: four `(d_model, d_model)` kernels scaled by
`1/sqrt(d_model)`, four zero biases, key split four ways.

### The one line that differs from self-attention
```python
q = W_q(x_q)     # queries from one sequence
k = W_k(x_kv)    # keys and values from the other
v = W_v(x_kv)
```

That is genuinely all of it — which is the point of doing this one right after
`b_26`. If your `apply_mha` was written without naming the batch axis, this is
almost a rename.

### The trap it adds
`seq_q` and `seq_kv` are **different**. The scores are
`(..., H, seq_q, seq_kv)`, the output length comes from `Q`, and softmax runs
over the last axis (the keys). Anything that assumed a square score matrix
breaks here, and a square test case would not notice.

### A property worth checking yourself
Feed the same array as both inputs and you must get exactly self-attention
back. That single assertion catches most wiring mistakes — a swapped `x_q` /
`x_kv`, or `W_k` fed the wrong sequence.
""",
    "stub": '''import jax
import jax.numpy as jnp


def init_cross_attention(key, d_model, num_heads):
    """Parameter pytree: W_q, W_k, W_v, W_o, each a kernel and a bias."""
    pass  # Replace this


def apply_cross_attention(params, x_q, x_kv, num_heads):
    """Queries from x_q attend over keys/values from x_kv."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def init_cross_attention(key, d_model, num_heads):
    keys = jax.random.split(key, 4)
    return {
        name: {
            "kernel": jax.random.normal(k, (d_model, d_model)) / jnp.sqrt(d_model),
            "bias": jnp.zeros((d_model,)),
        }
        for name, k in zip(("W_q", "W_k", "W_v", "W_o"), keys)
    }


def _dense(p, x):
    return x @ p["kernel"] + p["bias"]


def apply_cross_attention(params, x_q, x_kv, num_heads):
    d_model = x_q.shape[-1]
    d_k = d_model // num_heads

    def heads(t):
        return t.reshape(*t.shape[:-1], num_heads, d_k).swapaxes(-3, -2)

    # The whole difference from self-attention: q from x_q, k/v from x_kv.
    q = heads(_dense(params["W_q"], x_q))
    k = heads(_dense(params["W_k"], x_kv))
    v = heads(_dense(params["W_v"], x_kv))

    # (..., H, seq_q, seq_kv) — not square, so nothing may assume it is.
    scores = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
        jnp.asarray(d_k, x_q.dtype)
    )
    o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(scores, axis=-1), v)

    o = o.swapaxes(-3, -2)
    o = o.reshape(*o.shape[:-2], d_model)
    return _dense(params["W_o"], o)
''',
    "demo": '''import jax
import jax.numpy as jnp

params = init_cross_attention(jax.random.key(0), d_model=8, num_heads=2)

x_q = jax.random.normal(jax.random.key(1), (2, 3, 8))    # 3 queries
x_kv = jax.random.normal(jax.random.key(2), (2, 7, 8))   # 7 keys/values
print("seq_q=3, seq_kv=7 ->", apply_cross_attention(params, x_q, x_kv, 2).shape)

# Same input twice must reduce to self-attention.
x = jax.random.normal(jax.random.key(3), (2, 5, 8))
same = apply_cross_attention(params, x, x, 2)
print("cross(x, x) shape: ", same.shape)
print("that IS self-attention — the best single check of the wiring")
''',
    "tests": [
        {
            "name": "Pytree structure and independent kernels",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
assert set(p) == {'W_q', 'W_k', 'W_v', 'W_o'}, f'top-level keys {sorted(p)}'
for name in p:
    assert set(p[name]) == {'kernel', 'bias'}, f'{name} keys {sorted(p[name])}'
    assert p[name]['kernel'].shape == (8, 8), f"{name} kernel {p[name]['kernel'].shape}"
    assert p[name]['bias'].shape == (8,), f"{name} bias {p[name]['bias'].shape}"
assert len(jax.tree.leaves(p)) == 8, f'{len(jax.tree.leaves(p))} leaves, expected 8'

ks = [p[n]['kernel'] for n in ('W_q', 'W_k', 'W_v', 'W_o')]
for i in range(4):
    for j in range(i + 1, 4):
        assert not jnp.allclose(ks[i], ks[j]), 'Two projections share a kernel — split the key 4 ways'
""",
        },
        {
            "name": "seq_q and seq_kv may differ",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
for seq_q, seq_kv in [(3, 7), (7, 3), (1, 5), (4, 4)]:
    xq = jax.random.normal(jax.random.key(1), (2, seq_q, 8))
    xkv = jax.random.normal(jax.random.key(2), (2, seq_kv, 8))
    out = apply_cross_attention(p, xq, xkv, 2)
    assert out.shape == (2, seq_q, 8), (
        f'seq_q={seq_q}, seq_kv={seq_kv} gave {out.shape}, expected {(2, seq_q, 8)} '
        '— the output length comes from Q, the key axis from x_kv'
    )
    assert jnp.isfinite(out).all(), f'Non-finite output at {seq_q}x{seq_kv}'
""",
        },
        {
            "name": "cross(x, x) is exactly self-attention",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 5, 8, 2
p = {fn}(jax.random.key(0), D, H)
x = jax.random.normal(jax.random.key(1), (B, S, D))

d_k = D // H
def dense(pp, t): return t @ pp['kernel'] + pp['bias']
def heads(t): return t.reshape(*t.shape[:-1], H, d_k).swapaxes(-3, -2)
q, k, v = (heads(dense(p[n], x)) for n in ('W_q', 'W_k', 'W_v'))
sc = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(sc, axis=-1), v)
ref = dense(p['W_o'], o.swapaxes(-3, -2).reshape(*o.shape[:-3], S, D))

assert jnp.allclose(apply_cross_attention(p, x, x, H), ref, atol=1e-5), (
    'cross(x, x) must equal self-attention. If it does not, the wiring is off '
    '— check that W_k and W_v both read x_kv and W_q reads x_q.'
)
""",
        },
        {
            "name": "Queries and keys are not interchangeable",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
xq = jax.random.normal(jax.random.key(1), (1, 4, 8))
xkv = jax.random.normal(jax.random.key(2), (1, 4, 8))

a = apply_cross_attention(p, xq, xkv, 2)
b = apply_cross_attention(p, xkv, xq, 2)
assert not jnp.allclose(a, b, atol=1e-4), (
    'Swapping x_q and x_kv changed nothing — one of them is being ignored'
)

# Perturbing x_kv must move the output: keys and values come from there.
xkv2 = xkv.at[:, 0].add(5.0)
assert not jnp.allclose(apply_cross_attention(p, xq, xkv2, 2), a, atol=1e-4), (
    'Changing x_kv did not change the output — W_k / W_v are reading x_q'
)
# And perturbing x_q must move it too.
xq2 = xq.at[:, 0].add(5.0)
assert not jnp.allclose(apply_cross_attention(p, xq2, xkv, 2), a, atol=1e-4), (
    'Changing x_q did not change the output'
)
""",
        },
        {
            "name": "Gradients, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
xq = jax.random.normal(jax.random.key(1), (2, 3, 8))
xkv = jax.random.normal(jax.random.key(2), (2, 7, 8))

g = jax.grad(lambda q: jnp.sum(apply_cross_attention(q, xq, xkv, 2)))(p)
assert len(jax.tree.leaves(g)) == 8, f'{len(jax.tree.leaves(g))} grad leaves, expected 8'
assert all(jnp.isfinite(l).all() for l in jax.tree.leaves(g)), 'Non-finite gradient'
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert float(jnp.abs(g[name]['kernel']).max()) > 0, f'{name} got no gradient'

out = apply_cross_attention(p, xq, xkv, 2)
jf = jax.jit(apply_cross_attention, static_argnums=(3,))
assert jnp.allclose(jf(p, xq, xkv, 2), out, atol=1e-5), 'jit disagrees'

vm = jax.vmap(apply_cross_attention, in_axes=(None, 0, 0, None))(p, xq, xkv, 2)
assert jnp.allclose(vm, out, atol=1e-5), (
    'vmap over the batch disagrees — do not name the batch axis; use negative '
    'axes so vmap can strip it'
)
""",
        },
        {
            "name": "No Flax anywhere",
            "code": """
import sys

assert 'flax' not in sys.modules, 'flax got imported'
""",
        },
    ],
}
