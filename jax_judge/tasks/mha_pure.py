"""Problem 06 without Flax — four projections you own yourself."""

TASK = {
    "title": "Multi-Head Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_26",
    "difficulty": "Medium",
    "function_name": "init_mha",
    "extra_names": ["apply_mha"],
    "hint": (
        "init_mha: split the key four ways and build one (d_model, d_model) "
        "kernel plus a zero bias for each of W_q, W_k, W_v, W_o, scaled by "
        "1/sqrt(d_model). Return them as a NESTED dict keyed by those four "
        "names. apply_mha takes num_heads as an argument because it is a "
        "hyperparameter, not a parameter — pytrees hold arrays only. Split "
        "heads with reshape(B, S, H, d_k).transpose(0, 2, 1, 3) and merge by "
        "transposing back BEFORE the reshape."
    ),
    "description": r"""
Problem 06's multi-head attention with an explicit parameter pytree.

### Signature
```python
def init_mha(key, d_model, num_heads):
    ...   # -> nested pytree of the four projections

def apply_mha(params, Q, K, V, num_heads):
    ...   # (B, seq, d_model) x3 -> (B, seq, d_model)
```

### The parameter pytree
```python
{"W_q": {"kernel": (d_model, d_model), "bias": (d_model,)},
 "W_k": {...}, "W_v": {...}, "W_o": {...}}
```

Four projections, each `(d_model, d_model)`, kernels scaled by
`1/sqrt(d_model)` and biases at zero. Split the key **four ways** so the four
kernels are independent — one key reused four times gives four identical
matrices, silently.

### Hyperparameters are not parameters
`num_heads` is an argument to `apply_mha`, not an entry in `params`. A pytree
holds **arrays**; a stray Python int in it becomes a leaf that `jax.grad` will
try to differentiate and `jax.tree.map` will try to scale. Under `jit`,
`num_heads` is static because it determines shapes.

This is the split a module blurs: `self.num_heads` and `self.W_q` look alike
inside a class, but only one of them is a parameter.

### What is unchanged from 06
Everything else. Split into heads, scale by `1/sqrt(d_k)`, softmax over the key
axis, merge, project. Including the trap:

```python
o = o.swapaxes(-3, -2)                    # (..., H, S, d_k) -> (..., S, H, d_k)
o = o.reshape(*o.shape[:-2], d_model)     # only NOW is the reshape correct
```

Use **negative axes and never name the batch**. `vmap` strips the leading axis
off, so a function that unpacks `B, S, D = Q.shape` stops working the moment
anyone maps over it.

reshape does not reorder memory, so `H` and `d_k` have to be adjacent and in
that order before you collapse them.
""",
    "stub": '''import jax
import jax.numpy as jnp


def init_mha(key, d_model, num_heads):
    """Parameter pytree: W_q, W_k, W_v, W_o, each a kernel and a bias."""
    pass  # Replace this


def apply_mha(params, Q, K, V, num_heads):
    """(B, seq, d_model) x3 -> (B, seq, d_model)."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def init_mha(key, d_model, num_heads):
    # Four independent keys: reusing one would give four identical kernels.
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


def apply_mha(params, Q, K, V, num_heads):
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    def heads(t):
        # (..., S, d_model) -> (..., H, S, d_k). H before d_k in the reshape,
        # because d_model is laid out head-major in memory. Never name the
        # leading axes, so this survives vmap stripping the batch away.
        return t.reshape(*t.shape[:-1], num_heads, d_k).swapaxes(-3, -2)

    q = heads(_dense(params["W_q"], Q))
    k = heads(_dense(params["W_k"], K))
    v = heads(_dense(params["W_v"], V))

    scores = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
        jnp.asarray(d_k, Q.dtype)
    )
    o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(scores, axis=-1), v)

    # Swap FIRST so H and d_k are adjacent, then collapse them.
    o = o.swapaxes(-3, -2)
    o = o.reshape(*o.shape[:-2], d_model)
    return _dense(params["W_o"], o)
''',
    "demo": '''import jax
import jax.numpy as jnp

params = init_mha(jax.random.key(0), d_model=8, num_heads=2)
print("pytree:", jax.tree.map(lambda a: a.shape, params))
print("leaves:", len(jax.tree.leaves(params)))

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
print("\\nself-attention:", apply_mha(params, x, x, x, 2).shape)

xq = jax.random.normal(jax.random.key(2), (2, 3, 8))
print("cross shapes:  ", apply_mha(params, xq, x, x, 2).shape)

g = jax.grad(lambda p: jnp.sum(apply_mha(p, x, x, x, 2)))(params)
print("\\ngrad reaches", len(jax.tree.leaves(g)), "leaves")
''',
    "tests": [
        {
            "name": "Pytree structure: four projections, kernel and bias each",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
assert isinstance(p, dict), f'params should be a dict, got {type(p).__name__}'
assert set(p) == {'W_q', 'W_k', 'W_v', 'W_o'}, f'top-level keys {sorted(p)}'
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    sub = p[name]
    assert set(sub) == {'kernel', 'bias'}, f'{name} keys {sorted(sub)}'
    assert sub['kernel'].shape == (8, 8), f"{name} kernel {sub['kernel'].shape} vs (8, 8)"
    assert sub['bias'].shape == (8,), f"{name} bias {sub['bias'].shape} vs (8,)"
    assert jnp.allclose(sub['bias'], 0.0), f'{name} bias should start at zeros'

leaves = jax.tree.leaves(p)
assert len(leaves) == 8, f'{len(leaves)} leaves, expected 8 (4 kernels + 4 biases)'
assert all(hasattr(l, 'shape') for l in leaves), (
    'Every leaf must be an array — num_heads and d_model are hyperparameters '
    'and do not belong in the pytree'
)
""",
        },
        {
            "name": "The four kernels are independent",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 16, 4)
ks = [p[n]['kernel'] for n in ('W_q', 'W_k', 'W_v', 'W_o')]
for i in range(4):
    for j in range(i + 1, 4):
        assert not jnp.allclose(ks[i], ks[j]), (
            'Two projections got identical kernels — split the key four ways '
            'with jax.random.split(key, 4); one key used four times gives four '
            'identical matrices and no error'
        )

std = float(jnp.std(ks[0]))
assert abs(std - 1.0 / 16 ** 0.5) < 0.03, (
    f'kernel std {std:.4f}, expected ~{1/16**0.5:.4f} = 1/sqrt(d_model)'
)
assert jnp.allclose({fn}(jax.random.key(0), 16, 4)['W_q']['kernel'], ks[0]), (
    'Same key must be reproducible'
)
""",
        },
        {
            "name": "Matches attention computed the long way",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 5, 8, 2
p = {fn}(jax.random.key(0), D, H)
x = jax.random.normal(jax.random.key(1), (B, S, D))
out = apply_mha(p, x, x, x, H)
assert out.shape == (B, S, D), f'{out.shape} vs {(B, S, D)}'

d_k = D // H
def dense(pp, t): return t @ pp['kernel'] + pp['bias']
def heads(t): return t.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
q, k, v = (heads(dense(p[n], x)) for n in ('W_q', 'W_k', 'W_v'))
sc = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(sc, axis=-1), v)
ref = dense(p['W_o'], o.transpose(0, 2, 1, 3).reshape(B, S, D))
assert jnp.allclose(out, ref, atol=1e-5), (
    'Disagrees with the reference computation. Common causes: scaling by '
    'sqrt(d_model) instead of sqrt(d_k), or reshaping to merge heads before '
    'transposing them next to d_k.'
)
""",
        },
        {
            "name": "Heads are independent, and the merge order is right",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 1, 4, 8, 2
p = {fn}(jax.random.key(0), D, H)
x = jax.random.normal(jax.random.key(1), (B, S, D))

# Zero W_o so the output is exactly the merged head block, then check that the
# two heads land in the right halves. A transpose-after-reshape bug interleaves
# them instead: right shape, wrong bytes.
p0 = jax.tree.map(lambda a: a, p)
p0['W_o'] = {'kernel': jnp.eye(D), 'bias': jnp.zeros((D,))}
merged = apply_mha(p0, x, x, x, H)

d_k = D // H
def dense(pp, t): return t @ pp['kernel'] + pp['bias']
def heads(t): return t.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
q, k, v = (heads(dense(p0[n], x)) for n in ('W_q', 'W_k', 'W_v'))
sc = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(sc, axis=-1), v)

assert jnp.allclose(merged[..., :d_k], o[:, 0], atol=1e-5), (
    'The first d_k channels should be head 0 — transpose to (B, S, H, d_k) '
    'BEFORE the reshape, not after'
)
assert jnp.allclose(merged[..., d_k:], o[:, 1], atol=1e-5), 'Head 1 is misplaced'
""",
        },
        {
            "name": "Gradients reach all eight leaves; jit and vmap work",
            "code": """
import jax
import jax.numpy as jnp

B, S, D, H = 2, 4, 8, 2
p = {fn}(jax.random.key(0), D, H)
x = jax.random.normal(jax.random.key(1), (B, S, D))

g = jax.grad(lambda q: jnp.sum(apply_mha(q, x, x, x, H)))(p)
leaves = jax.tree.leaves(g)
assert len(leaves) == 8, f'grad has {len(leaves)} leaves, expected 8'
assert all(jnp.isfinite(l).all() for l in leaves), 'Non-finite gradient'
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert float(jnp.abs(g[name]['kernel']).max()) > 0, (
        f'{name} kernel got no gradient — it is not being used'
    )

out = apply_mha(p, x, x, x, H)
jf = jax.jit(apply_mha, static_argnums=(4,))
assert jnp.allclose(jf(p, x, x, x, H), out, atol=1e-5), 'jit disagrees'

vm = jax.vmap(apply_mha, in_axes=(None, 0, 0, 0, None))(p, x, x, x, H)
assert vm.shape == (B, S, D), f'{vm.shape}'
assert jnp.allclose(vm, out, atol=1e-5), 'vmap over the batch disagrees'
""",
        },
        {
            "name": "No Flax anywhere",
            "code": """
import sys

assert 'flax' not in sys.modules, (
    'flax got imported — the four projections are yours to build here'
)
""",
        },
    ],
}
