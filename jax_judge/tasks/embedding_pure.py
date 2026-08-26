"""Problem 18 without Flax."""

TASK = {
    "title": "Embedding without Flax",
    "category": "Core Ops & Layers",
    "number": "b_24",
    "difficulty": "Easy",
    "function_name": "init_embedding",
    "extra_names": ["apply_embedding", "attend_embedding"],
    "hint": (
        "init_embedding returns {'table': (num_embeddings, embedding_dim)} "
        "scaled by 0.02, the GPT-2 convention. apply_embedding is a gather — "
        "params['table'][indices] — and advanced indexing already handles "
        "indices of any shape, so there is no loop and no one-hot matmul. "
        "attend_embedding is x @ params['table'].T; with a plain array there "
        "is no Param wrapper to unwrap."
    ),
    "description": r"""
Problem 18's embedding table, with a plain pytree instead of an `nnx.Module`.

### Signature
```python
def init_embedding(key, num_embeddings, embedding_dim):
    ...   # -> {"table": (num_embeddings, embedding_dim)}

def apply_embedding(params, indices):
    ...   # (...) int -> (..., embedding_dim)

def attend_embedding(params, x):
    ...   # (..., embedding_dim) -> (..., num_embeddings)
```

Initialise the table with `jax.random.normal(...) * 0.02` — the GPT-2
convention, same as problem 18.

### What changes, and what does not
The **maths is identical**. What goes away is the wrapper:

```python
self.table[indices]          # 18: an nnx.Param, which proxies to the array
params["table"][indices]     # here: it IS the array
x @ self.table[...].T        # 18: [...] to unwrap explicitly
x @ params["table"].T        # here: nothing to unwrap
```

Every question about `.value` vs `[...]` vs `.get_value()` simply stops
existing. That is the trade: you lose the module's bookkeeping and you gain
one less layer between you and the array.

### Weight tying is now obvious
`attend_embedding` reuses the same array as `apply_embedding`, which is the
whole point of weight tying — and with an explicit pytree you can *see* that
there is only one `table` in it, rather than trusting a module to share it.
""",
    "stub": '''import jax
import jax.numpy as jnp


def init_embedding(key, num_embeddings, embedding_dim):
    """Build the parameter pytree: one (num_embeddings, embedding_dim) table."""
    pass  # Replace this


def apply_embedding(params, indices):
    """Integer indices of any shape -> (..., embedding_dim)."""
    pass  # Replace this


def attend_embedding(params, x):
    """(..., embedding_dim) -> (..., num_embeddings). Transpose projection."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def init_embedding(key, num_embeddings, embedding_dim):
    return {
        "table": jax.random.normal(key, (num_embeddings, embedding_dim)) * 0.02
    }


def apply_embedding(params, indices):
    # A gather. Advanced indexing already handles any leading shape, and it is
    # O(1) per token instead of the O(V) a one-hot matmul would cost.
    return params["table"][indices]


def attend_embedding(params, x):
    # Weight tying: the same array, transposed.
    return x @ params["table"].T
''',
    "demo": '''import jax
import jax.numpy as jnp

params = init_embedding(jax.random.key(0), 100, 8)
print("table:", params["table"].shape)

for idx in [jnp.array(5), jnp.array([1, 2, 3]), jnp.zeros((2, 4), dtype=jnp.int32)]:
    print(f"  indices {str(idx.shape):<8} -> {apply_embedding(params, idx).shape}")

print("attend:", attend_embedding(params, jnp.ones((2, 4, 8))).shape)

# Repeated indices must accumulate in the gradient.
g = jax.grad(lambda p: jnp.sum(apply_embedding(p, jnp.array([1, 1, 2]))))(params)
print("\\ngrad row 1 (used twice):", g["table"][1, 0])
print("grad row 2 (used once): ", g["table"][2, 0])
''',
    "tests": [
        {
            "name": "Table shape, scale, and reproducibility",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 100, 8)
assert isinstance(p, dict) and set(p) == {'table'}, f'keys {sorted(p)} vs [table]'
assert p['table'].shape == (100, 8), f"{p['table'].shape} vs (100, 8)"

std = float(jnp.std(p['table']))
assert abs(std - 0.02) < 0.004, f'table std {std:.4f}, expected ~0.02'

same = {fn}(jax.random.key(0), 100, 8)['table']
other = {fn}(jax.random.key(1), 100, 8)['table']
assert jnp.allclose(p['table'], same), 'Same key must be reproducible'
assert not jnp.allclose(p['table'], other), 'Different keys gave identical tables'
""",
        },
        {
            "name": "Lookup works for indices of any shape",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 100, 8)
cases = [(jnp.array(5), (8,)),
         (jnp.array([1, 2, 3]), (3, 8)),
         (jnp.zeros((2, 4), dtype=jnp.int32), (2, 4, 8))]
for idx, want in cases:
    got = apply_embedding(p, idx)
    assert got.shape == want, (
        f'indices {idx.shape} -> {got.shape}, expected {want}. Advanced '
        'indexing handles any leading shape with no loop.'
    )

# The rows must be the actual table rows.
idx = jnp.array([7, 0, 99])
assert jnp.allclose(apply_embedding(p, idx), p['table'][idx], atol=1e-7), 'Wrong rows'

# Same answer as the one-hot matmul, which is what a gather IS.
oh = jax.nn.one_hot(idx, 100) @ p['table']
assert jnp.allclose(apply_embedding(p, idx), oh, atol=1e-5), (
    'Disagrees with one_hot(indices) @ table'
)
""",
        },
        {
            "name": "attend is the transpose projection and ties the weights",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 50, 8)
x = jax.random.normal(jax.random.key(1), (2, 4, 8))
out = attend_embedding(p, x)
assert out.shape == (2, 4, 50), f'{out.shape} vs (2, 4, 50)'
assert jnp.allclose(out, x @ p['table'].T, atol=1e-5), 'Should be x @ table.T'

# Tied: perturbing the table must move BOTH directions.
p2 = {'table': p['table'].at[3].add(1.0)}
assert not jnp.allclose(attend_embedding(p2, x), out, atol=1e-4), 'attend ignored the table'
assert not jnp.allclose(apply_embedding(p2, jnp.array([3])),
                        apply_embedding(p, jnp.array([3])), atol=1e-4), (
    'Lookup ignored the table — attend and lookup must share one array'
)
""",
        },
        {
            "name": "Gradient is sparse and repeated indices accumulate",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 20, 4)
g = jax.grad(lambda q: jnp.sum(apply_embedding(q, jnp.array([1, 1, 2]))))(p)

assert set(g) == {'table'}, f'grad keys {sorted(g)}'
assert g['table'].shape == p['table'].shape, f"{g['table'].shape}"
touched = jnp.abs(g['table']).sum(axis=1)
assert float(touched[0]) == 0.0, 'Row 0 was never looked up; its gradient must be 0'
assert float(touched[1]) > 0 and float(touched[2]) > 0, 'Rows 1 and 2 should have gradient'
assert abs(float(touched[1]) - 2 * float(touched[2])) < 1e-5, (
    f'Index 1 appears twice, so its gradient should be double index 2: '
    f'{float(touched[1])} vs {float(touched[2])}'
)
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 30, 6)
idx = jnp.array([[1, 2], [3, 4]])
assert jnp.allclose(jax.jit(apply_embedding)(p, idx), apply_embedding(p, idx), atol=1e-6), 'jit disagrees'

batched = jax.vmap(apply_embedding, in_axes=(None, 0))(p, idx)
assert batched.shape == (2, 2, 6), f'{batched.shape} vs (2, 2, 6)'
assert jnp.allclose(batched, apply_embedding(p, idx), atol=1e-6), 'vmap disagrees'
""",
        },
        {
            "name": "No Flax anywhere",
            "code": """
import sys

assert 'flax' not in sys.modules, (
    'flax got imported — this problem exists to be runnable without it'
)
""",
        },
    ],
}
