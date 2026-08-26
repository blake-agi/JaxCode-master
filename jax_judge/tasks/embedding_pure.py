"""Problem 18 without Flax."""

_WHY = r"""
### Why this exists alongside problem 18
Interview sandboxes often ship `jax` alone. The API here is kept as close to
the `nnx` version as possible — same class name, same attribute, same
methods — so that practising it reinforces problem 18 rather than competing
with it. Only the key changes hands:

```python
MyEmbedding(100, 8, rngs=nnx.Rngs(params=0))    # nnx
MyEmbedding(100, 8, key=jax.random.key(0))      # here
```

A plain class is not a pytree, so `jax.grad(loss)(layer)` will not work.
Differentiate with respect to the input, or keep `table` outside the object.
"""

TASK = {
    "title": "Embedding without Flax",
    "category": "Core Ops & Layers",
    "number": "b_24",
    "difficulty": "Easy",
    "function_name": "MyEmbedding",
    "hint": (
        "self.table = jax.random.normal(key, (num_embeddings, embedding_dim)) "
        "* 0.02, the GPT-2 convention. __call__ is a gather — "
        "self.table[indices] — and advanced indexing already handles indices "
        "of any shape, so no loop and no one-hot matmul. attend is "
        "x @ self.table.T. Note there is no nnx.Param here, so there is "
        "nothing to unwrap: table IS the array."
    ),
    "description": r"""
Problem 18's embedding table, written with no Flax.

### Signature
```python
class MyEmbedding:
    def __init__(self, num_embeddings, embedding_dim, *, key): ...
    def __call__(self, indices): ...     # (...) int -> (..., embedding_dim)
    def attend(self, x): ...             # (..., embedding_dim) -> (..., num_embeddings)
```

`self.table` is `(num_embeddings, embedding_dim)`, initialised with
`jax.random.normal(...) * 0.02` — the GPT-2 convention, same as problem 18.

### What actually changes
The maths is identical. What disappears is the wrapper:

```python
self.table[indices]        # 18: an nnx.Param that proxies to the array
self.table[indices]        # here: it IS the array — same line, nothing to unwrap
x @ self.table[...].T      # 18: [...] to unwrap explicitly
x @ self.table.T           # here
```

Every question about `.value` vs `[...]` vs `.get_value()` simply stops
existing.

### Weight tying is now visible
`attend` reuses the same array as `__call__`. With a plain attribute you can
*see* there is only one `table`, rather than trusting a module to share it.
""" + _WHY,
    "stub": '''import jax
import jax.numpy as jnp


class MyEmbedding:
    """Integer indices -> dense vectors, and the transpose projection back."""

    def __init__(self, num_embeddings, embedding_dim, *, key):
        pass  # Replace this

    def __call__(self, indices):
        pass  # Replace this

    def attend(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


class MyEmbedding:
    def __init__(self, num_embeddings, embedding_dim, *, key):
        self.table = jax.random.normal(key, (num_embeddings, embedding_dim)) * 0.02

    def __call__(self, indices):
        # A gather. Advanced indexing handles any leading shape, and it costs
        # O(1) per token instead of the O(V) of a one-hot matmul.
        return self.table[indices]

    def attend(self, x):
        # Weight tying: the same array, transposed.
        return x @ self.table.T
''',
    "demo": '''import jax
import jax.numpy as jnp

emb = MyEmbedding(100, 8, key=jax.random.key(0))
print("table:", emb.table.shape)

for idx in [jnp.array(5), jnp.array([1, 2, 3]), jnp.zeros((2, 4), dtype=jnp.int32)]:
    print(f"  indices {str(idx.shape):<8} -> {emb(idx).shape}")

print("attend:", emb.attend(jnp.ones((2, 4, 8))).shape)

g = jax.grad(lambda t: jnp.sum(t[jnp.array([1, 1, 2])]))(emb.table)
print("\\ngrad row 1 (used twice):", float(g[1, 0]))
print("grad row 2 (used once): ", float(g[2, 0]))
''',
    "tests": [
        {
            "name": "Table shape, scale, and reproducibility",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(100, 8, key=jax.random.key(0))
assert hasattr(m, 'table'), "the array attribute must be called 'table', as in problem 18"
assert m.table.shape == (100, 8), f'{m.table.shape} vs (100, 8)'
assert isinstance(m.table, jax.Array), f'table is {type(m.table).__name__}, not a jax array'

std = float(jnp.std(m.table))
assert abs(std - 0.02) < 0.004, f'table std {std:.4f}, expected ~0.02'

assert jnp.allclose({fn}(100, 8, key=jax.random.key(0)).table, m.table), 'same key must be reproducible'
assert not jnp.allclose({fn}(100, 8, key=jax.random.key(1)).table, m.table), 'different keys gave the same table'
""",
        },
        {
            "name": "Lookup works for indices of any shape",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(100, 8, key=jax.random.key(0))
for idx, want in [(jnp.array(5), (8,)),
                  (jnp.array([1, 2, 3]), (3, 8)),
                  (jnp.zeros((2, 4), dtype=jnp.int32), (2, 4, 8))]:
    got = m(idx)
    assert got.shape == want, (
        f'indices {idx.shape} -> {got.shape}, expected {want}. Advanced '
        'indexing handles any leading shape with no loop.'
    )

idx = jnp.array([7, 0, 99])
assert jnp.allclose(m(idx), m.table[idx], atol=1e-7), 'wrong rows'
assert jnp.allclose(m(idx), jax.nn.one_hot(idx, 100) @ m.table, atol=1e-5), (
    'disagrees with one_hot(indices) @ table, which is what a gather IS'
)
""",
        },
        {
            "name": "attend is the transpose projection and ties the weights",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(50, 8, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 4, 8))
out = m.attend(x)
assert out.shape == (2, 4, 50), f'{out.shape} vs (2, 4, 50)'
assert jnp.allclose(out, x @ m.table.T, atol=1e-5), 'should be x @ table.T'

before_lookup = m(jnp.array([3]))
m.table = m.table.at[3].add(1.0)
assert not jnp.allclose(m.attend(x), out, atol=1e-4), 'attend ignored the table'
assert not jnp.allclose(m(jnp.array([3])), before_lookup, atol=1e-4), (
    'lookup ignored the table — attend and __call__ must share one array'
)
""",
        },
        {
            "name": "Gradient is sparse and repeated indices accumulate",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(20, 4, key=jax.random.key(0))
# The class is not a pytree, so differentiate the array directly.
g = jax.grad(lambda t: jnp.sum(t[jnp.array([1, 1, 2])]))(m.table)

assert g.shape == m.table.shape, f'{g.shape}'
touched = jnp.abs(g).sum(axis=1)
assert float(touched[0]) == 0.0, 'row 0 was never looked up; its gradient must be 0'
assert abs(float(touched[1]) - 2 * float(touched[2])) < 1e-5, (
    f'index 1 appears twice, so its gradient should be double index 2: '
    f'{float(touched[1])} vs {float(touched[2])}'
)
""",
        },
        {
            "name": "jit and vmap through __call__",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(30, 6, key=jax.random.key(0))
idx = jnp.array([[1, 2], [3, 4]])
assert jnp.allclose(jax.jit(lambda i: m(i))(idx), m(idx), atol=1e-6), 'jit disagrees'

batched = jax.vmap(lambda i: m(i))(idx)
assert batched.shape == (2, 2, 6), f'{batched.shape} vs (2, 2, 6)'
assert jnp.allclose(batched, m(idx), atol=1e-6), 'vmap disagrees'
""",
        },
    ],
}
