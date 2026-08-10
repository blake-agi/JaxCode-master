"""Embedding lookup — indexing vs one-hot matmul, and why gradients are sparse."""

TASK = {
    "title": "Embedding Layer",
    "category": "Core Ops & Layers",
    "order": 9,
    "difficulty": "Easy",
    "function_name": "MyEmbedding",
    "hint": (
        "The forward pass is one line: self.table[ids]. JAX's advanced indexing "
        "handles any shape of ids and appends the feature axis, so (B, T) ids "
        "gives (B, T, features) with no reshaping. For attend(), project back "
        "with x @ self.table.T."
    ),
    "description": r"""
Implement an **embedding table** as an `nnx.Module`.

Map integer token ids to dense vectors: `(...) -> (..., features)`.

### Rules
- Signature: `MyEmbedding(num_embeddings, features, *, rngs)`
- Table stored as `self.table`, an `nnx.Param` of shape `(num_embeddings, features)`
- Initialise with `jax.random.normal(...) * 0.02` (the GPT-2 convention)
- `__call__(ids)` accepts **any** shape of integer ids
- Also implement `attend(x)`: `(..., features) -> (..., num_embeddings)`,
  the transpose projection used for weight tying

### Indexing vs one-hot
These compute the same thing:

```python
table[ids]                        # gather
jax.nn.one_hot(ids, V) @ table    # matmul
```

The gather is `O(1)` per token; the matmul is `O(V)` per token and materialises a
`(B, T, V)` intermediate — with `V = 50257` that is enormous. Always gather.

(The one-hot form is not useless, though: on TPU it can be faster for small
vocabularies, and it is how you'd explain the *gradient*.)

### Why the gradient is sparse
`d(loss)/d(table)` is nonzero only at the rows you actually looked up, and
repeated ids **accumulate**. JAX handles this correctly through
`.at[].add()` semantics under the hood — but it produces a *dense* gradient array
with mostly zeros, which is why large-vocabulary models want sparse optimizer
support.

### Weight tying
`attend` exists because most language models share one matrix between the input
embedding and the output projection. It saves `V x d` parameters (about 40M for
GPT-2) and generally improves perplexity.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MyEmbedding(nnx.Module):
    """Integer ids -> dense vectors."""

    def __init__(self, num_embeddings: int, features: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, ids):
        """(...) integer ids -> (..., features)"""
        pass  # Replace this

    def attend(self, x):
        """(..., features) -> (..., num_embeddings). Transpose projection."""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MyEmbedding(nnx.Module):
    def __init__(self, num_embeddings: int, features: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.table = nnx.Param(
            jax.random.normal(key, (num_embeddings, features)) * 0.02
        )
        self.num_embeddings = num_embeddings
        self.features = features

    def __call__(self, ids):
        # A gather, not a one-hot matmul. Advanced indexing already handles
        # arbitrary leading shapes.
        return self.table[ids]

    def attend(self, x):
        # Weight tying: reuse the same matrix for the output projection.
        return x @ self.table[...].T
''',
    "demo": '''import jax.numpy as jnp
from flax import nnx

emb = MyEmbedding(100, 8, rngs=nnx.Rngs(params=0))

print("table:", emb.table.shape)
print("scalar id  ->", emb(jnp.array(5)).shape)
print("(3,) ids   ->", emb(jnp.array([1, 2, 3])).shape)
print("(2,4) ids  ->", emb(jnp.zeros((2, 4), dtype=jnp.int32)).shape)
print("attend     ->", emb.attend(jnp.ones((2, 4, 8))).shape)
''',
    "tests": [
        {
            "name": "Table shape and init",
            "code": """
import jax.numpy as jnp
from flax import nnx

emb = {fn}(100, 16, rngs=nnx.Rngs(params=0))

assert isinstance(emb.table, nnx.Param), f'table must be nnx.Param, got {type(emb.table)}'
assert emb.table.shape == (100, 16), f'{emb.table.shape} vs (100, 16)'

std = float(jnp.std(emb.table[...]))
assert 0.01 < std < 0.03, f'Init std should be ~0.02, got {std:.4f}'
""",
        },
        {
            "name": "Lookup shapes",
            "code": """
import jax.numpy as jnp
from flax import nnx

emb = {fn}(50, 8, rngs=nnx.Rngs(params=0))

assert emb(jnp.array(3)).shape == (8,), 'Scalar id should give (features,)'
assert emb(jnp.array([1, 2, 3])).shape == (3, 8), '(3,) ids should give (3, 8)'

ids = jnp.zeros((2, 4), dtype=jnp.int32)
assert emb(ids).shape == (2, 4, 8), f'(2, 4) ids should give (2, 4, 8), got {emb(ids).shape}'

ids3 = jnp.zeros((2, 3, 4), dtype=jnp.int32)
assert emb(ids3).shape == (2, 3, 4, 8), f'{emb(ids3).shape}'
""",
        },
        {
            "name": "Returns the correct rows",
            "code": """
import jax.numpy as jnp
from flax import nnx

emb = {fn}(10, 4, rngs=nnx.Rngs(params=0))
table = emb.table[...]

ids = jnp.array([0, 5, 9, 5])
out = emb(ids)

assert jnp.allclose(out[0], table[0], atol=1e-6), 'Row 0 mismatch'
assert jnp.allclose(out[1], table[5], atol=1e-6), 'Row 5 mismatch'
assert jnp.allclose(out[2], table[9], atol=1e-6), 'Row 9 mismatch'
assert jnp.allclose(out[1], out[3], atol=1e-6), 'The same id must give the same vector'
""",
        },
        {
            "name": "Matches the one-hot matmul",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

emb = {fn}(20, 6, rngs=nnx.Rngs(params=0))
ids = jax.random.randint(jax.random.key(0), (3, 5), 0, 20)

gathered = emb(ids)
one_hot = jax.nn.one_hot(ids, 20) @ emb.table[...]

assert jnp.allclose(gathered, one_hot, atol=1e-5), (
    'Gather must equal the one-hot matmul'
)
""",
        },
        {
            "name": "attend is the transpose projection",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

emb = {fn}(30, 8, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 4, 8))

logits = emb.attend(x)
assert logits.shape == (2, 4, 30), f'{logits.shape} vs (2, 4, 30)'
assert jnp.allclose(logits, x @ emb.table[...].T, atol=1e-4), 'attend must be x @ table.T'

# Weight tying sanity: a token's own embedding should score highest against itself.
row = emb.table[...][7]
scores = emb.attend(row)
assert int(jnp.argmax(scores)) == 7, (
    f'The embedding of token 7 should score highest on token 7, got {int(jnp.argmax(scores))}'
)
""",
        },
        {
            "name": "Gradient is sparse and accumulates",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

emb = {fn}(10, 4, rngs=nnx.Rngs(params=0))
ids = jnp.array([2, 2, 7])

grads = nnx.grad(lambda m: jnp.sum(m(ids)))(emb)
g = jax.tree.leaves(grads)[0]

assert g.shape == (10, 4), f'Gradient shape {g.shape} vs (10, 4)'
assert jnp.allclose(g[2], 2.0), f'Row 2 was looked up twice, grad should be 2.0, got {g[2]}'
assert jnp.allclose(g[7], 1.0), f'Row 7 grad should be 1.0, got {g[7]}'

untouched = jnp.array([0, 1, 3, 4, 5, 6, 8, 9])
assert jnp.allclose(g[untouched], 0.0), 'Rows that were never looked up must have zero gradient'
""",
        },
    ],
}
