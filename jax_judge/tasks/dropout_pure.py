"""Problem 17 without Flax — and the key has nowhere to hide."""

TASK = {
    "title": "Dropout without Flax",
    "category": "Core Ops & Layers",
    "number": "b_25",
    "difficulty": "Easy",
    "function_name": "apply_dropout",
    "hint": (
        "There are no parameters, so there is no init — only apply. The key is "
        "now an argument, which is the whole lesson: nnx.Rngs was holding it "
        "and advancing it for you. Early-return x unchanged when deterministic "
        "or when p == 0. Otherwise draw a keep mask with "
        "jax.random.bernoulli(key, 1 - p, x.shape) — bernoulli's p is the "
        "probability of True, so it is the KEEP probability — and scale the "
        "survivors by 1/(1-p)."
    ),
    "description": r"""
Problem 17's dropout, with no module to hold the RNG.

### Signature
```python
def apply_dropout(x, key, p, *, deterministic=False):
    ...   # -> same shape as x
```

There are no learnable parameters, so there is no `init_dropout`. What there
*is* — and what `nnx.Rngs` was quietly doing for you — is a key that has to
come from somewhere and be different on every call.

### Rules
- `deterministic=True` → return `x` unchanged.
- `p == 0.0` → return `x` unchanged, exactly. Both conditions, not just the first.
- Otherwise: keep each element with probability `1 - p`, and divide the
  survivors by `1 - p` so the expected value is unchanged (inverted dropout).

`p` and `deterministic` are hyperparameters, so under `jit` they are **static**
(`static_argnums=(2,)`, `static_argnames=('deterministic',)`) — the `p == 0`
exit is a Python branch and needs a concrete value.

### The key is the whole point
In problem 17 you wrote `self.rngs.dropout()` and a fresh mask appeared each
call, because the stream advanced itself. Here **nothing advances anything**:

```python
apply_dropout(x, key, 0.5)      # same key
apply_dropout(x, key, 0.5)      # SAME MASK — this is not a bug
```

Passing the same key twice gives the same mask, because JAX random functions
are pure. Two training steps need two keys, and that is the caller's job:

```python
key, sub = jax.random.split(key)
h = apply_dropout(h, sub, p)
```

That is what a module was hiding. It is also why a JAX training loop threads a
key through its carry — see `b_19`.

### Why divide by 1-p
So that `E[out] == x`. Doing it at training time ("inverted dropout") means
inference is a plain no-op instead of a rescale — which is exactly why
`deterministic=True` can just return `x`.
""",
    "stub": '''import jax
import jax.numpy as jnp


def apply_dropout(x, key, p, *, deterministic=False):
    """Inverted dropout. Returns an array shaped like x."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def apply_dropout(x, key, p, *, deterministic=False):
    # Both early exits: p == 0 must be an exact no-op, not "a mask that happens
    # to keep everything", which would still burn the key.
    if deterministic or p == 0.0:
        return x

    # bernoulli's p is the probability of True, i.e. the KEEP probability.
    keep = jax.random.bernoulli(key, 1.0 - p, x.shape)
    # Scale at train time so inference needs no rescale at all.
    return jnp.where(keep, x / (1.0 - p), 0.0)
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.ones((8,))
key = jax.random.key(0)

print("p=0.5      ", apply_dropout(x, key, 0.5))
print("same key   ", apply_dropout(x, key, 0.5), "  <- identical, by design")
print("split key  ", apply_dropout(x, jax.random.split(key)[0], 0.5))
print("determinist", apply_dropout(x, key, 0.5, deterministic=True))

big = jnp.ones((100000,))
out = apply_dropout(big, key, 0.3)
print(f"\\nmean over 100k: {float(jnp.mean(out)):.4f}  (should be ~1.0)")
print(f"fraction zeroed: {float(jnp.mean(out == 0)):.4f}  (should be ~0.3)")
''',
    "tests": [
        {
            "name": "Expectation is preserved and the drop rate is right",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.ones((200000,))
for p in (0.1, 0.5, 0.9):
    out = {fn}(x, jax.random.key(0), p)
    assert out.shape == x.shape, f'{out.shape} vs {x.shape}'
    mean = float(jnp.mean(out))
    assert abs(mean - 1.0) < 0.02, (
        f'p={p}: mean {mean:.4f}, expected ~1.0. Survivors must be divided by '
        '(1 - p) so the expectation is unchanged.'
    )
    zeroed = float(jnp.mean(out == 0.0))
    assert abs(zeroed - p) < 0.02, (
        f'p={p}: dropped {zeroed:.3f} of the elements. bernoulli(key, q) '
        'returns True with probability q, so q is the KEEP rate, i.e. 1 - p.'
    )
    kept = out[out != 0.0]
    assert jnp.allclose(kept, 1.0 / (1.0 - p), atol=1e-4), (
        f'p={p}: survivors should all equal 1/(1-p) = {1/(1-p):.4f}'
    )
""",
        },
        {
            "name": "deterministic and p=0 are exact no-ops",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (50,))
key = jax.random.key(0)

assert jnp.array_equal({fn}(x, key, 0.5, deterministic=True), x), (
    'deterministic=True must return x unchanged'
)
assert jnp.array_equal({fn}(x, key, 0.0), x), (
    'p=0.0 must be an exact no-op — it is a separate early exit, not something '
    'the mask happens to do'
)
assert jnp.array_equal({fn}(x, key, 0.0, deterministic=True), x), 'both flags together'
""",
        },
        {
            "name": "Same key gives the same mask; different keys do not",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.ones((500,))
k0 = jax.random.key(0)

a = {fn}(x, k0, 0.5)
b = {fn}(x, k0, 0.5)
assert jnp.array_equal(a, b), (
    'The same key must give the same mask — JAX random functions are pure. If '
    'these differ you are pulling entropy from somewhere else.'
)

c = {fn}(x, jax.random.key(1), 0.5)
assert not jnp.array_equal(a, c), 'Different keys produced the same mask'

# The idiom a caller uses to get a fresh mask each step.
k, sub = jax.random.split(k0)
d = {fn}(x, sub, 0.5)
assert not jnp.array_equal(a, d), 'split() should give an independent mask'
""",
        },
        {
            "name": "Shape and dtype are preserved for any input",
            "code": """
import jax
import jax.numpy as jnp

for shape in [(7,), (3, 4), (2, 3, 4)]:
    x = jax.random.normal(jax.random.key(1), shape)
    out = {fn}(x, jax.random.key(2), 0.5)
    assert out.shape == shape, f'{shape} -> {out.shape}'
    assert out.dtype == x.dtype, f'dtype {x.dtype} -> {out.dtype}'
    # Every surviving element is its input scaled by exactly 1/(1-p).
    surv = out != 0.0
    assert jnp.allclose(out[surv], x[surv] * 2.0, atol=1e-4), (
        'Survivors should be the ORIGINAL values scaled by 1/(1-p), not resampled'
    )
""",
        },
        {
            "name": "Gradient flows only through the survivors",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (200,))
key = jax.random.key(5)

out = {fn}(x, key, 0.5)
g = jax.grad(lambda v: jnp.sum({fn}(v, key, 0.5)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'Bad gradient'

dropped = out == 0.0
assert float(jnp.abs(g[dropped]).max()) == 0.0, 'Dropped elements must get zero gradient'
assert jnp.allclose(g[~dropped], 2.0, atol=1e-4), (
    'Surviving elements should have gradient 1/(1-p) = 2.0'
)

# p and deterministic are hyperparameters, so they are STATIC under jit — the
# p == 0 early exit is a Python branch and needs a concrete value.
jf = jax.jit({fn}, static_argnums=(2,), static_argnames=('deterministic',))
assert jnp.allclose(jf(x, key, 0.5), out, atol=1e-6), 'jit disagrees'
assert jnp.array_equal(jf(x, key, 0.0), x), 'jit with p=0 should still be a no-op'
""",
        },
        {
            "name": "No Flax anywhere",
            "code": """
import sys

assert 'flax' not in sys.modules, (
    'flax got imported — the key is supposed to be an argument here, not '
    'something nnx.Rngs hands you'
)
""",
        },
    ],
}
