"""Problem 17 without Flax — and the key has nowhere to hide."""

_WHY = r"""
### Why this exists alongside problem 17
Same class, same constructor, same `deterministic` flag as the `nnx` version —
only the key changes hands. In problem 17 `nnx.Rngs` held it and advanced it
for you; here it is an argument to `__call__`, which is the whole lesson.
"""

TASK = {
    "title": "Dropout without Flax",
    "category": "Core Ops & Layers",
    "number": "b_25",
    "difficulty": "Easy",
    "function_name": "MyDropout",
    "hint": (
        "There is nothing to store but p, so __init__ is one line. __call__ "
        "takes the key as an argument. Early-return x unchanged when "
        "deterministic or when self.p == 0. Otherwise draw the keep mask with "
        "jax.random.bernoulli(key, 1 - self.p, x.shape) — bernoulli's p is the "
        "probability of True, so it is the KEEP probability — and divide the "
        "survivors by (1 - p)."
    ),
    "description": r"""
Problem 17's dropout, with no module to hold the RNG.

### Signature
```python
class MyDropout:
    def __init__(self, p): ...
    def __call__(self, x, key, deterministic=False): ...
```

Same class as the `nnx` version except that `__call__` takes the **key**.
There is no `rngs=` in the constructor, because there is no stream to hold.

### Rules
- `deterministic=True` → return `x` unchanged.
- `self.p == 0.0` → return `x` unchanged, exactly. Both conditions, not one.
- Otherwise keep each element with probability `1 - p` and divide the
  survivors by `1 - p`, so the expectation is unchanged (inverted dropout).

`p` and `deterministic` decide a Python branch, so under `jit` they are
static.

### The key is the whole point
In problem 17 you wrote `self.rngs.dropout()` and a fresh mask appeared each
call, because the stream advanced itself. Here **nothing advances anything**:

```python
layer(x, key)      # same key
layer(x, key)      # SAME MASK — not a bug
```

JAX random functions are pure, so the same key gives the same mask. Two
training steps need two keys, and that is the caller's job:

```python
key, sub = jax.random.split(key)
h = layer(h, sub)
```

That is what the module was hiding — and why a JAX training loop threads a key
through its carry (see `b_19`).

### Why divide by 1-p
So that `E[out] == x`. Doing it at training time ("inverted dropout") is what
lets inference be a plain no-op instead of a rescale.
""" + _WHY,
    "stub": '''import jax
import jax.numpy as jnp


class MyDropout:
    """Inverted dropout. The key is an argument, not state."""

    def __init__(self, p):
        pass  # Replace this

    def __call__(self, x, key, deterministic=False):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


class MyDropout:
    def __init__(self, p):
        self.p = p

    def __call__(self, x, key, deterministic=False):
        # Both early exits: p == 0 must be an exact no-op, not "a mask that
        # happens to keep everything", which would still consume the key.
        if deterministic or self.p == 0.0:
            return x

        # bernoulli's p is the probability of True, i.e. the KEEP probability.
        keep = jax.random.bernoulli(key, 1.0 - self.p, x.shape)
        # Scale at train time so inference needs no rescale at all.
        return jnp.where(keep, x / (1.0 - self.p), 0.0)
''',
    "demo": '''import jax
import jax.numpy as jnp

drop = MyDropout(0.5)
x = jnp.ones((8,))
key = jax.random.key(0)

print("p=0.5       ", drop(x, key))
print("same key    ", drop(x, key), "  <- identical, by design")
print("split key   ", drop(x, jax.random.split(key)[0]))
print("deterministic", drop(x, key, deterministic=True))

big = jnp.ones((100000,))
out = MyDropout(0.3)(big, key)
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
    out = {fn}(p)(x, jax.random.key(0))
    assert out.shape == x.shape, f'{out.shape} vs {x.shape}'
    mean = float(jnp.mean(out))
    assert abs(mean - 1.0) < 0.02, (
        f'p={p}: mean {mean:.4f}, expected ~1.0. Survivors must be divided by '
        '(1 - p) so the expectation is unchanged.'
    )
    zeroed = float(jnp.mean(out == 0.0))
    assert abs(zeroed - p) < 0.02, (
        f'p={p}: dropped {zeroed:.3f} of the elements. bernoulli(key, q) is '
        'True with probability q, so q is the KEEP rate, i.e. 1 - p.'
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

assert jnp.array_equal({fn}(0.5)(x, key, deterministic=True), x), (
    'deterministic=True must return x unchanged'
)
assert jnp.array_equal({fn}(0.0)(x, key), x), (
    'p=0.0 must be an exact no-op — a separate early exit, not something the '
    'mask happens to do'
)
assert jnp.array_equal({fn}(0.0)(x, key, deterministic=True), x), 'both together'
""",
        },
        {
            "name": "Same key gives the same mask; different keys do not",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.ones((500,))
d = {fn}(0.5)
k0 = jax.random.key(0)

a, b = d(x, k0), d(x, k0)
assert jnp.array_equal(a, b), (
    'the same key must give the same mask — JAX random functions are pure. If '
    'these differ you are pulling entropy from somewhere else.'
)
assert not jnp.array_equal(a, d(x, jax.random.key(1))), 'different keys gave the same mask'

_, sub = jax.random.split(k0)
assert not jnp.array_equal(a, d(x, sub)), 'split() should give an independent mask'
""",
        },
        {
            "name": "Shape and dtype survive; survivors are the originals",
            "code": """
import jax
import jax.numpy as jnp

d = {fn}(0.5)
for shape in [(7,), (3, 4), (2, 3, 4)]:
    x = jax.random.normal(jax.random.key(1), shape)
    out = d(x, jax.random.key(2))
    assert out.shape == shape, f'{shape} -> {out.shape}'
    assert out.dtype == x.dtype, f'dtype {x.dtype} -> {out.dtype}'
    surv = out != 0.0
    assert jnp.allclose(out[surv], x[surv] * 2.0, atol=1e-4), (
        'survivors should be the ORIGINAL values scaled by 1/(1-p), not resampled'
    )
""",
        },
        {
            "name": "Gradient flows only through the survivors; jit works",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (200,))
key = jax.random.key(5)
d = {fn}(0.5)

out = d(x, key)
g = jax.grad(lambda v: jnp.sum(d(v, key)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient'

dropped = out == 0.0
assert float(jnp.abs(g[dropped]).max()) == 0.0, 'dropped elements must get zero gradient'
assert jnp.allclose(g[~dropped], 2.0, atol=1e-4), 'survivors should have gradient 1/(1-p) = 2.0'

# p and deterministic are hyperparameters, so they are static; here they live
# on the instance, which the closure captures.
assert jnp.allclose(jax.jit(lambda v, k: d(v, k))(x, key), out, atol=1e-6), 'jit disagrees'
assert jnp.array_equal(jax.jit(lambda v, k: {fn}(0.0)(v, k))(x, key), x), (
    'jit with p=0 should still be a no-op'
)
""",
        },
    ],
}
