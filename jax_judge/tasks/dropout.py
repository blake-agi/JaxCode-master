"""Inverted dropout — nnx.Rngs streams and the train/eval scaling question."""

TASK = {
    "title": "Implement Dropout",
    "category": "Core Ops & Layers",
    "number": "17",
    "difficulty": "Easy",
    "function_name": "MyDropout",
    "hint": (
        "Pull a fresh key from the rng stream with self.rngs.dropout(), then "
        "keep = jax.random.bernoulli(key, 1.0 - self.rate, x.shape). Return "
        "jnp.where(keep, x, 0.0) / (1.0 - self.rate) — dividing by the keep "
        "probability is what makes it INVERTED dropout. Return x untouched when "
        "deterministic=True, and short-circuit when rate == 0."
    ),
    "description": r"""
Implement **inverted dropout** as an `nnx.Module`.

**Training:** zero each element independently with probability `rate`, then
divide the survivors by `(1 - rate)`.

$$y_i = \frac{x_i \cdot m_i}{1 - p}, \qquad m_i \sim \text{Bernoulli}(1-p)$$

**Inference:** return `x` unchanged.

### Rules
- Signature: `MyDropout(rate, *, rngs)`
- `__call__(x, deterministic=False)`
- Draw the mask from the `dropout` rng stream: `self.rngs.dropout()`
- Each call must use a **fresh** key — two training calls must give different masks
- `rate == 0.0` must be an exact no-op
- `deterministic=True` returns `x` unchanged, with **no** scaling

### Why divide during training
The point is to keep $\mathbb{E}[y] = x$ so the network sees the same expected
activation magnitude in both modes:

$$\mathbb{E}[y_i] = (1-p)\cdot\frac{x_i}{1-p} + p \cdot 0 = x_i$$

The original 2014 paper scaled by $(1-p)$ at **test** time instead. "Inverted"
dropout moves that correction into training so the inference path is a plain
identity — which matters because inference runs far more often, and because it
means you can strip dropout entirely when exporting a model.

### The rng-stream part
`nnx.Rngs(params=0, dropout=1)` sets up **named streams**. Calling
`rngs.dropout()` returns a fresh key each time and advances the stream, so you
get a new mask per call without threading keys through your own code. Using
`rngs.params()` here would be wrong — it would consume the initialisation
stream.

### A gotcha you will hit
Advancing the stream **mutates** the module, so plain `jax.grad` refuses to
differentiate through a dropout call:

```
TraceContextError: Cannot mutate RngCount from a different trace level
```

Use NNX's own transforms instead — they know how to split the module's state
out and thread it through functionally:

```python
g = nnx.grad(lambda m, v: jnp.sum(m(v)), argnums=1)(drop, x)
```

The same applies to `nnx.jit` over anything holding rngs or `BatchStat`.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MyDropout(nnx.Module):
    """Inverted dropout."""

    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x, deterministic: bool = False):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MyDropout(nnx.Module):
    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        self.rate = rate
        self.rngs = rngs

    def __call__(self, x, deterministic: bool = False):
        if deterministic or self.rate == 0.0:
            return x

        keep_prob = 1.0 - self.rate
        # A fresh key per call — the stream advances automatically.
        key = self.rngs.dropout()
        keep = jax.random.bernoulli(key, keep_prob, x.shape)
        # Dividing here is what makes it "inverted": inference stays an identity.
        return jnp.where(keep, x, 0.0) / keep_prob
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

drop = MyDropout(0.5, rngs=nnx.Rngs(dropout=0))
x = jnp.ones((4, 6))

print("train call 1:\\n", drop(x))
print("train call 2 (different mask):\\n", drop(x))
print("eval:\\n", drop(x, deterministic=True))
print("\\nexpected value preserved:", float(jnp.mean(drop(jnp.ones((2000,))))), "(~1.0)")
''',
    "tests": [
        {
            "name": "Zeroes elements and rescales survivors",
            "code": """
import jax.numpy as jnp
from flax import nnx

drop = {fn}(0.5, rngs=nnx.Rngs(dropout=0))
x = jnp.ones((100, 100))
out = drop(x)

assert out.shape == x.shape, f'Shape mismatch: {out.shape}'

frac_zero = float(jnp.mean(out == 0.0))
assert 0.4 < frac_zero < 0.6, f'About half should be dropped at rate=0.5, got {frac_zero:.3f}'

survivors = out[out != 0.0]
assert jnp.allclose(survivors, 2.0, atol=1e-5), (
    f'Survivors should be scaled by 1/(1-0.5) = 2.0, got {survivors[:5]} — '
    'this is INVERTED dropout'
)
""",
        },
        {
            "name": "Expected value is preserved",
            "code": """
import jax.numpy as jnp
from flax import nnx

for rate in [0.1, 0.3, 0.5, 0.8]:
    drop = {fn}(rate, rngs=nnx.Rngs(dropout=0))
    x = jnp.ones((200, 200))
    m = float(jnp.mean(drop(x)))
    assert abs(m - 1.0) < 0.05, (
        f'rate={rate}: mean of the output is {m:.4f}, expected ~1.0. '
        'Divide the survivors by (1 - rate).'
    )
""",
        },
        {
            "name": "deterministic=True is an exact identity",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

drop = {fn}(0.5, rngs=nnx.Rngs(dropout=0))
x = jax.random.normal(jax.random.key(0), (8, 16))

out = drop(x, deterministic=True)
assert jnp.array_equal(out, x), (
    'Eval mode must return x completely unchanged — no mask and no scaling'
)

# Repeated eval calls must be identical.
assert jnp.array_equal(drop(x, deterministic=True), drop(x, deterministic=True)), (
    'Eval mode must be deterministic'
)
""",
        },
        {
            "name": "A fresh mask on every training call",
            "code": """
import jax.numpy as jnp
from flax import nnx

drop = {fn}(0.5, rngs=nnx.Rngs(dropout=0))
x = jnp.ones((50, 50))

a = drop(x)
b = drop(x)
c = drop(x)

assert not jnp.array_equal(a, b), (
    'Two training calls gave the identical mask — pull a fresh key with '
    'self.rngs.dropout() inside __call__, not once in __init__'
)
assert not jnp.array_equal(b, c), 'Third call repeated the second mask'
""",
        },
        {
            "name": "Reproducible from the same seed",
            "code": """
import jax.numpy as jnp
from flax import nnx

x = jnp.ones((20, 20))

d1 = {fn}(0.5, rngs=nnx.Rngs(dropout=42))
d2 = {fn}(0.5, rngs=nnx.Rngs(dropout=42))
assert jnp.array_equal(d1(x), d2(x)), 'Same seed must produce the same mask'

d3 = {fn}(0.5, rngs=nnx.Rngs(dropout=7))
assert not jnp.array_equal(d1(x), d3(x)), 'Different seeds must produce different masks'
""",
        },
        {
            "name": "rate=0 is a no-op",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

drop = {fn}(0.0, rngs=nnx.Rngs(dropout=0))
x = jax.random.normal(jax.random.key(1), (10, 10))

out = drop(x)
assert jnp.allclose(out, x, atol=1e-6), f'rate=0 must leave x unchanged'
assert not (out == 0.0).any() or (x == 0.0).any(), 'Nothing should be dropped at rate=0'
""",
        },
        {
            "name": "Gradient flows only through survivors",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

x = jnp.ones((100,))

# Replay the same seed twice so we know which positions the mask kept.
drop = {fn}(0.5, rngs=nnx.Rngs(dropout=0))
dropped = drop(x) == 0.0

# nnx.grad — NOT jax.grad — because __call__ mutates the rng stream's counter,
# and plain jax.grad rejects that with a TraceContextError.
drop2 = {fn}(0.5, rngs=nnx.Rngs(dropout=0))
g = nnx.grad(lambda m, v: jnp.sum(m(v)), argnums=1)(drop2, x)

assert g.shape == x.shape, f'Gradient shape {g.shape} vs {x.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'
assert jnp.allclose(g[~dropped], 2.0, atol=1e-5), (
    f'Surviving positions should have gradient 1/(1-rate) = 2.0, got {g[~dropped][:5]}'
)
assert jnp.allclose(g[dropped], 0.0, atol=1e-6), 'Dropped positions must have zero gradient'
""",
        },
    ],
}
