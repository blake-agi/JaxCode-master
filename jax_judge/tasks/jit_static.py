"""Masked mean with a static axis — jit, static_argnames, and safe division."""

TASK = {
    "title": "jit with static_argnames",
    "category": "JAX Fundamentals",
    "order": 3,
    "number": "b_03",
    "difficulty": "Easy",
    "function_name": "masked_mean",
    "hint": (
        "Decorate with @partial(jax.jit, static_argnames=('axis',)). `axis` changes "
        "the output SHAPE, so it must be a compile-time constant — a traced value "
        "would fail. For the mean itself: sum(x * mask) / sum(mask), and guard the "
        "denominator with jnp.maximum(count, 1) so an all-masked slice gives 0, not NaN."
    ),
    "description": r"""
Implement a **masked mean**: average `x` along `axis`, counting only the
positions where `mask` is `True`.

$$\text{out} = \frac{\sum_i x_i \cdot m_i}{\max\left(\sum_i m_i,\; 1\right)}$$

### Rules
- The function must be wrapped in `jax.jit`
- `axis` must be a **static** argument (`static_argnames`) — it determines the
  output shape, so JAX cannot trace it as a value
- A slice where the mask is entirely `False` must return `0.0`, **not** `NaN`
- `mask` is a boolean array broadcastable to `x`

### Signature
```python
@partial(jax.jit, static_argnames=("axis",))
def masked_mean(x, mask, axis):
    ...
```

### Why it matters
This is the single most common `jit` gotcha. Anything that affects a **shape**
— an axis, a `k`, a boolean that picks a branch returning different shapes —
must be static. Anything that is merely a **value** should stay traced, because
every distinct static argument triggers a fresh XLA compilation.

Padding-masked means show up constantly in real transformer code: averaging
token embeddings while ignoring `<pad>` positions.
""",
    "stub": '''from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("axis",))
def masked_mean(x, mask, axis):
    """Mean of x over `axis`, counting only positions where mask is True.

    Args:
        x:    float array
        mask: boolean array broadcastable to x
        axis: int (static) — axis to reduce

    Returns:
        Array with `axis` reduced. Fully-masked slices must be 0.0, not NaN.
    """
    pass  # Replace this
''',
    "solution": '''from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("axis",))
def masked_mean(x, mask, axis):
    m = mask.astype(x.dtype)
    total = jnp.sum(x * m, axis=axis)
    count = jnp.sum(m, axis=axis)
    # Guard the denominator so an all-masked slice yields 0.0 rather than NaN.
    return total / jnp.maximum(count, 1.0)
''',
    "demo": '''import jax.numpy as jnp

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
mask = jnp.array([[True, True, False, False],
                  [True, False, False, False]])

print("mean over axis=1:", masked_mean(x, mask, axis=1))  # [1.5, 5.0]
print("mean over axis=0:", masked_mean(x, mask, axis=0))
print("is jitted:", hasattr(masked_mean, "lower"))
''',
    "tests": [
        {
            "name": "Basic masked mean",
            "code": """
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
mask = jnp.array([[True, True, False, False],
                  [True, False, False, False]])

out = {fn}(x, mask, axis=1)
expected = jnp.array([1.5, 5.0])
assert out.shape == (2,), f'Shape mismatch: {out.shape} vs (2,)'
assert jnp.allclose(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Different axes trigger recompilation, not errors",
            "code": """
import jax.numpy as jnp

x = jnp.arange(12.0).reshape(3, 4)
mask = jnp.ones((3, 4), dtype=bool)

out0 = {fn}(x, mask, axis=0)
out1 = {fn}(x, mask, axis=1)

assert out0.shape == (4,), f'axis=0 shape {out0.shape} vs (4,)'
assert out1.shape == (3,), f'axis=1 shape {out1.shape} vs (3,)'
assert jnp.allclose(out0, x.mean(axis=0)), 'axis=0 values wrong with all-True mask'
assert jnp.allclose(out1, x.mean(axis=1)), 'axis=1 values wrong with all-True mask'
""",
        },
        {
            "name": "Fully masked slice returns 0, not NaN",
            "code": """
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
mask = jnp.array([[False, False], [True, True]])

out = {fn}(x, mask, axis=1)
assert not jnp.isnan(out).any(), f'NaN in output: {out} — guard the denominator'
assert jnp.allclose(out[0], 0.0), f'Fully-masked row should be 0.0, got {out[0]}'
assert jnp.allclose(out[1], 3.5), f'{out[1]} vs 3.5'
""",
        },
        {
            "name": "Actually wrapped in jit with a static axis",
            "code": """
import jax.numpy as jnp

assert hasattr({fn}, "lower"), (
    'masked_mean does not look like a jax.jit-wrapped function — '
    'apply @partial(jax.jit, static_argnames=("axis",))'
)

x = jnp.arange(6.0).reshape(2, 3)
mask = jnp.ones((2, 3), dtype=bool)

# A static argument must survive being passed positionally or by keyword.
a = {fn}(x, mask, 1)
b = {fn}(x, mask, axis=1)
assert jnp.allclose(a, b), 'Positional and keyword axis disagree'

# Distinct static values must each compile and give shape-correct results.
assert {fn}(x, mask, axis=0).shape == (3,)
assert {fn}(x, mask, axis=1).shape == (2,)
""",
        },
        {
            "name": "Gradient flows only through unmasked entries",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0, 4.0])
mask = jnp.array([True, True, False, False])

g = jax.grad(lambda v: {fn}(v, mask, axis=0))(x)
assert jnp.isfinite(g).all(), f'Non-finite gradient: {g}'
assert jnp.allclose(g[:2], 0.5), f'Unmasked grads should be 1/2, got {g[:2]}'
assert jnp.allclose(g[2:], 0.0), f'Masked grads should be 0, got {g[2:]}'
""",
        },
        {
            "name": "3-D input, middle axis",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 5, 3))
mask = jax.random.bernoulli(jax.random.key(1), 0.7, (2, 5, 3))

out = {fn}(x, mask, axis=1)
assert out.shape == (2, 3), f'Shape mismatch: {out.shape} vs (2, 3)'

m = mask.astype(x.dtype)
expected = jnp.sum(x * m, axis=1) / jnp.maximum(jnp.sum(m, axis=1), 1.0)
assert jnp.allclose(out, expected, atol=1e-5), 'Values differ from reference'
""",
        },
    ],
}
