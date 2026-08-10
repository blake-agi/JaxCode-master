"""Softmax with the max-subtraction trick — the numerical stability classic."""

TASK = {
    "title": "Implement Softmax",
    "category": "Core Ops & Layers",
    "number": "02",
    "difficulty": "Easy",
    "function_name": "my_softmax",
    "hint": (
        "exp() overflows in float32 above about 88, so shift before "
        "exponentiating: subtract the max along `axis` first. The shift cancels "
        "exactly in the numerator/denominator ratio, so the result is unchanged — "
        "only the intermediate magnitudes shrink. Watch keepdims on both "
        "reductions: without it the reduced array loses the axis and will not "
        "broadcast back against x."
    ),
    "description": r"""
Implement **softmax** from scratch, numerically stably.

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### Rules
- Do **not** use `jax.nn.softmax` or `jax.nn.logsumexp`
- Must be stable for large inputs (e.g. `x = [1000, 1001, 1002]`)
- Support an arbitrary `axis`, defaulting to `-1`

### Signature
```python
def my_softmax(x, axis=-1):
    ...
```

### Why the max subtraction is mandatory
`exp(1000)` overflows float32 to `inf`, and `inf / inf` is `NaN`. Subtracting the
row max first is exact, not an approximation:

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{-c} e^{x_i}}{e^{-c}\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

With `c = max(x)` the largest exponent becomes exactly `0`, so `exp` tops out at
`1` and nothing can overflow. Underflow to `0` in the small terms is harmless.

This is the most-asked numerical-stability question in ML interviews, and it is
the same trick inside `logsumexp`, cross-entropy, and Flash Attention's running
maximum.
""",
    "stub": '''import jax
import jax.numpy as jnp


def my_softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    Args:
        x:    array of any shape
        axis: axis to normalise over

    Returns:
        Array of the same shape; slices along `axis` sum to 1.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def my_softmax(x, axis=-1):
    # Shifting by the max is exact — it cancels in the ratio — and caps the
    # largest exponent at exp(0) = 1 so nothing can overflow.
    z = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(z)
    return e / jnp.sum(e, axis=axis, keepdims=True)
''',
    "demo": '''import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
print("softmax:", my_softmax(x))
print("sums to:", my_softmax(x).sum())

# The whole point — this is where the naive version dies:
big = jnp.array([1000.0, 1001.0, 1002.0])
print("stable on big input:", my_softmax(big))
print("naive would give:   ", jnp.exp(big) / jnp.sum(jnp.exp(big)))
''',
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
out = {fn}(x, axis=-1)
expected = jax.nn.softmax(x, axis=-1)

assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
assert jnp.allclose(jnp.sum(out), 1.0, atol=1e-6), f'Must sum to 1, got {jnp.sum(out)}'
""",
        },
        {
            "name": "2-D along the last axis",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (4, 8))
out = {fn}(x, axis=-1)
expected = jax.nn.softmax(x, axis=-1)

assert out.shape == expected.shape, 'Shape mismatch'
assert jnp.allclose(out, expected, atol=1e-5), 'Values differ'
assert jnp.allclose(jnp.sum(out, axis=-1), jnp.ones(4), atol=1e-5), 'Rows must sum to 1'
assert (out >= 0).all(), 'Softmax output must be non-negative'
""",
        },
        {
            "name": "Numerical stability on large inputs",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1000.0, 1001.0, 1002.0])
out = {fn}(x, axis=-1)

assert not jnp.isnan(out).any(), (
    f'NaN in output: {out} — subtract the max along the axis before exp()'
)
assert not jnp.isinf(out).any(), f'Inf in output: {out}'
assert jnp.allclose(jnp.sum(out), 1.0, atol=1e-5), f'Must still sum to 1, got {jnp.sum(out)}'
assert jnp.allclose(out, jax.nn.softmax(x, axis=-1), atol=1e-5), 'Values differ on large input'

# Very negative inputs must not produce NaN either.
neg = {fn}(jnp.array([-1000.0, -1001.0, -1002.0]))
assert not jnp.isnan(neg).any(), f'NaN on large negative input: {neg}'
assert jnp.allclose(jnp.sum(neg), 1.0, atol=1e-5), 'Large negative input must still sum to 1'
""",
        },
        {
            "name": "Non-default axis",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (3, 5, 2))

for axis in [0, 1, 2, -1, -2]:
    out = {fn}(x, axis=axis)
    expected = jax.nn.softmax(x, axis=axis)
    assert out.shape == x.shape, f'axis={axis}: shape {out.shape} vs {x.shape}'
    assert jnp.allclose(out, expected, atol=1e-5), f'axis={axis}: values differ'
    assert jnp.allclose(jnp.sum(out, axis=axis), 1.0, atol=1e-5), (
        f'axis={axis}: slices must sum to 1'
    )
""",
        },
        {
            "name": "Shift invariance and constant input",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (16,))
# softmax(x + c) == softmax(x) for any scalar c.
assert jnp.allclose({fn}(x), {fn}(x + 50.0), atol=1e-5), 'Softmax must be shift-invariant'

# A constant vector must give the uniform distribution.
u = {fn}(jnp.zeros(5))
assert jnp.allclose(u, 0.2, atol=1e-6), f'Constant input should give uniform, got {u}'
""",
        },
        {
            "name": "Gradient, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (4, 6))

g = jax.grad(lambda v: jnp.sum({fn}(v) ** 2))(x)
ref = jax.grad(lambda v: jnp.sum(jax.nn.softmax(v) ** 2))(x)
assert jnp.isfinite(g).all(), 'Non-finite gradient'
assert jnp.allclose(g, ref, atol=1e-4), 'Gradient differs from reference'

assert jnp.allclose(jax.jit({fn})(x), {fn}(x), atol=1e-6), 'jit changes the result'
assert jnp.allclose(jax.vmap({fn})(x), {fn}(x, axis=-1), atol=1e-6), 'vmap mismatch'
""",
        },
    ],
}
