"""logsumexp — the max trick again, but the shape discipline is inverted."""

TASK = {
    "title": "LogSumExp",
    "category": "Core Ops & Layers",
    "number": "b_12",
    "difficulty": "Medium",
    "function_name": "logsumexp",
    "hint": (
        "Subtract the per-slice max before exp(), exactly as in softmax — but "
        "watch what happens when you add it back. You need keepdims=True on the "
        "max so it broadcasts against x for the subtraction, and then the sum "
        "reduces the axis away, so the two operands no longer line up. Adding a "
        "(..., 1) max to a (...,) sum does not error: it broadcasts into an "
        "outer sum of the wrong rank. Squeeze the axis back out, or keep it on "
        "both and let keepdims decide at the end."
    ),
    "description": r"""
Implement **logsumexp** — numerically stable, with a working `axis` and
`keepdims`.

$$\text{logsumexp}(x) = \log \sum_i e^{x_i}
= m + \log \sum_i e^{x_i - m}, \qquad m = \max_i x_i$$

### Signature
```python
def logsumexp(x, axis=-1, keepdims=False):
    ...
```

### Rules
- No `jax.scipy.special.logsumexp` and no `jax.nn.logsumexp`
- Stable for large positive **and** large negative inputs
- `axis` may be any valid axis; `keepdims` must behave like every other
  reduction

### The trap that makes this its own problem
Softmax uses the same max trick, so it is tempting to assume the same shape
handling carries over. It does not, and the difference is the point.

Softmax **divides** by its sum. With `keepdims=True` on both reductions the
shapes cancel, so uniform `keepdims` is simply correct:

```python
z = x - x.max(axis, keepdims=True)      # (2, 3)
e = jnp.exp(z) / jnp.sum(..., keepdims=True)   # (2, 3) / (2, 1) -> (2, 3)  ✓
```

logsumexp **adds** the max back. The sum has already reduced the axis away, so
the operands no longer match:

```python
x_max = jnp.max(x, axis=axis, keepdims=True)              # (2, 1)
s = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis))       # (2,)
s + x_max     # (2,) + (2, 1) -> (2, 2)   WRONG, and it does not raise
```

That is an outer sum. On a `(2, 3)` input it silently returns `(2, 2)` with each
row's value duplicated. Nothing errors, and a test that only checks values at
`[0]` still passes. You need the max at the *reduced* rank —
`x_max.squeeze(axis)` — or `keepdims=True` on both and one squeeze at the end.

A square input hides this completely: `(3,) + (3, 1)` broadcasts to `(3, 3)`
without complaint, and so does the correct version's shape check if you only
compare `len(out)`. Test with something like `(2, 3)`.

### Why the max must be per-slice
Shifting by any constant is exact, so on a single vector a global `max(x)` also
works. Across slices on different scales it does not: subtract a global `1000`
from a row sitting near `-1000` and every term underflows to `0`, leaving
`log(0) = -inf`. Reduce along the axis you are reducing over.

### Where it shows up
Cross-entropy (as log-softmax), any log-domain probability accumulation, and
Flash Attention's running maximum — which is this function computed
incrementally, merging `(m, l)` pairs without ever holding all the scores.

### A useful identity
$$\frac{\partial}{\partial x_i}\,\text{logsumexp}(x) = \text{softmax}(x)_i$$

so the gradient is a probability distribution and sums to 1. Worth knowing: it
is the fastest way to check your implementation differentiates correctly.
""",
    "stub": '''import jax
import jax.numpy as jnp


def logsumexp(x, axis=-1, keepdims=False):
    """Numerically stable log(sum(exp(x))) along `axis`.

    Args:
        x:        array of any shape
        axis:     axis to reduce over
        keepdims: if True, keep the reduced axis as a length-1 dimension

    Returns:
        Array with `axis` reduced (or kept as length 1 when keepdims=True).
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def logsumexp(x, axis=-1, keepdims=False):
    # keepdims=True here so the max broadcasts against x for the subtraction.
    x_max = jnp.max(x, axis=axis, keepdims=True)
    # Guard against an all -inf slice, where max is -inf and x - x_max is nan.
    x_max = jnp.where(jnp.isfinite(x_max), x_max, 0.0)

    # Keep the axis on the sum too, so both operands still line up when the max
    # is added back. Adding a (..., 1) max to an already-reduced (...,) sum
    # broadcasts into an outer sum instead — silently, and with the wrong rank.
    out = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True)) + x_max

    return out if keepdims else jnp.squeeze(out, axis=axis)
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # (2, 3), deliberately not square

print("mine     :", logsumexp(x, axis=-1))
print("reference:", jax.scipy.special.logsumexp(x, axis=-1))

# Failure 1 — the max added back at the wrong rank. No error, wrong shape.
x_max = jnp.max(x, axis=-1, keepdims=True)          # (2, 1)
s = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=-1))   # (2,)
print("\\n(2,) + (2,1) ->", (s + x_max).shape, "  <- outer sum, silently wrong")
print(s + x_max)
print("squeezed     ->", (s + x_max.squeeze(-1)).shape, s + x_max.squeeze(-1))

# Failure 2 — a global max instead of a per-slice one.
rows = jnp.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
m = jnp.max(rows)                                   # one number for both rows
print("\\nglobal max :", jnp.log(jnp.sum(jnp.exp(rows - m), axis=-1)) + m)
print("per-slice  :", logsumexp(rows, axis=-1))

# Failure 3 — no shift at all.
big = jnp.array([1000.0, 1001.0, 1002.0])
print("\\nnaive      :", jnp.log(jnp.sum(jnp.exp(big))))
print("stable     :", logsumexp(big))

# The gradient is softmax — a quick correctness check that costs one line.
g = jax.grad(lambda v: logsumexp(v))(jnp.array([1.0, 2.0, 3.0]))
print("\\ngrad       :", g)
print("softmax    :", jax.nn.softmax(jnp.array([1.0, 2.0, 3.0])))
''',
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
out = {fn}(x)
expected = jax.scipy.special.logsumexp(x)

assert jnp.ndim(out) == 0, (
    f'A full reduction of a 1-D input must be a scalar, got shape {jnp.shape(out)}. '
    'If this is (1,), the max was added back with its keepdims axis still on.'
)
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "Shape is the reduced shape, not a broadcast of it",
            "code": """
import jax
import jax.numpy as jnp

# Deliberately NOT square: a square input hides the bug below, because
# (n,) + (n, 1) broadcasts to (n, n) and still looks plausible.
x = jax.random.normal(jax.random.key(0), (2, 3))

for axis in (-1, 0, 1):
    out = {fn}(x, axis=axis)
    expected = jax.scipy.special.logsumexp(x, axis=axis)
    assert out.shape == expected.shape, (
        f'axis={axis}: got shape {out.shape}, expected {expected.shape}. '
        'Adding a keepdims max to an already-reduced sum broadcasts into an '
        'outer sum — squeeze the axis back out, or keep it on both reductions.'
    )
    assert jnp.allclose(out, expected, atol=1e-5), (
        f'axis={axis}: {out} vs {expected}'
    )

# 3-D, middle axis — the reduced axis must be removed, not just resized.
y = jax.random.normal(jax.random.key(1), (2, 3, 4))
assert {fn}(y, axis=1).shape == (2, 4), f'3-D axis=1 gave {{fn}}(y, axis=1).shape'
""",
        },
        {
            "name": "keepdims",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (2, 3))

for axis in (-1, 0):
    kept = {fn}(x, axis=axis, keepdims=True)
    ref = jax.scipy.special.logsumexp(x, axis=axis, keepdims=True)
    assert kept.shape == ref.shape, (
        f'axis={axis}, keepdims=True: got {kept.shape}, expected {ref.shape}'
    )
    assert jnp.allclose(kept, ref, atol=1e-5), 'keepdims values differ'
    # And it must actually differ from the keepdims=False shape.
    assert kept.shape != {fn}(x, axis=axis).shape, (
        f'axis={axis}: keepdims=True and keepdims=False gave the same shape — '
        'the keepdims argument is being ignored'
    )
""",
        },
        {
            "name": "Stable on large positive and large negative inputs",
            "code": """
import jax
import jax.numpy as jnp

big = jnp.array([1000.0, 1001.0, 1002.0])
out = {fn}(big)
assert jnp.isfinite(out).all(), (
    f'Got {out} on a large input — exp() overflowed. Subtract the max first.'
)
assert jnp.allclose(out, jax.scipy.special.logsumexp(big), atol=1e-3), (
    f'{out} vs {jax.scipy.special.logsumexp(big)}'
)

small = jnp.array([-1000.0, -1001.0, -1002.0])
out_s = {fn}(small)
assert jnp.isfinite(out_s).all(), (
    f'Got {out_s} on a large negative input — every term underflowed to 0 and '
    'log(0) = -inf. Shifting by the max keeps the largest term at exp(0) = 1.'
)
assert jnp.allclose(out_s, jax.scipy.special.logsumexp(small), atol=1e-3), (
    f'{out_s} vs {jax.scipy.special.logsumexp(small)}'
)
""",
        },
        {
            "name": "The max is per-slice, not global",
            "code": """
import jax
import jax.numpy as jnp

# Two rows on wildly different scales. A single global max is exact for the
# big row and annihilates the small one.
rows = jnp.array([[1000.0, 1001.0, 1002.0],
                  [-1000.0, -1001.0, -1002.0]])
out = {fn}(rows, axis=-1)
expected = jax.scipy.special.logsumexp(rows, axis=-1)

assert jnp.isfinite(out).all(), (
    f'Got {out}. A global jnp.max(x) shifts the second row by +1000, so every '
    'term underflows to 0 and log(0) = -inf. Reduce along `axis`.'
)
assert jnp.allclose(out, expected, atol=1e-3), f'{out} vs {expected}'
""",
        },
        {
            "name": "Gradient is softmax",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
g = jax.grad(lambda v: {fn}(v))(x)

assert jnp.isfinite(g).all(), f'Non-finite gradient: {g}'
assert jnp.allclose(g, jax.nn.softmax(x), atol=1e-5), (
    f'd/dx logsumexp(x) must equal softmax(x); got {g} vs {jax.nn.softmax(x)}'
)
assert jnp.allclose(jnp.sum(g), 1.0, atol=1e-5), 'The gradient must sum to 1'

# Still finite where the naive formulation would have produced inf/nan.
gb = jax.grad(lambda v: {fn}(v))(jnp.array([1000.0, 1001.0, 1002.0]))
assert jnp.isfinite(gb).all(), f'Non-finite gradient on a large input: {gb}'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (2, 3))

jitted = jax.jit(lambda v: {fn}(v, axis=-1))
assert jnp.allclose(jitted(x), {fn}(x, axis=-1), atol=1e-5), 'jit changes the result'

# vmap over rows must agree with reducing the last axis directly.
mapped = jax.vmap(lambda row: {fn}(row))(x)
assert mapped.shape == (2,), f'vmap gave {mapped.shape}, expected (2,)'
assert jnp.allclose(mapped, {fn}(x, axis=-1), atol=1e-5), (
    'vmap over rows disagrees with axis=-1'
)
""",
        },
    ],
}
