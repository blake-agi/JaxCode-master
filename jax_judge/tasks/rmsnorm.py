"""RMSNorm as a pure function — no mean subtraction, no bias."""

TASK = {
    "title": "Implement RMSNorm",
    "category": "Core Ops & Layers",
    "number": "08",
    "difficulty": "Medium",
    "function_name": "rms_norm",
    "hint": (
        "RMS is the root of the MEAN of the squares — mean over the last axis "
        "with keepdims, then sqrt. Do not subtract the mean and do not add a "
        "bias; that is exactly what distinguishes RMSNorm from LayerNorm. Note "
        "eps defaults to 1e-6 here, and it goes INSIDE the sqrt."
    ),
    "description": r"""
Implement **RMSNorm**, the normalization used by LLaMA, Gemma and most modern LLMs.

$$y = \frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}} \odot w$$

### Signature
```python
def rms_norm(x, weight, eps=1e-6):
    ...
```

### Rules
- Do **not** use `nnx.RMSNorm`
- Normalise over the **last** axis
- **No mean subtraction** and **no bias** term — this is the whole point
- `eps` goes inside the sqrt, and defaults to `1e-6` (not `1e-5`)

### RMSNorm vs LayerNorm
LayerNorm centres *and* scales: it subtracts $\mu$ and divides by $\sigma$.
RMSNorm only scales, dividing by the root-mean-square. So it drops the mean
subtraction and the $\beta$ shift, leaving one learnable vector instead of two.

That turns out to cost nothing in quality while removing two reductions from
the critical path — and at LLM scale, normalization is bandwidth-bound, not
FLOP-bound, so removing a pass over the data is a real win. This is why
essentially every model after LLaMA switched.

### The trap
If your implementation still matches LayerNorm on zero-mean input, that proves
nothing — the two agree exactly when $\mu = 0$. The distinguishing test is
input with a large **non-zero mean**: LayerNorm centres it away, RMSNorm does
not. The tests below use exactly that.
""",
    "stub": '''import jax
import jax.numpy as jnp


def rms_norm(x, weight, eps=1e-6):
    """Root-mean-square normalization over the last axis.

    Args:
        x:      (..., D) array
        weight: (D,) learnable scale
        eps:    stability term inside the sqrt

    Returns:
        Array of the same shape as x.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def rms_norm(x, weight, eps=1e-6):
    # Root of the MEAN of the squares — no centring, so no mean subtraction.
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 8)) + 10.0   # note the big offset
w = jnp.ones(8)

out = rms_norm(x, w)
print("input mean :", float(x.mean()), "(far from 0)")
print("output mean:", float(out.mean()), "(still far from 0 — RMSNorm does NOT centre)")
print("output RMS :", float(jnp.sqrt(jnp.mean(out ** 2, axis=-1)).mean()), "(~1)")
''',
    "tests": [
        {
            "name": "Matches the reference formula",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 8))
weight = jnp.ones(8)
out = {fn}(x, weight)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
ref = x / rms * weight
assert jnp.allclose(out, ref, atol=1e-5), 'Value mismatch vs the reference formula'
""",
        },
        {
            "name": "Output RMS is 1",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (4, 32)) * 9.0
out = {fn}(x, jnp.ones(32))

rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
assert jnp.allclose(rms, 1.0, atol=1e-3), f'Per-row RMS should be ~1, got {rms}'
""",
        },
        {
            "name": "Does NOT centre the input",
            "code": """
import jax
import jax.numpy as jnp

# Large non-zero mean: this is where RMSNorm and LayerNorm visibly differ.
x = jax.random.normal(jax.random.key(2), (4, 16)) + 20.0
out = {fn}(x, jnp.ones(16))

assert not jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-2), (
    'Output rows have ~zero mean, so the mean was subtracted — that is LayerNorm. '
    'RMSNorm divides by the RMS and leaves the mean alone.'
)

mu = jnp.mean(x, axis=-1, keepdims=True)
ln = (x - mu) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-6)
assert not jnp.allclose(out, ln, atol=1e-2), 'Output matches LayerNorm, not RMSNorm'
""",
        },
        {
            "name": "weight scales per feature",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (4, 8))
base = {fn}(x, jnp.ones(8))

w = jnp.arange(1.0, 9.0)
assert jnp.allclose({fn}(x, w), base * w, atol=1e-5), (
    'weight must multiply element-wise along the last axis'
)

# No bias: scaling the weight by 0 must give exactly 0.
assert jnp.allclose({fn}(x, jnp.zeros(8)), 0.0, atol=1e-6), (
    'With weight=0 the output must be exactly 0 — a nonzero result means you '
    'added a bias term, which RMSNorm does not have'
)
""",
        },
        {
            "name": "eps is used and sits inside the sqrt",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.zeros((1, 4))
out = {fn}(x, jnp.ones(4), eps=1e-6)
assert jnp.isfinite(out).all(), 'All-zero input produced non-finite output — eps must guard the sqrt'
assert jnp.allclose(out, 0.0), f'All-zero input should give 0, got {out}'

# A big eps must visibly shrink the output.
x2 = jax.random.normal(jax.random.key(4), (2, 8))
small = {fn}(x2, jnp.ones(8), eps=1e-6)
large = {fn}(x2, jnp.ones(8), eps=100.0)
assert jnp.abs(large).max() < jnp.abs(small).max(), 'eps does not appear to be used'
""",
        },
        {
            "name": "Gradients, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(5), (4, 8))
w = jnp.ones(8)

gx, gw = jax.grad(lambda a, b: jnp.sum({fn}(a, b) ** 2), argnums=(0, 1))(x, w)
assert jnp.isfinite(gx).all() and jnp.isfinite(gw).all(), 'Non-finite gradient'
assert float(jnp.abs(gw).sum()) > 0, 'No gradient reached weight'

ref = {fn}(x, w)
assert jnp.allclose(jax.jit({fn})(x, w), ref, atol=1e-5), 'jit changes the result'
assert jnp.allclose(jax.vmap(lambda r: {fn}(r, w))(x), ref, atol=1e-5), 'vmap changes the result'
""",
        },
    ],
}
