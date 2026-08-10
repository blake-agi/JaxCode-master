"""LayerNorm as a pure function — params passed in, biased variance."""

TASK = {
    "title": "Implement LayerNorm",
    "category": "Core Ops & Layers",
    "number": "04",
    "difficulty": "Medium",
    "function_name": "my_layer_norm",
    "hint": (
        "Normalise over the LAST axis only, and use the biased variance — "
        "jnp.var defaults to ddof=0, which is what you want. Both reductions "
        "need keepdims so they broadcast back against x. gamma and beta are "
        "passed in as plain arrays; this is a pure function, there is no module."
    ),
    "description": r"""
Implement **Layer Normalization** as a pure function.

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

where $\mu$ and $\sigma^2$ are computed over the **last** axis only.

### Signature
```python
def my_layer_norm(x, gamma, beta, eps=1e-5):
    ...
```

### Rules
- Do **not** use `nnx.LayerNorm` or `jax.nn.standardize`
- Normalise over the **last** axis
- Use the **biased** variance (`ddof=0`) — this is what every framework does
- `gamma` / `beta` are plain arrays of shape `(D,)`, passed in — not module state

### The two traps
1. **Unbiased variance.** `jnp.var(x, ddof=1)` divides by `n-1` and will not match
   any reference implementation. The population variance is correct here.
2. **Wrong axis.** Normalising over the batch axis is BatchNorm, not LayerNorm.
   LayerNorm is per-example, which is exactly why it works with batch size 1 and
   why transformers use it.

### Why a pure function
PyTorch would wrap $\gamma,\beta$ in a module. Here they are just arguments, so
the whole thing is a pure function of its inputs — `jit`, `grad` and `vmap` all
apply directly with nothing to thread through. This is the JAX default; reach
for `nnx.Module` only when you actually want the parameters to live somewhere.
""",
    "stub": '''import jax
import jax.numpy as jnp


def my_layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization over the last axis.

    Args:
        x:     (..., D) array
        gamma: (D,) scale
        beta:  (D,) shift
        eps:   numerical-stability term inside the sqrt

    Returns:
        Array of the same shape as x.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # jnp.var is the BIASED (population) variance by default — ddof=0.
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 3, 8)) * 5.0 + 2.0
gamma = jnp.ones(8)
beta = jnp.zeros(8)

out = my_layer_norm(x, gamma, beta)
print("shape:", out.shape)
print("per-row mean:", jnp.mean(out, axis=-1).ravel()[:4], "(~0)")
print("per-row std: ", jnp.std(out, axis=-1).ravel()[:4], "(~1)")
''',
    "tests": [
        {
            "name": "Matches the reference formula",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 3, 8))
gamma = jnp.ones(8)
beta = jnp.zeros(8)
out = {fn}(x, gamma, beta)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'

mean = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
ref = (x - mean) / jnp.sqrt(var + 1e-5) * gamma + beta
assert jnp.allclose(out, ref, atol=1e-4), 'Value mismatch vs the reference formula'
""",
        },
        {
            "name": "Normalises to zero mean, unit variance",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (4, 16)) * 7.0 + 3.0
out = {fn}(x, jnp.ones(16), jnp.zeros(16))

means = jnp.mean(out, axis=-1)
stds = jnp.std(out, axis=-1)
assert jnp.allclose(means, 0.0, atol=1e-4), f'Row means should be ~0, got {means}'
assert jnp.allclose(stds, 1.0, atol=1e-3), (
    f'Row stds should be ~1, got {stds}. If they are slightly below 1 you used '
    'the unbiased variance (ddof=1) — LayerNorm uses the biased one.'
)
""",
        },
        {
            "name": "Biased variance, not unbiased",
            "code": """
import jax
import jax.numpy as jnp

# With D=4 the biased/unbiased gap is 4/3 — far outside any tolerance.
x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
out = {fn}(x, jnp.ones(4), jnp.zeros(4))

biased = (x - x.mean()) / jnp.sqrt(jnp.var(x) + 1e-5)
unbiased = (x - x.mean()) / jnp.sqrt(jnp.var(x, ddof=1) + 1e-5)

assert jnp.allclose(out, biased, atol=1e-4), (
    f'Got {out}, expected {biased} (biased variance). '
    f'The unbiased version would give {unbiased}.'
)
""",
        },
        {
            "name": "gamma and beta are applied",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (4, 8))
gamma = jnp.full((8,), 3.0)
beta = jnp.full((8,), -1.0)
out = {fn}(x, gamma, beta)

base = {fn}(x, jnp.ones(8), jnp.zeros(8))
assert jnp.allclose(out, base * 3.0 - 1.0, atol=1e-4), (
    'gamma should scale and beta should shift the normalised value'
)

# Per-feature gamma must apply per feature, not broadcast as a scalar.
g = jnp.arange(1.0, 9.0)
out_g = {fn}(x, g, jnp.zeros(8))
assert jnp.allclose(out_g, base * g, atol=1e-4), 'gamma must apply element-wise over the last axis'
""",
        },
        {
            "name": "Normalises the last axis, not the batch axis",
            "code": """
import jax
import jax.numpy as jnp

# Columns have wildly different scales; rows do not. LayerNorm works per ROW,
# so every row must come out standardised regardless of the column scales.
x = jnp.array([[1.0, 100.0, 1.0, 100.0],
               [2.0, 200.0, 2.0, 200.0]])
out = {fn}(x, jnp.ones(4), jnp.zeros(4))

assert jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-4), (
    f'Row means should be ~0, got {jnp.mean(out, axis=-1)}. '
    'Reducing over axis 0 is BatchNorm, not LayerNorm.'
)
assert jnp.allclose(out[0], out[1], atol=1e-4), (
    'The two rows are proportional, so LayerNorm must map them to the same output'
)
""",
        },
        {
            "name": "Gradients flow to x, gamma and beta",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (4, 8))
gamma = jnp.ones(8)
beta = jnp.zeros(8)

gx, gg, gb = jax.grad(lambda a, b, c: jnp.sum({fn}(a, b, c) ** 2), argnums=(0, 1, 2))(x, gamma, beta)

for name, g in (("x", gx), ("gamma", gg), ("beta", gb)):
    assert jnp.isfinite(g).all(), f'Non-finite gradient w.r.t. {name}'
assert float(jnp.abs(gg).sum()) > 0, 'No gradient reached gamma'
assert float(jnp.abs(gb).sum()) > 0, 'No gradient reached beta'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(4), (6, 8))
gamma, beta = jnp.ones(8), jnp.zeros(8)
ref = {fn}(x, gamma, beta)

assert jnp.allclose(jax.jit({fn})(x, gamma, beta), ref, atol=1e-5), 'jit changes the result'

per_row = jax.vmap(lambda r: {fn}(r, gamma, beta))(x)
assert jnp.allclose(per_row, ref, atol=1e-5), (
    'vmapping over rows must match the batched call — LayerNorm is per-example'
)
""",
        },
    ],
}
