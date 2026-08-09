"""RMSNorm — LayerNorm minus the mean subtraction, as used by Llama."""

TASK = {
    "title": "RMSNorm (nnx.Module)",
    "category": "Core Ops & Layers",
    "order": 6,
    "difficulty": "Easy",
    "function_name": "RMSNorm",
    "hint": (
        "rms = sqrt(mean(x**2, axis=-1, keepdims=True) + eps), then "
        "x / rms * scale. Note there is no mean subtraction and no bias term — "
        "that is the entire difference from LayerNorm. Compute the mean of the "
        "SQUARES, not the square of the mean."
    ),
    "description": r"""
Implement **RMSNorm** as an `nnx.Module`.

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

### Rules
- Subclass `nnx.Module`
- Signature: `RMSNorm(dim, *, eps=1e-6, rngs=None)`
- One parameter only: `self.scale`, initialised to ones
- **No mean subtraction** and **no bias** — that is the whole point
- Normalise over the last axis

### Why drop the mean
RMSNorm is LayerNorm with the re-centering removed. The 2019 paper's finding was
that the *re-scaling* is what stabilises training; the *re-centering* contributes
almost nothing. Dropping it saves a pass over the data and a subtraction, which
at Llama scale is a real win.

Llama, Mistral, Gemma, and T5 all use RMSNorm. GPT-2 and BERT use LayerNorm.

The follow-up worth being ready for: **when do the two coincide?** Exactly when
the input already has zero mean — then $\text{RMS}(x) = \sigma(x)$ and the two
are identical up to the missing bias.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root-mean-square normalization over the last axis."""

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs = None):
        pass  # Replace this

    def __call__(self, x):
        """(..., dim) -> (..., dim)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.scale = nnx.Param(jnp.ones((dim,)))   # no bias term
        self.eps = eps
        self.dim = dim

    def __call__(self, x):
        # Mean of the squares — no mean subtraction anywhere.
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x / jnp.sqrt(ms + self.eps) * self.scale
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

rms = RMSNorm(8)
x = jax.random.normal(jax.random.key(0), (2, 8)) * 3.0 + 10.0

out = rms(x)
print("input  RMS per row:", jnp.sqrt(jnp.mean(x ** 2, -1)))
print("output RMS per row:", jnp.sqrt(jnp.mean(out ** 2, -1)), "(~1)")
print("output mean per row:", out.mean(-1), "(NOT ~0 — no re-centering)")
''',
    "tests": [
        {
            "name": "Output has unit RMS",
            "code": """
import jax
import jax.numpy as jnp

rn = {fn}(16)
x = jax.random.normal(jax.random.key(0), (4, 16)) * 5.0
out = rn(x)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
assert jnp.allclose(rms, 1.0, atol=1e-3), f'Row RMS should be ~1, got {rms}'
""",
        },
        {
            "name": "Exact formula",
            "code": """
import jax
import jax.numpy as jnp

rn = {fn}(4, eps=1e-6)
x = jnp.array([[3.0, 4.0, 0.0, 0.0]])

# mean of squares = (9 + 16) / 4 = 6.25, rms = 2.5
out = rn(x)
expected = x / jnp.sqrt(6.25 + 1e-6)
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
assert jnp.allclose(out[0, 0], 1.2, atol=1e-4), f'3/2.5 = 1.2, got {out[0, 0]}'
""",
        },
        {
            "name": "No mean subtraction, no bias",
            "code": """
import jax.numpy as jnp
from flax import nnx

rn = {fn}(4)
assert not hasattr(rn, "bias") or rn.bias is None, (
    'RMSNorm has no bias parameter — that is part of what distinguishes it '
    'from LayerNorm'
)
assert isinstance(rn.scale, nnx.Param), f'scale must be nnx.Param, got {type(rn.scale)}'
assert jnp.allclose(rn.scale[...], 1.0), 'scale must be initialised to ones'

# An all-positive input keeps a positive mean: re-centering would zero it.
x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
out = rn(x)
assert float(jnp.mean(out)) > 0.5, (
    f'Output mean is {float(jnp.mean(out))}, close to 0 — you are subtracting the '
    'mean. RMSNorm must not re-center.'
)
assert (out > 0).all(), 'All-positive input must stay all-positive'
""",
        },
        {
            "name": "Scale parameter is applied",
            "code": """
import jax
import jax.numpy as jnp

rn = {fn}(4)
x = jax.random.normal(jax.random.key(1), (3, 4))

base = rn(x)
rn.scale[...] = jnp.full((4,), 3.0)
scaled = rn(x)

assert jnp.allclose(scaled, base * 3.0, atol=1e-4), 'scale is not being applied'

rn.scale[...] = jnp.array([1.0, 2.0, 3.0, 4.0])
per_channel = rn(x)
assert jnp.allclose(per_channel, base * jnp.array([1.0, 2.0, 3.0, 4.0]), atol=1e-4), (
    'scale must broadcast per feature'
)
""",
        },
        {
            "name": "Matches LayerNorm on zero-mean input",
            "code": """
import jax
import jax.numpy as jnp

rn = {fn}(8, eps=1e-6)
x = jax.random.normal(jax.random.key(2), (4, 8))
x = x - jnp.mean(x, axis=-1, keepdims=True)      # force zero mean

out = rn(x)
mu = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
ln_equivalent = (x - mu) / jnp.sqrt(var + 1e-6)

assert jnp.allclose(out, ln_equivalent, atol=1e-3), (
    'On zero-mean input RMSNorm and LayerNorm must agree'
)
""",
        },
        {
            "name": "Stability, 3-D input, and gradients",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

rn = {fn}(4, eps=1e-6)
zeros = rn(jnp.zeros((1, 4)))
assert jnp.isfinite(zeros).all(), f'All-zero input gave {zeros} — eps must guard the sqrt'

rn3 = {fn}(6)
x = jax.random.normal(jax.random.key(3), (2, 5, 6))
o = rn3(x)
assert o.shape == (2, 5, 6), f'{o.shape}'
assert jnp.allclose(jnp.sqrt(jnp.mean(o ** 2, -1)), 1.0, atol=1e-3), '3-D RMS'

grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(rn3)
leaves = jax.tree.leaves(grads)
assert len(leaves) >= 1, 'No gradient for scale'
assert all(jnp.isfinite(l).all() for l in leaves), 'Non-finite gradients'
""",
        },
    ],
}
