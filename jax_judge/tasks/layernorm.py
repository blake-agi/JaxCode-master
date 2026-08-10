"""LayerNorm as an nnx.Module — per-sample normalization plus affine params."""

TASK = {
    "title": "LayerNorm (nnx.Module)",
    "category": "Core Ops & Layers",
    "order": 5,
    "difficulty": "Easy",
    "function_name": "LayerNorm",
    "hint": (
        "Reduce over the LAST axis only, and pass keepdims=True — without it the "
        "reduced array has one fewer axis and the subtraction broadcasts against "
        "the wrong dimension. jnp.var already gives you the biased (divide-by-D) "
        "variance you want, so you do not need ddof at all. Initialise scale to "
        "ones and bias to zeros so an untrained layer is a pure normalization, "
        "and keep eps under the square root."
    ),
    "description": r"""
Implement **Layer Normalization** as an `nnx.Module`.

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance over the **last** axis only.
Every other axis is independent: for a `(B, T, D)` transformer activation you get
$B \times T$ separate means and variances, one per token, and the batch and time
axes are never reduced over.

### Rules
- Subclass `nnx.Module`; do **not** use `nnx.LayerNorm`
- Signature: `LayerNorm(dim, *, eps=1e-5, rngs=None)` — `rngs` is unused (the
  params are constants, not random), but every layer in this repo takes it so
  modules stay interchangeable
- `self.scale` initialised to ones, `self.bias` to zeros, both `nnx.Param`
- Normalise over the last axis only
- Use the **biased** variance (divide by `D`)
- `eps` goes **inside** the square root

### LayerNorm vs BatchNorm
This is the question that always follows.

|  | LayerNorm | BatchNorm |
|---|---|---|
| Normalises over | features, per sample | batch, per feature |
| Depends on batch | no | yes |
| Train ≠ eval | no | yes (running stats) |
| Batch size 1 | fine | broken |

Because LayerNorm never looks across the batch, it behaves identically at train
and inference time and needs no running statistics — which is exactly why every
transformer uses it. Sequence batches are padded to a common length, so
batch-wise statistics would be contaminated by however many pad positions
happened to land in the batch.

### Two conventions that are easy to get wrong
**`eps` placement.** It goes *inside*: $\sqrt{\sigma^2 + \epsilon}$, not
$\sqrt{\sigma^2} + \epsilon$. The outside form still produces finite,
plausible-looking output on ordinary data — it only bites when $\sigma^2$ is
tiny, which is precisely the case `eps` exists to handle.

**Biased variance.** Divide by $D$, not $D-1$. `jnp.var` and `np.var` do this by
default; `torch.var` does *not* — it defaults to the Bessel-corrected $D-1$ — so
a line-by-line port of PyTorch idiom silently normalises by the wrong denominator.
At $D = 768$ that is a 0.07% error in the divisor: far too small for any
tolerance-based test to catch, and exactly the kind of drift you hunt for when
your logits refuse to match a reference implementation.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    """Normalise over the last axis, then apply a learned affine transform."""

    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs = None):
        pass  # Replace this

    def __call__(self, x):
        """(..., dim) -> (..., dim)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs = None):
        # Starts as an identity transform: scale=1, bias=0.
        self.scale = nnx.Param(jnp.ones((dim,)))
        self.bias = nnx.Param(jnp.zeros((dim,)))
        self.eps = eps
        self.dim = dim

    def __call__(self, x):
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)          # biased: /D, not /(D-1)
        x_hat = (x - mu) / jnp.sqrt(var + self.eps)       # eps INSIDE the sqrt
        return x_hat * self.scale + self.bias
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

ln = LayerNorm(8)
x = jax.random.normal(jax.random.key(0), (2, 8)) * 5.0 + 3.0

out = ln(x)
print("input  mean/std per row:", x.mean(-1), x.std(-1))
print("output mean/std per row:", out.mean(-1), out.std(-1))
print("-> mean ~0, std ~1 regardless of the input scale")
''',
    "tests": [
        {
            "name": "Normalises to zero mean, unit variance",
            "code": """
import jax
import jax.numpy as jnp

ln = {fn}(16)
x = jax.random.normal(jax.random.key(0), (4, 16)) * 5.0 + 3.0
out = ln(x)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
assert jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-4), (
    f'Row means should be ~0, got {jnp.mean(out, axis=-1)}'
)
assert jnp.allclose(jnp.std(out, axis=-1), 1.0, atol=1e-3), (
    f'Row stds should be ~1, got {jnp.std(out, axis=-1)}'
)
""",
        },
        {
            "name": "Params exist with the right init",
            "code": """
import jax.numpy as jnp
from flax import nnx

ln = {fn}(8)
assert isinstance(ln.scale, nnx.Param), f'scale must be nnx.Param, got {type(ln.scale)}'
assert isinstance(ln.bias, nnx.Param), f'bias must be nnx.Param, got {type(ln.bias)}'
assert ln.scale.shape == (8,), f'scale shape {ln.scale.shape} vs (8,)'
assert ln.bias.shape == (8,), f'bias shape {ln.bias.shape} vs (8,)'
assert jnp.allclose(ln.scale[...], 1.0), 'scale must be initialised to ones'
assert jnp.allclose(ln.bias[...], 0.0), 'bias must be initialised to zeros'
""",
        },
        {
            "name": "Affine transform is applied",
            "code": """
import jax
import jax.numpy as jnp

ln = {fn}(4)
x = jax.random.normal(jax.random.key(1), (3, 4))

ln.scale[...] = jnp.array([2.0, 2.0, 2.0, 2.0])
ln.bias[...] = jnp.array([1.0, 1.0, 1.0, 1.0])
out = ln(x)

mu = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
expected = (x - mu) / jnp.sqrt(var + 1e-5) * 2.0 + 1.0

assert jnp.allclose(out, expected, atol=1e-4), 'scale/bias are not being applied'
assert jnp.allclose(jnp.mean(out, axis=-1), 1.0, atol=1e-3), 'bias should shift the mean to 1'
assert jnp.allclose(jnp.std(out, axis=-1), 2.0, atol=1e-2), 'scale should stretch std to 2'
""",
        },
        {
            "name": "Per-sample, not per-batch",
            "code": """
import jax
import jax.numpy as jnp

ln = {fn}(8)

# Rows on wildly different scales must each normalise independently.
x = jnp.stack([
    jnp.arange(8.0),
    jnp.arange(8.0) * 100.0,
    jnp.arange(8.0) - 50.0,
])
out = ln(x)

assert jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-4), 'Each row must have mean 0'
assert jnp.allclose(jnp.std(out, axis=-1), 1.0, atol=1e-3), 'Each row must have std 1'
# Because the rows are affine rescalings of each other, they normalise identically.
assert jnp.allclose(out[0], out[1], atol=1e-3), (
    'Rows differing only by scale should normalise to the same values — '
    'you may be reducing over the batch axis instead of the feature axis'
)

# Feeding one row alone must give the same answer as feeding the batch.
alone = ln(x[1:2])
assert jnp.allclose(alone[0], out[1], atol=1e-4), (
    'LayerNorm must not depend on the other samples in the batch'
)
""",
        },
        {
            "name": "eps is inside the sqrt",
            "code": """
import jax.numpy as jnp

ln = {fn}(4, eps=1e-2)
# A constant row has zero variance: output must be finite and exactly 0.
x = jnp.ones((1, 4)) * 7.0
out = ln(x)

assert jnp.isfinite(out).all(), f'Non-finite output on a constant row: {out}'
assert jnp.allclose(out, 0.0, atol=1e-6), f'Constant input should normalise to 0, got {out}'

# With a known variance, check the exact denominator.
y = jnp.array([[0.0, 2.0]])
ln2 = {fn}(2, eps=1e-2)
got = ln2(y)
var = 1.0                      # var([0, 2]) with /D is 1.0
expected = (y - 1.0) / jnp.sqrt(var + 1e-2)
assert jnp.allclose(got, expected, atol=1e-5), (
    f'{got} vs {expected} — check that eps is INSIDE the sqrt: sqrt(var + eps)'
)
""",
        },
        {
            "name": "Biased variance and 3-D input",
            "code": """
import jax
import jax.numpy as jnp

# var([0,2]) is 1.0 biased (/D) but 2.0 unbiased (/(D-1)).
ln = {fn}(2, eps=0.0)
out = ln(jnp.array([[0.0, 2.0]]))
assert jnp.allclose(jnp.abs(out), 1.0, atol=1e-4), (
    f'Expected +/-1 with the biased variance, got {out} — do not use ddof=1'
)

ln3 = {fn}(6)
x = jax.random.normal(jax.random.key(2), (2, 5, 6))
o = ln3(x)
assert o.shape == (2, 5, 6), f'{o.shape}'
assert jnp.allclose(jnp.mean(o, axis=-1), 0.0, atol=1e-4), '3-D: last-axis mean must be 0'
""",
        },
        {
            "name": "Gradient flows to both params",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

ln = {fn}(8)
x = jax.random.normal(jax.random.key(3), (4, 8))

grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(ln)
leaves = jax.tree.leaves(grads)
assert len(leaves) >= 2, f'Expected gradients for scale and bias, got {len(leaves)} leaves'
assert all(jnp.isfinite(l).all() for l in leaves), 'Non-finite gradients'
assert any(jnp.abs(l).sum() > 0 for l in leaves), 'All gradients are zero'
""",
        },
    ],
}
