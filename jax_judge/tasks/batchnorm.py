"""BatchNorm with running statistics — nnx.BatchStat and mutable module state."""

TASK = {
    "title": "Implement BatchNorm",
    "category": "Core Ops & Layers",
    "order": 7,
    "difficulty": "Medium",
    "function_name": "BatchNorm",
    "hint": (
        "Normalise over every axis EXCEPT the feature axis — for (N, C) that is "
        "axis 0, for (N, H, W, C) it is the first three. Derive the axes from "
        "x.ndim rather than hard-coding them, or the layer only works for one "
        "input rank. Training and eval differ in TWO ways: which statistics you "
        "normalise with, and whether you write to the running buffers at all. "
        "Store those buffers as nnx.BatchStat rather than nnx.Param so the "
        "optimizer never updates them."
    ),
    "description": r"""
Implement **Batch Normalization** with running statistics.

**Training** — normalise with the statistics of the current batch, and update
the running buffers:

$$\mu_{run} \leftarrow m\,\mu_{run} + (1-m)\,\mu_{batch}$$

**Inference** — normalise with the stored running statistics, update nothing.

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\gamma + \beta$$

### Rules
- Signature: `BatchNorm(num_features, *, momentum=0.9, eps=1e-5, rngs=None)`
- `__call__(x, use_running_average=False)`
- `self.scale`, `self.bias` → `nnx.Param` (ones / zeros)
- `self.running_mean`, `self.running_var` → `nnx.BatchStat` (zeros / ones)
- Reduce over **all axes except the last** — works for `(N, C)` and `(N, H, W, C)`
- Use the **biased** batch variance
- Running buffers must update **only** in training mode

### Why `nnx.BatchStat` and not `nnx.Param`
Running statistics are **state**, not parameters: they are updated by an
exponential moving average, not by gradient descent. Tagging them `BatchStat`
lets you filter them out when you build the optimizer:

```python
params = nnx.state(model, nnx.Param)        # optimizer sees only these
stats  = nnx.state(model, nnx.BatchStat)    # carried along, never differentiated
```

NNX modules are **mutable**, so `self.running_mean[...] = ...` inside `__call__`
just works — no threading of a `mutable` collection through every call the way
Linen requires. This is the clearest demonstration of what NNX buys you.

### The classic gotcha
Forgetting to switch to eval mode means inference normalises by whatever happens
to be in the current batch, so predictions change depending on what else you
batched alongside them — and with batch size 1 the variance is 0 and everything
collapses.

### ⚠️ Flax and PyTorch genuinely disagree here — twice
Both frameworks **normalise** with the biased (population) variance. Everything
else about the running buffer differs:

| | `flax.nnx.BatchNorm` | `torch.nn.BatchNorm1d` |
|---|---|---|
| `running_var` update uses | **biased** variance (`ddof=0`) | **unbiased** variance (`ddof=1`) |
| `momentum` means | weight kept on the **old** value | weight given to the **new** value |
| default `momentum` | `0.99` | `0.1` |
| feature axis | last (`axis=-1`, `NHWC`) | axis 1 (`NCHW`) |

So the two variance conventions differ by exactly the Bessel factor $n/(n-1)$ —
3.2% at batch size 32 — and Flax's `momentum=0.9` is PyTorch's `momentum=0.1`,
not `0.9`. Get that one backwards and your buffers track the *last* batch instead
of the running average.

Both bugs are invisible during training, because training never reads the
buffers. They surface at **inference**, after the weights already look fine.
This task follows the Flax convention throughout. (Verified empirically against
PyTorch — see `jax_pytorch_comparison/crosscheck_vs_torch.py`.)
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class BatchNorm(nnx.Module):
    """BatchNorm over the last (feature) axis, with running statistics."""

    def __init__(self, num_features: int, *, momentum: float = 0.9,
                 eps: float = 1e-5, rngs: nnx.Rngs = None):
        pass  # Replace this

    def __call__(self, x, use_running_average: bool = False):
        """(..., num_features) -> same shape."""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class BatchNorm(nnx.Module):
    def __init__(self, num_features: int, *, momentum: float = 0.9,
                 eps: float = 1e-5, rngs: nnx.Rngs = None):
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        # BatchStat, not Param: updated by EMA, never by the optimizer.
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,)))
        self.running_var = nnx.BatchStat(jnp.ones((num_features,)))
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features

    def __call__(self, x, use_running_average: bool = False):
        if use_running_average:
            mean = self.running_mean[...]
            var = self.running_var[...]
        else:
            # Every axis except the feature axis.
            reduce_axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=reduce_axes)
            var = jnp.var(x, axis=reduce_axes)
            m = self.momentum
            # NNX modules are mutable — assign the new buffers directly.
            self.running_mean[...] = m * self.running_mean[...] + (1 - m) * mean
            self.running_var[...] = m * self.running_var[...] + (1 - m) * var

        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        return x_hat * self.scale + self.bias
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

bn = BatchNorm(4)
x = jax.random.normal(jax.random.key(0), (32, 4)) * 3.0 + 5.0

print("running_mean before:", bn.running_mean[...])
out = bn(x)                       # training mode
print("running_mean after: ", bn.running_mean[...])
print("train-mode output mean:", out.mean(0), "(~0)")

eval_out = bn(x, use_running_average=True)
print("eval-mode output mean: ", eval_out.mean(0), "(NOT ~0 — uses running stats)")
''',
    "tests": [
        {
            "name": "Training mode normalises the batch",
            "code": """
import jax
import jax.numpy as jnp

bn = {fn}(8)
x = jax.random.normal(jax.random.key(0), (32, 8)) * 4.0 + 2.0
out = bn(x, use_running_average=False)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=1e-4), (
    f'Per-feature mean should be ~0, got {jnp.mean(out, axis=0)}'
)
assert jnp.allclose(jnp.std(out, axis=0), 1.0, atol=1e-3), (
    f'Per-feature std should be ~1, got {jnp.std(out, axis=0)}'
)
""",
        },
        {
            "name": "Correct variable kinds",
            "code": """
import jax.numpy as jnp
from flax import nnx

bn = {fn}(4)

assert isinstance(bn.scale, nnx.Param), f'scale must be nnx.Param, got {type(bn.scale)}'
assert isinstance(bn.bias, nnx.Param), f'bias must be nnx.Param, got {type(bn.bias)}'
assert isinstance(bn.running_mean, nnx.BatchStat), (
    f'running_mean must be nnx.BatchStat (state, not a learnable parameter), '
    f'got {type(bn.running_mean)}'
)
assert isinstance(bn.running_var, nnx.BatchStat), (
    f'running_var must be nnx.BatchStat, got {type(bn.running_var)}'
)

assert jnp.allclose(bn.running_mean[...], 0.0), 'running_mean must start at zeros'
assert jnp.allclose(bn.running_var[...], 1.0), 'running_var must start at ones'
assert jnp.allclose(bn.scale[...], 1.0) and jnp.allclose(bn.bias[...], 0.0)

# Params and stats must be separable — this is what the optimizer relies on.
params = nnx.state(bn, nnx.Param)
stats = nnx.state(bn, nnx.BatchStat)
import jax as _jax
assert len(_jax.tree.leaves(params)) == 2, 'Expected exactly 2 Param leaves'
assert len(_jax.tree.leaves(stats)) == 2, 'Expected exactly 2 BatchStat leaves'
""",
        },
        {
            "name": "Running stats update with the right EMA",
            "code": """
import jax
import jax.numpy as jnp

bn = {fn}(4, momentum=0.9)
x = jax.random.normal(jax.random.key(1), (64, 4)) * 2.0 + 3.0

batch_mean = jnp.mean(x, axis=0)
batch_var = jnp.var(x, axis=0)

bn(x, use_running_average=False)

expected_mean = 0.9 * 0.0 + 0.1 * batch_mean
expected_var = 0.9 * 1.0 + 0.1 * batch_var

assert jnp.allclose(bn.running_mean[...], expected_mean, atol=1e-4), (
    f'{bn.running_mean[...]} vs {expected_mean} — check momentum * old + (1 - momentum) * new'
)
assert jnp.allclose(bn.running_var[...], expected_var, atol=1e-4), (
    f'{bn.running_var[...]} vs {expected_var}'
)

# After many passes the running stats should approach the true batch stats.
for _ in range(200):
    bn(x, use_running_average=False)
assert jnp.allclose(bn.running_mean[...], batch_mean, atol=1e-2), 'EMA did not converge'
assert jnp.allclose(bn.running_var[...], batch_var, atol=1e-2), 'EMA did not converge'
""",
        },
        {
            "name": "Eval mode uses running stats and updates nothing",
            "code": """
import jax
import jax.numpy as jnp

bn = {fn}(4)
x = jax.random.normal(jax.random.key(2), (16, 4)) * 3.0 + 1.0

before_mean = bn.running_mean[...].copy()
before_var = bn.running_var[...].copy()

out = bn(x, use_running_average=True)

assert jnp.allclose(bn.running_mean[...], before_mean), (
    'running_mean changed in eval mode — buffers must only update during training'
)
assert jnp.allclose(bn.running_var[...], before_var), 'running_var changed in eval mode'

# With the initial buffers (mean 0, var 1) eval mode is nearly a no-op.
assert jnp.allclose(out, x, atol=1e-3), (
    'With running_mean=0 and running_var=1, eval output should be ~x'
)

# It must NOT normalise the incoming batch.
assert not jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=1e-2), (
    'Eval-mode output was batch-normalised — it should use the running stats'
)
""",
        },
        {
            "name": "Eval mode is independent of batch composition",
            "code": """
import jax
import jax.numpy as jnp

bn = {fn}(4)
train_x = jax.random.normal(jax.random.key(3), (64, 4)) * 2.0 + 1.0
for _ in range(50):
    bn(train_x, use_running_average=False)

sample = train_x[:1]
alone = bn(sample, use_running_average=True)
in_batch = bn(train_x, use_running_average=True)[:1]

assert jnp.allclose(alone, in_batch, atol=1e-5), (
    'Eval output for one sample changed depending on what it was batched with — '
    'eval mode must not look at the batch'
)
assert jnp.isfinite(alone).all(), 'Batch size 1 must work in eval mode'
""",
        },
        {
            "name": "4-D (N, H, W, C) input",
            "code": """
import jax
import jax.numpy as jnp

bn = {fn}(3)
x = jax.random.normal(jax.random.key(4), (8, 5, 5, 3)) * 2.0 + 1.0
out = bn(x, use_running_average=False)

assert out.shape == (8, 5, 5, 3), f'{out.shape}'
# Statistics are per-channel, reduced over N, H and W.
assert jnp.allclose(jnp.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-4), (
    'Per-channel mean should be ~0 — reduce over every axis except the last'
)
assert jnp.allclose(jnp.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3), 'Per-channel std'
assert bn.running_mean[...].shape == (3,), f'{bn.running_mean[...].shape} vs (3,)'
""",
        },
        {
            "name": "Affine params and gradients",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

bn = {fn}(4)
x = jax.random.normal(jax.random.key(5), (16, 4))

bn.scale[...] = jnp.full((4,), 2.0)
bn.bias[...] = jnp.full((4,), 5.0)
out = bn(x, use_running_average=False)
assert jnp.allclose(jnp.mean(out, axis=0), 5.0, atol=1e-3), 'bias should shift the mean'
assert jnp.allclose(jnp.std(out, axis=0), 2.0, atol=1e-2), 'scale should stretch the std'

grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(bn)
leaves = jax.tree.leaves(grads)
assert all(jnp.isfinite(l).all() for l in leaves), 'Non-finite gradients'
assert len(leaves) == 2, (
    f'nnx.grad should differentiate only the 2 Params, got {len(leaves)} leaves — '
    'running stats must be BatchStat so they are excluded'
)
""",
        },
    ],
}
