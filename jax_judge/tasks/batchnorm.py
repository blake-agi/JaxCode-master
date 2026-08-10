"""BatchNorm as a pure function — and why JAX must return the running stats."""

TASK = {
    "title": "Implement BatchNorm",
    "category": "Core Ops & Layers",
    "number": "07",
    "difficulty": "Medium",
    "function_name": "my_batch_norm",
    "hint": (
        "Reduce over axis 0 (the batch), not the last axis — that is the whole "
        "difference from LayerNorm. Use the biased variance for both the "
        "normalisation and the running-buffer update. In training you normalise "
        "with the batch statistics and move the buffers toward them; at eval you "
        "use the buffers and touch nothing. JAX arrays are immutable, so the new "
        "buffers have to be RETURNED, not updated in place."
    ),
    "description": r"""
Implement **Batch Normalization** with running statistics, as a pure function.

**Training** — normalise with the current batch's statistics, and move the
running buffers toward them:

$$\mu_{run} \leftarrow (1-m)\,\mu_{run} + m\,\mu_{batch}$$

**Inference** — normalise with the stored buffers, update nothing.

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

### Signature
```python
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    ...  # -> (out, running_mean, running_var)
```

### Rules
- Do **not** use `nnx.BatchNorm`
- Reduce over the **batch** axis (axis 0), not the feature axis
- Use the **biased** variance (`ddof=0`)
- `momentum=0.1` means the new batch gets weight `0.1`
- Buffers update **only** in training mode

### ⚠️ The one place JAX forces a different signature
PyTorch updates the buffers in place:

```python
running_mean.mul_(1 - momentum).add_(momentum * batch_mean)   # mutates the caller's tensor
```

JAX arrays are **immutable** — there is no `mul_`. So this version **returns**
the new buffers instead:

```python
out, running_mean, running_var = my_batch_norm(x, gamma, beta,
                                               running_mean, running_var)
```

This is not a workaround, it is the JAX model: state is threaded through as
values rather than hidden in objects, which is exactly what makes the function
`jit`-able and free of side effects. (`nnx.BatchStat` exists for when you *do*
want the buffers to live in a module — see how `nnx.BatchNorm` does it.)

### BatchNorm vs LayerNorm
LayerNorm reduces over features, per example — so it is independent of batch
size and works with a batch of 1. BatchNorm reduces over the batch, which
couples examples to each other: predictions change depending on what else was
batched alongside them, and with batch size 1 the variance is 0 and everything
collapses. That coupling is why transformers use LayerNorm.

### A note on the variance convention
This task uses the biased variance for the running buffer, matching
`flax.nnx.BatchNorm`. Real `torch.nn.BatchNorm1d` differs — it updates the
buffer with the *unbiased* variance while normalising with the biased one. The
gap is the Bessel factor $n/(n-1)$, it only shows up at inference, and it is a
classic porting bug. See `jax_pytorch_comparison/crosscheck_vs_torch.py`.
""",
    "stub": '''import jax
import jax.numpy as jnp


def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    """BatchNorm over the batch axis, with running statistics.

    Args:
        x:            (N, D) array
        gamma:        (D,) scale
        beta:         (D,) shift
        running_mean: (D,) buffer
        running_var:  (D,) buffer
        eps:          stability term inside the sqrt
        momentum:     weight given to the new batch statistics
        training:     use batch stats and update buffers, or use buffers as-is

    Returns:
        (out, running_mean, running_var) — JAX cannot mutate the buffers
        in place, so the updated ones come back as return values.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if training:
        # Reduce over the BATCH axis. Biased variance (ddof=0).
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)

        # PyTorch would do running_mean.mul_(1-m).add_(m*batch_mean) in place.
        # JAX arrays are immutable, so we build new buffers and return them.
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

        mean, var = batch_mean, batch_var
    else:
        mean, var = running_mean, running_var

    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta, running_mean, running_var
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (32, 4)) * 3.0 + 5.0
gamma, beta = jnp.ones(4), jnp.zeros(4)
rm, rv = jnp.zeros(4), jnp.ones(4)

out, rm, rv = my_batch_norm(x, gamma, beta, rm, rv, training=True)
print("train-mode column means:", out.mean(0), "(~0)")
print("running_mean after 1 step:", rm, "(moved 10% toward the batch mean)")

eval_out, _, _ = my_batch_norm(x, gamma, beta, rm, rv, training=False)
print("eval-mode column means: ", eval_out.mean(0), "(NOT ~0 — uses the buffers)")
''',
    "tests": [
        {
            "name": "Training mode normalises the batch",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (32, 4)) * 4.0 + 2.0
out, rm, rv = {fn}(x, jnp.ones(4), jnp.zeros(4), jnp.zeros(4), jnp.ones(4), training=True)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
col_means = jnp.mean(out, axis=0)
col_stds = jnp.std(out, axis=0)
assert jnp.allclose(col_means, 0.0, atol=1e-4), f'Column means should be ~0, got {col_means}'
assert jnp.allclose(col_stds, 1.0, atol=1e-3), (
    f'Column stds should be ~1, got {col_stds}. Below 1 means you used the '
    'unbiased variance — BatchNorm normalises with the biased one.'
)
""",
        },
        {
            "name": "Reduces over the batch axis, not the features",
            "code": """
import jax
import jax.numpy as jnp

# Rows are proportional; columns have very different scales. BatchNorm works
# per COLUMN, so every column must come out standardised.
x = jnp.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
out, _, _ = {fn}(x, jnp.ones(2), jnp.zeros(2), jnp.zeros(2), jnp.ones(2), training=True)

assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=1e-4), (
    f'Column means should be ~0, got {jnp.mean(out, axis=0)}. '
    'Reducing over the last axis is LayerNorm, not BatchNorm.'
)
assert jnp.allclose(out[:, 0], out[:, 1], atol=1e-4), (
    'Both columns are the same up to scale, so BatchNorm must map them identically'
)
""",
        },
        {
            "name": "Running buffers update with the right momentum",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (16, 4)) * 2.0 + 3.0
rm0, rv0 = jnp.zeros(4), jnp.ones(4)
_, rm, rv = {fn}(x, jnp.ones(4), jnp.zeros(4), rm0, rv0, momentum=0.1, training=True)

bm, bv = jnp.mean(x, axis=0), jnp.var(x, axis=0)
assert jnp.allclose(rm, 0.9 * rm0 + 0.1 * bm, atol=1e-5), (
    f'running_mean should be (1-m)*old + m*batch, got {rm}'
)
assert jnp.allclose(rv, 0.9 * rv0 + 0.1 * bv, atol=1e-5), (
    f'running_var should use the BIASED batch variance, got {rv}'
)

# Repeated steps must converge toward the batch statistics.
rm_i, rv_i = jnp.zeros(4), jnp.ones(4)
for _ in range(200):
    _, rm_i, rv_i = {fn}(x, jnp.ones(4), jnp.zeros(4), rm_i, rv_i, training=True)
assert jnp.allclose(rm_i, bm, atol=1e-3), 'EMA did not converge to the batch mean'
""",
        },
        {
            "name": "Eval mode uses the buffers and updates nothing",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (16, 4)) * 5.0 + 7.0
rm = jnp.full((4,), 1.0)
rv = jnp.full((4,), 4.0)

out, rm2, rv2 = {fn}(x, jnp.ones(4), jnp.zeros(4), rm, rv, training=False)

assert jnp.allclose(rm2, rm) and jnp.allclose(rv2, rv), (
    'Buffers must not change in eval mode'
)
expected = (x - rm) / jnp.sqrt(rv + 1e-5)
assert jnp.allclose(out, expected, atol=1e-4), (
    'Eval mode must normalise with the running buffers, not the batch statistics'
)
assert not jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=1e-2), (
    'Eval output has ~zero column means, so it used the batch statistics — '
    'that is the train-mode path'
)
""",
        },
        {
            "name": "gamma and beta are applied",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (16, 4))
base, _, _ = {fn}(x, jnp.ones(4), jnp.zeros(4), jnp.zeros(4), jnp.ones(4), training=True)

g = jnp.arange(1.0, 5.0)
b = jnp.full((4,), -2.0)
out, _, _ = {fn}(x, g, b, jnp.zeros(4), jnp.ones(4), training=True)
assert jnp.allclose(out, base * g + b, atol=1e-4), 'gamma must scale and beta must shift'
""",
        },
        {
            "name": "Buffers are returned, not mutated",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(4), (16, 4))
rm = jnp.zeros(4)
rv = jnp.ones(4)
rm_before = rm.copy()
rv_before = rv.copy()

out, rm_new, rv_new = {fn}(x, jnp.ones(4), jnp.zeros(4), rm, rv, training=True)

assert jnp.allclose(rm, rm_before) and jnp.allclose(rv, rv_before), (
    'The arrays passed in must be untouched — JAX arrays are immutable'
)
assert not jnp.allclose(rm_new, rm_before), 'The returned running_mean should have moved'
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(5), (16, 4))
g, b = jnp.ones(4), jnp.zeros(4)
rm, rv = jnp.zeros(4), jnp.ones(4)

def loss(x_, g_, b_):
    out, _, _ = {fn}(x_, g_, b_, rm, rv, training=True)
    return jnp.sum(out ** 2)

gx, gg, gb = jax.grad(loss, argnums=(0, 1, 2))(x, g, b)
for name, gr in (("x", gx), ("gamma", gg), ("beta", gb)):
    assert jnp.isfinite(gr).all(), f'Non-finite gradient w.r.t. {name}'
assert float(jnp.abs(gg).sum()) > 0, 'No gradient reached gamma'

jitted = jax.jit(functools.partial({fn}, training=True))
o1, m1, v1 = jitted(x, g, b, rm, rv)
o2, m2, v2 = {fn}(x, g, b, rm, rv, training=True)
assert jnp.allclose(o1, o2, atol=1e-5), 'jit changes the output'
assert jnp.allclose(m1, m2, atol=1e-6), 'jit changes the running_mean'
""",
        },
    ],
}
