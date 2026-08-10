"""Gradient accumulation — simulate a big batch out of small ones."""

TASK = {
    "title": "Gradient Accumulation",
    "category": "Training",
    "order": 6,
    "number": "31",
    "difficulty": "Easy",
    "function_name": "accumulated_step",
    "hint": (
        "Divide each micro-batch's loss by the number of micro-batches BEFORE "
        "differentiating, then sum the gradient trees — that sum is the "
        "full-batch gradient. nnx.value_and_grad gives you loss and grads "
        "together, and jax.tree.map adds two gradient States leaf by leaf. Apply "
        "the optimizer once, after the loop. There is no zero_grad to call."
    ),
    "description": r"""
Implement **gradient accumulation**: run several small micro-batches and apply
a single optimizer step, so the update matches what one large batch would have
produced.

### Signature
```python
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    ...  # -> total_loss (a float)
```

- `model`: an `nnx.Module`
- `optimizer`: an `nnx.Optimizer`
- `loss_fn`: `loss_fn(predictions, targets) -> scalar`
- `micro_batches`: a list of `(x, y)` pairs

### Rules
- Scale each micro-batch loss by `1 / len(micro_batches)` **before** taking
  gradients
- Accumulate the gradient pytrees, then call `optimizer.update` **once**
- Return the summed (already-scaled) loss as a Python float

### Why divide by n
Gradients are linear in the loss, so
$\nabla(\tfrac{1}{n}\sum_i L_i) = \tfrac{1}{n}\sum_i \nabla L_i$. Dividing each
micro-loss by `n` and summing the gradients reproduces the mean-reduced
full-batch gradient exactly.

Skip the division and your effective learning rate is multiplied by `n` — the
model still trains, often looks fine for a while, then diverges. That silent
scaling is the classic bug.

### The subtlety worth knowing
This is exact **only when the micro-batches are the same size**. `loss_fn`
typically returns a *mean* over its micro-batch, and an unweighted mean of
means is not the mean of the whole unless every group has equal weight. With a
ragged last batch you must weight each micro-batch by its row count and divide
by the total. Interviewers like this one because the code looks correct either
way.

### ⚠️ What is different from PyTorch
There is **no `zero_grad`**. PyTorch accumulates into `p.grad` as a side effect,
so you must clear it first — and forgetting is a classic bug. `nnx.grad` returns
a fresh gradient tree on every call, so accumulation is something you do
explicitly with `jax.tree.map`, and the bug cannot be written.

`nnx.Optimizer` does mutate the model in place, which is the one stateful
concession NNX makes for ergonomics — underneath it is still a functional
update applied to the module's parameter State.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


def accumulated_step(model, optimizer, loss_fn, micro_batches):
    """Accumulate gradients over micro-batches, then take one optimizer step.

    Args:
        model:         an nnx.Module
        optimizer:     an nnx.Optimizer
        loss_fn:       loss_fn(preds, targets) -> scalar
        micro_batches: list of (x, y) pairs

    Returns:
        The summed scaled loss, as a float.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


def accumulated_step(model, optimizer, loss_fn, micro_batches):
    n = len(micro_batches)
    total_loss = 0.0
    acc = None

    # No optimizer.zero_grad() — nnx.grad hands back a fresh tree each call,
    # so nothing accumulates behind our back.
    for x, y in micro_batches:
        loss, grads = nnx.value_and_grad(
            lambda m: loss_fn(m(x), y) / n
        )(model)

        # Sum the gradient trees; this is the full-batch gradient.
        acc = grads if acc is None else jax.tree.map(jnp.add, acc, grads)
        total_loss += float(loss)

    optimizer.update(model, acc)
    return total_loss
''',
    "demo": '''import jax
import jax.numpy as jnp
import optax
from flax import nnx

model = nnx.Linear(4, 1, rngs=nnx.Rngs(params=0))
opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

x = jax.random.normal(jax.random.key(1), (8, 4))
y = jax.random.normal(jax.random.key(2), (8,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

micro = [(x[:4], y[:4]), (x[4:], y[4:])]
print("loss:", accumulated_step(model, opt, mse, micro))
''',
    "tests": [
        {
            "name": "Equivalent to one full-batch step",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(0), (8, 4))
y = jax.random.normal(jax.random.key(1), (8,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

# Path A: two micro-batches through the function under test.
a = nnx.Linear(4, 1, rngs=nnx.Rngs(params=0))
opt_a = nnx.Optimizer(a, optax.sgd(0.1), wrt=nnx.Param)
{fn}(a, opt_a, mse, [(x[:4], y[:4]), (x[4:], y[4:])])

# Path B: one full batch, same init, same optimizer.
b = nnx.Linear(4, 1, rngs=nnx.Rngs(params=0))
opt_b = nnx.Optimizer(b, optax.sgd(0.1), wrt=nnx.Param)
grads = nnx.grad(lambda m: mse(m(x), y))(b)
opt_b.update(b, grads)

assert jnp.allclose(a.kernel[...], b.kernel[...], atol=1e-5), (
    'Accumulating two micro-batches must match a single full-batch step. '
    'If the weights moved further than expected, the 1/n scaling is missing.'
)
assert jnp.allclose(a.bias[...], b.bias[...], atol=1e-5), 'Bias differs from the full-batch step'
""",
        },
        {
            "name": "Returns the summed scaled loss",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(2), (8, 4))
y = jax.random.normal(jax.random.key(3), (8,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

m = nnx.Linear(4, 1, rngs=nnx.Rngs(params=1))
expected = float(mse(m(x), y))          # loss BEFORE the step

opt = nnx.Optimizer(m, optax.sgd(0.0), wrt=nnx.Param)   # lr=0 so nothing moves
got = {fn}(m, opt, mse, [(x[:4], y[:4]), (x[4:], y[4:])])

assert isinstance(got, float), f'Should return a Python float, got {type(got)}'
assert abs(got - expected) < 1e-4, (
    f'Returned {got}, expected the mean loss over the full batch {expected}. '
    'Each micro-loss is divided by n, so the SUM is the full-batch mean.'
)
""",
        },
        {
            "name": "The 1/n scaling is present",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(4), (8, 4))
y = jax.random.normal(jax.random.key(5), (8,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

# Four micro-batches vs one full batch: without the 1/n the step is 4x too big.
a = nnx.Linear(4, 1, rngs=nnx.Rngs(params=2))
opt_a = nnx.Optimizer(a, optax.sgd(0.1), wrt=nnx.Param)
{fn}(a, opt_a, mse, [(x[i*2:(i+1)*2], y[i*2:(i+1)*2]) for i in range(4)])

b = nnx.Linear(4, 1, rngs=nnx.Rngs(params=2))
opt_b = nnx.Optimizer(b, optax.sgd(0.1), wrt=nnx.Param)
opt_b.update(b, nnx.grad(lambda m: mse(m(x), y))(b))

start = nnx.Linear(4, 1, rngs=nnx.Rngs(params=2)).kernel[...]
step_a = jnp.abs(a.kernel[...] - start).sum()
step_b = jnp.abs(b.kernel[...] - start).sum()
assert jnp.allclose(step_a, step_b, rtol=1e-3), (
    f'Update magnitude {float(step_a):.5f} vs full-batch {float(step_b):.5f}. '
    'A ratio near 4 means each micro-loss was not divided by n=4.'
)
""",
        },
        {
            "name": "A single micro-batch is a plain step",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(6), (6, 4))
y = jax.random.normal(jax.random.key(7), (6,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

a = nnx.Linear(4, 1, rngs=nnx.Rngs(params=3))
opt_a = nnx.Optimizer(a, optax.sgd(0.05), wrt=nnx.Param)
{fn}(a, opt_a, mse, [(x, y)])

b = nnx.Linear(4, 1, rngs=nnx.Rngs(params=3))
opt_b = nnx.Optimizer(b, optax.sgd(0.05), wrt=nnx.Param)
opt_b.update(b, nnx.grad(lambda m: mse(m(x), y))(b))

assert jnp.allclose(a.kernel[...], b.kernel[...], atol=1e-6), (
    'With one micro-batch this must reduce to an ordinary step'
)
""",
        },
        {
            "name": "The optimizer steps exactly once",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(8), (8, 4))
y = jax.random.normal(jax.random.key(9), (8,))
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

m = nnx.Linear(4, 1, rngs=nnx.Rngs(params=4))
opt = nnx.Optimizer(m, optax.sgd(0.1), wrt=nnx.Param)

calls = {"n": 0}
real_update = opt.update
def counting_update(model, grads, **kw):
    calls["n"] += 1
    return real_update(model, grads, **kw)
opt.update = counting_update

{fn}(m, opt, mse, [(x[:2], y[:2]), (x[2:4], y[2:4]), (x[4:6], y[4:6]), (x[6:], y[6:])])
assert calls["n"] == 1, (
    f'optimizer.update should be called once after the loop, got {calls["n"]} '
    'calls — stepping per micro-batch defeats the purpose'
)
""",
        },
        {
            "name": "Training actually reduces the loss",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

x = jax.random.normal(jax.random.key(10), (16, 4))
w_true = jnp.array([2.0, -1.0, 0.5, 3.0])
y = x @ w_true
mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)

m = nnx.Linear(4, 1, rngs=nnx.Rngs(params=5))
opt = nnx.Optimizer(m, optax.sgd(0.1), wrt=nnx.Param)
micro = [(x[i*4:(i+1)*4], y[i*4:(i+1)*4]) for i in range(4)]

first = {fn}(m, opt, mse, micro)
for _ in range(200):
    last = {fn}(m, opt, mse, micro)

assert last < first, f'Loss did not decrease: {first} -> {last}'
assert last < 1e-2, f'Should converge on a linear problem, final loss {last}'
""",
        },
        {
            "name": "Gradients from every micro-batch contribute",
            "code": """
import jax
import jax.numpy as jnp
import optax
from flax import nnx

mse = lambda p, t: jnp.mean((p.squeeze(-1) - t) ** 2)
x = jax.random.normal(jax.random.key(11), (8, 4))
y = jax.random.normal(jax.random.key(12), (8,))

# Dropping any micro-batch must change the resulting weights.
full = nnx.Linear(4, 1, rngs=nnx.Rngs(params=6))
opt_f = nnx.Optimizer(full, optax.sgd(0.1), wrt=nnx.Param)
{fn}(full, opt_f, mse, [(x[:4], y[:4]), (x[4:], y[4:])])

part = nnx.Linear(4, 1, rngs=nnx.Rngs(params=6))
opt_p = nnx.Optimizer(part, optax.sgd(0.1), wrt=nnx.Param)
{fn}(part, opt_p, mse, [(x[:4], y[:4])])

assert not jnp.allclose(full.kernel[...], part.kernel[...], atol=1e-6), (
    'Dropping the second micro-batch changed nothing — only the last (or first) '
    'gradient is being used instead of the accumulated sum'
)
""",
        },
    ],
}
