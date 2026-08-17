"""Linear regression where the training loop itself is a lax.scan."""

TASK = {
    "title": "Linear Regression with lax.scan",
    "category": "Training",
    "number": "b_18",
    "difficulty": "Medium",
    "function_name": "LinearRegressionScan",
    "hint": (
        "gradient_descent: the carry is (w, b) and there are no xs — pass "
        "None and give scan `length=steps`. The per-step output is the loss, "
        "so `return (w_new, b_new), loss` and scan stacks the losses into "
        "(steps,) for free. Record the loss BEFORE the update, computed from "
        "the same error vector you build the gradients from. closed_form and "
        "nn_linear are unchanged from problem 40."
    ),
    "description": r"""
Problem 40 again, with the training loop rewritten as a `jax.lax.scan` —
and the loss curve falling out of it for free.

$$\hat{y} = Xw + b, \qquad \mathcal{L} = \frac{1}{N}\|Xw + b - y\|^2$$

### Signature
```python
class LinearRegressionScan:
    def closed_form(self, X, y): ...                          # -> (w, b)
    def gradient_descent(self, X, y, lr=0.01, steps=1000): ...# -> (w, b, losses)
    def nn_linear(self, X, y, lr=0.01, steps=1000): ...       # -> (w, b)
```

`X` is `(N, D)`, `y` is `(N,)`, `w` comes back `(D,)` and `b` is a scalar.

### What changes from problem 40
Only `gradient_descent`, and it changes in two ways:

1. **No Python `for` loop.** The update goes in a `lax.scan` body.
2. **It returns a third value**, `losses` of shape `(steps,)` — the MSE
   *before* each update, so `losses[0]` is the loss at the starting point
   `w = 0, b = 0`. This is not extra work: it is the `ys` that scan stacks
   for you, and it is the reason to reach for scan rather than
   `fori_loop`.

`closed_form` and `nn_linear` are exactly as in problem 40. Keep them —
`closed_form` is what the scan version has to agree with.

### Fitting a training loop into scan
Scan wants `(carry, x) -> (carry, y)`. A training loop has no per-step input,
so `xs` is `None` and the trip count comes from `length=`:

```python
def step(carry, _):
    w, b = carry
    ...
    return (w_new, b_new), loss        # (new carry, per-step output)

(w, b), losses = jax.lax.scan(step, (w0, b0), None, length=steps)
```

The gradients are still the ones you derive by hand — no `jax.grad`:

$$\nabla_w = \frac{2}{N}X^\top(\hat{y}-y), \qquad
  \nabla_b = \frac{2}{N}\sum(\hat{y}-y)$$

### Why this is the version that matters
A Python loop is unrolled at trace time: 2000 steps means a 2000-node graph,
and XLA has to compile every one of them. `scan` compiles the body **once**
and loops it, so compile time is flat in `steps`. Measured on this problem:

| | steps | wall clock |
|---|---|---|
| Python loop (problem 40) | 2 000 | 31.4 s |
| `lax.scan` | 20 000 | 0.04 s |

Ten times the steps, roughly a thousandth of the time. That gap is not an
optimisation detail — it is the difference between a training loop you can
`jit` and one you cannot.

`steps` has to be **static**, because it is the length of the scan and
therefore part of the shape of `losses`. Under `jit` that means
`static_argnames=('steps',)`.

### Where you have met this before
This is the same shape as `b_06`'s discounted returns — carry a state, emit
one value per step. An optimizer loop, an RNN, a KV-cache decode and a
diffusion sampler are all this one pattern; only the carry changes.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class LinearRegressionScan:
    """Least squares, with the training loop as a scan."""

    def closed_form(self, X, y):
        """Normal equations via an augmented matrix. -> (w, b)"""
        pass  # Replace this

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        """Manual gradients inside a lax.scan. -> (w, b, losses)

        losses is (steps,): the MSE BEFORE each update, so losses[0] is the
        loss at w = 0, b = 0.
        """
        pass  # Replace this

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        """nnx.Linear trained with nnx.grad. -> (w, b)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class LinearRegressionScan:
    def closed_form(self, X, y):
        N, D = X.shape
        X_aug = jnp.concatenate([X, jnp.ones((N, 1))], axis=1)   # (N, D+1)
        theta = jnp.linalg.lstsq(X_aug, y)[0]                    # (D+1,)
        return theta[:D], theta[D]

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        N, D = X.shape

        def step(carry, _):
            w, b = carry
            error = X @ w + b - y                # (N,)
            loss = jnp.mean(error ** 2)          # BEFORE the update
            grad_w = (2.0 / N) * (X.T @ error)   # (D,)
            grad_b = (2.0 / N) * jnp.sum(error)  # scalar
            return (w - lr * grad_w, b - lr * grad_b), loss

        # No per-step input, so xs=None and the trip count comes from length=.
        init = (jnp.zeros(D), jnp.array(0.0))
        (w, b), losses = jax.lax.scan(step, init, None, length=steps)
        return w, b, losses

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        D = X.shape[1]
        layer = nnx.Linear(D, 1, rngs=nnx.Rngs(params=0))

        def loss_fn(model, X, y):
            return jnp.mean((model(X).squeeze(-1) - y) ** 2)

        grad_fn = nnx.grad(loss_fn)          # transform once, not per step

        for _ in range(steps):
            grads = grad_fn(layer, X, y)
            # One tree.map updates every parameter, whatever the module is.
            params = nnx.state(layer, nnx.Param)
            nnx.update(layer, jax.tree.map(lambda p, g: p - lr * g, params, grads))

        # Flax kernel is (D, 1) — the transpose of torch's (1, D).
        return layer.kernel[...].squeeze(-1), layer.bias[...].squeeze()
''',
    "demo": '''import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (200, 3))
w_true = jnp.array([2.0, -3.0, 0.5])
y = X @ w_true + 1.5

model = LinearRegressionScan()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd, losses = model.gradient_descent(X, y, lr=0.1, steps=2000)

print(f"closed_form       w={jnp.round(w_cf, 3)}  b={float(b_cf):.3f}")
print(f"gradient_descent  w={jnp.round(w_gd, 3)}  b={float(b_gd):.3f}")
print(f"truth             w={w_true}  b=1.5")

# The loss curve came free with the scan — no extra pass over the data.
print(f"\\nlosses {losses.shape}: {float(losses[0]):.4f} -> {float(losses[-1]):.6f}")
for i in (0, 1, 2, 10, 100, 1999):
    print(f"  step {i:>4}: {float(losses[i]):.6f}")
''',
    "tests": [
        {
            "name": "Closed form recovers exact coefficients",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (100, 3))
w_true = jnp.array([2.0, -3.0, 0.5])
y = X @ w_true + 1.5              # noiseless

w, b = {fn}().closed_form(X, y)

assert w.shape == (3,), f'w shape {w.shape} vs (3,)'
assert jnp.ndim(b) == 0, f'b should be a scalar, got shape {jnp.shape(b)}'
assert jnp.allclose(w, w_true, atol=1e-3), f'w = {w}, expected {w_true}'
assert jnp.allclose(b, 1.5, atol=1e-3), f'b = {float(b)}, expected 1.5'
""",
        },
        {
            "name": "Scan gradient descent converges to the closed form",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(1), (100, 3))
w_true = jnp.array([1.0, -2.0, 3.0])
y = X @ w_true - 0.5

model = {fn}()
w_cf, b_cf = model.closed_form(X, y)

out = model.gradient_descent(X, y, lr=0.1, steps=3000)
assert isinstance(out, tuple) and len(out) == 3, (
    f'gradient_descent must return (w, b, losses), got {type(out)} of '
    f'length {len(out) if isinstance(out, tuple) else "n/a"}'
)
w_gd, b_gd, losses = out

assert w_gd.shape == (3,), f'w shape {w_gd.shape}'
assert jnp.ndim(b_gd) == 0, f'b should be a scalar, got shape {jnp.shape(b_gd)}'
assert jnp.allclose(w_gd, w_cf, atol=1e-2), (
    f'Gradient descent {w_gd} should converge to the closed form {w_cf}'
)
assert jnp.allclose(b_gd, b_cf, atol=1e-2), f'b: {float(b_gd)} vs {float(b_cf)}'
""",
        },
        {
            "name": "losses is the per-step curve, recorded before each update",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(2), (50, 2))
y = X @ jnp.array([4.0, -1.0]) + 2.0
model = {fn}()

w, b, losses = model.gradient_descent(X, y, lr=0.1, steps=200)

assert losses.shape == (200,), (
    f'losses should be one value per step: {losses.shape} vs (200,). '
    'Return the loss as the scan body\\'s second output and scan stacks it.'
)

# Recorded BEFORE the update, so the first entry is the loss at w=0, b=0.
loss_at_zero = float(jnp.mean(y ** 2))
assert abs(float(losses[0]) - loss_at_zero) < 1e-4, (
    f'losses[0] = {float(losses[0])}, expected the loss at the starting point '
    f'w=0, b=0 which is {loss_at_zero}'
)

assert bool(jnp.all(jnp.diff(losses) <= 1e-9)), 'Loss did not decrease monotonically'
assert float(losses[-1]) < float(losses[0]), 'Loss did not decrease at all'

# The returned parameters are AFTER the last update, so they beat losses[-1].
final = float(jnp.mean((X @ w + b - y) ** 2))
assert final <= float(losses[-1]) + 1e-9, (
    'The returned (w, b) should be the state after the final update, so its '
    f'loss {final} cannot exceed the last recorded one {float(losses[-1])}'
)

# steps=0 must leave the parameters untouched and give an empty curve.
w0, b0, l0 = model.gradient_descent(X, y, lr=0.1, steps=0)
assert jnp.allclose(w0, 0.0), f'w should start at zeros, got {w0}'
assert jnp.allclose(b0, 0.0), f'b should start at 0, got {float(b0)}'
assert l0.shape == (0,), f'steps=0 should give an empty loss curve, got {l0.shape}'
""",
        },
        {
            "name": "Actually a scan: the graph stays flat, and 20000 steps are cheap",
            "code": """
import time
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(3), (100, 3))
y = X @ jnp.array([1.0, -2.0, 3.0]) - 0.5
model = {fn}()

# The defining property of a scan: the body is compiled once and looped, so
# the graph SIZE does not depend on the trip count. A Python loop unrolls and
# grows with it. Checked at tiny step counts, so a wrong answer fails here in
# milliseconds instead of hanging the timing check below while it unrolls
# 20000 steps.
def _graph(steps):
    return str(jax.make_jaxpr(
        lambda X, y: model.gradient_descent(X, y, 0.1, steps))(X, y))

small, large = _graph(10), _graph(50)
n_small, n_large = len(small.splitlines()), len(large.splitlines())
assert n_small == n_large, (
    f'The jaxpr grew from {n_small} lines at 10 steps to {n_large} at 50, so '
    'the loop is being unrolled by Python at trace time. A lax.scan compiles '
    'its body once, so the graph is the same size at any step count.'
)
assert 'scan' in small, 'No scan in the jaxpr'

# Now the payoff that flat graph buys.
f = jax.jit(model.gradient_descent, static_argnames=('steps',))
t0 = time.perf_counter()
w, b, losses = f(X, y, lr=0.1, steps=20000)
jax.block_until_ready((w, b, losses))
elapsed = time.perf_counter() - t0

assert losses.shape == (20000,), f'{losses.shape} vs (20000,)'
assert jnp.isfinite(w).all() and jnp.isfinite(b), 'Non-finite parameters'
assert elapsed < 20.0, (
    f'Took {elapsed:.1f}s for 20000 steps; scan should do this in well under '
    'a second'
)
""",
        },
        {
            "name": "Composes with vmap and grad",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(4), (60, 3))
y = X @ jnp.array([1.0, -2.0, 3.0]) - 0.5
model = {fn}()

# One run per learning rate, batched.
lrs = jnp.array([0.01, 0.05, 0.1])
vw, vb, vl = jax.vmap(lambda r: model.gradient_descent(X, y, lr=r, steps=200))(lrs)
assert vw.shape == (3, 3), f'vmapped w {vw.shape} vs (3, 3)'
assert vl.shape == (3, 200), f'vmapped losses {vl.shape} vs (3, 200)'
assert float(vl[2, -1]) < float(vl[0, -1]), (
    'A larger learning rate should get further in the same number of steps'
)

# Differentiable through the whole loop — scan supports reverse mode
# (lax.while_loop would not).
g = jax.grad(lambda r: model.gradient_descent(X, y, lr=r, steps=50)[2][-1])(0.05)
assert jnp.isfinite(g), f'd(final loss)/d(lr) should be finite, got {g}'
assert float(g) < 0.0, (
    f'Raising the learning rate from 0.05 should still be lowering the loss '
    f'here, so the derivative should be negative, got {float(g)}'
)
""",
        },
        {
            "name": "nn_linear agrees, and D=1 does not collapse",
            "code": """
import jax
import jax.numpy as jnp

model = {fn}()

X = jax.random.normal(jax.random.key(5), (80, 2))
y = X @ jnp.array([1.5, -0.5]) + 0.25
w_cf, b_cf = model.closed_form(X, y)
w_nn, b_nn = model.nn_linear(X, y, lr=0.1, steps=3000)

assert w_nn.shape == (2,), (
    f'w shape {w_nn.shape} vs (2,) — the Flax kernel is (D, 1), so squeeze it'
)
assert jnp.allclose(w_nn, w_cf, atol=2e-2), f'nn_linear {w_nn} vs closed form {w_cf}'
assert jnp.allclose(b_nn, b_cf, atol=2e-2), f'b: {float(b_nn)} vs {float(b_cf)}'

# D = 1 must survive the scan carry without losing its axis.
X1 = jax.random.normal(jax.random.key(6), (30, 1))
y1 = X1[:, 0] * 3.0 + 1.0
w1, b1, l1 = model.gradient_descent(X1, y1, lr=0.1, steps=2000)
assert w1.shape == (1,), f'D=1 should still give w shape (1,), got {w1.shape}'
assert jnp.allclose(w1, 3.0, atol=1e-2), f'w = {w1}, expected [3.0]'
assert jnp.allclose(b1, 1.0, atol=1e-2), f'b = {float(b1)}, expected 1.0'
""",
        },
    ],
}
