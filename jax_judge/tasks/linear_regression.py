"""Linear regression three ways — closed form, hand-rolled GD, and nnx.Linear."""

TASK = {
    "title": "Linear Regression",
    "category": "Training",
    "order": 2,
    "number": "40",
    "difficulty": "Medium",
    "function_name": "LinearRegression",
    "hint": (
        "closed_form: append a ones column to X so the bias comes out of the "
        "same solve, then use jnp.linalg.lstsq and split the last entry back "
        "off. gradient_descent: the MSE gradient is (2/N) X^T(Xw + b - y) and "
        "(2/N) sum(error) — write it by hand, no autodiff. nn_linear: an "
        "nnx.Linear(D, 1) plus a manual SGD loop; remember its kernel is "
        "(D, 1), so squeeze to get a (D,) weight vector back."
    ),
    "description": r"""
Solve linear regression three ways and get the same answer each time.

$$\hat{y} = Xw + b, \qquad \mathcal{L} = \frac{1}{N}\|Xw + b - y\|^2$$

### Signature
```python
class LinearRegression:
    def closed_form(self, X, y): ...                        # -> (w, b)
    def gradient_descent(self, X, y, lr=0.01, steps=1000): ...
    def nn_linear(self, X, y, lr=0.01, steps=1000): ...
```

`X` is `(N, D)`, `y` is `(N,)`, and every method returns `w` of shape `(D,)`
and a scalar `b`.

### Method 1 — closed form
Append a column of ones to `X` so the bias is just one more coefficient, then
solve the least-squares system with `jnp.linalg.lstsq`. Split the last entry
back off as `b`.

### Method 2 — gradient descent, by hand
Start at `w = 0`, `b = 0` and derive the gradients yourself:

$$\nabla_w = \frac{2}{N}X^\top(\hat{y}-y), \qquad
  \nabla_b = \frac{2}{N}\sum(\hat{y}-y)$$

No `jax.grad` here — the point is to show you can differentiate MSE on paper.

### Method 3 — with a layer
Build an `nnx.Linear(D, 1)`, take gradients with `nnx.grad`, and run a manual
SGD loop. Note the kernel is `(D, 1)` in Flax — the transpose of PyTorch's
`(1, D)` — so squeeze rather than transpose to recover `w`.

### Why anyone uses gradient descent for a problem with a closed form
The normal equations need $X^\top X$, which is $O(ND^2)$ to form and $O(D^3)$
to invert — hopeless once $D$ is large. Worse, $X^\top X$ **squares the
condition number**, so with correlated features the closed form loses roughly
twice the digits that an iterative method would.

That is also why you should reach for `lstsq` (which goes through QR or SVD)
rather than literally computing `inv(X.T @ X) @ X.T @ y`. Same answer in exact
arithmetic, very different numerics — and "what's wrong with inverting X^T X?"
is a standard interview follow-up.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class LinearRegression:
    """Least squares three ways."""

    def closed_form(self, X, y):
        """Normal equations via an augmented matrix. -> (w, b)"""
        pass  # Replace this

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        """Manual gradients — no autodiff. -> (w, b)"""
        pass  # Replace this

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        """nnx.Linear trained with nnx.grad. -> (w, b)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class LinearRegression:
    def closed_form(self, X, y):
        N, D = X.shape
        # Augment with a ones column so the bias falls out of the same solve.
        X_aug = jnp.concatenate([X, jnp.ones((N, 1))], axis=1)   # (N, D+1)
        # lstsq goes through QR/SVD — better conditioned than inv(X.T @ X).
        theta = jnp.linalg.lstsq(X_aug, y)[0]                    # (D+1,)
        return theta[:D], theta[D]

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        N, D = X.shape
        w = jnp.zeros(D)
        b = jnp.array(0.0)

        for _ in range(steps):
            error = X @ w + b - y                # (N,)
            grad_w = (2.0 / N) * (X.T @ error)   # (D,)
            grad_b = (2.0 / N) * jnp.sum(error)  # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b

        return w, b

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        D = X.shape[1]
        layer = nnx.Linear(D, 1, rngs=nnx.Rngs(params=0))

        def loss_fn(m):
            return jnp.mean((m(X).squeeze(-1) - y) ** 2)

        for _ in range(steps):
            grads = nnx.grad(loss_fn)(layer)
            state = nnx.state(grads)
            layer.kernel[...] = layer.kernel[...] - lr * state["kernel"][...]
            layer.bias[...] = layer.bias[...] - lr * state["bias"][...]

        # Flax kernel is (D, 1) — the transpose of torch's (1, D).
        return layer.kernel[...].squeeze(-1), layer.bias[...].squeeze()
''',
    "demo": '''import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (200, 3))
w_true = jnp.array([2.0, -3.0, 0.5])
y = X @ w_true + 1.5

lr = LinearRegression()
for name in ("closed_form", "gradient_descent", "nn_linear"):
    fn = getattr(lr, name)
    w, b = fn(X, y) if name == "closed_form" else fn(X, y, lr=0.1, steps=2000)
    print(f"{name:18s} w={jnp.round(w, 3)}  b={float(b):.3f}")
print(f"{'truth':18s} w={w_true}  b=1.5")
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
            "name": "Gradient descent converges to the same answer",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(1), (100, 3))
w_true = jnp.array([1.0, -2.0, 3.0])
y = X @ w_true - 0.5

model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.1, steps=3000)

assert w_gd.shape == (3,), f'w shape {w_gd.shape}'
assert jnp.allclose(w_gd, w_cf, atol=1e-2), (
    f'Gradient descent {w_gd} should converge to the closed form {w_cf}'
)
assert jnp.allclose(b_gd, b_cf, atol=1e-2), f'b: {float(b_gd)} vs {float(b_cf)}'
""",
        },
        {
            "name": "Gradient descent starts at zero and reduces the loss",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(2), (50, 2))
y = X @ jnp.array([4.0, -1.0]) + 2.0
model = {fn}()

# Zero steps must leave the parameters at their initial value.
w0, b0 = model.gradient_descent(X, y, lr=0.1, steps=0)
assert jnp.allclose(w0, 0.0), f'w should start at zeros, got {w0}'
assert jnp.allclose(b0, 0.0), f'b should start at 0, got {float(b0)}'

mse = lambda w, b: float(jnp.mean((X @ w + b - y) ** 2))
w1, b1 = model.gradient_descent(X, y, lr=0.1, steps=10)
w2, b2 = model.gradient_descent(X, y, lr=0.1, steps=500)
assert mse(w2, b2) < mse(w1, b1) < mse(w0, b0), 'Loss did not decrease monotonically'
""",
        },
        {
            "name": "nn_linear agrees with the other two",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(3), (80, 2))
y = X @ jnp.array([1.5, -0.5]) + 0.25

model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_nn, b_nn = model.nn_linear(X, y, lr=0.1, steps=3000)

assert w_nn.shape == (2,), (
    f'w shape {w_nn.shape} vs (2,) — the Flax kernel is (D, 1), so squeeze it'
)
assert jnp.ndim(b_nn) == 0, f'b should be a scalar, got shape {jnp.shape(b_nn)}'
assert jnp.allclose(w_nn, w_cf, atol=2e-2), f'nn_linear {w_nn} vs closed form {w_cf}'
assert jnp.allclose(b_nn, b_cf, atol=2e-2), f'b: {float(b_nn)} vs {float(b_cf)}'
""",
        },
        {
            "name": "Handles a bias-only fit and D=1",
            "code": """
import jax
import jax.numpy as jnp

model = {fn}()

# Constant target: w should vanish, b should be the mean.
X = jax.random.normal(jax.random.key(4), (40, 3))
y = jnp.full((40,), 7.0)
w, b = model.closed_form(X, y)
assert jnp.allclose(w, 0.0, atol=1e-3), f'w should be ~0 for a constant target, got {w}'
assert jnp.allclose(b, 7.0, atol=1e-3), f'b should be 7.0, got {float(b)}'

# D = 1 must not collapse a dimension.
X1 = jax.random.normal(jax.random.key(5), (30, 1))
y1 = X1[:, 0] * 3.0 + 1.0
w1, b1 = model.closed_form(X1, y1)
assert w1.shape == (1,), f'D=1 should still give w shape (1,), got {w1.shape}'
assert jnp.allclose(w1, 3.0, atol=1e-3) and jnp.allclose(b1, 1.0, atol=1e-3)
""",
        },
        {
            "name": "Least-squares fit on noisy data",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.key(6)
kx, kn = jax.random.split(key)
X = jax.random.normal(kx, (400, 3))
w_true = jnp.array([2.0, -1.0, 0.5])
y = X @ w_true + 1.0 + jax.random.normal(kn, (400,)) * 0.1

w, b = {fn}().closed_form(X, y)
assert jnp.allclose(w, w_true, atol=0.05), f'{w} vs {w_true}'

# The residual must be orthogonal to the design matrix — the defining property
# of a least-squares solution.
resid = X @ w + b - y
assert jnp.allclose(X.T @ resid, 0.0, atol=1e-2), (
    'Residuals are not orthogonal to the columns of X, so this is not the '
    'least-squares solution'
)
assert abs(float(jnp.sum(resid))) < 1e-2, 'Residuals should sum to ~0 when a bias is fitted'
""",
        },
    ],
}
