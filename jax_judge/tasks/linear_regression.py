"""Least squares two ways — QR/lstsq closed form with ridge, and gradient descent."""

TASK = {
    "title": "Linear Regression",
    "category": "Training",
    "order": 2,
    "difficulty": "Medium",
    "function_name": "linear_regression",
    "hint": (
        "Absorb the bias as a constant feature so both methods solve for a single "
        "vector theta = [w; b], and split it apart again at the end. "
        "For the closed form, jnp.linalg.lstsq factorises whatever matrix you hand "
        "it, so the trick is to hand it a matrix whose least-squares residual "
        "ALREADY contains the ridge penalty — never to build the normal equations "
        "and add lambda to their diagonal. Ask what extra ROWS you could stack "
        "under the design matrix (and what to stack under y) so that their squared "
        "residual is exactly l2 * ||w||^2; whatever those rows put in their last "
        "column is what decides whether the bias gets penalised. "
        "For gd, differentiate J by hand — it is one matvec and its transpose — and "
        "give the ridge term the same bias-excluding treatment. Drive the iteration "
        "with jax.lax.fori_loop starting from theta = 0; a Python for loop unrolls "
        "into `steps` copies of the graph, and there is a compile-time test for it."
    ),
    "description": r"""
Fit a linear model $\hat y = Xw + b$ two ways and return `(w, b)`.

Both methods minimise the same ridge objective (note that the bias is **not**
penalised):

$$J(w, b) = \frac{1}{N}\Bigl( \lVert Xw + b\mathbf{1} - y \rVert_2^2 + \lambda \lVert w \rVert_2^2 \Bigr)$$

Writing $\tilde X = [\,X \mid \mathbf{1}\,]$ and $\theta = [w; b]$, the stationary
point is the **normal equations**

$$(\tilde X^\top \tilde X + \lambda M)\,\theta = \tilde X^\top y, \qquad M = \mathrm{diag}(1,\dots,1,0)$$

### Rules
- Signature: `linear_regression(X, y, *, l2=0.0, method="closed_form", lr=0.1, steps=200)`
- `X` is `(N, D)`, `y` is `(N,)`; return `(w, b)` with `w` of shape `(D,)` and `b` a **scalar**
- Banned: `jnp.linalg.inv`, `jnp.linalg.pinv`, `optax`, `flax` optimizers
- `method="closed_form"`: solve with `jnp.linalg.lstsq` (or an explicit `jnp.linalg.qr`).
  **Do not form $\tilde X^\top \tilde X$** — see below. Handle a rank-deficient `X` at
  $\lambda = 0$ without producing `nan`
- `method="gd"`: full-batch gradient descent on $J$, starting from $\theta = 0$, for
  exactly `steps` iterations with step size `lr`. Use `jax.lax.fori_loop` — a Python
  loop unrolls into `steps` copies of the graph and blows up compile time
- Both paths must run under `jit`, and `vmap` over a stack of datasets

### Why lstsq beats inverting $\tilde X^\top \tilde X$
Forming the Gram matrix **squares the condition number**:
$\kappa(\tilde X^\top \tilde X) = \kappa(\tilde X)^2$. The relative error of a
solve is roughly $\kappa \cdot \epsilon_{\text{machine}}$, and float32 has
$\epsilon \approx 1.2 \times 10^{-7}$ — about 7 decimal digits.

Take a degree-6 polynomial design on $t \in [0, 1]$, which is nothing exotic:
$\kappa(\tilde X) \approx 2 \times 10^4$, so $\kappa(\tilde X^\top \tilde X)
\approx 4 \times 10^8$. Every digit is gone. Run it and the normal equations
return coefficients wrong in the *first* digit — an absolute error of order 1
(about 8 on the machine this was written on; once $\kappa \epsilon > 1$ the
answer is noise, so the exact figure is not reproducible), against
$\sim 4\times 10^{-5}$ from `lstsq` on identical inputs. QR factorises
$\tilde X$ directly and never squares anything, so its error tracks
$\kappa(\tilde X)$ — you keep half the digits the normal equations throw away.

This is also the real reason ridge "stabilises" the fit. Adding $\lambda$ shifts
every squared singular value up, so the effective condition number becomes
$(\sigma_{\max}^2 + \lambda)/(\sigma_{\min}^2 + \lambda)$. At $\lambda = 0$ with
duplicated (collinear) columns the Gram matrix is exactly singular and a solve
returns `nan`; `lstsq` instead returns the **minimum-norm** solution, which fits
just as well and splits the weight evenly between the collinear features.

### So why would you ever iterate?
The closed form costs $O(ND^2 + D^3)$ and needs all $N$ rows resident. Gradient
descent costs $O(ND)$ per step, streams minibatches, and never materialises
anything of size $D \times D$ — at $D = 10^6$ (an embedding table) the Gram
matrix alone would be $10^{12}$ entries. And the moment the model stops being
linear in its parameters, there is no closed form at all. Linear regression is
the one place you can check your optimizer against ground truth, which is
exactly why interviewers ask for both and then ask which one they should ship.
""",
    "stub": '''import jax
import jax.numpy as jnp


def linear_regression(X, y, *, l2=0.0, method="closed_form", lr=0.1, steps=200):
    """Fit y ~ X @ w + b.

    Args:
        X:      (N, D) design matrix
        y:      (N,) targets
        l2:     ridge strength; penalises w only, never the bias
        method: "closed_form" (lstsq/QR) or "gd" (full-batch gradient descent)
        lr:     step size, used only by "gd"
        steps:  number of iterations, used only by "gd"

    Returns:
        (w, b) with w of shape (D,) and b a scalar array.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def linear_regression(X, y, *, l2=0.0, method="closed_form", lr=0.1, steps=200):
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n, d = X.shape
    # Absorb the bias as a constant feature: theta = [w; b].
    Xb = jnp.concatenate([X, jnp.ones((n, 1), X.dtype)], axis=1)

    if method == "closed_form":
        # Ridge by data augmentation: the extra rows add sqrt(l2)*w to the
        # residual, i.e. l2*||w||^2 to the objective. The trailing zero column
        # is what keeps the bias out of the penalty.
        reg = jnp.sqrt(jnp.asarray(l2, Xb.dtype)) * jnp.concatenate(
            [jnp.eye(d, dtype=Xb.dtype), jnp.zeros((d, 1), Xb.dtype)], axis=1
        )
        Xa = jnp.concatenate([Xb, reg], axis=0)
        ya = jnp.concatenate([y, jnp.zeros((d,), y.dtype)])
        # lstsq factorises Xa itself — it never forms Xa.T @ Xa.
        theta = jnp.linalg.lstsq(Xa, ya)[0]

    elif method == "gd":
        mask = jnp.concatenate(
            [jnp.ones((d,), Xb.dtype), jnp.zeros((1,), Xb.dtype)]
        )

        def body(_, theta):
            resid = Xb @ theta - y
            grad = 2.0 / n * (Xb.T @ resid + l2 * mask * theta)
            return theta - lr * grad

        # fori_loop compiles ONE body and loops it; a Python for would unroll.
        theta = jax.lax.fori_loop(0, steps, body, jnp.zeros((d + 1,), Xb.dtype))

    else:
        raise ValueError(f"unknown method {method!r}")

    return theta[:d], theta[d]
''',
    "demo": '''import jax.numpy as jnp

# The conditioning blow-up, on a degree-6 polynomial design.
t = jnp.linspace(0.0, 1.0, 60)
X = jnp.stack([t ** k for k in range(1, 7)], axis=1)
true_w = jnp.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0])
y = X @ true_w + 0.3

Xb = jnp.concatenate([X, jnp.ones((60, 1))], axis=1)
print("cond(X~)     =", float(jnp.linalg.cond(Xb)))
# ~2.6e8 rather than the exact 4.0e8: estimating the condition number of the
# Gram matrix in float32 is itself past the point where float32 can answer.
print("cond(X~^T X~)=", float(jnp.linalg.cond(Xb.T @ Xb)))

w_ls, b_ls = linear_regression(X, y)
theta_ne = jnp.linalg.solve(Xb.T @ Xb, Xb.T @ y)      # the tempting one-liner
print("lstsq  max coef error:", float(jnp.max(jnp.abs(w_ls - true_w))))
print("normal max coef error:", float(jnp.max(jnp.abs(theta_ne[:6] - true_w))))

# Gradient descent lands in the same place on a well-conditioned problem.
import jax
Xg = jax.random.normal(jax.random.key(0), (200, 3))
yg = Xg @ jnp.array([2.0, -1.0, 0.5]) + 3.0
print("closed form:", linear_regression(Xg, yg))
print("gd  2000 it:", linear_regression(Xg, yg, method="gd", lr=0.1, steps=2000))
''',
    "tests": [
        {
            "name": "Shapes and exact recovery",
            "code": """
import jax
import jax.numpy as jnp

true_w = jnp.array([2.0, -1.0, 0.5])
X = jax.random.normal(jax.random.key(42), (100, 3))
y = X @ true_w + 3.0

w, b = {fn}(X, y)
assert w.shape == (3,), f'w shape {w.shape}, expected (3,)'
assert jnp.ndim(b) == 0, f'b must be a scalar, got shape {jnp.shape(b)}'
assert jnp.allclose(w, true_w, atol=1e-4), f'w = {w} vs true {true_w}'
assert jnp.allclose(b, 3.0, atol=1e-4), f'b = {float(b):.5f} vs true 3.0'

# Hand-computed: the points (0,1), (1,3), (2,5) lie exactly on y = 2x + 1.
w1, b1 = {fn}(jnp.array([[0.0], [1.0], [2.0]]), jnp.array([1.0, 3.0, 5.0]))
assert jnp.allclose(w1, 2.0, atol=1e-5), f'slope {w1} vs 2.0'
assert jnp.allclose(b1, 1.0, atol=1e-5), f'intercept {float(b1)} vs 1.0'
""",
        },
        {
            "name": "Ridge shrinks w but never the bias",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (80, 4))
true_w = jnp.array([1.5, -2.0, 0.75, 3.0])
y = X @ true_w + 5.0

# Matches the analytic ridge solution on a well-conditioned problem.
l2 = 2.0
w, b = {fn}(X, y, l2=l2)
Xb = jnp.concatenate([X, jnp.ones((80, 1))], axis=1)
M = jnp.eye(5).at[4, 4].set(0.0)
theta = jnp.linalg.solve(Xb.T @ Xb + l2 * M, Xb.T @ y)
assert jnp.allclose(w, theta[:4], atol=1e-3), f'ridge w {w} vs analytic {theta[:4]}'
assert jnp.allclose(b, theta[4], atol=1e-3), f'ridge b {float(b)} vs {float(theta[4])}'

# Monotone shrinkage of ||w|| as l2 grows.
norms = [float(jnp.linalg.norm({fn}(X, y, l2=v)[0])) for v in (0.0, 1.0, 10.0, 100.0)]
for i in range(3):
    assert norms[i] > norms[i + 1], f'||w|| should shrink with l2, got {norms}'

# l2 -> infinity: w collapses to 0 but b must go to mean(y), NOT to 0.
w_big, b_big = {fn}(X, y, l2=1e7)
assert jnp.allclose(w_big, 0.0, atol=1e-3), f'huge l2 should drive w to 0, got {w_big}'
assert jnp.allclose(b_big, jnp.mean(y), atol=1e-2), (
    f'huge l2 gave b={float(b_big):.4f}, expected mean(y)={float(jnp.mean(y)):.4f} — '
    'the bias must be excluded from the penalty (the 0 in diag(1,...,1,0))'
)
""",
        },
        {
            "name": "Ill-conditioned design: QR survives, normal equations do not",
            "code": """
import jax.numpy as jnp

# Degree-6 polynomial features: cond(X~) ~ 2e4, so cond(X~^T X~) ~ 4e8,
# which is past float32's ~1e7 of usable precision.
t = jnp.linspace(0.0, 1.0, 60)
X = jnp.stack([t ** k for k in range(1, 7)], axis=1)
true_w = jnp.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0])
y = X @ true_w + 0.3

w, b = {fn}(X, y)
err = float(jnp.max(jnp.abs(w - true_w)))
assert jnp.isfinite(w).all() and jnp.isfinite(b), f'Non-finite fit: {w}, {b}'
assert err < 1e-2, (
    f'Max coefficient error {err:.4f}. Solving the normal equations gives an error '
    'of order 1 here (~8) because forming X~^T X~ squares the condition number. '
    'Factorise X~ itself with jnp.linalg.lstsq / jnp.linalg.qr instead.'
)
assert abs(float(b) - 0.3) < 1e-2, f'intercept {float(b):.5f} vs 0.3'
""",
        },
        {
            "name": "Rank-deficient design does not blow up",
            "code": """
import jax.numpy as jnp

# Column 1 is an exact copy of column 0, so X~^T X~ is singular and a solve
# returns nan. lstsq returns the minimum-norm solution instead.
t = jnp.linspace(-1.0, 1.0, 40)
X = jnp.stack([t, t, t ** 2], axis=1)
y = 2.0 * t + 3.0 * t ** 2 + 0.5

w, b = {fn}(X, y, l2=0.0)
assert jnp.isfinite(w).all() and jnp.isfinite(b), (
    f'Got {w}, {b} on a rank-deficient design — jnp.linalg.solve on X~^T X~ '
    'produces nan here; lstsq handles it'
)
pred_err = float(jnp.max(jnp.abs(X @ w + b - y)))
assert pred_err < 1e-3, f'Predictions are off by {pred_err:.5f} despite a consistent system'
# Minimum-norm: the two collinear columns share the weight instead of one
# taking a huge value and the other cancelling it.
assert float(jnp.linalg.norm(w)) < 4.0, (
    f'||w|| = {float(jnp.linalg.norm(w)):.3f}; expected the min-norm solution '
    '(weight split evenly across the duplicated column)'
)

# A small ridge makes the system unique and must not change the predictions much.
w2, b2 = {fn}(X, y, l2=1e-4)
assert jnp.isfinite(w2).all(), f'nan with l2=1e-4: {w2}'
assert float(jnp.max(jnp.abs(X @ w2 + b2 - y))) < 1e-2, 'Ridge fit lost the signal'
""",
        },
        {
            "name": "Gradient descent reaches the closed-form optimum",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(7), (120, 3))
y = X @ jnp.array([2.0, -1.0, 0.5]) + 3.0 + 0.05 * jax.random.normal(
    jax.random.key(8), (120,))

w_cf, b_cf = {fn}(X, y)
w_gd, b_gd = {fn}(X, y, method="gd", lr=0.1, steps=3000)
assert w_gd.shape == (3,), f'gd w shape {w_gd.shape}'
assert jnp.allclose(w_gd, w_cf, atol=1e-3), f'GD w {w_gd} vs closed form {w_cf}'
assert abs(float(b_gd) - float(b_cf)) < 1e-3, f'GD b {float(b_gd)} vs {float(b_cf)}'

# Starting from theta = 0 means 0 steps returns exactly zeros.
w0, b0 = {fn}(X, y, method="gd", lr=0.1, steps=0)
assert jnp.allclose(w0, 0.0) and jnp.allclose(b0, 0.0), (
    f'steps=0 must return the initial theta=0, got {w0}, {b0}'
)

# The objective must decrease monotonically at this step size.
def obj(w, b, lam=0.0):
    return float(jnp.mean((X @ w + b - y) ** 2) + lam * jnp.sum(w ** 2) / X.shape[0])

losses = [obj(*{fn}(X, y, method="gd", lr=0.1, steps=s)) for s in (1, 5, 20, 100)]
for i in range(3):
    assert losses[i] > losses[i + 1], f'Loss should fall every step, got {losses}'

# GD on the ridge objective must agree with the ridge closed form.
w_r, b_r = {fn}(X, y, l2=5.0, method="gd", lr=0.1, steps=6000)
w_c, b_c = {fn}(X, y, l2=5.0)
assert jnp.allclose(w_r, w_c, atol=2e-3), (
    f'GD ridge {w_r} vs closed-form ridge {w_c} — both must minimise the SAME J, '
    'with the l2 term scaled by 1/N and the bias left unpenalised'
)
""",
        },
        {
            "name": "Fits under jit and vmap over a batch of datasets",
            "code": """
import functools
import time
import jax
import jax.numpy as jnp

Xs = jax.random.normal(jax.random.key(1), (5, 60, 3))
ws = jax.random.normal(jax.random.key(2), (5, 3))
ys = jnp.einsum('knd,kd->kn', Xs, ws) + 1.0

cf = functools.partial({fn}, method="closed_form")
w_b, b_b = jax.vmap(cf)(Xs, ys)
assert w_b.shape == (5, 3), f'vmapped w shape {w_b.shape} vs (5, 3)'
assert b_b.shape == (5,), f'vmapped b shape {b_b.shape} vs (5,)'
assert jnp.allclose(w_b, ws, atol=1e-3), 'vmapped closed form did not recover the weights'
for i in range(5):
    wi, bi = cf(Xs[i], ys[i])
    assert jnp.allclose(w_b[i], wi, atol=1e-4), f'vmap row {i} differs from the single fit'

jf = jax.jit(cf)
assert jnp.allclose(jf(Xs[0], ys[0])[0], cf(Xs[0], ys[0])[0], atol=1e-5), 'jit changed the fit'

# fori_loop keeps compile time flat: 50k GD steps must not unroll the graph.
gd = jax.jit(functools.partial({fn}, method="gd", lr=0.05, steps=50000))
t0 = time.perf_counter()
w_g, b_g = gd(Xs[0], ys[0])
jax.block_until_ready((w_g, b_g))
elapsed = time.perf_counter() - t0
assert jnp.allclose(w_g, ws[0], atol=1e-3), f'50k GD steps gave {w_g} vs {ws[0]}'
assert elapsed < 30.0, (
    f'50000 steps took {elapsed:.1f}s — a Python loop unrolls into 50000 copies of '
    'the graph; use jax.lax.fori_loop'
)
""",
        },
    ],
}
