"""Mini-batch SGD where both loops are scans and the PRNG key rides the carry."""

TASK = {
    "title": "Mini-batch SGD with nested lax.scan",
    "category": "Training",
    "number": "b_19",
    "difficulty": "Hard",
    "function_name": "sgd_epochs",
    "hint": (
        "Two scans. The OUTER one runs epochs: its carry is "
        "((w, b), key) and it has no xs, so pass None with length=epochs. "
        "Inside it, split the key — `key, sub = jax.random.split(key)` — "
        "permute with `sub` and carry `key` onward; reusing the same key is "
        "the silent bug this problem is about. The INNER scan runs batches: "
        "reshape the shuffled X into (n_batches, batch_size, D) and pass it "
        "as xs, so scan iterates the leading axis for you. n_batches = "
        "N // batch_size — the remainder has to be dropped, because shapes "
        "must be static."
    ),
    "description": r"""
Train linear regression with **mini-batch SGD**, where the epoch loop and the
batch loop are both `lax.scan`, and the PRNG key is part of the carry.

### Signature
```python
def sgd_epochs(X, y, key, lr=0.1, batch_size=10, epochs=20):
    ...   # -> (w, b, losses, key)
```

`X` is `(N, D)`, `y` is `(N,)`. Start at `w = 0`, `b = 0`. Returns:

- `w` `(D,)`, `b` scalar — the trained parameters
- `losses` `(epochs,)` — the mean of that epoch's batch losses
- `key` — the key **after** training, so the caller can resume

Gradients are the same hand-derived MSE ones as problem 40 and `b_18`, but
computed on each mini-batch:

$$\nabla_w = \frac{2}{B}X_b^\top(\hat{y}_b-y_b), \qquad
  \nabla_b = \frac{2}{B}\sum(\hat{y}_b-y_b)$$

### The two scans

**Outer — epochs.** No per-step input, so `xs=None` and `length=epochs`.
Its carry is `((w, b), key)`.

**Inner — batches.** Here there *is* a per-step input: the batches themselves.
Reshape the shuffled data to `(n_batches, batch_size, D)` and hand it to scan
as `xs`; scan walks the leading axis, so the body sees one batch at a time.
Its carry is just `(w, b)`.

```python
(w, b), batch_losses = jax.lax.scan(batch_body, (w, b), (Xs, ys))
```

`n_batches = N // batch_size`, and the remainder is **dropped** — every shape
inside a scan has to be static, and a short final batch is not.

### Why returning the key is part of the exercise
Splitting inside the loop is the whole point:

```python
key, sub = jax.random.split(key)   # shuffle with sub, carry key onward
perm = jax.random.permutation(sub, N)
```

Reuse the incoming `key` every epoch instead and **the model still converges** —
you get the same permutation each time, which on a small problem costs you
almost nothing measurable. That is exactly what makes it dangerous: a silent
bug that only shows up as a mysteriously worse model months later. Returning
the final key makes it visible, and it is what a real training loop does
anyway, because you need somewhere to resume from.

### What is hard here
1. **The carry structure must match exactly** — same pytree in and out. Nest
   it as `((w, b), key)` and keep it nested; returning `(w, b, key)` from the
   body will not match the init.
2. **The key must advance.** Split, use the child, carry the parent.
3. **`batch_size` and `epochs` are static.** They set the shapes of `xs` and
   of `losses`, so under `jit` they are `static_argnames`.
4. **Permute `X` and `y` with the SAME indices.** Drawing two permutations, or
   shuffling only `X`, destroys the pairing and the fit never converges.

### Where you have met the pieces
`b_06` and `b_18` are the carry; `b_05` is key splitting; the difference here
is that they have to work at the same time, one nested inside the other.
""",
    "stub": '''import jax
import jax.numpy as jnp


def sgd_epochs(X, y, key, lr=0.1, batch_size=10, epochs=20):
    """Mini-batch SGD with both loops as lax.scan.

    Returns (w, b, losses, key):
      w       (D,)        trained weights, started from zeros
      b       scalar      trained bias, started from 0
      losses  (epochs,)   mean batch loss for each epoch
      key                 the key AFTER training, for resuming
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def sgd_epochs(X, y, key, lr=0.1, batch_size=10, epochs=20):
    N, D = X.shape
    n_batches = N // batch_size          # remainder dropped: shapes must be static
    used = n_batches * batch_size

    def epoch(carry, _):
        (w, b), key = carry
        # Split, shuffle with the child, carry the parent onward. Reusing `key`
        # here would give every epoch the same permutation — and still converge.
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, N)[:used]
        Xs = X[perm].reshape(n_batches, batch_size, D)
        ys = y[perm].reshape(n_batches, batch_size)

        def batch(params, data):
            w, b = params
            Xb, yb = data                        # scan walks the leading axis
            error = Xb @ w + b - yb              # (batch_size,)
            loss = jnp.mean(error ** 2)
            grad_w = (2.0 / batch_size) * (Xb.T @ error)
            grad_b = (2.0 / batch_size) * jnp.sum(error)
            return (w - lr * grad_w, b - lr * grad_b), loss

        (w, b), batch_losses = jax.lax.scan(batch, (w, b), (Xs, ys))
        return ((w, b), key), jnp.mean(batch_losses)

    init = ((jnp.zeros(D), jnp.array(0.0)), key)
    ((w, b), key), losses = jax.lax.scan(epoch, init, None, length=epochs)
    return w, b, losses, key
''',
    "demo": '''import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (100, 3))
w_true = jnp.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.3

key = jax.random.key(42)
w, b, losses, key2 = sgd_epochs(X, y, key, lr=0.1, batch_size=10, epochs=50)

print(f"w = {jnp.round(w, 4)}   truth {w_true}")
print(f"b = {float(b):.4f}      truth 0.3")
print(f"losses {losses.shape}: {float(losses[0]):.4f} -> {float(losses[-1]):.3e}")

# The key came back advanced, so training can pick up where it left off.
same = jnp.array_equal(jax.random.key_data(key), jax.random.key_data(key2))
print(f"\\nkey advanced? {not same}")
w2, b2, l2, _ = sgd_epochs(X, y, key2, lr=0.1, batch_size=10, epochs=50)
print(f"resumed from the returned key, first epoch loss {float(l2[0]):.3e}")
''',
    "tests": [
        {
            "name": "Converges to the least-squares solution",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (100, 3))
w_true = jnp.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.3

out = {fn}(X, y, jax.random.key(42), lr=0.1, batch_size=10, epochs=200)
assert isinstance(out, tuple) and len(out) == 4, (
    f'sgd_epochs must return (w, b, losses, key), got a {type(out).__name__} '
    f'of length {len(out) if isinstance(out, tuple) else "n/a"}'
)
w, b, losses, key = out

assert w.shape == (3,), f'w shape {w.shape} vs (3,)'
assert jnp.ndim(b) == 0, f'b should be a scalar, got shape {jnp.shape(b)}'
assert jnp.allclose(w, w_true, atol=1e-2), f'w = {w}, expected {w_true}'
assert jnp.allclose(b, 0.3, atol=1e-2), f'b = {float(b)}, expected 0.3'
""",
        },
        {
            "name": "One loss per epoch, and it goes down",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(1), (100, 2))
y = X @ jnp.array([1.5, -0.5]) + 1.0

w, b, losses, key = {fn}(X, y, jax.random.key(0), lr=0.1, batch_size=10, epochs=30)

assert losses.shape == (30,), (
    f'losses should be one value per EPOCH: {losses.shape} vs (30,). '
    'Average the inner scan\\'s per-batch losses.'
)
assert float(losses[-1]) < float(losses[0]), (
    f'Loss did not decrease: {float(losses[0])} -> {float(losses[-1])}'
)
assert jnp.isfinite(losses).all(), 'Non-finite loss'

# epochs=0 leaves the parameters at their starting point.
w0, b0, l0, _ = {fn}(X, y, jax.random.key(0), lr=0.1, batch_size=10, epochs=0)
assert jnp.allclose(w0, 0.0), f'w should start at zeros, got {w0}'
assert jnp.allclose(b0, 0.0), f'b should start at 0, got {float(b0)}'
assert l0.shape == (0,), f'epochs=0 should give an empty loss curve, got {l0.shape}'
""",
        },
        {
            "name": "The key is threaded through and advances",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(2), (100, 3))
y = X @ jnp.array([2.0, -1.0, 0.5]) + 0.3

k0 = jax.random.key(42)
w, b, losses, k_out = {fn}(X, y, k0, lr=0.1, batch_size=10, epochs=20)

# Returned unchanged means the same permutation was reused every epoch. That
# still converges, which is exactly why it has to be tested directly.
assert not jnp.array_equal(jax.random.key_data(k_out), jax.random.key_data(k0)), (
    'The returned key is identical to the one passed in, so the key was never '
    'split inside the epoch loop — every epoch shuffled the data the same way. '
    'Use `key, sub = jax.random.split(key)`, permute with sub, carry key on.'
)

# A different starting key must give a different trajectory, or the shuffle is
# not actually driving anything.
_, _, losses_b, _ = {fn}(X, y, jax.random.key(7), lr=0.1, batch_size=10, epochs=20)
assert not jnp.allclose(losses, losses_b), (
    'Two different keys produced identical loss curves — the permutation is '
    'not being used to build the batches'
)
""",
        },
        {
            "name": "Both loops really are scans",
            "code": """
import time
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(3), (100, 3))
y = X @ jnp.array([2.0, -1.0, 0.5]) + 0.3

# The defining property of a scan: the graph is compiled once and looped, so
# its SIZE does not depend on the trip count. An unrolled Python loop grows
# with it. Checked at tiny epoch counts, so a wrong answer fails here in
# milliseconds instead of unrolling 500 epochs in the timing check below.
def _graph(epochs):
    return str(jax.make_jaxpr(
        lambda X, y, k: {fn}(X, y, k, 0.1, 10, epochs))(X, y, jax.random.key(0)))

small, large = _graph(5), _graph(20)
n_small, n_large = len(small.splitlines()), len(large.splitlines())
assert n_small == n_large, (
    f'The jaxpr grew from {n_small} lines at 5 epochs to {n_large} at 20, so '
    'the epoch loop is being unrolled by Python at trace time. A lax.scan '
    'compiles its body once, so the graph is the same size at any epoch count.'
)
assert small.count('scan') >= 2, (
    f"Only {small.count('scan')} scan(s) in the jaxpr — the epoch loop and the "
    'batch loop should BOTH be lax.scan'
)

f = jax.jit({fn}, static_argnames=('batch_size', 'epochs'))
t0 = time.perf_counter()
out = f(X, y, jax.random.key(0), lr=0.1, batch_size=10, epochs=500)
jax.block_until_ready(out)
elapsed = time.perf_counter() - t0
assert elapsed < 20.0, f'500 epochs took {elapsed:.1f}s; scan should be well under a second'
""",
        },
        {
            "name": "Drops the remainder when N is not divisible by batch_size",
            "code": """
import jax
import jax.numpy as jnp

w_true = jnp.array([2.0, -1.0, 0.5])

# 95 rows with batch_size 10 -> 9 batches of 10, last 5 rows dropped each epoch.
X = jax.random.normal(jax.random.key(4), (95, 3))
y = X @ w_true + 0.3
w, b, losses, _ = {fn}(X, y, jax.random.key(0), lr=0.1, batch_size=10, epochs=200)
assert losses.shape == (200,), f'{losses.shape}'
assert jnp.isfinite(losses).all(), 'Non-finite loss — a ragged final batch?'
assert jnp.allclose(w, w_true, atol=2e-2), f'w = {w}, expected {w_true}'

# batch_size == N degenerates to full-batch gradient descent.
X2 = jax.random.normal(jax.random.key(5), (60, 3))
y2 = X2 @ w_true + 0.3
w2, b2, _, _ = {fn}(X2, y2, jax.random.key(0), lr=0.1, batch_size=60, epochs=400)
assert jnp.allclose(w2, w_true, atol=1e-2), f'full batch: {w2} vs {w_true}'
""",
        },
        {
            "name": "X and y stay paired through the shuffle",
            "code": """
import jax
import jax.numpy as jnp

# A fit this clean is only reachable if every row's features stayed with its
# own target. Permuting X and y separately, or shuffling only one of them,
# leaves a model no better than predicting the mean.
X = jax.random.normal(jax.random.key(6), (80, 2))
w_true = jnp.array([3.0, -2.0])
y = X @ w_true - 1.0

w, b, losses, _ = {fn}(X, y, jax.random.key(1), lr=0.1, batch_size=8, epochs=300)

assert jnp.allclose(w, w_true, atol=1e-2), (
    f'w = {w}, expected {w_true}. If the shuffle used different indices for X '
    'and y, the pairing is destroyed and this cannot converge.'
)
assert jnp.allclose(b, -1.0, atol=1e-2), f'b = {float(b)}, expected -1.0'

resid_mse = float(jnp.mean((X @ w + b - y) ** 2))
var_y = float(jnp.var(y))
assert resid_mse < 1e-3 * var_y, (
    f'Residual MSE {resid_mse:.4g} vs target variance {var_y:.4g} — the fit is '
    'far worse than it should be on noiseless data'
)

# D = 1 must not collapse the weight axis inside the carry.
X1 = jax.random.normal(jax.random.key(7), (40, 1))
y1 = X1[:, 0] * 4.0 + 0.5
w1, b1, _, _ = {fn}(X1, y1, jax.random.key(2), lr=0.1, batch_size=8, epochs=300)
assert w1.shape == (1,), f'D=1 should give w shape (1,), got {w1.shape}'
assert jnp.allclose(w1, 4.0, atol=2e-2), f'w = {w1}, expected [4.0]'
""",
        },
    ],
}
