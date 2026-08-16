"""Cross-entropy from logits — the log-sum-exp trick. Part 1 of notebook 16."""

TASK = {
    "title": "Cross-Entropy Loss",
    "category": "Training",
    "number": "16",
    "difficulty": "Easy",
    "function_name": "cross_entropy_loss",
    "hint": (
        "Never log(softmax(x)) — compute log-softmax in one step: subtract the "
        "row max, then subtract the log of the sum of the shifted exponentials. "
        "Pull out the target's log-probability by gathering that index with "
        "jnp.take_along_axis rather than materialising a one-hot matrix, then "
        "take the mean over the batch."
    ),
    "description": r"""
Implement **cross-entropy loss directly from logits** — the plain form, no extras.

$$\ell_i = -\log p_{i,t_i}, \qquad
p_{i,c} = \frac{e^{z_{i,c}}}{\sum_{k} e^{z_{i,k}}}$$

Return the **mean over the batch**.

### Rules
- Signature: `cross_entropy_loss(logits, targets)`
- `logits` is `(B, C)`, `targets` is `(B,)` of integer class ids; the output is a **scalar**
- Banned: `jax.nn.log_softmax`, `jax.nn.softmax`, `jax.scipy.special.logsumexp`, `optax`
- Compute $\log p$ in one fused expression; never form $p$ and then take its log
- It has to survive extreme logits, and work under `jit`

### Why you never softmax-then-log
The naive route dies twice.

**Overflow.** `exp(z)` is `inf` above $z \approx 88.7$ in float32 and above
$z \approx 11.1$ in float16. The fix is the shift
$\log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}$ with $m = \max_k z_k$: every
exponent is now $\le 0$, so the largest term is exactly `1.0` and the sum can
never overflow.

**Underflow — the one that actually bites.** Even with no overflow, a confidently
*wrong* prediction pushes $p_t$ under the float32 floor: normals stop at
$\approx 1.2\times10^{-38}$ and subnormals at $\approx 1.4\times10^{-45}$, and
XLA flushes subnormals to zero on accelerators anyway. Once $p_t$ rounds to
`0.0`, `log(0) = -inf` makes the loss `inf` and every gradient `nan`. The fused
form never materialises $p_t$: it computes $z_t - \log\sum_k e^{z_k}$, a
perfectly finite number like $-120$ (that is $p_t \approx 10^{-52}$, hopelessly
unrepresentable, yet its logarithm is an ordinary float). Your loss stays
large-but-finite and training recovers instead of poisoning every parameter
with `nan`.

There is a gradient bonus too. $\partial \ell / \partial z = p - q$ — a clean,
bounded expression that autodiff derives exactly from the fused form. Compose
`log` on top of a separate `softmax` and you hand XLA a division of two tiny
numbers to differentiate through.

The `max` shift cancels analytically ($\ell$ is invariant to it), so wrapping it
in `stop_gradient` changes nothing mathematically and keeps the backward graph
smaller.

### Gathering the target
$-\log p_{i,t_i}$ needs one entry per row. `jnp.take_along_axis(log_probs,
targets[:, None], axis=-1)` reads exactly those and nothing else. A one-hot
matmul gets the same answer by building a `(B, C)` array of zeros to multiply
against — correct, but it is `B×C` work and memory for `B` numbers, which at
vocabulary sizes is the difference between a loss that fits and one that does
not.

Once this passes, **`b_14` (Cross-Entropy: Smoothing & Padding Mask)** picks it
up and adds the two arguments real training code always passes.
""",
    "stub": '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets):
    """Mean cross-entropy over the batch.

    Args:
        logits:  (B, C) unnormalised scores
        targets: (B,) integer class ids

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets):
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets)

    # log-softmax, fused. Shifting by the row max makes every exponent <= 0, so
    # the sum is in [1, C] and can never overflow. The shift cancels exactly in
    # the final expression, hence stop_gradient.
    shift = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    z = logits - shift
    log_probs = z - jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))

    # Gather the target's log-probability — no one-hot matrix needed.
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)
''',
    "demo": '''import jax
import jax.numpy as jnp

# Uniform logits over 3 classes -> loss is exactly log(3).
print("uniform:", cross_entropy_loss(jnp.zeros((1, 3)), jnp.array([0])), "vs", jnp.log(3.0))

# Random data against the library log-softmax.
logits = jax.random.normal(jax.random.key(0), (4, 5)) * 3.0
targets = jnp.array([1, 2, 0, 4])
ref = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(logits, axis=-1), targets[:, None], axis=-1))
print("mine:", float(cross_entropy_loss(logits, targets)), " ref:", float(ref))

# The stability trap: huge logits.
big = jnp.array([[1000.0, 0.0, 0.0]])
print("\\nbig logits, correct class:", cross_entropy_loss(big, jnp.array([0])))
print("naive softmax-then-log would give:",
      -jnp.log(jnp.exp(big) / jnp.exp(big).sum(-1, keepdims=True))[0, 0])

# Confidently WRONG stays finite — the loss is ~1000, not inf.
print("big logits, wrong class:  ", float(cross_entropy_loss(big, jnp.array([1]))))

# The gradient is p - onehot, averaged over the batch.
g = jax.grad(cross_entropy_loss)(jnp.array([[2.0, 1.0, 0.0]]), jnp.array([0]))
print("\\ngrad:", g, " sums to", float(jnp.sum(g)))
''',
    "tests": [
        {
            "name": "Hand-computed values",
            "code": """
import jax
import jax.numpy as jnp

# Uniform logits over C classes -> log(C) regardless of the target.
for c in (2, 3, 10):
    out = {fn}(jnp.zeros((1, c)), jnp.array([0]))
    assert jnp.ndim(out) == 0, f'Loss must be a scalar, got shape {out.shape}'
    assert jnp.allclose(out, jnp.log(float(c)), atol=1e-5), (
        f'Uniform logits over {c} classes should give log({c})='
        f'{float(jnp.log(float(c))):.4f}, got {float(out):.4f}'
    )

# Two rows, worked out by hand.
logits = jnp.array([[0.0, jnp.log(3.0)], [jnp.log(3.0), 0.0]])
# Row 0: p = [1/4, 3/4], target 1 -> -log(3/4).  Row 1: target 1 -> -log(1/4).
expected = 0.5 * (-jnp.log(0.75) - jnp.log(0.25))
out = {fn}(logits, jnp.array([1, 1]))
assert jnp.allclose(out, expected, atol=1e-6), f'{float(out):.6f} vs {float(expected):.6f}'

# The mean is over the batch: duplicating a row must not change the loss.
one = {fn}(logits[:1], jnp.array([1]))
dup = {fn}(jnp.concatenate([logits[:1]] * 4), jnp.array([1, 1, 1, 1]))
assert jnp.allclose(one, dup, atol=1e-6), (
    f'Reduce with the MEAN over the batch, not the sum: {float(dup):.6f} vs {float(one):.6f}'
)
""",
        },
        {
            "name": "Matches the library log-softmax on random data",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(0), (16, 12)) * 4.0
targets = jax.random.randint(jax.random.key(1), (16,), 0, 12)

ref = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(logits, axis=-1), targets[:, None], axis=-1))
out = {fn}(logits, targets)
assert jnp.ndim(out) == 0, f'Loss must be a scalar, got shape {out.shape}'
assert jnp.allclose(out, ref, atol=1e-5), f'{float(out):.6f} vs reference {float(ref):.6f}'

# A second batch, wider and shallower, so a hardcoded axis or shape shows up.
lg = jax.random.normal(jax.random.key(2), (3, 40)) * 2.0
tg = jax.random.randint(jax.random.key(3), (3,), 0, 40)
ref2 = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(lg, axis=-1), tg[:, None], axis=-1))
out2 = {fn}(lg, tg)
assert jnp.allclose(out2, ref2, atol=1e-5), f'{float(out2):.6f} vs {float(ref2):.6f}'

# Softmax is invariant to adding a constant to a whole row, so the loss must
# be too. Reducing down the batch axis instead of the class axis breaks this.
bump = logits + jnp.arange(16.0)[:, None]
assert jnp.allclose({fn}(bump, targets), out, atol=1e-4), (
    'Adding a per-row constant must not change the loss — you are reducing '
    'over the wrong axis'
)
""",
        },
        {
            "name": "Numerical stability at extreme logits",
            "code": """
import jax.numpy as jnp

# exp(1000) = inf: softmax-then-log gives nan here.
big = jnp.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
out = {fn}(big, jnp.array([0, 1]))
assert jnp.isfinite(out), f'Non-finite loss on huge logits ({out}) — subtract the row max'
assert float(out) < 1e-3, f'Confident and correct should be ~0, got {float(out)}'

# Confidently WRONG: loss must be large but finite, never inf.
wrong = {fn}(big, jnp.array([1, 0]))
assert jnp.isfinite(wrong), (
    f'Confidently-wrong gave {wrong}. p_target underflows to 0.0, so log(p) is -inf. '
    'The fused form z_t - logsumexp(z) stays finite.'
)
assert 900.0 < float(wrong) < 1100.0, f'Expected ~1000, got {float(wrong)}'

# exp(-1000) = 0 everywhere: the naive sum is 0 and 0/0 = nan.
tiny = jnp.full((2, 4), -1000.0)
out2 = {fn}(tiny, jnp.array([0, 3]))
assert jnp.isfinite(out2), f'Non-finite on uniformly tiny logits: {out2}'
assert jnp.allclose(out2, jnp.log(4.0), atol=1e-4), (
    f'All-equal logits should give log(4), got {float(out2)}'
)
""",
        },
        {
            "name": "Gradients are p - onehot, and stay finite",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(7), (4, 5)) * 2.0
targets = jnp.array([2, 0, 1, 4])

g = jax.grad({fn})(logits, targets)
assert g.shape == logits.shape, f'Gradient shape {g.shape} vs {logits.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'

# d/dz of -log p sums to zero across classes: sum_c (p_c - q_c) = 1 - 1 = 0.
assert jnp.allclose(jnp.sum(g, axis=-1), 0.0, atol=1e-6), (
    f'Per-row gradients should sum to 0 across classes, got {jnp.sum(g, axis=-1)}'
)

# Explicit: grad = (p - onehot) / B.
p = jax.nn.softmax(logits, axis=-1)
expected = (p - jax.nn.one_hot(targets, 5)) / 4.0
assert jnp.allclose(g, expected, atol=1e-5), 'Gradient does not match (p - onehot) / B'

# Gradient stays finite where the naive softmax-then-log route would give nan.
gbig = jax.grad({fn})(jnp.array([[800.0, -800.0]]), jnp.array([1]))
assert jnp.isfinite(gbig).all(), f'nan/inf gradient at extreme logits: {gbig}'
""",
        },
        {
            "name": "jit",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(8), (6, 9)) * 3.0
targets = jax.random.randint(jax.random.key(9), (6,), 0, 9)

eager = {fn}(logits, targets)
jitted = jax.jit({fn})(logits, targets)
assert jnp.allclose(jitted, eager, atol=1e-6), (
    'jit changed the answer — no Python branching on array values'
)
""",
        },
    ],
}
