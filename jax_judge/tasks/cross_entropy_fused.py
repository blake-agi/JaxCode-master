"""Cross-entropy with logsumexp banned — write the max-shift yourself.

Problem 16 is the same loss with jax.scipy.special.logsumexp allowed. This is
where the shift, the underflow and the stop_gradient get their explanation.
"""

TASK = {
    "title": "Cross-Entropy Without logsumexp",
    "category": "Training",
    "number": "b_14",
    "difficulty": "Medium",
    "function_name": "cross_entropy_loss",
    "hint": (
        "logsumexp(z) = m + log(sum(exp(z - m))) with m = max(z), keepdims=True "
        "so it broadcasts back over the class axis. Every exponent is then <= 0, "
        "so the sum sits in [1, C] and cannot overflow. Subtract the gathered "
        "target logit from that and take the mean — still no exp() in the final "
        "expression. m is a constant with respect to the loss (it cancels), so "
        "jax.lax.stop_gradient around it keeps the backward graph smaller."
    ),
    "description": r"""
The same loss as **problem 16**, with the library function taken away:

$$\ell_i = \operatorname{logsumexp}(z_i) - z_{i,t_i}, \qquad
\operatorname{logsumexp}(z) = \log \sum_k e^{z_k}$$

Write that `logsumexp` yourself. Return the **mean over the batch**.

### Rules
- Signature: `cross_entropy_loss(logits, targets)` — identical to problem 16
- `logits` is `(B, C)`, `targets` is `(B,)` of integer class ids; the output is a **scalar**
- Banned: **`jax.scipy.special.logsumexp`** and `jax.nn.logsumexp`, plus
  `jax.nn.log_softmax`, `jax.nn.softmax`, `optax`
- Only `jnp.max`, `jnp.exp`, `jnp.log`, `jnp.sum` and friends
- Must stay finite at extreme logits — including in `float16` — and work under `jit`

### The shift, and why the naive route dies twice

$$\log \sum_k e^{z_k} \;=\; m + \log \sum_k e^{z_k - m}, \qquad m = \max_k z_k$$

The identity is exact: pull $e^m$ out of the sum and the $\log$ turns it into a
`+m`. What it buys is that every exponent is now $\le 0$, so the largest term is
exactly `1.0` and the sum lands in $[1, C]$ — it cannot overflow, whatever the
inputs were.

**Overflow.** Without the shift, `exp(z)` is `inf` above $z \approx 88.7$ in
float32, and above $z \approx 11.1$ in **float16** — which is why this bites in
mixed-precision training long before anyone's logits look extreme.

**Underflow — the one that actually bites.** Even with no overflow, a
confidently *wrong* prediction pushes $p_t$ under the float32 floor: normals
stop at $\approx 1.2\times10^{-38}$, subnormals at $\approx 1.4\times10^{-45}$,
and XLA flushes subnormals to zero on accelerators anyway. Once $p_t$ rounds to
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

### stop_gradient on the max
$\ell$ is invariant to the shift — it cancels analytically — so $m$ carries no
gradient information. `jax.lax.stop_gradient(m)` says so explicitly: the answer
and its derivatives are unchanged, and the backward pass no longer threads
through the `max`'s argmax-shaped subgradient. Leaving it out is not *wrong*
here; knowing why it costs nothing is the point.

### Watch the axis
`jnp.max(logits, axis=-1)` drops the class axis, so `logits - m` fails to
broadcast (or worse, broadcasts against the batch axis when `B == C` and
silently gives a wrong answer). `keepdims=True` on both the `max` and the `sum`
is what keeps the shapes honest — the same discipline as **`b_12`**.

**`b_15`** takes this further with label smoothing and a padding mask.
""",
    "stub": '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets):
    """Mean cross-entropy over the batch — logsumexp written by hand.

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

    # logsumexp, by hand. Shifting by the row max makes every exponent <= 0, so
    # the sum is in [1, C] and can never overflow. The shift cancels exactly in
    # the final expression, hence stop_gradient.
    m = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    lse = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))

    # Gather the target logit — one entry per row, no one-hot matrix needed.
    z_t = jnp.take_along_axis(logits, targets[:, None], axis=-1)

    return jnp.mean(lse - z_t)
''',
    "demo": '''import jax
import jax.numpy as jnp

# Uniform logits over 3 classes -> loss is exactly log(3).
print("uniform:", cross_entropy_loss(jnp.zeros((1, 3)), jnp.array([0])), "vs", jnp.log(3.0))

# Against the library routines you are not allowed to call.
logits = jax.random.normal(jax.random.key(0), (4, 5)) * 3.0
targets = jnp.array([1, 2, 0, 4])
ref = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(logits, axis=-1), targets[:, None], axis=-1))
print("mine:", float(cross_entropy_loss(logits, targets)), " ref:", float(ref))

# What the shift is worth. float32 first: exp(1000) = inf.
big = jnp.array([[1000.0, 0.0, 0.0]])
print("\\nshifted   :", float(cross_entropy_loss(big, jnp.array([0]))))
print("unshifted :", float(jnp.log(jnp.sum(jnp.exp(big))) - big[0, 0]), " <- inf - inf")

# float16 overflows at exp(11.1), so this is not an exotic case at all.
h = jnp.array([[20.0, 0.0, 0.0]], dtype=jnp.float16)
print("\\nfp16 shifted  :", float(cross_entropy_loss(h, jnp.array([0]))))
print("fp16 exp(20)  :", float(jnp.exp(h[0, 0])), " <- inf, in a dtype people train in")

# Confidently WRONG stays finite: ~1000, not inf.
print("\\nwrong class:", float(cross_entropy_loss(big, jnp.array([1]))))

# The gradient is p - onehot, averaged over the batch.
g = jax.grad(cross_entropy_loss)(jnp.array([[2.0, 1.0, 0.0]]), jnp.array([0]))
print("grad:", g, " sums to", float(jnp.sum(g)))
''',
    # Implementations the ban test MUST reject. probe_tests.py cannot check
    # these — it execs attacks from strings, where inspect.getsource is blind —
    # so `python scripts/smoke_notebooks.py --banned` runs them through a real
    # kernel, which is the only place the check is live.
    "banned_examples": ['''import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def cross_entropy_loss(logits, targets):
    z_t = jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
    return jnp.mean(logsumexp(logits, axis=-1) - z_t)
''', '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))
'''],
    "tests": [
        {
            "name": "logsumexp is not called for you",
            "code": """
import ast
import inspect
import textwrap

try:
    src = inspect.getsource({fn})
except (OSError, TypeError):
    src = ''          # defined by exec() rather than a cell — nothing to read

if src:
    # Identifiers only: comments and docstrings are not code, and this problem
    # is about what you CALL, not what you write about.
    used = set()
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)

    for banned in ('logsumexp', 'log_softmax', 'softmax', 'optax'):
        assert banned not in used, (
            f'{banned}() is banned here — with it, this is just problem 16. '
            'Write the shift yourself: m + log(sum(exp(z - m))).'
        )
""",
        },
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

# Square batch: B == C, so a max/sum without keepdims broadcasts against the
# WRONG axis instead of failing loudly.
sq = jax.random.normal(jax.random.key(4), (7, 7)) * 2.0
tg = jax.random.randint(jax.random.key(5), (7,), 0, 7)
ref_sq = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(sq, axis=-1), tg[:, None], axis=-1))
assert jnp.allclose({fn}(sq, tg), ref_sq, atol=1e-5), (
    f'{float({fn}(sq, tg)):.6f} vs {float(ref_sq):.6f} on a (7, 7) batch — reduce over '
    'the CLASS axis with keepdims=True'
)

# Softmax is invariant to adding a constant to a whole row, so the loss is too.
bump = logits + jnp.arange(16.0)[:, None]
assert jnp.allclose({fn}(bump, targets), out, atol=1e-4), (
    'Adding a per-row constant must not change the loss'
)
""",
        },
        {
            "name": "The shift: extreme logits, float32 and float16",
            "code": """
import jax.numpy as jnp

# exp(1000) = inf without the max shift.
big = jnp.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
out = {fn}(big, jnp.array([0, 1]))
assert jnp.isfinite(out), (
    f'Non-finite loss on huge logits ({out}) — subtract the row max before exp()'
)
assert float(out) < 1e-3, f'Confident and correct should be ~0, got {float(out)}'

# Confidently WRONG: large but finite, never inf.
wrong = {fn}(big, jnp.array([1, 0]))
assert jnp.isfinite(wrong), (
    f'Confidently-wrong gave {wrong}. p_target underflows to 0.0, so log(p) is -inf. '
    'The fused form z_t - logsumexp(z) stays finite.'
)
assert 900.0 < float(wrong) < 1100.0, f'Expected ~1000, got {float(wrong)}'

# float16 overflows at exp(11.1) — the shift matters at everyday logit sizes.
h = jnp.array([[20.0, 0.0, 0.0], [0.0, 30.0, 0.0]], dtype=jnp.float16)
oh = {fn}(h, jnp.array([0, 1]))
assert jnp.isfinite(oh), (
    f'Non-finite in float16 ({oh}): exp(20) is already inf in fp16, so the shift '
    'is not optional in mixed precision'
)
assert float(oh) < 1e-2, f'Confident and correct in fp16 should be ~0, got {float(oh)}'

# exp(-1000) = 0 everywhere: an unshifted sum is 0 and log(0) = -inf.
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

# Explicit: grad = (p - onehot) / B. stop_gradient on the max must not change
# this — the shift cancels, so its derivative contributes nothing either way.
p = jax.nn.softmax(logits, axis=-1)
expected = (p - jax.nn.one_hot(targets, 5)) / 4.0
assert jnp.allclose(g, expected, atol=1e-5), 'Gradient does not match (p - onehot) / B'

# Gradient stays finite where the naive route would give nan.
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
