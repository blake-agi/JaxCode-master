"""Cross-entropy from logits — the log-sum-exp trick, label smoothing, padding mask."""

TASK = {
    "title": "Cross-Entropy Loss from Logits",
    "category": "Training",
    "order": 1,
    "difficulty": "Easy",
    "function_name": "cross_entropy_loss",
    "hint": (
        "Never log(softmax(x)) — compute log-softmax in one step, subtracting the "
        "row max and then the log-sum-exp of the shifted logits. Pull out the "
        "target's log-probability by gathering that index rather than "
        "materialising a one-hot matrix. Label smoothing is a convex blend of the "
        "target's NLL and the mean log-probability over the vocabulary. For "
        "ignore_index the trap is that an out-of-range index does NOT raise in "
        "JAX — it silently clamps — so sanitise the indices before gathering, "
        "then mask the per-token losses and divide by the number of valid tokens, "
        "not the total."
    ),
    "description": r"""
Implement **cross-entropy loss directly from logits**, with optional label
smoothing and a padding mask.

$$\ell_i = -\sum_{c} q_{i,c} \log p_{i,c}, \qquad
p_{i,c} = \frac{e^{z_{i,c}}}{\sum_{k} e^{z_{i,k}}}$$

With smoothing $\alpha$ over $C$ classes the target distribution is
$q_{i,c} = (1-\alpha)\,\mathbb{1}[c = t_i] + \alpha/C$, so

$$\ell_i = (1-\alpha)\bigl(-\log p_{i,t_i}\bigr) \;+\; \frac{\alpha}{C}\sum_c \bigl(-\log p_{i,c}\bigr)$$

Return the **mean over the non-ignored positions only**.

### Rules
- Signature: `cross_entropy_loss(logits, targets, *, label_smoothing=0.0, ignore_index=-1)`
- `logits` is `(..., C)`, `targets` is `(...)` of integer class ids; the output is a **scalar**
- Banned: `jax.nn.log_softmax`, `jax.nn.softmax`, `jax.scipy.special.logsumexp`, `optax`
- Compute $\log p$ in one fused expression; never form $p$ and then take its log
- Positions where `targets == ignore_index` contribute **zero** loss and are excluded
  from the denominator; if every position is ignored, return `0.0` (not `nan`)
- No `if` on array values — the whole thing must work under `jit` and `vmap`

### Why you never softmax-then-log
The naive route dies twice.

**Overflow.** `exp(z)` is `inf` above $z \approx 88.7$ in float32 and above
$z \approx 11.1$ in float16. The fix is the shift
$\log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}$ with $m = \max_k z_k$: every
exponent is now $\le 0$, so the largest term is exactly `1.0` and the sum can
never overflow.

**Underflow — the one that actually bites.** Even with no overflow, a confidently
*wrong* prediction gives $p_{t} < 10^{-38}$, which flushes to `0.0`, and
`log(0) = -inf` makes the loss `inf` and every gradient `nan`. The fused form
never materialises $p_t$: it computes $z_t - \log\sum_k e^{z_k}$, a perfectly
finite number like $-120$. Your loss stays large-but-finite and training
recovers instead of poisoning every parameter with `nan`.

There is a gradient bonus too. $\partial \ell / \partial z = p - q$ — a clean,
bounded expression that autodiff derives exactly from the fused form. Compose
`log` on top of a separate `softmax` and you hand XLA a division of two tiny
numbers to differentiate through.

The `max` shift cancels analytically ($\ell$ is invariant to it), so wrapping it
in `stop_gradient` changes nothing mathematically and keeps the backward graph
smaller.

### The interview angle
`ignore_index` is where candidates lose the plot. In a padded LM batch, 30–60%
of positions are `<pad>`. Average over all of them and two things go wrong: the
loss is scaled down by the pad fraction (so it silently changes meaning when the
batch composition changes), and the gradient actively teaches the model to
predict `<pad>`. Note also that you must clamp the index *before* the gather —
JAX has no bounds checking, so `take_along_axis` with `-1` quietly reads the
last class instead of raising, and you get a plausible-looking wrong loss.
""",
    "stub": '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets, *, label_smoothing=0.0, ignore_index=-1):
    """Mean cross-entropy over the non-ignored positions.

    Args:
        logits:          (..., C) unnormalised scores
        targets:         (...) integer class ids
        label_smoothing: alpha in [0, 1); target is (1-a)*onehot + a/C
        ignore_index:    positions equal to this are masked out entirely

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets, *, label_smoothing=0.0, ignore_index=-1):
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets)
    num_classes = logits.shape[-1]

    # log-softmax, fused. Shifting by the row max makes every exponent <= 0, so
    # the sum is in [1, C] and can never overflow. The shift cancels exactly in
    # the final expression, hence stop_gradient.
    shift = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    z = logits - shift
    log_probs = z - jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))

    valid = targets != ignore_index
    # Clamp BEFORE gathering: JAX silently clips out-of-range indices.
    safe = jnp.where(valid, targets, 0)
    nll = -jnp.take_along_axis(log_probs, safe[..., None], axis=-1)[..., 0]

    # Smoothing is a convex blend with the uniform target: (a/C) * sum_c -log p_c.
    uniform = -jnp.mean(log_probs, axis=-1)
    per_position = (1.0 - label_smoothing) * nll + label_smoothing * uniform

    per_position = jnp.where(valid, per_position, 0.0)
    n_valid = jnp.sum(valid)
    return jnp.sum(per_position) / jnp.maximum(n_valid, 1)
''',
    "demo": '''import jax.numpy as jnp

# Uniform logits over 3 classes -> loss is exactly log(3).
print("uniform:", cross_entropy_loss(jnp.zeros((1, 3)), jnp.array([0])), "vs", jnp.log(3.0))

# The stability trap: huge logits.
big = jnp.array([[1000.0, 0.0, 0.0]])
print("big logits, correct class:", cross_entropy_loss(big, jnp.array([0])))
print("naive softmax-then-log would give:",
      -jnp.log(jnp.exp(big) / jnp.exp(big).sum(-1, keepdims=True))[0, 0])

# Padding: the last two positions are masked out.
logits = jnp.zeros((4, 5))
targets = jnp.array([1, 2, -1, -1])
print("masked mean:", cross_entropy_loss(logits, targets, ignore_index=-1))
print("unmasked mean would also be log(5) here, but the gradient would not be:")
print("  smoothing 0.1:", cross_entropy_loss(logits, targets, label_smoothing=0.1,
                                             ignore_index=-1))
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
assert jnp.allclose(out, ref, atol=1e-5), f'{float(out):.6f} vs reference {float(ref):.6f}'

# Sequence shape (B, T, C) must work too, averaging over B*T.
lg = jax.random.normal(jax.random.key(2), (3, 5, 7))
tg = jax.random.randint(jax.random.key(3), (3, 5), 0, 7)
ref3 = -jnp.mean(jnp.take_along_axis(
    jax.nn.log_softmax(lg, axis=-1), tg[..., None], axis=-1))
o3 = {fn}(lg, tg)
assert jnp.ndim(o3) == 0, f'(B, T, C) input must still give a scalar, got {o3.shape}'
assert jnp.allclose(o3, ref3, atol=1e-5), f'(B, T, C): {float(o3):.6f} vs {float(ref3):.6f}'
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
            "name": "Label smoothing",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(4), (8, 6)) * 2.0
targets = jax.random.randint(jax.random.key(5), (8,), 0, 6)
lp = jax.nn.log_softmax(logits, axis=-1)

# alpha = 0 must be identical to plain cross-entropy.
assert jnp.allclose({fn}(logits, targets, label_smoothing=0.0),
                    {fn}(logits, targets), atol=1e-7), 'label_smoothing=0.0 changed the loss'

# alpha = 0.1 against the explicit convex blend.
a = 0.1
nll = -jnp.take_along_axis(lp, targets[:, None], axis=-1)[:, 0]
ref = jnp.mean((1 - a) * nll + a * (-jnp.mean(lp, axis=-1)))
got = {fn}(logits, targets, label_smoothing=a)
assert jnp.allclose(got, ref, atol=1e-5), (
    f'{float(got):.6f} vs {float(ref):.6f} — the uniform part must spread alpha over '
    'ALL C classes (including the true one), i.e. q = (1-a)*onehot + a/C'
)

# alpha = 1.0 is the pure uniform target: mean over classes of -log p.
full = {fn}(logits, targets, label_smoothing=1.0)
assert jnp.allclose(full, jnp.mean(-jnp.mean(lp, axis=-1)), atol=1e-5), (
    f'label_smoothing=1.0 should ignore the target entirely, got {float(full)}'
)

# Smoothing penalises over-confidence: a near-perfect prediction gets a HIGHER loss.
sharp = jnp.array([[20.0, 0.0, 0.0]])
t = jnp.array([0])
assert float({fn}(sharp, t, label_smoothing=0.1)) > float({fn}(sharp, t)) + 1e-3, (
    'Smoothing must increase the loss of an over-confident correct prediction'
)
""",
        },
        {
            "name": "ignore_index masks positions and the denominator",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(6), (6, 5)) * 3.0
targets = jnp.array([1, 4, -1, 0, -1, 2])

masked = {fn}(logits, targets, ignore_index=-1)
keep = jnp.array([0, 1, 3, 5])
dense = {fn}(logits[keep], targets[keep])
assert jnp.allclose(masked, dense, atol=1e-6), (
    f'{float(masked):.6f} vs {float(dense):.6f} — the mean must divide by the number of '
    'VALID positions (4 here), not by 6'
)

# The masked positions must not leak in through the gather either.
poison = logits.at[2].set(jnp.array([1e4, -1e4, 0.0, 0.0, 0.0]))
poison = poison.at[4].set(jnp.array([-1e4, 1e4, 0.0, 0.0, 0.0]))
assert jnp.allclose({fn}(poison, targets, ignore_index=-1), masked, atol=1e-5), (
    'Changing the logits at an ignored position changed the loss'
)

# A different sentinel value.
t2 = jnp.array([1, 4, -100, 0, -100, 2])
assert jnp.allclose({fn}(logits, t2, ignore_index=-100), dense, atol=1e-6), (
    'ignore_index must be honoured for values other than -1'
)

# Everything ignored -> 0.0, not nan (0/0).
allpad = {fn}(logits, jnp.full((6,), -1), ignore_index=-1)
assert jnp.isfinite(allpad), f'All-ignored batch gave {allpad} — guard the zero denominator'
assert jnp.allclose(allpad, 0.0, atol=1e-7), f'All-ignored should be 0.0, got {float(allpad)}'
""",
        },
        {
            "name": "Gradients are p - q and vanish on ignored rows",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(7), (4, 5)) * 2.0
targets = jnp.array([2, 0, -1, 4])

g = jax.grad(lambda z: {fn}(z, targets, ignore_index=-1))(logits)
assert g.shape == logits.shape, f'Gradient shape {g.shape} vs {logits.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'

# d/dz of -log p sums to zero across classes: sum_c (p_c - q_c) = 1 - 1 = 0.
assert jnp.allclose(jnp.sum(g, axis=-1), 0.0, atol=1e-6), (
    f'Per-row gradients should sum to 0 across classes, got {jnp.sum(g, axis=-1)}'
)
# Ignored rows get exactly zero gradient.
assert jnp.allclose(g[2], 0.0, atol=1e-9), f'Ignored row has gradient {g[2]}'

# Explicit check: grad = (p - onehot) / n_valid on the valid rows.
p = jax.nn.softmax(logits, axis=-1)
onehot = jax.nn.one_hot(jnp.clip(targets, 0), 5)
expected = jnp.where((targets != -1)[:, None], (p - onehot) / 3.0, 0.0)
assert jnp.allclose(g, expected, atol=1e-5), 'Gradient does not match (p - q) / n_valid'

# Gradient stays finite where the naive softmax-then-log route would produce nan.
gbig = jax.grad(lambda z: {fn}(z, jnp.array([1])))(jnp.array([[800.0, -800.0]]))
assert jnp.isfinite(gbig).all(), f'nan/inf gradient at extreme logits: {gbig}'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import functools
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(8), (4, 9, 7))
targets = jax.random.randint(jax.random.key(9), (4, 9), 0, 7)

f = jax.jit(functools.partial({fn}, label_smoothing=0.05, ignore_index=-1))
eager = {fn}(logits, targets, label_smoothing=0.05, ignore_index=-1)
assert jnp.allclose(f(logits, targets), eager, atol=1e-6), 'jit changed the answer'

# Per-example losses via vmap over the batch axis.
per = jax.vmap(f)(logits, targets)
assert per.shape == (4,), f'vmap gave {per.shape}, expected (4,)'
assert jnp.allclose(jnp.mean(per), eager, atol=1e-5), (
    'Mean of the per-example losses should equal the batched loss'
)
for i in range(4):
    assert jnp.allclose(per[i], f(logits[i], targets[i]), atol=1e-6), f'vmap row {i} differs'
""",
        },
    ],
}
