"""PPO's clipped surrogate — and why the min() is the whole trick."""

TASK = {
    "title": "PPO Clipped Surrogate Loss",
    "category": "RLHF & Preference Losses",
    "order": 3,
    "difficulty": "Medium",
    "function_name": "ppo_loss",
    "hint": (
        "Five lines, three of which are easy to get subtly wrong. The ratio is a "
        "ratio of probabilities but you are handed log-probs, so form it in log "
        "space — a real sequence log-prob is around -800 and exp() of that is 0. "
        "The min() is over the two signed PRODUCTS, not over the two ratios: "
        "clipping the ratio first and multiplying afterwards is a different "
        "function, and it differs exactly where the advantage is negative. And "
        "the formula in the spec is a surrogate you maximise while the thing you "
        "return is a loss you minimise — the test with ratio 1 catches that sign "
        "immediately."
    ),
    "description": r"""
Implement PPO's **clipped surrogate objective**.

$$L^{\text{CLIP}} = \mathbb{E}\Big[\min\big(r_t A_t,\ \
\text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big],
\qquad r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

Return the **loss**, i.e. $-L^{\text{CLIP}}$, averaged over all elements
(honouring an optional mask).

### Signature
```python
def ppo_loss(new_logps, old_logps, advantages, clip_eps=0.2, mask=None):
    ...  # -> scalar loss
```

`new_logps`, `old_logps`, `advantages` are `(batch, seq)` (or any matching
shape). `mask` is 1 for real tokens, 0 for padding.

### Rules
- Compute the ratio in log space: `exp(new - old)`, never `exp(new)/exp(old)`
- The `min` is taken **after** multiplying by the advantage, on the two signed
  products
- Return a scalar; with a mask, average over unmasked elements only
- Do not use any RL library

### Why min() and not just clip()
This is the question interviewers actually ask. Clipping alone is **not**
conservative — it is the `min` that makes the bound pessimistic, and the two
sides behave asymmetrically:

- **$A > 0$** (action was better than expected). The objective wants $r$ up. The
  clip caps the reward at $r = 1+\epsilon$, so past that there is **no gradient**
  — you cannot keep pushing a good action arbitrarily far in one update.
- **$A < 0$** (action was worse). The objective wants $r$ down. Now $rA$ becomes
  *more* negative as $r$ grows, and `min` selects that unclipped branch. So for
  a bad action that has become *more* likely, the gradient is **not** clipped —
  PPO always retains the ability to fix a mistake.

With `clip` alone and no `min`, that second case would also flatten out, leaving
a policy that had drifted badly with no gradient to recover. The `min` deliberately
keeps the penalty live in exactly the direction where you want it live.

### The trap
$\pi_{\text{old}}$ is frozen when the rollouts are collected, so on the very
first gradient step of an update $\theta = \theta_{\text{old}}$, every $r_t$ is
exactly 1, nothing clips, and the loss is just $-\bar{A}$. Clipping only starts
biting as later minibatches and inner epochs push $\theta$ away. Two
consequences worth saying out loud: a fresh-rollout loss that is not
$-\bar{A}$ is a bug in your ratio or your sign, and if you only ever take one
inner epoch, the clip is dead code and PPO degenerates to vanilla policy
gradient.
""",
    "stub": '''import jax
import jax.numpy as jnp


def ppo_loss(new_logps, old_logps, advantages, clip_eps=0.2, mask=None):
    """PPO clipped surrogate loss.

    Args:
        new_logps:  (batch, seq) log-probs under the current policy
        old_logps:  (batch, seq) log-probs under the sampling policy
        advantages: (batch, seq) advantage estimates
        clip_eps:   clipping range epsilon
        mask:       optional (batch, seq) of 1.0 real / 0.0 padding

    Returns:
        Scalar loss (negated surrogate).
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def ppo_loss(new_logps, old_logps, advantages, clip_eps=0.2, mask=None):
    # Probability ratio, formed in log space so nothing overflows.
    ratio = jnp.exp(new_logps - old_logps)

    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    # Pessimistic bound: take the WORSE of the two signed products.
    surrogate = jnp.minimum(unclipped, clipped)

    if mask is None:
        return -jnp.mean(surrogate)

    mask = mask.astype(surrogate.dtype)
    return -jnp.sum(surrogate * mask) / jnp.maximum(jnp.sum(mask), 1.0)
''',
    "demo": '''import jax.numpy as jnp

old = jnp.zeros((1, 5))
adv = jnp.array([[1.0, 1.0, 1.0, -1.0, -1.0]])

for delta in (0.0, 0.1, 0.5, 2.0):
    new = old + delta
    r = float(jnp.exp(delta))
    print(f"ratio={r:6.2f}  loss={float(ppo_loss(new, old, adv)):+.4f}")
# Positive-advantage terms stop improving past r = 1.2;
# negative-advantage terms keep getting punished.
''',
    "tests": [
        {
            "name": "Ratio is 1 at the start of an epoch",
            "code": """
import jax
import jax.numpy as jnp

logps = jax.random.normal(jax.random.key(0), (4, 6))
adv = jax.random.normal(jax.random.key(1), (4, 6))

loss = {fn}(logps, logps, adv, 0.2)
assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'
assert jnp.allclose(loss, -jnp.mean(adv), atol=1e-6), (
    f'With new_logps == old_logps the ratio is exactly 1 and nothing clips, so '
    f'the loss must be -mean(advantages) = {float(-jnp.mean(adv)):.6f}. '
    f'Got {float(loss)}. A sign error shows up here first.'
)
""",
        },
        {
            "name": "Matches the closed form",
            "code": """
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(2), 3)
new = jax.random.normal(k[0], (8, 5)) * 0.3
old = jax.random.normal(k[1], (8, 5)) * 0.3
adv = jax.random.normal(k[2], (8, 5))

for eps in (0.1, 0.2, 0.4):
    ratio = jnp.exp(new - old)
    expected = -jnp.mean(jnp.minimum(ratio * adv,
                                     jnp.clip(ratio, 1 - eps, 1 + eps) * adv))
    got = {fn}(new, old, adv, eps)
    assert jnp.allclose(got, expected, atol=1e-6), f'eps={eps}: {float(got)} vs {float(expected)}'
""",
        },
        {
            "name": "Positive advantage: gradient dies past 1+eps",
            "code": """
import jax
import jax.numpy as jnp

old = jnp.zeros((1, 1))
adv = jnp.ones((1, 1))

# Ratio well above 1 + eps => clipped branch is selected => flat => no gradient.
new_far = jnp.full((1, 1), jnp.log(2.0))     # ratio = 2.0, eps = 0.2
g = jax.grad(lambda n: {fn}(n, old, adv, 0.2))(new_far)
assert jnp.allclose(g, 0.0, atol=1e-6), (
    f'With A>0 and ratio=2.0 far past 1+eps=1.2, the clipped branch wins and the '
    f'gradient must be exactly 0. Got {float(g[0, 0])} — this is what clipping is for.'
)

# Just inside the clip range the gradient is alive.
new_near = jnp.full((1, 1), jnp.log(1.1))
g_near = jax.grad(lambda n: {fn}(n, old, adv, 0.2))(new_near)
assert abs(float(g_near[0, 0])) > 1e-3, (
    f'Inside the trust region the gradient must be non-zero, got {float(g_near[0, 0])}'
)

# The loss also saturates in value, not just in gradient.
l2 = float({fn}(jnp.full((1, 1), jnp.log(2.0)), old, adv, 0.2))
l5 = float({fn}(jnp.full((1, 1), jnp.log(5.0)), old, adv, 0.2))
assert jnp.allclose(l2, l5, atol=1e-6), f'Loss should saturate at -(1+eps): {l2} vs {l5}'
assert jnp.allclose(l2, -1.2, atol=1e-6), f'Saturated value should be -(1+eps) = -1.2, got {l2}'
""",
        },
        {
            "name": "Negative advantage: penalty stays live (the min matters)",
            "code": """
import jax
import jax.numpy as jnp

old = jnp.zeros((1, 1))
adv = -jnp.ones((1, 1))   # bad action

# A bad action that became MUCH more likely must keep producing gradient.
new_far = jnp.full((1, 1), jnp.log(3.0))    # ratio = 3.0
g = jax.grad(lambda n: {fn}(n, old, adv, 0.2))(new_far)
assert abs(float(g[0, 0])) > 1e-3, (
    f'With A<0 and ratio=3.0, min() selects the UNCLIPPED branch (-3.0 < -1.2), '
    f'so the gradient must stay alive to push this action back down. Got {float(g[0, 0])}. '
    'Using clip() alone, without the min, wrongly kills it here.'
)

loss_far = float({fn}(new_far, old, adv, 0.2))
loss_near = float({fn}(jnp.full((1, 1), jnp.log(1.5)), old, adv, 0.2))
assert loss_far > loss_near, (
    f'A worse ratio on a negative-advantage action must cost more: '
    f'{loss_far} (r=3) should exceed {loss_near} (r=1.5)'
)
assert jnp.allclose(loss_far, 3.0, atol=1e-5), (
    f'Loss should be -min(-3.0, -1.2) = 3.0, got {loss_far}'
)
""",
        },
        {
            "name": "Ratio computed in log space",
            "code": """
import jax
import jax.numpy as jnp

# Log-probs of long sequences are large and negative; exp() of them underflows
# to 0, so exp(new)/exp(old) becomes 0/0 = nan.
new = jnp.full((2, 3), -800.0)
old = jnp.full((2, 3), -800.5)
adv = jnp.ones((2, 3))

loss = {fn}(new, old, adv, 0.2)
assert jnp.isfinite(loss), (
    f'Got {loss} for log-probs around -800. Compute exp(new - old), not '
    'exp(new) / exp(old) — the latter underflows to 0/0.'
)
assert jnp.allclose(loss, -jnp.exp(0.5), atol=1e-5) or jnp.allclose(loss, -1.2, atol=1e-5), (
    f'Unexpected value {float(loss)} for ratio exp(0.5)=1.6487 with eps=0.2'
)
""",
        },
        {
            "name": "Mask averages over real tokens only",
            "code": """
import jax
import jax.numpy as jnp

new = jnp.zeros((2, 4))
old = jnp.zeros((2, 4))
adv = jnp.array([[1.0, 1.0, 999.0, 999.0],
                 [2.0, 2.0, -999.0, -999.0]])
mask = jnp.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0, 0.0]])

loss = {fn}(new, old, adv, 0.2, mask)
assert jnp.allclose(loss, -1.5, atol=1e-6), (
    f'Masked mean of [1,1,2,2] is 1.5 so the loss is -1.5, got {float(loss)}. '
    'Padding must be excluded from BOTH the numerator and the denominator.'
)

# An all-ones mask must reproduce the unmasked result.
full = jnp.ones((2, 4))
a = {fn}(new, old, adv, 0.2, full)
b = {fn}(new, old, adv, 0.2)
assert jnp.allclose(a, b, atol=1e-6), f'All-ones mask changed the result: {float(a)} vs {float(b)}'
""",
        },
        {
            "name": "jit and gradient sanity",
            "code": """
import functools
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(4), 3)
new = jax.random.normal(k[0], (4, 5)) * 0.2
old = jax.random.normal(k[1], (4, 5)) * 0.2
adv = jax.random.normal(k[2], (4, 5))

jitted = jax.jit(functools.partial({fn}, clip_eps=0.2))
assert jnp.allclose(jitted(new, old, adv), {fn}(new, old, adv, 0.2), atol=1e-6), (
    'jit changes the result'
)

g = jax.grad({fn})(new, old, adv, 0.2)
assert g.shape == new.shape, f'Gradient shape {g.shape} vs {new.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'
""",
        },
    ],
}
