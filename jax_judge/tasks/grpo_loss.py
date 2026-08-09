"""GRPO — PPO with the value network replaced by a group mean."""

TASK = {
    "title": "GRPO (Group Relative Policy Optimization) Loss",
    "category": "RLHF & Preference Losses",
    "order": 2,
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "Two stages. First the advantages: rewards are (n_prompts, group_size), "
        "so normalise WITHIN each group — subtract the row mean and divide by "
        "the row std (+eps), keepdims=True on both. Then broadcast each "
        "sequence-level advantage across its tokens and run the usual PPO "
        "clipped surrogate, adding beta * the k3 KL estimator. Everything is "
        "masked-averaged over real tokens."
    ),
    "description": r"""
Implement the **GRPO** loss — the objective behind DeepSeek-R1-style reasoning
training.

**1. Group-relative advantages.** For a prompt with a group of $G$ sampled
completions and rewards $r_1 \dots r_G$:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

Every token of completion $i$ gets that same scalar advantage.

**2. Clipped surrogate with a KL penalty:**

$$\mathcal{L} = -\frac{1}{\sum m}\sum m \Big[
\min\big(r_t A,\ \text{clip}(r_t, 1-\epsilon_c, 1+\epsilon_c) A\big)
- \beta\, \mathbb{D}_{KL}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]_t \Big]$$

**3. The KL term** uses the low-variance, always-positive **k3** estimator:

$$\mathbb{D}_{KL} = \exp(\ell_{\text{ref}} - \ell_\theta) - (\ell_{\text{ref}} - \ell_\theta) - 1$$

### Signature
```python
def grpo_loss(new_logps, old_logps, ref_logps, rewards,
              clip_eps=0.2, beta=0.04, mask=None):
    # new_logps / old_logps / ref_logps: (n_prompts, group_size, seq)
    # rewards:                           (n_prompts, group_size)
    # mask:                              (n_prompts, group_size, seq) or None
    ...  # -> scalar loss
```

### Rules
- Normalise rewards **within each group**, never across the whole batch — that
  is the entire idea
- Broadcast the per-sequence advantage across the token axis
- Use the k3 KL estimator above, not `logp_ratio` and not a plain difference
- Mask-average over real tokens
- Do not use any RL library

### Why the group baseline replaces the value network
PPO needs an advantage, and an advantage needs a baseline $V(s)$ — normally a
second network of the same size as the policy, trained alongside it. That is
expensive, and for LLM reasoning it is also *hard*: the value of a
half-finished chain of thought is a terrible regression target.

GRPO's move is to sample $G$ completions for the **same** prompt and use the
group's own mean reward as the baseline. Same prompt means the comparison is
apples-to-apples, so the mean is an unbiased baseline — and it costs no
parameters. Dividing by the group std additionally whitens the advantage scale,
which is what lets a single learning rate work across prompts of wildly
different difficulty.

### The degenerate case to watch
If every completion in a group gets the **same** reward — all correct, or all
wrong — then $\text{std} = 0$ and every advantage is $0$. That group
contributes no policy gradient at all, only the KL term. This is not a bug; it
is why GRPO implementations care about prompts landing at neither 0% nor 100%
pass rate.

### Why k3 and not the naive estimator
The naive single-sample KL, $\ell_\theta - \ell_{\text{ref}}$, is unbiased but
can go **negative** for an individual sample, which makes a "penalty" that
sometimes pays you. The k3 form $e^{-x} - (-x) - 1$ with
$x = \ell_\theta - \ell_{\text{ref}}$ is non-negative everywhere, is zero
exactly when the policies agree, and has much lower variance.
""",
    "stub": '''import jax
import jax.numpy as jnp


def grpo_loss(new_logps, old_logps, ref_logps, rewards,
              clip_eps=0.2, beta=0.04, mask=None):
    """GRPO loss.

    Args:
        new_logps: (n_prompts, group_size, seq) current policy log-probs
        old_logps: (n_prompts, group_size, seq) sampling policy log-probs
        ref_logps: (n_prompts, group_size, seq) reference policy log-probs
        rewards:   (n_prompts, group_size) scalar reward per completion
        clip_eps:  PPO clip range
        beta:      KL penalty weight
        mask:      optional (n_prompts, group_size, seq), 1 real / 0 padding

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def grpo_loss(new_logps, old_logps, ref_logps, rewards,
              clip_eps=0.2, beta=0.04, mask=None):
    # 1. Advantages, normalised WITHIN each prompt's group of completions.
    mean = jnp.mean(rewards, axis=-1, keepdims=True)
    std = jnp.std(rewards, axis=-1, keepdims=True)
    advantages = (rewards - mean) / (std + 1e-8)

    # One scalar advantage per completion, shared by all of its tokens.
    adv = advantages[..., None]

    # 2. Clipped surrogate, exactly as in PPO.
    ratio = jnp.exp(new_logps - old_logps)
    surrogate = jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    )

    # 3. k3 KL estimator: non-negative, low variance, zero iff the policies match.
    log_ratio = ref_logps - new_logps
    kl = jnp.exp(log_ratio) - log_ratio - 1.0

    per_token = surrogate - beta * kl

    if mask is None:
        return -jnp.mean(per_token)

    mask = mask.astype(per_token.dtype)
    return -jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)
''',
    "demo": '''import jax.numpy as jnp

# One prompt, four completions, rewards 0/0/1/1.
rewards = jnp.array([[0.0, 0.0, 1.0, 1.0]])
mean = rewards.mean(-1, keepdims=True)
std = rewards.std(-1, keepdims=True)
print("advantages:", ((rewards - mean) / (std + 1e-8))[0])

lp = jnp.zeros((1, 4, 3))
print("loss (on-policy, no KL):", float(grpo_loss(lp, lp, lp, rewards, beta=0.0)))

# All-correct group -> std 0 -> every advantage 0 -> no policy gradient.
flat = jnp.ones((1, 4))
print("loss (all rewards equal):", float(grpo_loss(lp, lp, lp, flat, beta=0.0)))
''',
    "tests": [
        {
            "name": "Advantages are normalised within each group",
            "code": """
import jax
import jax.numpy as jnp

# Two prompts on wildly different reward scales. Group-relative normalisation
# makes them produce IDENTICAL advantages; batch-wide normalisation would not.
rewards = jnp.array([[0.0, 1.0, 2.0],
                     [100.0, 101.0, 102.0]])
lp = jnp.zeros((2, 3, 4))

loss = {fn}(lp, lp, lp, rewards, 0.2, 0.0)
assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'

# On-policy (ratio = 1) and beta = 0, so loss = -mean(advantages) = 0 for
# per-group normalisation, since each group's advantages sum to zero.
assert jnp.allclose(loss, 0.0, atol=1e-5), (
    f'Got {float(loss)}. Each group normalised on its own has zero mean, so the '
    'total must be 0. A non-zero value means the rewards were normalised across '
    'the whole batch instead of within each prompt group.'
)
""",
        },
        {
            "name": "Matches the closed form",
            "code": """
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(0), 4)
new = jax.random.normal(k[0], (3, 4, 5)) * 0.2
old = jax.random.normal(k[1], (3, 4, 5)) * 0.2
ref = jax.random.normal(k[2], (3, 4, 5)) * 0.2
rew = jax.random.normal(k[3], (3, 4))

for eps, beta in ((0.2, 0.04), (0.1, 0.0), (0.3, 0.5)):
    mean = jnp.mean(rew, -1, keepdims=True)
    std = jnp.std(rew, -1, keepdims=True)
    adv = ((rew - mean) / (std + 1e-8))[..., None]

    ratio = jnp.exp(new - old)
    surr = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - eps, 1 + eps) * adv)
    lr_ = ref - new
    kl = jnp.exp(lr_) - lr_ - 1.0
    expected = -jnp.mean(surr - beta * kl)

    got = {fn}(new, old, ref, rew, eps, beta)
    assert jnp.allclose(got, expected, atol=1e-4), (
        f'eps={eps} beta={beta}: {float(got)} vs {float(expected)}'
    )
""",
        },
        {
            "name": "Zero-variance group produces no policy gradient",
            "code": """
import jax
import jax.numpy as jnp

# Every completion equally good -> std 0 -> all advantages 0.
rewards = jnp.full((2, 4), 1.0)
new = jax.random.normal(jax.random.key(1), (2, 4, 3)) * 0.1
old = jnp.zeros((2, 4, 3))
ref = jnp.zeros((2, 4, 3))

loss = {fn}(new, old, ref, rewards, 0.2, 0.0)
assert jnp.isfinite(loss), f'std=0 produced {loss} — the +eps guard is missing'
assert jnp.allclose(loss, 0.0, atol=1e-5), (
    f'With beta=0 and every reward identical, all advantages are 0 so the loss '
    f'must be 0. Got {float(loss)}.'
)

g = jax.grad({fn})(new, old, ref, rewards, 0.2, 0.0)
assert jnp.allclose(g, 0.0, atol=1e-5), (
    'A group where every completion scores the same carries no learning signal, '
    f'so the policy gradient must vanish. Got max |g| = {float(jnp.abs(g).max())}'
)
""",
        },
        {
            "name": "k3 KL is non-negative and zero at the reference",
            "code": """
import jax
import jax.numpy as jnp

rew = jnp.array([[0.0, 1.0]])
lp = jnp.zeros((1, 2, 4))

# new == ref => KL is exactly 0 => beta cannot matter.
a = float({fn}(lp, lp, lp, rew, 0.2, 0.0))
b = float({fn}(lp, lp, lp, rew, 0.2, 10.0))
assert jnp.allclose(a, b, atol=1e-6), (
    f'When the policy equals the reference the KL term is 0, so beta must have '
    f'no effect: beta=0 gave {a}, beta=10 gave {b}'
)

# Diverging from the reference must INCREASE the loss, in both directions.
for delta in (0.5, -0.5):
    new = jnp.full((1, 2, 4), delta)
    pen = float({fn}(new, new, lp, rew, 0.2, 1.0))
    nopen = float({fn}(new, new, lp, rew, 0.2, 0.0))
    assert pen > nopen, (
        f'delta={delta}: the KL penalty must raise the loss ({pen} vs {nopen}). '
        'k3 is non-negative for a divergence of EITHER sign — a plain '
        '(ref - new) difference would go negative for one of them.'
    )
""",
        },
        {
            "name": "Clipping behaves like PPO",
            "code": """
import jax
import jax.numpy as jnp

rew = jnp.array([[0.0, 2.0]])          # advantages: -1, +1
old = jnp.zeros((1, 2, 1))
ref = jnp.zeros((1, 2, 1))

# Push the positive-advantage completion far past 1 + eps.
new = jnp.array([[[0.0], [jnp.log(3.0)]]])
g = jax.grad({fn})(new, old, ref, rew, 0.2, 0.0)

assert jnp.allclose(g[0, 1, 0], 0.0, atol=1e-6), (
    f'The A>0 completion at ratio 3.0 is far past 1+eps, so its gradient must be '
    f'clipped to 0. Got {float(g[0, 1, 0])}'
)

# And the negative-advantage completion pushed up keeps its gradient.
new2 = jnp.array([[[jnp.log(3.0)], [0.0]]])
g2 = jax.grad({fn})(new2, old, ref, rew, 0.2, 0.0)
assert abs(float(g2[0, 0, 0])) > 1e-3, (
    f'The A<0 completion at ratio 3.0 must keep its gradient (min picks the '
    f'unclipped branch). Got {float(g2[0, 0, 0])}'
)
""",
        },
        {
            "name": "Mask averages over real tokens only",
            "code": """
import jax
import jax.numpy as jnp

rew = jnp.array([[0.0, 2.0]])
lp = jnp.zeros((1, 2, 4))
garbage = jnp.array([[[0.0, 0.0, 50.0, 50.0], [0.0, 0.0, -50.0, -50.0]]])
mask = jnp.array([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

masked = {fn}(lp + garbage * (1 - mask), lp, lp, rew, 0.2, 0.04, mask)
clean = {fn}(lp[:, :, :2], lp[:, :, :2], lp[:, :, :2], rew, 0.2, 0.04)

assert jnp.allclose(masked, clean, atol=1e-5), (
    f'Padding positions must be excluded entirely: {float(masked)} vs {float(clean)}. '
    'Both the numerator and the denominator have to honour the mask.'
)

full = {fn}(lp, lp, lp, rew, 0.2, 0.04, jnp.ones_like(lp))
none = {fn}(lp, lp, lp, rew, 0.2, 0.04)
assert jnp.allclose(full, none, atol=1e-6), 'An all-ones mask changed the result'
""",
        },
        {
            "name": "jit and gradient sanity",
            "code": """
import functools
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(2), 4)
new = jax.random.normal(k[0], (2, 3, 4)) * 0.2
old = jax.random.normal(k[1], (2, 3, 4)) * 0.2
ref = jax.random.normal(k[2], (2, 3, 4)) * 0.2
rew = jax.random.normal(k[3], (2, 3))

jitted = jax.jit(functools.partial({fn}, clip_eps=0.2, beta=0.04))
assert jnp.allclose(jitted(new, old, ref, rew),
                    {fn}(new, old, ref, rew, 0.2, 0.04), atol=1e-5), 'jit changes the result'

g = jax.grad({fn})(new, old, ref, rew, 0.2, 0.04)
assert g.shape == new.shape, f'Gradient shape {g.shape} vs {new.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'
""",
        },
    ],
}
