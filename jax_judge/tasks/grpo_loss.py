"""GRPO — advantages normalised within a group, no value network."""

TASK = {
    "title": "GRPO (Group Relative Policy Optimization) Loss",
    "category": "RLHF & Preference Losses",
    "number": "38",
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "For each distinct group id, take the rewards belonging to it and "
        "standardise them: (r - mean) / (std + eps), using the BIASED std "
        "(ddof=0). JAX has no boolean-mask assignment, so build the advantages "
        "with jnp.where per group rather than advantages[mask] = ... . Stop the "
        "gradient through the advantages — they are a target, not something to "
        "differentiate — then the loss is -mean(A * logps)."
    ),
    "description": r"""
Implement the **GRPO** loss.

For each prompt you sample a *group* of responses. The advantage of a response
is its reward standardised **within its own group**:

$$A_i = \frac{r_i - \mu_{g(i)}}{\sigma_{g(i)} + \epsilon},
\qquad \mathcal{L} = -\frac{1}{B}\sum_i \text{sg}[A_i]\,\log \pi(y_i)$$

### Signature
```python
def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    ...  # -> scalar
```

- `logps`: `(B,)` policy log-probability of each sampled response
- `rewards`: `(B,)` scalar reward per response
- `group_ids`: `(B,)` integers — equal ids mean the same prompt

### Rules
- Standardise within each group using the **biased** std (`ddof=0`)
- **Stop the gradient** through the advantages
- Return the mean over the batch, as a scalar

### Why the group replaces the value network
PPO needs a learned value function $V(s)$ to compute advantages, which means a
second network the size of the policy — extra memory, extra training, and a
common source of instability when it lags behind.

GRPO's observation: if you sample $G$ responses to the *same* prompt, the group
mean is already an unbiased baseline for that prompt. Subtracting it removes the
prompt-difficulty confound that the value network existed to model, and dividing
by the group std normalises the scale. So the critic disappears entirely — which
is most of why GRPO is cheaper than PPO and why DeepSeek-R1 used it.

### Why the advantages are detached
$A$ is a *target*, computed from rewards the policy does not differentiate
through. Letting gradient flow into it would optimise the baseline rather than
the policy — the model could lower the loss by manipulating the normalisation
instead of by producing better responses.

### The trap
A group of identical rewards has $\sigma = 0$, so every advantage is 0 and that
group contributes nothing — correct, and the reason $\epsilon$ sits in the
denominator rather than being a division guard you can skip. Getting a `NaN`
here means $\epsilon$ was left out.

### ⚠️ A JAX note
PyTorch writes `advantages[mask] = ...` per group. JAX arrays are immutable and
a boolean mask has data-dependent shape, so build the result with `jnp.where`
over the whole batch, one group at a time.
""",
    "stub": '''import jax
import jax.numpy as jnp


def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    """Group Relative Policy Optimization loss.

    Args:
        logps:     (B,) policy log-probs for each sampled response
        rewards:   (B,) scalar reward per response
        group_ids: (B,) integers; equal ids belong to the same prompt
        eps:       stability term in the advantage denominator

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    advantages = jnp.zeros_like(rewards)

    # JAX has no advantages[mask] = ..., so accumulate with jnp.where.
    for gid in jnp.unique(group_ids):
        m = group_ids == gid
        n = jnp.sum(m)
        mean = jnp.sum(jnp.where(m, rewards, 0.0)) / n
        # Biased (population) std, matching ddof=0.
        var = jnp.sum(jnp.where(m, (rewards - mean) ** 2, 0.0)) / n
        a = (rewards - mean) / (jnp.sqrt(var) + eps)
        advantages = jnp.where(m, a, advantages)

    # The advantages are a target, not something to differentiate through.
    return -jnp.mean(jax.lax.stop_gradient(advantages) * logps)
''',
    "demo": '''import jax.numpy as jnp

# Two prompts, three sampled responses each.
logps = jnp.array([-1.0, -2.0, -0.5, -1.5, -0.8, -2.2])
rewards = jnp.array([1.0, 0.0, 2.0, 5.0, 4.0, 6.0])   # group 1 is easier
group_ids = jnp.array([0, 0, 0, 1, 1, 1])

print("loss:", float(grpo_loss(logps, rewards, group_ids)))
print("\\nGroup 1 has far higher raw rewards, but after within-group")
print("normalisation both groups contribute advantages on the same scale —")
print("that is the prompt-difficulty confound the value network used to model.")
''',
    "tests": [
        {
            "name": "Scalar output and the closed form",
            "code": """
import jax
import jax.numpy as jnp

logps = jnp.array([-1.0, -2.0, -0.5, -1.5])
rewards = jnp.array([1.0, 3.0, 2.0, 4.0])
gids = jnp.array([0, 0, 1, 1])

loss = {fn}(logps, rewards, gids)
assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'

adv = jnp.zeros_like(rewards)
for g in (0, 1):
    m = gids == g
    r = rewards[m]
    a = (r - r.mean()) / (r.std() + 1e-5)
    adv = adv.at[jnp.where(m)[0]].set(a)
expected = -jnp.mean(adv * logps)
assert jnp.allclose(loss, expected, atol=1e-5), f'{float(loss)} vs {float(expected)}'
""",
        },
        {
            "name": "Advantages are normalised WITHIN each group",
            "code": """
import jax
import jax.numpy as jnp

# Group 1's rewards are shifted far up. Because normalisation is per group,
# shifting a whole group must not change the loss at all.
logps = jnp.array([-1.0, -2.0, -0.5, -1.5])
gids = jnp.array([0, 0, 1, 1])

a = {fn}(logps, jnp.array([1.0, 3.0, 2.0, 4.0]), gids)
b = {fn}(logps, jnp.array([1.0, 3.0, 102.0, 104.0]), gids)

assert jnp.allclose(a, b, atol=1e-4), (
    f'Shifting group 1 by +100 changed the loss ({float(a)} -> {float(b)}). '
    'Advantages must be standardised within each group, so a constant offset '
    'to one group cancels — that is the whole point of the group baseline.'
)
""",
        },
        {
            "name": "Biased std, and group scale is removed",
            "code": """
import jax
import jax.numpy as jnp

# logps must CORRELATE with the advantages, otherwise the weighted sum is
# ~0 under either std convention and the test proves nothing.
logps = jnp.array([-4.0, -3.0, -2.0, -1.0])
gids = jnp.array([0, 0, 0, 0])
rewards = jnp.array([1.0, 2.0, 3.0, 4.0])

loss = {fn}(logps, rewards, gids)
biased = (rewards - rewards.mean()) / (rewards.std(ddof=0) + 1e-5)
unbiased = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-5)

exp_b = -jnp.mean(biased * logps)
exp_u = -jnp.mean(unbiased * logps)
assert not jnp.allclose(exp_b, exp_u, atol=1e-4), 'test is not discriminating'
assert jnp.allclose(loss, exp_b, atol=1e-5), (
    f'Got {float(loss)}; biased std gives {float(exp_b)}, unbiased {float(exp_u)}. '
    'Use ddof=0.'
)
""",
        },
        {
            "name": "Gradient flows to logps but not through the advantages",
            "code": """
import jax
import jax.numpy as jnp

logps = jnp.array([-1.0, -2.0, -0.5, -1.5])
rewards = jnp.array([1.0, 3.0, 2.0, 4.0])
gids = jnp.array([0, 0, 1, 1])

g_logps = jax.grad({fn}, argnums=0)(logps, rewards, gids)
assert jnp.isfinite(g_logps).all(), 'Non-finite gradient w.r.t. logps'
assert float(jnp.abs(g_logps).sum()) > 0, 'No gradient reached logps'

# d(loss)/d(logps_i) = -A_i / B, so it must not depend on logps at all.
adv = jnp.zeros_like(rewards)
for g in (0, 1):
    m = gids == g
    r = rewards[m]
    adv = adv.at[jnp.where(m)[0]].set((r - r.mean()) / (r.std() + 1e-5))
assert jnp.allclose(g_logps, -adv / 4, atol=1e-5), f'{g_logps} vs {-adv/4}'

g_rewards = jax.grad({fn}, argnums=1)(logps, rewards, gids)
assert jnp.allclose(g_rewards, 0.0, atol=1e-6), (
    f'Gradient w.r.t. rewards should be exactly 0 ({g_rewards}) — the '
    'advantages must be wrapped in stop_gradient'
)
""",
        },
        {
            "name": "A constant-reward group contributes nothing",
            "code": """
import jax
import jax.numpy as jnp

logps = jnp.array([-1.0, -2.0, -0.5])
rewards = jnp.array([5.0, 5.0, 5.0])       # zero variance
gids = jnp.array([0, 0, 0])

loss = {fn}(logps, rewards, gids)
assert jnp.isfinite(loss), (
    f'Got {loss}. A zero-variance group divides by std=0 — eps must guard it.'
)
assert jnp.allclose(loss, 0.0, atol=1e-4), (
    f'All rewards equal means every advantage is 0, so the loss is 0, got {float(loss)}'
)
""",
        },
        {
            "name": "Groups of unequal size, and unsorted ids",
            "code": """
import jax
import jax.numpy as jnp

# Group 7 has three members, group 3 has two, and the ids are interleaved.
logps = jnp.array([-1.0, -2.0, -0.5, -1.5, -0.2])
rewards = jnp.array([1.0, 5.0, 3.0, 6.0, 2.0])
gids = jnp.array([7, 3, 7, 3, 7])

loss = {fn}(logps, rewards, gids)
adv = jnp.zeros_like(rewards)
for g in (3, 7):
    m = gids == g
    r = rewards[m]
    adv = adv.at[jnp.where(m)[0]].set((r - r.mean()) / (r.std() + 1e-5))
expected = -jnp.mean(adv * logps)
assert jnp.allclose(loss, expected, atol=1e-5), (
    f'{float(loss)} vs {float(expected)} — groups need not be contiguous, '
    'equal-sized, or zero-indexed'
)
""",
        },
        {
            "name": "Rewarding a response raises its log-prob",
            "code": """
import jax
import jax.numpy as jnp

# The above-average response should get a negative gradient on its logp,
# i.e. gradient descent pushes that log-prob UP.
logps = jnp.array([-1.0, -1.0])
rewards = jnp.array([10.0, 0.0])
gids = jnp.array([0, 0])

g = jax.grad({fn}, argnums=0)(logps, rewards, gids)
assert g[0] < 0 < g[1], (
    f'Expected the better response (index 0) to have a negative logp gradient '
    f'and the worse one positive, got {g}'
)
""",
        },
    ],
}
