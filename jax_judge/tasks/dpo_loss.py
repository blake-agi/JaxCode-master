"""DPO — the loss that turned RLHF into a supervised problem."""

TASK = {
    "title": "DPO (Direct Preference Optimization) Loss",
    "category": "RLHF & Preference Losses",
    "order": 1,
    "difficulty": "Medium",
    "function_name": "dpo_loss",
    "hint": (
        "Form the two log-ratios pi_chosen - ref_chosen and pi_rejected - "
        "ref_rejected, subtract them to get the margin, scale by beta, then the "
        "loss is -log_sigmoid(beta * margin) averaged over the batch. Write "
        "log-sigmoid as -softplus(-x) (or use jax.nn.log_sigmoid) — never "
        "jnp.log(jax.nn.sigmoid(x)), which underflows to -inf for margins "
        "below about -90."
    ),
    "description": r"""
Implement the **DPO** loss.

$$\mathcal{L} = -\log\sigma\!\left(\beta\Big[
\big(\log\pi_\theta(y_w|x) - \log\pi_{\text{ref}}(y_w|x)\big) -
\big(\log\pi_\theta(y_l|x) - \log\pi_{\text{ref}}(y_l|x)\big)
\Big]\right)$$

where $y_w$ is the **chosen** (preferred) completion and $y_l$ the **rejected**
one. All four inputs are already-summed sequence log-probabilities of shape
`(batch,)`.

### Signature
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    ...  # -> scalar, mean over the batch
```

### Rules
- Return the **mean** over the batch, as a scalar
- Use a numerically stable log-sigmoid — `jnp.log(sigmoid(x))` is not acceptable
- Do not detach or stop-gradient the policy terms; the reference terms are
  plain constants here (already computed under no-grad upstream)

### What the reference model is doing there
Without $\pi_{\text{ref}}$ the objective would happily drive
$\log\pi_\theta(y_w)$ to zero and $\log\pi_\theta(y_l)$ to $-\infty$ — perfect
preference accuracy, destroyed model. Subtracting the reference log-probs turns
the quantity being ranked into a *log-ratio*, and that ratio is exactly the
implicit reward $r(x,y) = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ of a
KL-constrained RLHF problem. So the KL penalty is not an extra term bolted on —
it is baked into the parameterisation. $\beta$ is the KL strength: large $\beta$
keeps you near the reference, small $\beta$ lets you drift.

### Why this replaced PPO-style RLHF
Classic RLHF needs a separately-trained reward model plus an online RL loop with
a value network ([[ppo_loss]]). DPO's insight is that for the KL-constrained
objective the optimal policy has a closed form, which can be inverted to express
the reward *in terms of the policy itself* — so the reward model cancels out and
what remains is a binary classification loss on offline preference pairs. No
sampling, no value network, no reward model.

### The trap
At initialisation $\pi_\theta = \pi_{\text{ref}}$, so every margin is exactly 0
and the loss is $-\log\sigma(0) = \log 2 \approx 0.6931$. If your first training
step does not report ~0.693, something is wrong — that constant is the standard
sanity check, and it is what the tests pin down.
""",
    "stub": '''import jax
import jax.numpy as jnp


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO loss, averaged over the batch.

    Args:
        policy_chosen_logps:   (batch,) log pi_theta(y_w | x)
        policy_rejected_logps: (batch,) log pi_theta(y_l | x)
        ref_chosen_logps:      (batch,) log pi_ref(y_w | x)
        ref_rejected_logps:    (batch,) log pi_ref(y_l | x)
        beta:                  KL strength

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # Implicit rewards, up to the shared beta factor.
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_ratio - rejected_ratio)

    # log sigmoid(x) == -softplus(-x); stable for large |x| in both directions.
    return -jnp.mean(jax.nn.log_sigmoid(logits))
''',
    "demo": '''import jax.numpy as jnp

# At init the policy IS the reference -> every margin is 0 -> loss = log 2.
z = jnp.zeros(4)
print("at init:      ", float(dpo_loss(z, z, z, z)))
print("log 2 =       ", float(jnp.log(2.0)))

# Policy prefers the chosen completion more than the reference does -> lower loss.
pc = jnp.array([0.5, 0.5, 0.5, 0.5])
print("chosen boosted:", float(dpo_loss(pc, z, z, z)))
''',
    "tests": [
        {
            "name": "log 2 at initialisation",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.key(0)
ref_c = jax.random.normal(key, (8,))
ref_r = jax.random.normal(jax.random.key(1), (8,))

# policy == reference => every margin is exactly 0
loss = {fn}(ref_c, ref_r, ref_c, ref_r, beta=0.1)

assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'
assert jnp.allclose(loss, jnp.log(2.0), atol=1e-5), (
    f'When the policy equals the reference every margin is 0, so the loss must '
    f'be -log(sigmoid(0)) = log 2 = 0.6931. Got {float(loss)}.'
)
""",
        },
        {
            "name": "Matches the closed form",
            "code": """
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(2), 4)
pc = jax.random.normal(k[0], (16,))
pr = jax.random.normal(k[1], (16,))
rc = jax.random.normal(k[2], (16,))
rr = jax.random.normal(k[3], (16,))

for beta in (0.05, 0.1, 0.5):
    logits = beta * ((pc - rc) - (pr - rr))
    expected = jnp.mean(-jax.nn.log_sigmoid(logits))
    got = {fn}(pc, pr, rc, rr, beta)
    assert jnp.allclose(got, expected, atol=1e-6), (
        f'beta={beta}: {float(got)} vs {float(expected)}'
    )
""",
        },
        {
            "name": "Reference model is actually subtracted",
            "code": """
import jax
import jax.numpy as jnp

pc = jnp.array([1.0, 1.0])
pr = jnp.array([0.0, 0.0])

# Shifting BOTH reference terms by the same constant shifts nothing:
# the margin is a difference of differences.
a = {fn}(pc, pr, jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), 0.1)
b = {fn}(pc, pr, jnp.array([5.0, 5.0]), jnp.array([5.0, 5.0]), 0.1)
assert jnp.allclose(a, b, atol=1e-6), (
    f'A constant shift applied to both reference terms must cancel: {float(a)} vs {float(b)}'
)

# But shifting only the chosen reference must change the loss.
c = {fn}(pc, pr, jnp.array([2.0, 2.0]), jnp.array([0.0, 0.0]), 0.1)
assert not jnp.allclose(a, c, atol=1e-4), (
    'Changing ref_chosen alone did not change the loss — are the reference '
    'log-probs being ignored?'
)
assert c > a, 'Raising ref_chosen lowers the implicit reward, so the loss should rise'
""",
        },
        {
            "name": "Monotone in the margin",
            "code": """
import jax.numpy as jnp

z = jnp.zeros(4)
losses = []
for m in (-2.0, -0.5, 0.0, 0.5, 2.0, 5.0):
    losses.append(float({fn}(jnp.full((4,), m), z, z, z, beta=1.0)))

for i in range(1, len(losses)):
    assert losses[i] < losses[i - 1], (
        f'Loss must decrease as the chosen margin grows, got {losses}'
    )
assert losses[-1] < 0.01, f'A margin of +5 should give a near-zero loss, got {losses[-1]}'
assert losses[0] > 1.0, f'A margin of -2 should give a large loss, got {losses[0]}'
""",
        },
        {
            "name": "beta scales the margin",
            "code": """
import jax
import jax.numpy as jnp

z = jnp.zeros(4)
pc = jnp.full((4,), 1.0)

small = float({fn}(pc, z, z, z, beta=0.01))
large = float({fn}(pc, z, z, z, beta=1.0))

assert large < small, (
    f'A larger beta amplifies a positive margin and should lower the loss: '
    f'beta=1.0 gave {large}, beta=0.01 gave {small}'
)
assert abs(small - float(jnp.log(2.0))) < 0.01, (
    f'With beta ~ 0 the margin vanishes and the loss should approach log 2, got {small}'
)
""",
        },
        {
            "name": "Numerically stable at extreme margins",
            "code": """
import jax
import jax.numpy as jnp

z = jnp.zeros(3)

big_neg = {fn}(jnp.full((3,), -500.0), z, z, z, beta=1.0)
assert jnp.isfinite(big_neg), (
    f'Got {big_neg} for a margin of -500. jnp.log(sigmoid(x)) underflows to '
    '-inf here — use log_sigmoid / -softplus(-x).'
)
assert jnp.allclose(big_neg, 500.0, rtol=1e-3), (
    f'For a very negative margin the loss should approach |margin|, got {float(big_neg)}'
)

big_pos = {fn}(jnp.full((3,), 500.0), z, z, z, beta=1.0)
assert jnp.isfinite(big_pos) and big_pos >= 0.0, f'Got {big_pos} for a margin of +500'
assert big_pos < 1e-6, f'A margin of +500 should give ~0 loss, got {float(big_pos)}'
""",
        },
        {
            "name": "Gradients flow in the right direction",
            "code": """
import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(3), 4)
pc = jax.random.normal(k[0], (8,))
pr = jax.random.normal(k[1], (8,))
rc = jax.random.normal(k[2], (8,))
rr = jax.random.normal(k[3], (8,))

g_c, g_r = jax.grad({fn}, argnums=(0, 1))(pc, pr, rc, rr, 0.1)

assert jnp.isfinite(g_c).all() and jnp.isfinite(g_r).all(), 'Non-finite gradient'
assert (g_c < 0).all(), (
    'd(loss)/d(chosen logp) must be negative — raising the chosen log-prob '
    f'should reduce the loss. Got {g_c}'
)
assert (g_r > 0).all(), (
    'd(loss)/d(rejected logp) must be positive. Got {}'.format(g_r)
)
# The two are exact mirror images of each other.
assert jnp.allclose(g_c, -g_r, atol=1e-6), 'Chosen and rejected gradients must be symmetric'

assert jnp.allclose(jax.jit({fn})(pc, pr, rc, rr, 0.1),
                    {fn}(pc, pr, rc, rr, 0.1), atol=1e-6), 'jit changes the result'
""",
        },
    ],
}
