"""Cosine decay with linear warmup — a pure function of the step counter."""

TASK = {
    "title": "Cosine LR Schedule with Warmup",
    "category": "Training",
    "order": 4,
    "difficulty": "Easy",
    "function_name": "cosine_schedule",
    "hint": (
        "Two regimes. Below warmup_steps the factor is step / warmup_steps. "
        "After it, let progress = (step - warmup) / (total - warmup) clipped to "
        "[0, 1], then min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress)). "
        "Use jnp.where rather than a Python if, so the whole thing stays "
        "vectorised and jittable when step is a traced array."
    ),
    "description": r"""
Implement the learning-rate schedule that essentially every modern transformer
is trained with: **linear warmup, then cosine decay**.

$$
\eta(t) =
\begin{cases}
\eta_{\max}\dfrac{t}{T_w} & t < T_w \\[2ex]
\eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})
\left(1 + \cos\left(\pi\dfrac{t-T_w}{T-T_w}\right)\right) & t \ge T_w
\end{cases}
$$

### Signature
```python
def cosine_schedule(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    ...  # -> learning rate at `step`
```

### Rules
- `step` may be a **scalar or an array** of steps — the function must be
  vectorised, so branch with `jnp.where`, not `if`
- Must be `jax.jit`-able with `step` traced
- Clamp past the end: for `step >= total_steps` the rate stays at `min_lr`
- Do not use `optax.warmup_cosine_decay_schedule`

### Boundary conventions this is graded on
- $\eta(0) = 0$
- $\eta(T_w) = \eta_{\max}$ exactly — warmup ends *at* the peak
- $\eta(T) = \eta_{\min}$ exactly
- the halfway point of decay is $(\eta_{\max}+\eta_{\min})/2$

### Why warmup exists
At step 0 Adam's second-moment estimate $v$ is pure noise — it has seen exactly
one gradient — so $\hat{m}/\sqrt{\hat{v}}$ is an unreliable direction with
magnitude pinned near 1. Taking full-size steps in a badly-estimated direction
is how early training diverges, and the deeper the network the worse it is.
Warmup buys the moment estimates time to become meaningful. This is also why
[[adam]]'s bias correction and warmup are usually discussed together, and why
architectures that stabilise early gradients (pre-norm) need much less warmup.

Cosine decay then matters at the *other* end: it spends a long time at a high
rate and anneals smoothly to near zero, which empirically beats step decay and,
unlike a linear ramp to zero, does not waste the final steps at a rate too
small to make progress.
""",
    "stub": '''import jax
import jax.numpy as jnp


def cosine_schedule(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    """Learning rate at `step` — linear warmup then cosine decay.

    Args:
        step:         scalar or array of step indices
        base_lr:      peak learning rate, reached at step == warmup_steps
        warmup_steps: length of the linear ramp
        total_steps:  step at which the rate reaches min_lr
        min_lr:       floor

    Returns:
        Learning rate(s), same shape as `step`.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def cosine_schedule(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    step = jnp.asarray(step, dtype=jnp.float32)

    warmup_lr = base_lr * step / jnp.maximum(warmup_steps, 1)

    # Clip so the schedule flattens at min_lr instead of turning back upward
    # once step runs past total_steps.
    decay_span = jnp.maximum(total_steps - warmup_steps, 1)
    progress = jnp.clip((step - warmup_steps) / decay_span, 0.0, 1.0)
    cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))

    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
''',
    "demo": '''import jax.numpy as jnp

steps = jnp.arange(0, 1001, 100)
lrs = cosine_schedule(steps, base_lr=1e-3, warmup_steps=100, total_steps=1000)
for s, lr in zip(steps.tolist(), lrs.tolist()):
    bar = "█" * int(lr / 1e-3 * 40)
    print(f"{s:>5}  {lr:.6f}  {bar}")
''',
    "tests": [
        {
            "name": "Boundary values",
            "code": """
import jax.numpy as jnp

base, warm, total, lo = 1e-3, 100, 1000, 1e-5

assert jnp.allclose({fn}(0, base, warm, total, lo), 0.0, atol=1e-9), (
    f'lr(0) should be 0, got {float({fn}(0, base, warm, total, lo))}'
)
assert jnp.allclose({fn}(warm, base, warm, total, lo), base, rtol=1e-6), (
    f'lr(warmup_steps) should be exactly base_lr, got '
    f'{float({fn}(warm, base, warm, total, lo))}'
)
assert jnp.allclose({fn}(total, base, warm, total, lo), lo, rtol=1e-6), (
    f'lr(total_steps) should be exactly min_lr, got '
    f'{float({fn}(total, base, warm, total, lo))}'
)
""",
        },
        {
            "name": "Linear during warmup",
            "code": """
import jax.numpy as jnp

base, warm, total = 1.0, 100, 1000
half = float({fn}(50, base, warm, total))
assert jnp.allclose(half, 0.5, rtol=1e-6), f'lr(50) should be 0.5*base, got {half}'

quarter = float({fn}(25, base, warm, total))
assert jnp.allclose(quarter, 0.25, rtol=1e-6), f'lr(25) should be 0.25*base, got {quarter}'

# Equal spacing => equal increments, i.e. genuinely linear.
d1 = float({fn}(40, base, warm, total)) - float({fn}(20, base, warm, total))
d2 = float({fn}(80, base, warm, total)) - float({fn}(60, base, warm, total))
assert jnp.allclose(d1, d2, rtol=1e-5), f'Warmup is not linear: {d1} vs {d2}'
""",
        },
        {
            "name": "Cosine shape during decay",
            "code": """
import jax.numpy as jnp

base, warm, total, lo = 1.0, 0, 1000, 0.0

mid = float({fn}(500, base, warm, total, lo))
assert jnp.allclose(mid, 0.5, atol=1e-6), (
    f'Halfway through decay should be (base+min)/2 = 0.5, got {mid}. '
    'A linear decay also gives 0.5 here — the quarter points below separate them.'
)

# cos-specific: at 25% progress the factor is 0.5*(1+cos(pi/4)) = 0.85355
q1 = float({fn}(250, base, warm, total, lo))
assert jnp.allclose(q1, 0.5 * (1 + jnp.cos(jnp.pi * 0.25)), atol=1e-5), (
    f'At 25% of decay expected 0.85355 (cosine), got {q1} — linear would give 0.75'
)
q3 = float({fn}(750, base, warm, total, lo))
assert jnp.allclose(q3, 0.5 * (1 + jnp.cos(jnp.pi * 0.75)), atol=1e-5), (
    f'At 75% of decay expected 0.14645, got {q3}'
)
""",
        },
        {
            "name": "min_lr floor and clamping past the end",
            "code": """
import jax.numpy as jnp

base, warm, total, lo = 1.0, 10, 100, 0.2

for s in (100, 150, 1000):
    v = float({fn}(s, base, warm, total, lo))
    assert jnp.allclose(v, lo, atol=1e-6), (
        f'lr({s}) should stay clamped at min_lr={lo}, got {v}. '
        'Without clipping progress to [0, 1] the cosine turns back upward.'
    )

# The floor applies to the DECAY phase. Warmup deliberately starts at 0 and
# ramps up to base_lr, so it may legitimately sit below min_lr on the way.
decay_lrs = {fn}(jnp.arange(warm, 201), base, warm, total, lo)
assert (decay_lrs >= lo - 1e-6).all(), (
    f'Decay phase went below min_lr: min={float(decay_lrs.min())}'
)

all_lrs = {fn}(jnp.arange(0, 201), base, warm, total, lo)
assert (all_lrs <= base + 1e-6).all(), (
    f'Schedule exceeded base_lr: max={float(all_lrs.max())}'
)
assert (all_lrs >= -1e-9).all(), 'Learning rate must never be negative'
""",
        },
        {
            "name": "Monotonic decay after warmup",
            "code": """
import jax.numpy as jnp

lrs = {fn}(jnp.arange(100, 1001), 1e-3, 100, 1000, 0.0)
diffs = jnp.diff(lrs)
assert (diffs <= 1e-9).all(), 'Learning rate must not increase after warmup'

warm = {fn}(jnp.arange(0, 101), 1e-3, 100, 1000, 0.0)
assert (jnp.diff(warm) >= -1e-9).all(), 'Learning rate must not decrease during warmup'
""",
        },
        {
            "name": "Vectorised and jittable",
            "code": """
import functools
import jax
import jax.numpy as jnp

steps = jnp.arange(0, 1000, 7)
out = {fn}(steps, 1e-3, 100, 1000, 1e-5)
assert out.shape == steps.shape, (
    f'Must accept an array of steps and return the same shape: '
    f'{out.shape} vs {steps.shape} — use jnp.where, not a Python if'
)

one_by_one = jnp.array([float({fn}(int(s), 1e-3, 100, 1000, 1e-5)) for s in steps])
assert jnp.allclose(out, one_by_one, atol=1e-7), 'Vectorised result differs from scalar calls'

jitted = jax.jit(functools.partial({fn}, base_lr=1e-3, warmup_steps=100,
                                    total_steps=1000, min_lr=1e-5))
assert jnp.allclose(jitted(steps), out, atol=1e-7), 'jit changes the result'
""",
        },
        {
            "name": "Zero warmup does not divide by zero",
            "code": """
import jax.numpy as jnp

v0 = float({fn}(0, 1.0, 0, 100, 0.0))
assert jnp.isfinite(v0), f'warmup_steps=0 produced {v0} — guard the division'
assert jnp.allclose(v0, 1.0, rtol=1e-6), (
    f'With no warmup, step 0 should already be at base_lr, got {v0}'
)
""",
        },
    ],
}
