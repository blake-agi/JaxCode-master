"""Cosine decay with linear warmup — a pure function of the step counter."""

TASK = {
    "title": "Cosine LR Scheduler with Warmup",
    "category": "Training",
    "order": 4,
    "number": "30",
    "difficulty": "Medium",
    "function_name": "cosine_lr_schedule",
    "hint": (
        "Two regimes joined at warmup_steps. Below it the factor rises linearly "
        "from 0; above it, map (step - warmup) / (total - warmup) into [0, 1] and "
        "feed that through 0.5 * (1 + cos(pi * progress)). Clip the progress or "
        "the cosine turns back upward past total_steps. Use jnp.where rather than "
        "a Python if so the schedule stays vectorised and jittable."
    ),
    "description": r"""
Implement the learning-rate schedule essentially every modern transformer is
trained with: **linear warmup, then cosine decay**.

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
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    ...
```

Note the argument order — `total_steps` comes **before** `warmup_steps` and
`max_lr`.

### Rules
- Do not use `optax.warmup_cosine_decay_schedule`
- `step` may be a scalar **or an array** of steps, so branch with `jnp.where`,
  not a Python `if`
- Must be `jax.jit`-able with `step` traced
- Past `total_steps` the rate stays clamped at `min_lr`

### Boundary conventions this is graded on
- $\eta(0) = 0$
- $\eta(T_w) = \eta_{\max}$ exactly — warmup ends *at* the peak
- $\eta(T) = \eta_{\min}$ exactly
- the halfway point of decay is $(\eta_{\max}+\eta_{\min})/2$

### Why warmup exists
At step 0 Adam's second-moment estimate has seen exactly one gradient, so
$\hat{m}/\sqrt{\hat{v}}$ is an unreliable direction with magnitude pinned near 1.
Taking full-size steps in a badly-estimated direction is how early training
diverges, and the deeper the network the worse it gets. Warmup buys the moment
estimates time to become meaningful.

Cosine decay then matters at the other end: it holds a high rate for a long
time and anneals smoothly to near zero, which beats step decay empirically and,
unlike a linear ramp, does not waste the final steps at a rate too small to
make progress.

### A JAX note
The PyTorch original branches with a Python `if` and returns a float, which is
fine for a scalar step. Writing it with `jnp.where` instead costs nothing, and
buys you a schedule that works on a whole array of steps at once and survives
`jit` when the step is traced.
""",
    "stub": '''import jax
import jax.numpy as jnp


def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    """Learning rate at `step` — linear warmup then cosine decay.

    Args:
        step:         scalar or array of step indices
        total_steps:  step at which the rate reaches min_lr
        warmup_steps: length of the linear ramp
        max_lr:       peak rate, reached at step == warmup_steps
        min_lr:       floor

    Returns:
        Learning rate(s), same shape as `step`.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    step = jnp.asarray(step, dtype=jnp.float32)

    warmup_lr = max_lr * step / jnp.maximum(warmup_steps, 1)

    # Clip so the schedule flattens at min_lr instead of turning back upward
    # once step runs past total_steps.
    decay_span = jnp.maximum(total_steps - warmup_steps, 1)
    progress = jnp.clip((step - warmup_steps) / decay_span, 0.0, 1.0)
    cosine_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))

    # jnp.where, not a Python if — keeps this vectorised and jittable.
    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
''',
    "demo": '''import jax.numpy as jnp

steps = jnp.arange(0, 1001, 100)
lrs = cosine_lr_schedule(steps, total_steps=1000, warmup_steps=100, max_lr=1e-3)
for s, lr in zip(steps.tolist(), lrs.tolist()):
    bar = "█" * int(lr / 1e-3 * 40)
    print(f"{s:>5}  {lr:.6f}  {bar}")
''',
    "tests": [
        {
            "name": "Boundary values",
            "code": """
import jax.numpy as jnp

total, warm, mx, lo = 1000, 100, 1e-3, 1e-5

assert jnp.allclose({fn}(0, total, warm, mx, lo), 0.0, atol=1e-9), (
    f'lr(0) should be 0, got {float({fn}(0, total, warm, mx, lo))}'
)
assert jnp.allclose({fn}(warm, total, warm, mx, lo), mx, rtol=1e-6), (
    f'lr(warmup_steps) should be exactly max_lr, got {float({fn}(warm, total, warm, mx, lo))}'
)
assert jnp.allclose({fn}(total, total, warm, mx, lo), lo, rtol=1e-6), (
    f'lr(total_steps) should be exactly min_lr, got {float({fn}(total, total, warm, mx, lo))}'
)
""",
        },
        {
            "name": "Linear during warmup",
            "code": """
import jax.numpy as jnp

total, warm, mx = 1000, 100, 1.0
half = float({fn}(50, total, warm, mx))
assert jnp.allclose(half, 0.5, rtol=1e-6), f'lr(50) should be 0.5*max_lr, got {half}'

quarter = float({fn}(25, total, warm, mx))
assert jnp.allclose(quarter, 0.25, rtol=1e-6), f'lr(25) should be 0.25*max_lr, got {quarter}'

d1 = float({fn}(40, total, warm, mx)) - float({fn}(20, total, warm, mx))
d2 = float({fn}(80, total, warm, mx)) - float({fn}(60, total, warm, mx))
assert jnp.allclose(d1, d2, rtol=1e-5), f'Warmup is not linear: {d1} vs {d2}'
""",
        },
        {
            "name": "Cosine shape during decay",
            "code": """
import jax.numpy as jnp

total, warm, mx, lo = 1000, 0, 1.0, 0.0

mid = float({fn}(500, total, warm, mx, lo))
assert jnp.allclose(mid, 0.5, atol=1e-6), (
    f'Halfway through decay should be (max+min)/2 = 0.5, got {mid}'
)

# cos-specific: linear decay would give 0.75 and 0.25 at these points.
q1 = float({fn}(250, total, warm, mx, lo))
assert jnp.allclose(q1, 0.5 * (1 + jnp.cos(jnp.pi * 0.25)), atol=1e-5), (
    f'At 25% of decay expected 0.85355 (cosine), got {q1} — linear would give 0.75'
)
q3 = float({fn}(750, total, warm, mx, lo))
assert jnp.allclose(q3, 0.5 * (1 + jnp.cos(jnp.pi * 0.75)), atol=1e-5), (
    f'At 75% of decay expected 0.14645, got {q3}'
)
""",
        },
        {
            "name": "min_lr floor and clamping past the end",
            "code": """
import jax.numpy as jnp

total, warm, mx, lo = 100, 10, 1.0, 0.2

for s in (100, 150, 1000):
    v = float({fn}(s, total, warm, mx, lo))
    assert jnp.allclose(v, lo, atol=1e-6), (
        f'lr({s}) should stay clamped at min_lr={lo}, got {v}. Without clipping '
        'progress to [0, 1] the cosine turns back upward.'
    )

# The floor applies to the DECAY phase; warmup deliberately starts at 0.
decay = {fn}(jnp.arange(warm, 201), total, warm, mx, lo)
assert (decay >= lo - 1e-6).all(), f'Decay went below min_lr: {float(decay.min())}'

allv = {fn}(jnp.arange(0, 201), total, warm, mx, lo)
assert (allv <= mx + 1e-6).all(), f'Schedule exceeded max_lr: {float(allv.max())}'
assert (allv >= -1e-9).all(), 'Learning rate must never be negative'
""",
        },
        {
            "name": "Monotonic decay after warmup",
            "code": """
import jax.numpy as jnp

lrs = {fn}(jnp.arange(100, 1001), 1000, 100, 1e-3, 0.0)
assert (jnp.diff(lrs) <= 1e-9).all(), 'Learning rate must not increase after warmup'

warm = {fn}(jnp.arange(0, 101), 1000, 100, 1e-3, 0.0)
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
out = {fn}(steps, 1000, 100, 1e-3, 1e-5)
assert out.shape == steps.shape, (
    f'Must accept an array of steps and return the same shape: {out.shape} vs '
    f'{steps.shape} — use jnp.where, not a Python if'
)

one_by_one = jnp.array([float({fn}(int(s), 1000, 100, 1e-3, 1e-5)) for s in steps])
assert jnp.allclose(out, one_by_one, atol=1e-7), 'Vectorised result differs from scalar calls'

jitted = jax.jit(functools.partial({fn}, total_steps=1000, warmup_steps=100,
                                    max_lr=1e-3, min_lr=1e-5))
assert jnp.allclose(jitted(steps), out, atol=1e-7), 'jit changes the result'
""",
        },
        {
            "name": "Zero warmup does not divide by zero",
            "code": """
import jax.numpy as jnp

v0 = float({fn}(0, 100, 0, 1.0, 0.0))
assert jnp.isfinite(v0), f'warmup_steps=0 produced {v0} — guard the division'
assert jnp.allclose(v0, 1.0, rtol=1e-6), (
    f'With no warmup, step 0 should already be at max_lr, got {v0}'
)
""",
        },
    ],
}
