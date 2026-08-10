"""Cosine decay with linear warmup — a pure function of the step counter."""

TASK = {
    "title": "Cosine LR Schedule with Warmup",
    "category": "Training",
    "order": 4,
    "difficulty": "Easy",
    "function_name": "cosine_schedule",
    "hint": (
        "Evaluate BOTH regimes unconditionally and select with jnp.where — a "
        "Python if cannot see a traced step, and it cannot handle an array of "
        "steps either. Then worry about the two denominators, because "
        "jnp.where evaluates both sides: warmup_steps may be 0, and the decay "
        "span total_steps - warmup_steps may be 0 too. Clamp the decay "
        "*progress* into [0, 1] rather than clamping the resulting rate — past "
        "the end cos turns back upward, and a min_lr floor applied afterwards "
        "only raises values, so it would never catch that."
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
- $\eta(0) = 0$ when $T_w > 0$ (and $\eta_{\max}$ when $T_w = 0$ — the ramp is empty)
- $\eta(T_w) = \eta_{\max}$ exactly — warmup ends *at* the peak
- $\eta(T) = \eta_{\min}$ exactly
- the halfway point of decay is $(\eta_{\max}+\eta_{\min})/2$

The two branches are chosen so that they **agree at the seam**: warmup's
$\eta_{\max} t / T_w$ hits $\eta_{\max}$ at $t = T_w$, and the cosine at
progress 0 is also $\eta_{\max}$, so it does not matter whether the comparison
is `<` or `<=` and the schedule is continuous either way. The off-by-one that
*does* bite is the warmup numerator: writing `(step + 1) / T_w` (or dividing by
`T_w - 1`) gives $\eta(0) \neq 0$ or overshoots the peak.

### Why warmup exists
At step 0 Adam's second-moment estimate $v$ has seen exactly one gradient, so
$\hat{m}/\sqrt{\hat{v}}$ is an unreliable *direction* whose magnitude is pinned
near 1 — full-size steps along a badly-estimated direction is how early training
diverges, and the deeper the network the worse it is. Warmup buys the moment
estimates time to become meaningful. That is also why [[adam]]'s bias correction
and warmup are usually discussed together (get the correction wrong and the
early steps are several times *too large*, which warmup then partially hides),
and why architectures that stabilise early gradients (pre-norm) need much less
warmup.

### Why cosine rather than linear
Both start at $\eta_{\max}$ and end at $\eta_{\min}$; the difference is how the
budget in between is spent. Cosine is **flatter at both ends and steeper in the
middle**: at 25% through decay it is still at $0.854\,\eta_{\max}$ where a linear
ramp is already down to $0.75$, and at 75% it is at $0.146$ where linear is still
at $0.25$. So cosine holds a near-peak rate through the early part of decay —
where most of the learning happens — and then anneals hard.

Its derivative $-\tfrac{\pi}{2}(\eta_{\max}-\eta_{\min})\sin(\pi p)$ vanishes at
both $p = 0$ and $p = 1$, so the rate leaves the peak and arrives at the floor
smoothly, with none of the loss spikes that step decay's discontinuities cause.
The cost is that the whole shape is pinned to $T$: stop early and you stop
mid-decay at a high rate, extend $T$ and every previous step was on the wrong
curve. That non-resumability is the standard interview follow-up, and it is
exactly what constant-then-decay ("WSD"-style) schedules were introduced to fix.
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
