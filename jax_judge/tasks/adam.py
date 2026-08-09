"""Adam from scratch over a pytree — and what bias correction actually fixes."""

TASK = {
    "title": "Adam Optimizer",
    "category": "Training",
    "order": 3,
    "difficulty": "Medium",
    "function_name": "adam_update",
    "hint": (
        "Two running averages, both initialised at zero: m tracks the gradient, "
        "v tracks its square. Because they start at zero they are biased toward "
        "zero early on, so divide by (1 - b1**t) and (1 - b2**t) before using "
        "them — t is 1-based on the first step. Everything is elementwise, so "
        "jax.tree.map over (params, grads, m, v) does the whole update at once."
    ),
    "description": r"""
Implement the **Adam** update as a pure function over pytrees.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad
  \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Signature
```python
def adam_update(params, grads, state, step, lr=1e-3,
                b1=0.9, b2=0.999, eps=1e-8):
    # state: {"m": pytree_like_params, "v": pytree_like_params}
    # step:  1-based — the first call has step=1
    # returns (new_params, new_state)
```

### Rules
- Do **not** use `optax`
- `params`, `grads`, `m` and `v` are all the same pytree structure — use
  `jax.tree.map`, do not assume a flat array
- `state` starts as all-zeros `m` and `v`
- Must work under `jax.jit`

### What bias correction actually fixes
This is the part people get wrong. $m$ and $v$ start at **zero**, so at $t=1$,
$m_1 = (1-\beta_1) g_1 = 0.1 g_1$ — a tenth of the true gradient. Without
correction the first steps are far too small, and because $\beta_2 = 0.999$ the
second-moment estimate takes *thousands* of steps to warm up.

The correction has a sharp observable signature: **with** it, the very first
update has magnitude $\approx \eta$ regardless of how large or small the
gradient is (since $\hat{m}_1/\sqrt{\hat{v}_1} = g/|g| = \pm 1$). That property
is exactly what the tests check, and it is the cleanest way to tell a correct
Adam from one missing the correction.

### Where AdamW differs
AdamW does **not** add weight decay to the gradient. It applies
$\theta \mathrel{-}= \eta \lambda \theta$ separately, so the decay is not scaled
by $\sqrt{\hat{v}}$. Folding L2 into the gradient instead — plain "Adam + L2" —
decays large-gradient parameters *less*, which is why AdamW generalises better
and is the default for transformers.
""",
    "stub": '''import jax
import jax.numpy as jnp


def adam_update(params, grads, state, step, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step.

    Args:
        params: pytree of parameters
        grads:  pytree of gradients, same structure as params
        state:  {"m": pytree, "v": pytree} — both zeros on the first step
        step:   1-based step counter (first call is 1)

    Returns:
        (new_params, new_state)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def adam_update(params, grads, state, step, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, state["v"], grads)

    # step is 1-based, so at the first call the denominators are (1 - b1) and
    # (1 - b2) — exactly cancelling the shrinkage of the zero-initialised means.
    mc = 1 - b1 ** step
    vc = 1 - b2 ** step

    new_params = jax.tree.map(
        lambda p, m_, v_: p - lr * (m_ / mc) / (jnp.sqrt(v_ / vc) + eps),
        params, m, v,
    )
    return new_params, {"m": m, "v": v}
''',
    "demo": '''import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, -2.0])}
state = {"m": jax.tree.map(jnp.zeros_like, params),
         "v": jax.tree.map(jnp.zeros_like, params)}

# Wildly different gradient magnitudes...
for g in (jnp.array([1e-4, 1e-4]), jnp.array([1e4, 1e4])):
    p, _ = adam_update(params, {"w": g}, state, step=1, lr=0.1)
    print(f"grad {g[0]:>8.0e} -> step size {abs(float(p['w'][0] - 1.0)):.4f}")
# ...both give a first step of ~lr. That is bias correction doing its job.
''',
    "tests": [
        {
            "name": "Matches an independent reference",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, -2.0, 0.5]), "b": jnp.array(0.25)}
grads = {"w": jnp.array([0.1, -0.3, 2.0]), "b": jnp.array(-1.5)}
state = {"m": jax.tree.map(jnp.zeros_like, params),
         "v": jax.tree.map(jnp.zeros_like, params)}

lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
p, s = {fn}(params, grads, state, 1, lr, b1, b2, eps)

for k in ("w", "b"):
    g = grads[k]
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    ref = params[k] - lr * (m / (1 - b1 ** 1)) / (jnp.sqrt(v / (1 - b2 ** 1)) + eps)
    assert jnp.allclose(p[k], ref, atol=1e-6), f'param {k}: {p[k]} vs {ref}'
    assert jnp.allclose(s["m"][k], m, atol=1e-7), f'state m[{k}] wrong'
    assert jnp.allclose(s["v"][k], v, atol=1e-7), f'state v[{k}] wrong'
""",
        },
        {
            "name": "Bias correction: first step is ~lr at any gradient scale",
            "code": """
import jax
import jax.numpy as jnp

lr = 0.01
params = jnp.array([0.0, 0.0])
state = {"m": jnp.zeros(2), "v": jnp.zeros(2)}

for scale in (1e-5, 1.0, 1e5):
    g = jnp.array([scale, -scale])
    p, _ = {fn}(params, g, state, 1, lr)
    mag = jnp.abs(p)
    assert jnp.allclose(mag, lr, rtol=1e-3), (
        f'First step with gradient {scale:g} moved {float(mag[0]):.3g}, expected ~{lr}. '
        'Adam normalises by sqrt(v_hat), so step 1 should be ~lr for ANY gradient '
        'magnitude — this almost always means bias correction is missing.'
    )
    assert p[0] < 0 and p[1] > 0, 'Step must go downhill (opposite the gradient)'
""",
        },
        {
            "name": "Nested pytree structure preserved",
            "code": """
import jax
import jax.numpy as jnp

params = {"l1": {"w": jnp.ones((3, 2)), "b": jnp.zeros(2)},
          "l2": [jnp.full((4,), 0.5), jnp.array(1.0)]}
grads = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
state = {"m": jax.tree.map(jnp.zeros_like, params),
         "v": jax.tree.map(jnp.zeros_like, params)}

p, s = {fn}(params, grads, state, 1)

assert jax.tree.structure(p) == jax.tree.structure(params), 'params structure changed'
assert jax.tree.structure(s["m"]) == jax.tree.structure(params), 'state m structure changed'
assert jax.tree.structure(s["v"]) == jax.tree.structure(params), 'state v structure changed'
for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(params)):
    assert a.shape == b.shape, f'leaf shape changed: {a.shape} vs {b.shape}'
""",
        },
        {
            "name": "Moments accumulate across steps",
            "code": """
import jax
import jax.numpy as jnp

params = jnp.zeros(1)
state = {"m": jnp.zeros(1), "v": jnp.zeros(1)}
g = jnp.array([2.0])
b1, b2 = 0.9, 0.999

m_ref, v_ref = 0.0, 0.0
for t in (1, 2, 3, 4):
    params, state = {fn}(params, g, state, t, 0.01, b1, b2)
    m_ref = b1 * m_ref + (1 - b1) * 2.0
    v_ref = b2 * v_ref + (1 - b2) * 4.0
    assert jnp.allclose(state["m"], m_ref, atol=1e-6), (
        f'step {t}: m={float(state["m"][0]):.6f} vs {m_ref:.6f} — is the '
        'returned state being fed back in?'
    )
    assert jnp.allclose(state["v"], v_ref, atol=1e-6), f'step {t}: v mismatch'
""",
        },
        {
            "name": "Converges on a quadratic",
            "code": """
import jax
import jax.numpy as jnp

# minimise sum((p - target)^2)
target = jnp.array([3.0, -1.0, 0.5])
params = jnp.zeros(3)
state = {"m": jnp.zeros(3), "v": jnp.zeros(3)}

for t in range(1, 801):
    g = 2 * (params - target)
    params, state = {fn}(params, g, state, t, 0.05)

assert jnp.allclose(params, target, atol=1e-2), (
    f'Did not converge: {params} vs {target}'
)
""",
        },
        {
            "name": "Hyperparameters are actually used",
            "code": """
import jax
import jax.numpy as jnp

params = jnp.zeros(1)
state = {"m": jnp.zeros(1), "v": jnp.zeros(1)}
g = jnp.array([1.0])

p_a, _ = {fn}(params, g, state, 1, 0.01)
p_b, _ = {fn}(params, g, state, 1, 0.02)
assert jnp.allclose(jnp.abs(p_b), 2 * jnp.abs(p_a), rtol=1e-3), 'lr is not scaling the step'

_, s1 = {fn}(params, g, state, 1, 0.01, 0.5, 0.999)
assert jnp.allclose(s1["m"], 0.5, atol=1e-6), f'b1 unused: m={s1["m"]}'
_, s2 = {fn}(params, g, state, 1, 0.01, 0.9, 0.5)
assert jnp.allclose(s2["v"], 0.5, atol=1e-6), f'b2 unused: v={s2["v"]}'
""",
        },
        {
            "name": "Works under jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0])}
grads = {"w": jnp.array([0.5, -0.5])}
state = {"m": jax.tree.map(jnp.zeros_like, params),
         "v": jax.tree.map(jnp.zeros_like, params)}

jitted = jax.jit(functools.partial({fn}, lr=0.01))
p_j, s_j = jitted(params, grads, state, 1)
p_e, s_e = {fn}(params, grads, state, 1, 0.01)

assert jnp.allclose(p_j["w"], p_e["w"], atol=1e-6), 'jit changes the result'
assert jnp.allclose(s_j["m"]["w"], s_e["m"]["w"], atol=1e-7), 'jit changes the state'
""",
        },
    ],
}
