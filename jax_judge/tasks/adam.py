"""Adam from scratch — optax-style init/update, and what bias correction fixes."""

TASK = {
    "title": "Implement Adam Optimizer",
    "category": "Training",
    "order": 3,
    "number": "29",
    "difficulty": "Medium",
    "function_name": "MyAdam",
    "hint": (
        "init() builds two zero trees, m and v, plus a step counter. update() "
        "does the EMAs, the bias correction, and the parameter step in one "
        "jax.tree.map over (params, grads, m, v). The step counter is 1-based on "
        "the first update, which is what makes the correction cancel exactly. "
        "Nothing is mutated: update returns the new params and the new state."
    ),
    "description": r"""
Implement the **Adam** optimizer.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad
  \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Signature
```python
class MyAdam:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8): ...
    def init(self, params): ...                    # -> state
    def update(self, params, grads, state): ...    # -> (new_params, new_state)
```

### Rules
- Do **not** use `optax`
- `params` and `grads` are matching **pytrees** — use `jax.tree.map`, do not
  assume a flat array
- The step counter is **1-based**: the first `update` uses $t=1$
- Must work under `jax.jit`

### What bias correction actually fixes
$m$ and $v$ start at **zero**, so at $t=1$, $m_1 = (1-\beta_1)g_1 = 0.1g_1$ —
a tenth of the true gradient. Without correction the first steps are far too
small, and with $\beta_2 = 0.999$ the second-moment estimate takes thousands of
steps to warm up.

The correction has a sharp observable signature: **with** it, the very first
update has magnitude $\approx \eta$ regardless of the gradient's size, since
$\hat{m}_1/\sqrt{\hat{v}_1} = g/|g| = \pm 1$. That is exactly what the tests
check, and the cleanest way to tell a correct Adam from one missing it.

### Where AdamW differs
AdamW does **not** fold weight decay into the gradient. It applies
$\theta \mathrel{-}= \eta\lambda\theta$ separately, so the decay is not scaled
by $\sqrt{\hat{v}}$. Plain "Adam + L2" decays large-gradient parameters *less*,
which is why AdamW generalises better and is the default for transformers.

### ⚠️ Why init/update instead of step()/zero_grad()
The PyTorch original holds the parameters, reads `p.grad`, and mutates `p` in
place inside `step()`. None of that translates: JAX arrays are immutable, there
is no `.grad` attribute, and a method with hidden mutable state cannot be
`jit`-ed or differentiated through.

So `MyAdam` keeps its name but takes the shape every JAX optimizer has —
`init(params) -> state` and `update(params, grads, state) -> (params, state)`,
the same contract as `optax.adam`. There is also no `zero_grad`: `jax.grad`
returns a fresh gradient tree each call, so gradients never accumulate and the
"forgot to zero" bug cannot be written.
""",
    "stub": '''import jax
import jax.numpy as jnp


class MyAdam:
    """Adam optimizer, optax-style: pure init/update over pytrees."""

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        pass  # Replace this

    def init(self, params):
        """Build the initial optimizer state for a parameter pytree."""
        pass  # Replace this

    def update(self, params, grads, state):
        """One Adam step. Returns (new_params, new_state)."""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


class MyAdam:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

    def init(self, params):
        # Zero-initialised moments — which is exactly why bias correction is needed.
        return {
            "m": jax.tree.map(jnp.zeros_like, params),
            "v": jax.tree.map(jnp.zeros_like, params),
            "t": 0,
        }

    def update(self, params, grads, state):
        t = state["t"] + 1                      # 1-based on the first update

        m = jax.tree.map(
            lambda m_, g: self.beta1 * m_ + (1 - self.beta1) * g, state["m"], grads
        )
        v = jax.tree.map(
            lambda v_, g: self.beta2 * v_ + (1 - self.beta2) * g * g, state["v"], grads
        )

        mc = 1 - self.beta1 ** t
        vc = 1 - self.beta2 ** t

        new_params = jax.tree.map(
            lambda p, m_, v_: p - self.lr * (m_ / mc) / (jnp.sqrt(v_ / vc) + self.eps),
            params, m, v,
        )
        return new_params, {"m": m, "v": v, "t": t}
''',
    "demo": '''import jax
import jax.numpy as jnp

opt = MyAdam(lr=0.1)
params = {"w": jnp.array([1.0, -2.0])}
state = opt.init(params)

# Wildly different gradient magnitudes both give a first step of ~lr.
for g in (jnp.array([1e-4, 1e-4]), jnp.array([1e4, 1e4])):
    p, _ = opt.update(params, {"w": g}, state)
    print(f"grad {g[0]:>8.0e} -> step {abs(float(p['w'][0] - 1.0)):.4f}")
print("\\nThat invariance is bias correction doing its job.")
''',
    "tests": [
        {
            "name": "Matches an independent reference",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, -2.0, 0.5]), "b": jnp.array(0.25)}
grads = {"w": jnp.array([0.1, -0.3, 2.0]), "b": jnp.array(-1.5)}

lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
opt = {fn}(lr=lr, betas=(b1, b2), eps=eps)
state = opt.init(params)
p, s = opt.update(params, grads, state)

for k in ("w", "b"):
    g = grads[k]
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    ref = params[k] - lr * (m / (1 - b1)) / (jnp.sqrt(v / (1 - b2)) + eps)
    assert jnp.allclose(p[k], ref, atol=1e-6), f'param {k}: {p[k]} vs {ref}'
    assert jnp.allclose(s["m"][k], m, atol=1e-7), f'state m[{k}] wrong'
    assert jnp.allclose(s["v"][k], v, atol=1e-7), f'state v[{k}] wrong'
assert s["t"] == 1, f'Step counter should be 1 after one update, got {s["t"]}'
""",
        },
        {
            "name": "Bias correction: first step is ~lr at any gradient scale",
            "code": """
import jax
import jax.numpy as jnp

lr = 0.01
opt = {fn}(lr=lr)
params = jnp.array([0.0, 0.0])
state = opt.init(params)

for scale in (1e-5, 1.0, 1e5):
    g = jnp.array([scale, -scale])
    p, _ = opt.update(params, g, state)
    mag = jnp.abs(p)
    assert jnp.allclose(mag, lr, rtol=1e-3), (
        f'First step with gradient {scale:g} moved {float(mag[0]):.3g}, expected '
        f'~{lr}. Adam normalises by sqrt(v_hat), so step 1 is ~lr for ANY '
        'gradient magnitude — this almost always means bias correction is missing.'
    )
    assert p[0] < 0 and p[1] > 0, 'Step must go downhill'
""",
        },
        {
            "name": "State is threaded, not mutated",
            "code": """
import jax
import jax.numpy as jnp

opt = {fn}(lr=0.01)
params = jnp.zeros(3)
state = opt.init(params)
m_before = state["m"].copy()

p1, s1 = opt.update(params, jnp.ones(3), state)

assert jnp.allclose(state["m"], m_before), 'The state passed in must not be mutated'
assert state["t"] == 0, f'The original state t changed to {state["t"]}'
assert s1["t"] == 1, 'The RETURNED state should carry the incremented counter'
assert jnp.allclose(params, 0.0), 'The params passed in must not be mutated'
""",
        },
        {
            "name": "Moments accumulate across steps",
            "code": """
import jax
import jax.numpy as jnp

b1, b2 = 0.9, 0.999
opt = {fn}(lr=0.01, betas=(b1, b2))
params = jnp.zeros(1)
state = opt.init(params)
g = jnp.array([2.0])

m_ref = v_ref = 0.0
for t in (1, 2, 3, 4):
    params, state = opt.update(params, g, state)
    m_ref = b1 * m_ref + (1 - b1) * 2.0
    v_ref = b2 * v_ref + (1 - b2) * 4.0
    assert jnp.allclose(state["m"], m_ref, atol=1e-6), (
        f'step {t}: m={float(state["m"][0]):.6f} vs {m_ref:.6f} — is the returned '
        'state being fed back in?'
    )
    assert state["t"] == t, f'Step counter should be {t}, got {state["t"]}'
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

opt = {fn}()
state = opt.init(params)
p, s = opt.update(params, grads, state)

assert jax.tree.structure(p) == jax.tree.structure(params), 'params structure changed'
assert jax.tree.structure(s["m"]) == jax.tree.structure(params), 'state m structure changed'
for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(params)):
    assert a.shape == b.shape, f'leaf shape changed: {a.shape} vs {b.shape}'
""",
        },
        {
            "name": "Converges on a quadratic",
            "code": """
import jax
import jax.numpy as jnp

target = jnp.array([3.0, -1.0, 0.5])
params = jnp.zeros(3)
opt = {fn}(lr=0.05)
state = opt.init(params)

for _ in range(800):
    g = 2 * (params - target)
    params, state = opt.update(params, g, state)

assert jnp.allclose(params, target, atol=1e-2), f'Did not converge: {params} vs {target}'
""",
        },
        {
            "name": "Hyperparameters used, and jit works",
            "code": """
import jax
import jax.numpy as jnp

params = jnp.zeros(1)
g = jnp.array([1.0])

a = {fn}(lr=0.01); pa, _ = a.update(params, g, a.init(params))
b = {fn}(lr=0.02); pb, _ = b.update(params, g, b.init(params))
assert jnp.allclose(jnp.abs(pb), 2 * jnp.abs(pa), rtol=1e-3), 'lr is not scaling the step'

c = {fn}(lr=0.01, betas=(0.5, 0.999)); _, sc = c.update(params, g, c.init(params))
assert jnp.allclose(sc["m"], 0.5, atol=1e-6), f'beta1 unused: m={sc["m"]}'
d = {fn}(lr=0.01, betas=(0.9, 0.5)); _, sd = d.update(params, g, d.init(params))
assert jnp.allclose(sd["v"], 0.5, atol=1e-6), f'beta2 unused: v={sd["v"]}'

opt = {fn}(lr=0.01)
st = opt.init(params)
p_j, s_j = jax.jit(opt.update)(params, g, st)
p_e, s_e = opt.update(params, g, st)
assert jnp.allclose(p_j, p_e, atol=1e-6), 'jit changes the result'
""",
        },
    ],
}
