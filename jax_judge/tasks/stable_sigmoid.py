"""Sigmoid where the forward pass is easy and the backward pass is not."""

TASK = {
    "title": "Numerically Stable Sigmoid",
    "category": "Core Ops & Layers",
    "number": "b_22",
    "difficulty": "Medium",
    "function_name": "stable_sigmoid",
    "extra_names": ["stable_log_sigmoid"],
    "hint": (
        "The trick is to never call exp on a positive number. For sigmoid, "
        "let z = exp(-|x|) — always in (0, 1] — and select: 1/(1+z) when "
        "x >= 0, z/(1+z) when x < 0. (0.5 * (1 + tanh(x/2)) also works and is "
        "shorter, since tanh is already stable.) For log-sigmoid, "
        "log(sigmoid(x)) = -log(1 + exp(-x)) = -logaddexp(0, -x), and "
        "jnp.logaddexp is stable by construction. Beware the obvious fix: "
        "jnp.where(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x))) evaluates BOTH "
        "branches, so each one overflows at one end and you get NaN gradients "
        "at both."
    ),
    "description": r"""
Implement sigmoid and log-sigmoid so that they survive large inputs — in the
**backward** pass, which is where the naive versions actually break.

$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad
\log\sigma(x) = -\log\!\left(1+e^{-x}\right)$$

### Signature
```python
def stable_sigmoid(x): ...        # elementwise, same shape and dtype as x
def stable_log_sigmoid(x): ...    # elementwise, same shape and dtype as x
```

Both must work on arrays of any shape. Do not call `jax.nn.sigmoid`,
`jax.nn.log_sigmoid` or `jax.scipy.special.expit` — implementing them is the
exercise.

### The forward pass is a red herring
Write `1 / (1 + jnp.exp(-x))` and evaluate it at `x = -800`. In float32
`exp(800)` is `inf`, and `1 / (1 + inf)` is `0.0` — which is the *right answer*.
The naive sigmoid looks completely fine:

```
x       [-800,  -50,     0,      50,   800]
naive   [   0,  1.9e-22, 0.5,     1,     1]     ✅ matches jax.nn.sigmoid exactly
```

Now differentiate it. The chain rule has to divide by that `inf`:

```
grad naive    [ nan,  0,  0.25,  1.9e-22,  0]      ← NaN at x = -800
```

### The obvious fix makes it worse
The textbook branch — one form for each sign — looks like it solves this:

```python
jnp.where(x >= 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x)))
```

`jnp.where` is not a branch. **Both sides are evaluated** and then selected
between, so `exp(-x)` still overflows for very negative `x` and `exp(x)` still
overflows for very positive `x`. The forward pass is saved by the select, but
the gradient is not:

```
grad where    [ nan,  1.9e-22,  0.25,  1.9e-22,  nan]     ← now NaN at BOTH ends
```

This is the general trap: `jnp.where(c, f(x), g(x))` computes `g(x)` even where
`c` is true, so a NaN or inf in the *unused* branch still reaches the backward
pass.

### What actually works
Make the exponent never positive, so nothing can overflow in the first place:

```python
z = jnp.exp(-jnp.abs(x))            # always in (0, 1]
jnp.where(x >= 0, 1 / (1 + z), z / (1 + z))
```

Both branches are now finite everywhere, so selecting between them is safe.

For log-sigmoid the naive form is worse still — it breaks in the **forward**
pass, returning `-inf` where the answer is simply `-800`:

```
x               [-800,  -50,   0,      50]
log(sigmoid(x)) [-inf,  -50,  -0.693,   0]     ❌
correct         [-800,  -50,  -0.693,  -1.9e-22]
```

`jnp.logaddexp` already handles this, and $\log\sigma(x) = -\log(1+e^{-x})$.

### Why it matters
`log_sigmoid` is the core of every pairwise preference loss — DPO is
`-log σ(β(Δ_policy - Δ_ref))`. Those logits are unbounded, so a naive
implementation trains fine until one batch produces a large margin and the loss
becomes `inf` or the gradient becomes `NaN`. The model does not recover.
""",
    "stub": '''import jax
import jax.numpy as jnp


def stable_sigmoid(x):
    """Elementwise sigmoid, finite in both value and gradient for any input."""
    pass  # Replace this


def stable_log_sigmoid(x):
    """Elementwise log(sigmoid(x)), finite in both value and gradient."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def stable_sigmoid(x):
    # exp is only ever called on -|x| <= 0, so z is in (0, 1] and nothing
    # overflows. Both branches are finite everywhere, which is what makes the
    # select safe in the BACKWARD pass too.
    z = jnp.exp(-jnp.abs(x))
    return jnp.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    # 0.5 * (1.0 + jnp.tanh(x / 2.0)) is equally stable and shorter.


def stable_log_sigmoid(x):
    # log(sigmoid(x)) = -log(1 + exp(-x)) = -logaddexp(0, -x).
    # logaddexp factors out the max internally, so it never overflows and
    # never underflows to -inf the way log(1/(1+exp(-x))) does.
    return -jnp.logaddexp(jnp.zeros_like(x), -x)
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.array([-800.0, -50.0, 0.0, 50.0, 800.0])

naive = lambda v: 1.0 / (1.0 + jnp.exp(-v))
branch = lambda v: jnp.where(v >= 0, 1.0 / (1.0 + jnp.exp(-v)),
                             jnp.exp(v) / (1.0 + jnp.exp(v)))

print("x                ", x)
print("naive   value    ", naive(x))
print("yours   value    ", stable_sigmoid(x))
print("-- the forward pass does not distinguish them --\\n")

for name, f in (("naive ", naive), ("where ", branch), ("yours ", stable_sigmoid)):
    g = jax.grad(lambda v: jnp.sum(f(v)))(x)
    print(f"grad {name}      {g}")

print("\\nlog-sigmoid, where the FORWARD pass already breaks:")
print("  log(sigmoid(x))", jnp.log(naive(x)))
print("  yours          ", stable_log_sigmoid(x))
''',
    "tests": [
        {
            "name": "Matches the reference on ordinary inputs",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.linspace(-8.0, 8.0, 33)
s = {fn}(x)
assert s.shape == x.shape, f'{s.shape} vs {x.shape}'
assert jnp.allclose(s, jax.nn.sigmoid(x), atol=1e-6), 'Disagrees with jax.nn.sigmoid'

ls = stable_log_sigmoid(x)
assert ls.shape == x.shape, f'{ls.shape} vs {x.shape}'
assert jnp.allclose(ls, jax.nn.log_sigmoid(x), atol=1e-6), 'Disagrees with jax.nn.log_sigmoid'

# Defining identities.
assert jnp.allclose({fn}(x) + {fn}(-x), 1.0, atol=1e-6), 'sigmoid(x) + sigmoid(-x) != 1'
assert jnp.allclose(jnp.exp(stable_log_sigmoid(x)), {fn}(x), atol=1e-6), (
    'exp(log_sigmoid(x)) should equal sigmoid(x)'
)

# Any shape, elementwise.
X = jax.random.normal(jax.random.key(0), (2, 3, 4)) * 5
assert {fn}(X).shape == (2, 3, 4), f'{ {fn}(X).shape }'
assert jnp.allclose({fn}(X), jax.nn.sigmoid(X), atol=1e-6), 'N-D case disagrees'
""",
        },
        {
            "name": "Values stay finite and correct at extreme inputs",
            "code": """
import jax.numpy as jnp

x = jnp.array([-800.0, -100.0, 0.0, 100.0, 800.0])

s = {fn}(x)
assert jnp.isfinite(s).all(), f'Non-finite sigmoid values: {s}'
assert jnp.all((s >= 0.0) & (s <= 1.0)), f'sigmoid left [0, 1]: {s}'
assert float(s[0]) == 0.0 and float(s[-1]) == 1.0, (
    f'Saturating ends should be exactly 0 and 1, got {s[0]}, {s[-1]}'
)
assert abs(float(s[2]) - 0.5) < 1e-6, f'sigmoid(0) should be 0.5, got {s[2]}'

ls = stable_log_sigmoid(x)
assert jnp.isfinite(ls).all(), (
    f'Non-finite log-sigmoid: {ls}. log(1/(1+exp(-x))) underflows to -inf for '
    'very negative x — use -logaddexp(0, -x), whose answer there is just x.'
)
assert jnp.all(ls <= 0.0), f'log-sigmoid must be <= 0, got {ls}'
# For x << 0, log sigmoid(x) -> x.
assert abs(float(ls[0]) - (-800.0)) < 1e-3, f'log_sigmoid(-800) should be ~-800, got {ls[0]}'
""",
        },
        {
            "name": "Gradients are finite at both extremes",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([-800.0, -100.0, -1.0, 0.0, 1.0, 100.0, 800.0])

g = jax.grad(lambda v: jnp.sum({fn}(v)))(x)
assert jnp.isfinite(g).all(), (
    f'Non-finite sigmoid gradient: {g}\\n'
    'The naive 1/(1+exp(-x)) gives NaN at very negative x, and the two-branch '
    'jnp.where version gives NaN at BOTH ends because where evaluates both '
    'sides. Keep the exponent non-positive: z = exp(-|x|).'
)

gl = jax.grad(lambda v: jnp.sum(stable_log_sigmoid(v)))(x)
assert jnp.isfinite(gl).all(), f'Non-finite log-sigmoid gradient: {gl}'

# Second derivative must survive too — a NaN hiding in an unused branch shows
# up here even when the first derivative looked clean.
h = jax.grad(lambda v: jnp.sum(jax.grad(lambda u: jnp.sum({fn}(u)))(v)))(x)
assert jnp.isfinite(h).all(), f'Non-finite second derivative: {h}'
""",
        },
        {
            "name": "Gradients have the right value, not just finite ones",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.linspace(-6.0, 6.0, 25)

g = jax.grad(lambda v: jnp.sum({fn}(v)))(x)
s = jax.nn.sigmoid(x)
assert jnp.allclose(g, s * (1.0 - s), atol=1e-6), (
    "d/dx sigmoid should be s*(1-s) — finite is not enough, it has to be right"
)

gl = jax.grad(lambda v: jnp.sum(stable_log_sigmoid(v)))(x)
assert jnp.allclose(gl, jax.nn.sigmoid(-x), atol=1e-6), (
    'd/dx log_sigmoid(x) should be sigmoid(-x)'
)
""",
        },
        {
            "name": "Shape, dtype and jit/vmap all survive",
            "code": """
import jax
import jax.numpy as jnp

for dt in (jnp.float32, jnp.float16, jnp.bfloat16):
    xd = jnp.array([-4.0, 0.0, 4.0], dtype=dt)
    assert {fn}(xd).dtype == dt, (
        f'{dt.__name__} in, {{ {fn}(xd).dtype }} out — the output dtype should '
        'follow the input'
    )
    assert stable_log_sigmoid(xd).dtype == dt, f'log-sigmoid changed dtype for {dt.__name__}'

x = jnp.linspace(-5.0, 5.0, 11)
assert jnp.allclose(jax.jit({fn})(x), {fn}(x), atol=1e-6), 'jit disagrees'
assert jnp.allclose(jax.vmap({fn})(x.reshape(11, 1)).ravel(), {fn}(x), atol=1e-6), (
    'vmap disagrees — the implementation should be elementwise'
)

# Scalar input must give a scalar back.
out0 = {fn}(jnp.array(0.0))
assert jnp.ndim(out0) == 0, f'Scalar in, shape {jnp.shape(out0)} out'
""",
        },
        {
            "name": "Monotonic, and saturates without wobbling",
            "code": """
import jax.numpy as jnp

x = jnp.linspace(-40.0, 40.0, 401)
s = {fn}(x)
d = jnp.diff(s)
assert jnp.all(d >= -1e-7), (
    'sigmoid must be non-decreasing; a dip means the two branches disagree '
    'where they meet'
)

# The two halves must agree at the seam x = 0.
eps = 1e-4
lo, hi = {fn}(jnp.array(-eps)), {fn}(jnp.array(eps))
assert abs(float(hi - lo)) < 1e-3, (
    f'Discontinuity at x=0: sigmoid(-{eps})={lo}, sigmoid(+{eps})={hi}'
)

ls = stable_log_sigmoid(x)
assert jnp.all(jnp.diff(ls) >= -1e-7), 'log-sigmoid must be non-decreasing'
assert jnp.allclose(ls[-1], 0.0, atol=1e-6), 'log-sigmoid should approach 0 as x grows'
""",
        },
    ],
}
