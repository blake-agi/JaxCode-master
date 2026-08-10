"""GELU — exact erf form and the tanh approximation GPT-2 actually shipped."""

TASK = {
    "title": "GELU Activation",
    "category": "Core Ops & Layers",
    "number": "19",
    "difficulty": "Easy",
    "function_name": "my_gelu",
    "hint": (
        "Both formulas are written out in the description — transcribe them "
        "literally, including the 0.044715 and the sqrt(2/pi). Branch on the "
        "`approximate` flag with a plain Python `if`: it is a static Python bool, "
        "not a traced array, so this is jit-safe (declare it with "
        "static_argnames if you jit the function yourself). Use "
        "jax.scipy.special.erf for the exact form — jnp has no erf."
    ),
    "description": r"""
Implement **GELU** (Gaussian Error Linear Unit) in both of its standard forms.

**Exact:**
$$\text{GELU}(x) = x \cdot \Phi(x) = 0.5\,x\left(1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)$$

**Tanh approximation:**
$$0.5\,x\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)$$

### Rules
- Do **not** use `jax.nn.gelu`
- `approximate=False` (default) → exact erf form
- `approximate=True` → tanh form
- `erf` is available as `jax.scipy.special.erf`

### Signature
```python
def my_gelu(x, approximate=False):
    ...
```

### Why two versions exist
$\Phi$ is the Gaussian CDF, so GELU weights each input by the probability that a
standard normal falls below it — a smooth, probabilistic gate, unlike ReLU's hard
cutoff. The tanh form was published alongside the exact one in the original 2016
paper as a cheaper stand-in for `erf`, and it is what BERT and GPT-2 actually
shipped — so their released weights were trained against *that* curve.

The two agree closely but not exactly: the largest gap is about **4.7e-4**, near
$|x| \approx 2.7$, shrinking to zero at the origin and in both tails. Small
enough to ignore when training from scratch, large enough to notice when you are
chasing a logit mismatch against a reference implementation.

### The default that catches people
`jax.nn.gelu` defaults to `approximate=True` — the **tanh** form. PyTorch's
`F.gelu` defaults to the exact one. This task follows the PyTorch convention
(`approximate=False` by default), so read the flag carefully.

Being asked "why does GELU beat ReLU?" is common: it is smooth everywhere
(so the gradient does not jump discontinuously at 0) and it is non-monotonic,
letting small negative activations survive instead of hard-zeroing them.
""",
    "stub": '''import jax
import jax.numpy as jnp
from jax.scipy.special import erf


def my_gelu(x, approximate=False):
    """GELU activation.

    Args:
        x: array of any shape
        approximate: if True use the tanh approximation, else the exact erf form

    Returns:
        Array of the same shape.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from jax.scipy.special import erf


def my_gelu(x, approximate=False):
    if approximate:
        # The form GPT-2 / BERT shipped.
        c = jnp.sqrt(2.0 / jnp.pi)
        return 0.5 * x * (1.0 + jnp.tanh(c * (x + 0.044715 * x ** 3)))
    # x * Phi(x), where Phi is the standard normal CDF.
    return 0.5 * x * (1.0 + erf(x / jnp.sqrt(2.0)))
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("exact: ", my_gelu(x))
print("tanh:  ", my_gelu(x, approximate=True))
print("max gap:", jnp.max(jnp.abs(my_gelu(x) - my_gelu(x, approximate=True))))
print("grad:  ", jax.grad(lambda v: jnp.sum(my_gelu(v)))(x))
''',
    "tests": [
        {
            "name": "Exact form matches jax.nn.gelu",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
out = {fn}(x)
expected = jax.nn.gelu(x, approximate=False)

assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
assert jnp.allclose({fn}(jnp.array(0.0)), 0.0, atol=1e-6), 'GELU(0) must be 0'
""",
        },
        {
            "name": "Tanh approximation",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (64,))
out = {fn}(x, approximate=True)
expected = jax.nn.gelu(x, approximate=True)

assert jnp.allclose(out, expected, atol=1e-5), 'Tanh form disagrees with jax.nn.gelu'

# The two forms must actually differ, but only slightly: the largest possible
# gap between them is ~4.7e-4, anywhere on the real line.
exact = {fn}(x, approximate=False)
gap = float(jnp.max(jnp.abs(out - exact)))
assert gap < 1e-3, f'Approximation is {gap} away from exact — too far'
""",
        },
        {
            "name": "Shape and asymptotic behaviour",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (4, 8))
assert {fn}(x).shape == (4, 8), 'Shape not preserved on 2-D input'

# Large positive -> approaches x; large negative -> approaches 0.
big = jnp.array([10.0, -10.0])
out = {fn}(big)
assert abs(float(out[0]) - 10.0) < 1e-3, f'GELU(10) should be ~10, got {out[0]}'
assert abs(float(out[1])) < 1e-3, f'GELU(-10) should be ~0, got {out[1]}'

# GELU dips slightly negative around x = -1 — that non-monotonicity is the point.
assert float({fn}(jnp.array(-1.0))) < 0.0, 'GELU(-1) should be negative'
""",
        },
        {
            "name": "Gradient",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

for approx in [False, True]:
    g = jax.grad(lambda v: jnp.sum({fn}(v, approximate=approx)))(x)
    ref = jax.grad(lambda v: jnp.sum(jax.nn.gelu(v, approximate=approx)))(x)
    assert g.shape == x.shape, f'Gradient shape {g.shape}'
    assert jnp.isfinite(g).all(), f'Non-finite gradient (approximate={approx})'
    assert jnp.allclose(g, ref, atol=1e-4), (
        f'Gradient disagrees with reference (approximate={approx}): {g} vs {ref}'
    )

# d/dx GELU at 0 is 0.5.
g0 = float(jax.grad(lambda v: jnp.sum({fn}(v)))(jnp.array([0.0]))[0])
assert abs(g0 - 0.5) < 1e-4, f'grad at 0 should be 0.5, got {g0}'
""",
        },
        {
            "name": "jit with the flag as a static bool",
            "code": """
import functools
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (16,))

f = jax.jit({fn}, static_argnames=("approximate",))
assert jnp.allclose(f(x), {fn}(x), atol=1e-5), 'jit changes the exact result'
assert jnp.allclose(f(x, approximate=True), {fn}(x, approximate=True), atol=1e-5), (
    'jit changes the approximate result'
)
""",
        },
    ],
}
