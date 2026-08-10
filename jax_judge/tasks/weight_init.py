"""Xavier/Glorot and He/Kaiming initialisation — variance preservation from scratch."""

TASK = {
    "title": "Kaiming Initialization",
    "category": "Core Ops & Layers",
    "order": 11,
    "difficulty": "Easy",
    "function_name": "init_weights",
    "hint": (
        "Unpack fan_in, fan_out = shape — in JAX the weight is (fan_in, fan_out) "
        "because the forward pass is x @ W, so fan_in is shape[0]. "
        "Xavier std = sqrt(2 / (fan_in + fan_out)); He std = sqrt(2 / fan_in). "
        "For the normal variants: jax.random.normal(key, shape) * std. "
        "For the uniform variants remember Var(U(-a, a)) = a^2 / 3, so to hit the "
        "same std you need the limit a = sqrt(3) * std — pass it as minval/maxval "
        "to jax.random.uniform. Raise ValueError on an unknown mode."
    ),
    "description": r"""
Implement the two classic weight initialisers as one plain function, selected by
a string `mode`.

$$\sigma_{\text{xavier}} = \sqrt{\frac{2}{f_{in} + f_{out}}}
\qquad
\sigma_{\text{he}} = \sqrt{\frac{2}{f_{in}}}$$

Four modes: `"xavier_normal"`, `"xavier_uniform"`, `"he_normal"`, `"he_uniform"`.
(Glorot is another name for Xavier; Kaiming is another name for He.)

### Rules
- Signature: `init_weights(key, shape, mode="he_normal") -> jnp.ndarray`
- `shape` is `(fan_in, fan_out)` — the JAX layout, since the forward pass is `x @ W`
- Do **not** use `jax.nn.initializers` or `flax.nnx.initializers`
- Normal variants: zero-mean Gaussian with standard deviation $\sigma$
- Uniform variants: $U(-a, a)$ with $a$ chosen so the **variance matches** the
  normal variant
- Unknown `mode` must raise `ValueError`
- Pure and deterministic: the same key must give the same array

### The derivation, in three lines
Take one layer $y = xW$ with $f_{in}$ inputs, all entries i.i.d. and zero-mean:

$$\mathrm{Var}(y) = f_{in}\,\mathrm{Var}(W)\,\mathrm{Var}(x)$$

so the forward pass keeps its scale when $\mathrm{Var}(W) = 1/f_{in}$. Running the
same argument through the backward pass — where the gradient flows through $W^\top$
and therefore sees $f_{out}$ terms — demands $\mathrm{Var}(W) = 1/f_{out}$. You
cannot have both unless the layer is square, so **Xavier splits the difference**
with the harmonic-style compromise $2/(f_{in} + f_{out})$.

He then adds one observation: ReLU zeroes half its inputs, so it destroys half the
variance at every layer. Doubling the numerator, $2/f_{in}$, exactly cancels that.

### Which pairs with which
| Init | Activation | Why |
|---|---|---|
| Xavier | tanh, sigmoid, linear/attention projections | roughly linear and symmetric near 0, no variance lost |
| He | ReLU, GELU, SiLU | compensates for the half of the signal the gate kills |

Use Xavier with a 30-layer ReLU net and every layer shrinks the signal by
$1/\sqrt{2}$; after 20 layers the activations are down by ~$2^{-10}$ and the
gradient with them. That is the failure the tests below reproduce, and it is the
reason "we couldn't train deep nets before 2015" is only half a story about
architectures.

### Uniform vs normal
$\mathrm{Var}(U(-a,a)) = a^2/3$, so matching a target $\sigma$ needs
$a = \sqrt{3}\,\sigma$ — i.e. $a = \sqrt{6/(f_{in}+f_{out})}$ for Xavier. That
$\sqrt{6}$ in every framework's `xavier_uniform` comes from precisely here, not
from anywhere magic.
""",
    "stub": '''import math

import jax
import jax.numpy as jnp


def init_weights(key, shape, mode="he_normal"):
    """Draw an initialised weight matrix.

    Args:
        key:   a jax.random key
        shape: (fan_in, fan_out)
        mode:  "xavier_normal" | "xavier_uniform" | "he_normal" | "he_uniform"

    Returns:
        Array of the given shape.
    """
    pass  # Replace this
''',
    "solution": '''import math

import jax
import jax.numpy as jnp


def init_weights(key, shape, mode="he_normal"):
    fan_in, fan_out = shape          # x @ W, so fan_in is the leading axis

    if mode in ("xavier_normal", "xavier_uniform"):
        std = math.sqrt(2.0 / (fan_in + fan_out))
    elif mode in ("he_normal", "he_uniform"):
        std = math.sqrt(2.0 / fan_in)
    else:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of "
            "'xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform'"
        )

    if mode.endswith("_normal"):
        return jax.random.normal(key, shape) * std

    # Var(U(-a, a)) = a^2 / 3, so a = sqrt(3) * std matches the normal variant.
    limit = math.sqrt(3.0) * std
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
''',
    "demo": '''import jax
import jax.numpy as jnp

key = jax.random.key(0)
for mode in ("xavier_normal", "xavier_uniform", "he_normal", "he_uniform"):
    w = init_weights(key, (256, 64), mode)
    print(f"{mode:>15s}  std={float(jnp.std(w)):.4f}  max={float(jnp.abs(w).max()):.4f}")

print("\\ntarget xavier std:", (2 / (256 + 64)) ** 0.5)
print("target he     std:", (2 / 256) ** 0.5)

# Signal propagation through 20 ReLU layers, He vs Xavier.
x = jax.random.normal(jax.random.key(1), (64, 256))
for mode in ("he_normal", "xavier_normal"):
    h, k = x, jax.random.key(2)
    for _ in range(20):
        k, sub = jax.random.split(k)
        h = jax.nn.relu(h @ init_weights(sub, (256, 256), mode))
    print(f"{mode:>15s}  rms after 20 layers: {float(jnp.sqrt(jnp.mean(h ** 2))):.6f}")
''',
    "tests": [
        {
            "name": "he_normal: shape, mean and std",
            "code": """
import jax
import jax.numpy as jnp

w = {fn}(jax.random.key(0), (1024, 256), 'he_normal')
assert w.shape == (1024, 256), f'Shape {w.shape} vs (1024, 256)'
assert jnp.isfinite(w).all(), 'Non-finite values in the initialised matrix'
assert abs(float(jnp.mean(w))) < 0.005, f'Mean should be ~0, got {float(jnp.mean(w)):.5f}'

expected = (2.0 / 1024) ** 0.5
got = float(jnp.std(w))
assert abs(got - expected) < 0.03 * expected, (
    f'he std is {got:.5f}, expected sqrt(2/fan_in) = {expected:.5f}'
)
""",
        },
        {
            "name": "xavier_normal uses both fans",
            "code": """
import jax
import jax.numpy as jnp

w = {fn}(jax.random.key(0), (1024, 256), 'xavier_normal')
expected = (2.0 / (1024 + 256)) ** 0.5
got = float(jnp.std(w))
assert abs(got - expected) < 0.03 * expected, (
    f'xavier std is {got:.5f}, expected sqrt(2/(fan_in+fan_out)) = {expected:.5f}'
)
""",
        },
        {
            "name": "fan_in is axis 0, not axis 1",
            "code": """
import jax
import jax.numpy as jnp

# A deliberately lopsided matrix: 64 inputs, 1024 outputs.
w = {fn}(jax.random.key(0), (64, 1024), 'he_normal')
right = (2.0 / 64) ** 0.5       # 0.1768 — fan_in = shape[0]
wrong = (2.0 / 1024) ** 0.5     # 0.0442 — fan_in read off the wrong axis
got = float(jnp.std(w))
assert abs(got - right) < 0.05 * right, (
    f'std is {got:.5f}, expected {right:.5f}. Got ~{wrong:.5f}? Then you used '
    'shape[1] as fan_in — JAX weights are (fan_in, fan_out) because the forward '
    'pass is x @ W.'
)

# Xavier is symmetric in the two fans, so transposing must not change its scale.
a = float(jnp.std({fn}(jax.random.key(1), (64, 1024), 'xavier_normal')))
b = float(jnp.std({fn}(jax.random.key(2), (1024, 64), 'xavier_normal')))
assert abs(a - b) < 0.05 * a, f'xavier should be symmetric in fan_in/fan_out: {a:.5f} vs {b:.5f}'
""",
        },
        {
            "name": "Uniform variants are bounded and variance-matched",
            "code": """
import jax
import jax.numpy as jnp

for mode, std in [('he_uniform', (2.0 / 512) ** 0.5),
                  ('xavier_uniform', (2.0 / (512 + 512)) ** 0.5)]:
    w = {fn}(jax.random.key(0), (512, 512), mode)
    limit = (3.0 ** 0.5) * std
    hi = float(jnp.max(jnp.abs(w)))

    assert hi <= limit * 1.001, (
        f'{mode}: values reach {hi:.5f} but the limit should be sqrt(3)*std = {limit:.5f}'
    )
    assert hi > 0.98 * limit, (
        f'{mode}: max |w| is only {hi:.5f} vs a limit of {limit:.5f} — the range is too narrow. '
        'Var(U(-a,a)) = a^2/3, so a must be sqrt(3)*std, not std.'
    )
    got = float(jnp.std(w))
    assert abs(got - std) < 0.03 * std, (
        f'{mode}: std {got:.5f} should match the normal variant {std:.5f}'
    )
""",
        },
        {
            "name": "He survives 20 ReLU layers, Xavier does not",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (64, 256))
start = float(jnp.sqrt(jnp.mean(x ** 2)))


def propagate(mode):
    h, k = x, jax.random.key(1)
    for _ in range(20):
        k, sub = jax.random.split(k)
        h = jax.nn.relu(h @ {fn}(sub, (256, 256), mode))
    return float(jnp.sqrt(jnp.mean(h ** 2))) / start


he = propagate('he_normal')
xavier = propagate('xavier_normal')

assert 0.4 < he < 2.5, (
    f'He init should roughly preserve the activation RMS through 20 ReLU layers, '
    f'got a factor of {he:.4f}'
)
assert xavier < 0.1, (
    f'Xavier + ReLU must decay (~2^-10 over 20 layers), got a factor of {xavier:.4f} — '
    'the two modes are not actually using different variances'
)
""",
        },
        {
            "name": "Deterministic, pure, and validates mode",
            "code": """
import jax
import jax.numpy as jnp

k0, k1 = jax.random.key(0), jax.random.key(1)
a = {fn}(k0, (16, 8), 'he_normal')
b = {fn}(k0, (16, 8), 'he_normal')
c = {fn}(k1, (16, 8), 'he_normal')

assert jnp.allclose(a, b), 'Same key must give the same array — do not fold in fresh entropy'
assert not jnp.allclose(a, c), 'Different keys must give different arrays'

# The mode really has to change the answer.
assert not jnp.allclose(a, {fn}(k0, (16, 8), 'xavier_normal')), 'mode is being ignored'
assert not jnp.allclose(a, {fn}(k0, (16, 8), 'he_uniform')), 'normal/uniform are identical'

try:
    {fn}(k0, (16, 8), 'lecun_normal')
except ValueError:
    pass
except Exception as e:
    raise AssertionError(f'Unknown mode should raise ValueError, raised {type(e).__name__}')
else:
    raise AssertionError('Unknown mode should raise ValueError, but it returned normally')
""",
        },
        {
            "name": "Jittable with static shape and mode",
            "code": """
import jax
import jax.numpy as jnp

# shape and mode are Python objects, not arrays: they must be static under jit.
f = jax.jit({fn}, static_argnums=(1, 2))
key = jax.random.key(0)

out = f(key, (128, 32), 'he_normal')
assert out.shape == (128, 32), f'{out.shape} vs (128, 32)'
assert jnp.allclose(out, {fn}(key, (128, 32), 'he_normal'), atol=1e-6), (
    'jitted result differs from the eager result'
)
assert out.dtype == jnp.float32, f'Expected float32, got {out.dtype}'

# Default mode must be usable positionally-free.
d = {fn}(key, (8, 4))
assert d.shape == (8, 4), f'Default-mode call gave {d.shape}'
""",
        },
    ],
}
