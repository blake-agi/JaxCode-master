"""A Linear layer as an nnx.Module — parameters, rngs, and shape conventions."""

TASK = {
    "title": "Linear Layer (nnx.Module)",
    "category": "Core Ops & Layers",
    "order": 4,
    "difficulty": "Easy",
    "function_name": "Linear",
    "hint": (
        "Store parameters as nnx.Param(...) attributes in __init__, and draw the "
        "weights with jax.random.normal(rngs.params(), (din, dout)) — calling "
        "rngs.params() yields a fresh key each time. Scale by 1/sqrt(din) for "
        "sane variance. In __call__ just do x @ self.w, adding self.b only when "
        "use_bias is True. Set self.b = None when use_bias is False so the "
        "attribute always exists."
    ),
    "description": r"""
Implement a **fully-connected layer** as a `flax.nnx.Module`.

$$y = xW + b, \qquad W \in \mathbb{R}^{d_{in} \times d_{out}},\; b \in \mathbb{R}^{d_{out}}$$

### Rules
- Subclass `nnx.Module`; do **not** use `nnx.Linear`
- Signature: `Linear(din, dout, *, use_bias=True, rngs)`
- Weights stored as `self.w`, bias as `self.b` (or `None` when `use_bias=False`)
- Both must be wrapped in `nnx.Param`
- Initialise `w` with `jax.random.normal(...) / sqrt(din)`; `b` with zeros
- `__call__(x)` maps `(..., din) -> (..., dout)` — any number of leading axes

### The NNX mental model
```python
class Linear(nnx.Module):
    def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)))
        ...
    def __call__(self, x):
        return x @ self.w
```

Unlike Flax Linen, NNX modules hold their parameters **directly as attributes**,
so `layer.w` is a real array you can inspect and mutate — much closer to
PyTorch's feel. The functional purity you need for `jit` and `grad` comes from
`nnx.split` / `nnx.merge`, which NNX's own `nnx.jit` and `nnx.grad` apply for you.

Note the shape convention: JAX and Flax use `(din, dout)` and compute `x @ W`,
whereas PyTorch stores `(dout, din)` and computes `x @ W.T`. Mixing them up is
the most common bug when porting weights between the two.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class Linear(nnx.Module):
    """y = x @ w + b"""

    def __init__(self, din: int, dout: int, *, use_bias: bool = True, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(..., din) -> (..., dout)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, use_bias: bool = True, rngs: nnx.Rngs):
        # rngs.params() hands back a fresh key on every call.
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)) / jnp.sqrt(din))
        self.b = nnx.Param(jnp.zeros((dout,))) if use_bias else None
        self.din = din
        self.dout = dout

    def __call__(self, x):
        y = x @ self.w
        if self.b is not None:
            y = y + self.b
        return y
''',
    "demo": '''import jax.numpy as jnp
from flax import nnx

layer = Linear(4, 3, rngs=nnx.Rngs(params=0))
x = jnp.ones((2, 4))

print("w shape:", layer.w.shape, " b shape:", layer.b.shape)
print("out shape:", layer(x).shape)

no_bias = Linear(4, 3, use_bias=False, rngs=nnx.Rngs(params=0))
print("bias when use_bias=False:", no_bias.b)
print("batched (5, 6, 4) ->", Linear(4, 3, rngs=nnx.Rngs(params=1))(jnp.ones((5, 6, 4))).shape)
''',
    "tests": [
        {
            "name": "Shapes and parameter layout",
            "code": """
import jax.numpy as jnp
from flax import nnx

layer = {fn}(4, 3, rngs=nnx.Rngs(params=0))

assert layer.w.shape == (4, 3), (
    f'w should be (din, dout) = (4, 3), got {layer.w.shape}. '
    'JAX uses x @ W, not PyTorch\\'s W.T convention.'
)
assert layer.b.shape == (3,), f'b should be (3,), got {layer.b.shape}'

out = layer(jnp.ones((2, 4)))
assert out.shape == (2, 3), f'Output shape {out.shape} vs (2, 3)'
""",
        },
        {
            "name": "Params are nnx.Param and discoverable",
            "code": """
import jax
from flax import nnx

layer = {fn}(6, 5, rngs=nnx.Rngs(params=0))

assert isinstance(layer.w, nnx.Param), (
    f'w must be wrapped in nnx.Param, got {type(layer.w)}'
)
assert isinstance(layer.b, nnx.Param), f'b must be wrapped in nnx.Param, got {type(layer.b)}'

# nnx.split must find both parameters in the module state.
_, state = nnx.split(layer)
leaves = jax.tree.leaves(state)
assert len(leaves) >= 2, f'Expected at least 2 param leaves in the state, got {len(leaves)}'
shapes = sorted(tuple(l.shape) for l in leaves)
assert (5,) in shapes and (6, 5) in shapes, f'Unexpected param shapes: {shapes}'
""",
        },
        {
            "name": "Computes x @ w + b exactly",
            "code": """
import jax.numpy as jnp
from flax import nnx

layer = {fn}(4, 3, rngs=nnx.Rngs(params=0))
x = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]])

expected = x @ layer.w.value + layer.b.value
assert jnp.allclose(layer(x), expected, atol=1e-5), 'Output is not x @ w + b'

# Bias must actually be added: setting it to a known value should shift the output.
layer.b.value = jnp.array([10.0, 20.0, 30.0])
shifted = layer(x)
assert jnp.allclose(shifted, x @ layer.w.value + jnp.array([10.0, 20.0, 30.0]), atol=1e-5), (
    'Bias is not being added'
)
""",
        },
        {
            "name": "use_bias=False",
            "code": """
import jax.numpy as jnp
from flax import nnx

layer = {fn}(4, 3, use_bias=False, rngs=nnx.Rngs(params=0))
x = jnp.ones((2, 4))

assert layer.b is None, f'b should be None when use_bias=False, got {layer.b}'
assert jnp.allclose(layer(x), x @ layer.w.value, atol=1e-5), 'Output should be x @ w only'
assert layer(x).shape == (2, 3), f'{layer(x).shape}'
""",
        },
        {
            "name": "Arbitrary leading axes",
            "code": """
import jax.numpy as jnp
from flax import nnx

layer = {fn}(8, 4, rngs=nnx.Rngs(params=0))

for shape in [(8,), (3, 8), (2, 5, 8), (2, 3, 4, 8)]:
    out = layer(jnp.ones(shape))
    assert out.shape == shape[:-1] + (4,), (
        f'Input {shape} should give {shape[:-1] + (4,)}, got {out.shape}'
    )
""",
        },
        {
            "name": "Initialisation scale and independence",
            "code": """
import jax.numpy as jnp
from flax import nnx

big = {fn}(256, 256, rngs=nnx.Rngs(params=0))
std = float(jnp.std(big.w.value))
expected = 1.0 / jnp.sqrt(256.0)
assert abs(std - expected) < 0.3 * expected, (
    f'w std is {std:.5f}, expected ~{expected:.5f} (normal / sqrt(din))'
)
assert jnp.allclose(big.b.value, 0.0), 'Bias must be initialised to zeros'

# Different seeds must give different weights; the same seed must reproduce.
a = {fn}(4, 3, rngs=nnx.Rngs(params=0))
b = {fn}(4, 3, rngs=nnx.Rngs(params=1))
c = {fn}(4, 3, rngs=nnx.Rngs(params=0))
assert not jnp.allclose(a.w.value, b.w.value), 'Different seeds gave identical weights'
assert jnp.allclose(a.w.value, c.w.value), 'Same seed must reproduce the same weights'
""",
        },
        {
            "name": "Trains under nnx.grad",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(3, 1, rngs=nnx.Rngs(params=0))
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 1.0, 0.0]])
y = jnp.array([[1.0], [2.0], [0.5]])


def loss_fn(m):
    return jnp.mean((m(x) - y) ** 2)


before = float(loss_fn(layer))
for _ in range(500):
    grads = nnx.grad(loss_fn)(layer)
    state = nnx.state(layer, nnx.Param)
    nnx.update(layer, jax.tree.map(lambda p, g: p - 0.02 * g, state, grads))
after = float(loss_fn(layer))

assert jnp.isfinite(after), f'Training diverged to {after}'
assert after < before, f'Loss did not decrease: {before} -> {after}'
assert after < 1e-3, f'Should fit this tiny linear system, final loss {after}'
""",
        },
    ],
}
