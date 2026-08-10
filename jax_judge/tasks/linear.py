"""A linear layer as an nnx.Module — parameters, rngs, and shape conventions."""

TASK = {
    "title": "Simple Linear Layer",
    "category": "Core Ops & Layers",
    "number": "03",
    "difficulty": "Medium",
    "function_name": "SimpleLinear",
    "hint": (
        "Registration in NNX is just assignment: any nnx.Param you set on self in "
        "__init__ is found automatically. Get a key by CALLING the stream — "
        "rngs.params() returns a fresh key and advances it — and note the weight "
        "shape is (in_features, out_features), because the forward pass is x @ w, not w @ x. "
        "__call__ needs no reshaping: @ already contracts the last axis of x with "
        "the first of w, and the bias broadcasts. When use_bias is False, still "
        "define self.b (as None) so the attribute always exists."
    ),
    "description": r"""
Implement a **fully-connected layer** as a `flax.nnx.Module`.

$$y = xW + b, \qquad W \in \mathbb{R}^{d_{in} \times d_{out}},\; b \in \mathbb{R}^{d_{out}}$$

### Rules
- Subclass `nnx.Module`; do **not** use `nnx.Linear`
- Signature: `SimpleLinear(in_features, out_features, *, use_bias=True, rngs)`
- Weights stored as `self.w`, bias as `self.b` (or `None` when `use_bias=False`)
- Both must be wrapped in `nnx.Param`
- Initialise `w` with `jax.random.normal(...) / sqrt(in_features)`; `b` with zeros
- `__call__(x)` maps `(..., in_features) -> (..., out_features)` — any number of leading axes

### The NNX mental model
```python
class SimpleLinear(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (in_features, out_features)))
        ...
    def __call__(self, x):
        return x @ self.w
```

Unlike Flax Linen, NNX modules hold their parameters **directly as attributes**,
so `layer.w` is a live object you can inspect and mutate in place — much closer
to PyTorch's feel, with no separate `params` dict threaded through every call.
The functional purity that `jit` and `grad` need comes from `nnx.split` /
`nnx.merge`, which NNX's own `nnx.jit` and `nnx.grad` apply for you.

`layer.w` is an `nnx.Param`, not a bare array. It forwards `.shape` and the
arithmetic operators, so `x @ self.w` works unchanged, but to read or write the
array itself use **`layer.w[...]`** — `layer.w[...] = new_weights` assigns,
`layer.w[...]` reads. (The old `.value` property still exists but is deprecated
in Flax 0.12.)

Note the shape convention: JAX and Flax use `(in_features, out_features)` and compute `x @ W`,
whereas PyTorch stores `(out_features, in_features)` and computes `x @ W.T`. Mixing them up is
the most common bug when porting weights between the two — and because a
transposed square matrix has the right shape, it fails silently on square layers.

The `1/sqrt(in_features)` scale is LeCun-style init: it keeps `Var(y) ≈ Var(x)` through
the layer, so activations neither explode nor vanish as you stack them.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class SimpleLinear(nnx.Module):
    """y = x @ w + b"""

    def __init__(self, in_features: int, out_features: int, *, use_bias: bool = True, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(..., in_features) -> (..., out_features)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class SimpleLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, use_bias: bool = True, rngs: nnx.Rngs):
        # rngs.params() hands back a fresh key on every call.
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (in_features, out_features)) / jnp.sqrt(in_features))
        self.b = nnx.Param(jnp.zeros((out_features,))) if use_bias else None
        self.din = in_features
        self.dout = out_features

    def __call__(self, x):
        y = x @ self.w
        if self.b is not None:
            y = y + self.b
        return y
''',
    "demo": '''import jax.numpy as jnp
from flax import nnx

layer = SimpleLinear(4, 3, rngs=nnx.Rngs(params=0))
x = jnp.ones((2, 4))

print("w shape:", layer.w.shape, " b shape:", layer.b.shape)
print("out shape:", layer(x).shape)

no_bias = SimpleLinear(4, 3, use_bias=False, rngs=nnx.Rngs(params=0))
print("bias when use_bias=False:", no_bias.b)
print("batched (5, 6, 4) ->", SimpleLinear(4, 3, rngs=nnx.Rngs(params=1))(jnp.ones((5, 6, 4))).shape)
''',
    "tests": [
        {
            "name": "Shapes and parameter layout",
            "code": """
import jax.numpy as jnp
from flax import nnx

layer = {fn}(4, 3, rngs=nnx.Rngs(params=0))

assert layer.w.shape == (4, 3), (
    f'w should be (in_features, out_features) = (4, 3), got {layer.w.shape}. '
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

expected = x @ layer.w[...] + layer.b[...]
assert jnp.allclose(layer(x), expected, atol=1e-5), 'Output is not x @ w + b'

# Bias must actually be added: setting it to a known value should shift the output.
layer.b[...] = jnp.array([10.0, 20.0, 30.0])
shifted = layer(x)
assert jnp.allclose(shifted, x @ layer.w[...] + jnp.array([10.0, 20.0, 30.0]), atol=1e-5), (
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
assert jnp.allclose(layer(x), x @ layer.w[...], atol=1e-5), 'Output should be x @ w only'
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
std = float(jnp.std(big.w[...]))
expected = 1.0 / jnp.sqrt(256.0)
assert abs(std - expected) < 0.3 * expected, (
    f'w std is {std:.5f}, expected ~{expected:.5f} (normal / sqrt(in_features))'
)
assert jnp.allclose(big.b[...], 0.0), 'Bias must be initialised to zeros'

# Different seeds must give different weights; the same seed must reproduce.
a = {fn}(4, 3, rngs=nnx.Rngs(params=0))
b = {fn}(4, 3, rngs=nnx.Rngs(params=1))
c = {fn}(4, 3, rngs=nnx.Rngs(params=0))
assert not jnp.allclose(a.w[...], b.w[...]), 'Different seeds gave identical weights'
assert jnp.allclose(a.w[...], c.w[...]), 'Same seed must reproduce the same weights'
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
