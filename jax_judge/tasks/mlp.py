"""SwiGLU MLP — the LLaMA-style feed-forward block, built from nnx.Linear."""

TASK = {
    "title": "SwiGLU MLP",
    "category": "Core Ops & Layers",
    "order": 10,
    "number": "15",
    "difficulty": "Medium",
    "function_name": "SwiGLUMLP",
    "hint": (
        "Three nnx.Linear layers, not two: gate_proj and up_proj both map "
        "d_model -> d_ff, and down_proj maps d_ff -> d_model. The gate is the "
        "one that goes through the activation; up_proj is passed through "
        "untouched and multiplied in element-wise. SiLU is x * sigmoid(x), "
        "available as jax.nn.silu."
    ),
    "description": r"""
Implement the **SwiGLU MLP** — the feed-forward block used in LLaMA, Mistral,
Gemma and most modern LLMs.

$$\text{SwiGLU}(x) = \text{down\_proj}\big(\text{SiLU}(\text{gate\_proj}(x))
\odot \text{up\_proj}(x)\big)$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$.

### Signature
```python
class SwiGLUMLP(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs): ...
    def __call__(self, x): ...
```

### Requirements
- `self.gate_proj`: `nnx.Linear(d_model, d_ff)`
- `self.up_proj`:   `nnx.Linear(d_model, d_ff)`
- `self.down_proj`: `nnx.Linear(d_ff, d_model)`
- Activation: **SiLU** (a.k.a. Swish) — `jax.nn.silu`, or write `x * jax.nn.sigmoid(x)`

`nnx.Linear` is an allowed building block here — you are implementing the
*block*, not the linear layer. It also brings its own initialization
(lecun_normal kernel, zero bias), which is why you do not touch `nnx.Param`
directly in this problem.

### Why three projections and not two
A classic MLP is `down(act(up(x)))` — two matrices. SwiGLU splits the
expansion into two parallel projections and uses one to **gate** the other:
the network can suppress a channel by driving the gate negative, independently
of what the value projection says. That multiplicative interaction is what a
plain activation cannot express.

The cost is a third matrix. To keep the parameter count comparable, models
using SwiGLU shrink $d_{ff}$ from the classic $4d$ to about $\tfrac{8}{3}d$ —
LLaMA-7B uses $d_{ff} = 11008$ against $d_{model} = 4096$, a ratio of 2.69,
which is exactly $\tfrac{8}{3}$ rounded to a hardware-friendly multiple.

### The trap
It is easy to apply the activation to the wrong branch, or to both. Only the
**gate** goes through SiLU; `up_proj(x)` is multiplied in linearly. Swapping
them still runs, still trains, and is silently a different architecture — the
tests below pin down which branch is which.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class SwiGLUMLP(nnx.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(..., d_model) -> (..., d_model)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class SwiGLUMLP(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.up_proj = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.down_proj = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x):
        # Only the GATE goes through the activation; up_proj is linear.
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

mlp = SwiGLUMLP(d_model=8, d_ff=21, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))

print("in :", x.shape)
print("out:", mlp(x).shape, "(same as input)")
print("d_ff/d_model =", 21 / 8, "— LLaMA uses ~8/3, not 4")

params = nnx.state(mlp, nnx.Param)
print("param leaves:", len(jax.tree.leaves(params)), "(3 kernels + 3 biases)")
''',
    "tests": [
        {
            "name": "Shapes and required sub-layers",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (4, 8))
out = m(x)

assert out.shape == (4, 8), f'Shape mismatch: {out.shape} vs (4, 8)'

for name in ("gate_proj", "up_proj", "down_proj"):
    assert hasattr(m, name), f'Missing self.{name}'
    assert isinstance(getattr(m, name), nnx.Linear), (
        f'self.{name} must be an nnx.Linear, got {type(getattr(m, name))}'
    )

assert m.gate_proj.kernel.shape == (8, 16), f'gate_proj kernel {m.gate_proj.kernel.shape}'
assert m.up_proj.kernel.shape == (8, 16), f'up_proj kernel {m.up_proj.kernel.shape}'
assert m.down_proj.kernel.shape == (16, 8), f'down_proj kernel {m.down_proj.kernel.shape}'
""",
        },
        {
            "name": "Matches the SwiGLU formula",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(6, 12, rngs=nnx.Rngs(params=1))
x = jax.random.normal(jax.random.key(2), (3, 6))

ref = m.down_proj(jax.nn.silu(m.gate_proj(x)) * m.up_proj(x))
assert jnp.allclose(m(x), ref, atol=1e-5), 'Output does not match the SwiGLU formula'
""",
        },
        {
            "name": "Only the gate branch is activated",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 4, rngs=nnx.Rngs(params=2))

# Make down_proj the identity so we can read the inner product directly.
m.down_proj.kernel[...] = jnp.eye(4)
m.down_proj.bias[...] = jnp.zeros(4)
m.gate_proj.bias[...] = jnp.zeros(4)
m.up_proj.bias[...] = jnp.zeros(4)

x = jax.random.normal(jax.random.key(3), (5, 4))
g = m.gate_proj(x)
u = m.up_proj(x)
out = m(x)

correct = jax.nn.silu(g) * u
swapped = g * jax.nn.silu(u)
both = jax.nn.silu(g) * jax.nn.silu(u)

assert jnp.allclose(out, correct, atol=1e-5), (
    'Expected silu(gate_proj(x)) * up_proj(x). '
    + ('Looks like you activated up_proj instead of gate_proj.'
       if jnp.allclose(out, swapped, atol=1e-5)
       else 'Looks like you activated both branches.'
       if jnp.allclose(out, both, atol=1e-5) else '')
)
""",
        },
        {
            "name": "SiLU, not ReLU or GELU",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 4, rngs=nnx.Rngs(params=4))
m.down_proj.kernel[...] = jnp.eye(4)
m.down_proj.bias[...] = jnp.zeros(4)
m.gate_proj.bias[...] = jnp.zeros(4)
m.up_proj.bias[...] = jnp.zeros(4)

x = jax.random.normal(jax.random.key(5), (16, 4)) * 2.0
g, u = m.gate_proj(x), m.up_proj(x)
out = m(x)

assert jnp.allclose(out, jax.nn.silu(g) * u, atol=1e-5), 'Not SiLU'
assert not jnp.allclose(out, jax.nn.relu(g) * u, atol=1e-3), (
    'Output matches ReLU gating — SwiGLU uses SiLU (x * sigmoid(x)), which is '
    'smooth and lets small negative values through'
)
""",
        },
        {
            "name": "Batch dimensions pass through",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, rngs=nnx.Rngs(params=6))

for shape in ((8,), (4, 8), (2, 5, 8)):
    x = jax.random.normal(jax.random.key(7), shape)
    assert m(x).shape == shape, f'Shape {shape} -> {m(x).shape}, expected unchanged'

# A leading batch axis must not mix examples.
x = jax.random.normal(jax.random.key(8), (3, 8))
stacked = m(x)
one_by_one = jnp.stack([m(x[i]) for i in range(3)])
assert jnp.allclose(stacked, one_by_one, atol=1e-5), 'Rows are not independent'
""",
        },
        {
            "name": "Gradients reach all three projections",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, rngs=nnx.Rngs(params=9))
x = jax.random.normal(jax.random.key(10), (4, 8))
y = jax.random.normal(jax.random.key(11), (4, 8))

grads = nnx.grad(lambda mod: jnp.mean((mod(x) - y) ** 2))(m)
state = nnx.state(grads)

for name in ("gate_proj", "up_proj", "down_proj"):
    k = state[name]["kernel"]
    v = k[...] if isinstance(k, nnx.Variable) else k
    assert jnp.isfinite(v).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(v).sum()) > 0, (
        f'Gradient for {name} is all zeros — that projection is not in the forward path'
    )
""",
        },
        {
            "name": "jit via split/merge",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, rngs=nnx.Rngs(params=12))
x = jax.random.normal(jax.random.key(13), (4, 8))
ref = m(x)

graphdef, state = nnx.split(m)

@jax.jit
def run(state, x):
    return nnx.merge(graphdef, state)(x)

assert jnp.allclose(run(state, x), ref, atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
