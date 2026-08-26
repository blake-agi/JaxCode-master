"""Problem 15 without Flax."""

_LINEAR = '''class Linear:
    """Given to you, exactly as nnx.Linear is given to you in problem 15."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

TASK = {
    "title": "SwiGLU MLP without Flax",
    "category": "Core Ops & Layers",
    "number": "b_29",
    "difficulty": "Easy",
    "function_name": "SwiGLUMLP",
    "hint": (
        "Three Linears from jax.random.split(key, 3): gate_proj and up_proj "
        "are (d_model, d_ff), down_proj is (d_ff, d_model). __call__ is one "
        "line: down_proj(silu(gate_proj(x)) * up_proj(x)). The multiply is "
        "elementwise, which is why gate and up must have the SAME output width."
    ),
    "description": r"""
Problem 15's SwiGLU feed-forward block with no Flax.

### Signature
```python
class SwiGLUMLP:
    def __init__(self, d_model, d_ff, *, key): ...
    def __call__(self, x): ...        # (..., d_model) -> (..., d_model)
```

Three projections, named as in problem 15:

| | shape |
|---|---|
| `self.gate_proj` | `(d_model, d_ff)` |
| `self.up_proj` | `(d_model, d_ff)` |
| `self.down_proj` | `(d_ff, d_model)` |

Built from `jax.random.split(key, 3)` — one key reused three times gives three
identical matrices, which makes `gate` and `up` the same tensor and quietly
turns SwiGLU into `silu(z) * z`.

### The formula
$$\text{SwiGLU}(x) = W_{\text{down}}\big(\text{silu}(W_{\text{gate}}x)\odot W_{\text{up}}x\big)$$

One line. The `*` is **elementwise**, which is exactly why `gate_proj` and
`up_proj` must produce the same width — the gate multiplies the value stream
channel by channel.

### Why two projections up and one down
A vanilla MLP is `down(gelu(up(x)))` — two matrices. SwiGLU spends three, so
implementations shrink `d_ff` (Llama uses about `8/3 · d_model` instead of
`4 · d_model`) to keep the parameter count level. `silu(x) = x · sigmoid(x)`,
which you built in `b_22`.

### Why this exists alongside problem 15
Interview sandboxes ship `jax` but not `flax`. Same class name, same argument
names, same attribute names — only `rngs=nnx.Rngs(params=0)` becomes
`key=jax.random.key(0)`, and `Linear` is handed to you the way `nnx.Linear`
is.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class SwiGLUMLP:
    """down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model, d_ff, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class SwiGLUMLP:
    def __init__(self, d_model, d_ff, *, key):
        # Three independent keys: one reused would make gate and up identical,
        # collapsing SwiGLU into silu(z) * z.
        kg, ku, kd = jax.random.split(key, 3)
        self.gate_proj = Linear(d_model, d_ff, key=kg)
        self.up_proj = Linear(d_model, d_ff, key=ku)
        self.down_proj = Linear(d_ff, d_model, key=kd)

    def __call__(self, x):
        # The * is elementwise, so gate and up must share a width.
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
''',
    "demo": '''import jax
import jax.numpy as jnp

mlp = SwiGLUMLP(8, 16, key=jax.random.key(0))
print("gate", mlp.gate_proj.kernel.shape,
      " up", mlp.up_proj.kernel.shape,
      " down", mlp.down_proj.kernel.shape)

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
print("out:", mlp(x).shape)

# The gate really does gate: zeroing it kills the output.
mlp.gate_proj.kernel = jnp.zeros_like(mlp.gate_proj.kernel)
mlp.gate_proj.bias = jnp.zeros_like(mlp.gate_proj.bias)
print("gate forced to 0 -> out is down(0):", jnp.allclose(mlp(x), mlp.down_proj(jnp.zeros((2, 5, 16)))))
''',
    "tests": [
        {
            "name": "Three projections, named and shaped like problem 15",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 16, key=jax.random.key(0))
for name, want in (('gate_proj', (8, 16)), ('up_proj', (8, 16)), ('down_proj', (16, 8))):
    assert hasattr(m, name), f'missing {name} — keep problem 15 names'
    k = getattr(m, name).kernel
    assert k.shape == want, f'{name}.kernel {k.shape} vs {want}'

assert not jnp.allclose(m.gate_proj.kernel, m.up_proj.kernel), (
    'gate_proj and up_proj got identical kernels — split the key three ways. '
    'Identical gate and up turns SwiGLU into silu(z) * z.'
)
""",
        },
        {
            "name": "Computes down(silu(gate(x)) * up(x))",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 16, key=jax.random.key(0))
for shape in [(8,), (4, 8), (2, 5, 8)]:
    x = jax.random.normal(jax.random.key(1), shape)
    out = m(x)
    assert out.shape == shape, f'{shape} -> {out.shape}, should be unchanged'
    ref = m.down_proj(jax.nn.silu(m.gate_proj(x)) * m.up_proj(x))
    assert jnp.allclose(out, ref, atol=1e-5), f'wrong values for {shape}'
""",
        },
        {
            "name": "The gate is a gate, and the two branches are not swapped",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 16, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))
base = m(x)

# silu(0) == 0, so a zero gate must annihilate the value stream.
m.gate_proj.kernel = jnp.zeros_like(m.gate_proj.kernel)
m.gate_proj.bias = jnp.zeros_like(m.gate_proj.bias)
zeroed = m(x)
assert jnp.allclose(zeroed, m.down_proj(jnp.zeros((2, 5, 16))), atol=1e-5), (
    'forcing gate_proj to zero should send silu(0)=0 through the multiply, '
    'leaving down_proj(0). The activation goes on the GATE branch, not up.'
)

# And the reverse: a zero up branch must also annihilate it.
m2 = {fn}(8, 16, key=jax.random.key(0))
m2.up_proj.kernel = jnp.zeros_like(m2.up_proj.kernel)
m2.up_proj.bias = jnp.zeros_like(m2.up_proj.bias)
assert jnp.allclose(m2(x), m2.down_proj(jnp.zeros((2, 5, 16))), atol=1e-5), (
    'a zero up_proj should also zero the product'
)
assert not jnp.allclose(base, zeroed, atol=1e-4), 'the gate had no effect at all'
""",
        },
        {
            "name": "silu, not relu or gelu",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(4, 4, key=jax.random.key(0))
# Force up_proj and down_proj to the identity so the output isolates the gate.
m.up_proj.kernel = jnp.eye(4)
m.up_proj.bias = jnp.ones((4,))
m.down_proj.kernel = jnp.eye(4)
m.down_proj.bias = jnp.zeros((4,))

x = jax.random.normal(jax.random.key(1), (32, 4))
got = m(x)
g = m.gate_proj(x)
assert jnp.allclose(got, jax.nn.silu(g) * (x + 1.0), atol=1e-5), (
    'the activation should be silu (x * sigmoid(x)). relu and gelu both differ '
    'for negative inputs, which this test feeds.'
)
assert not jnp.allclose(got, jax.nn.relu(g) * (x + 1.0), atol=1e-3), 'looks like relu'
""",
        },
        {
            "name": "Gradient w.r.t. the input, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 16, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))
out = m(x)

g = jax.grad(lambda v: jnp.sum(m(v)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. the input'

assert jnp.allclose(jax.jit(lambda v: m(v))(x), out, atol=1e-5), 'jit disagrees'
vm = jax.vmap(lambda v: m(v))(x)
assert jnp.allclose(vm, out, atol=1e-5), 'vmap disagrees — do not name the batch axis'
""",
        },
    ],
}
