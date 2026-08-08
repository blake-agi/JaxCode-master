"""ReLU from scratch — and the subgradient question at x=0."""

TASK = {
    "title": "Implement ReLU",
    "category": "Core Ops & Layers",
    "order": 1,
    "difficulty": "Easy",
    "function_name": "relu",
    "hint": (
        "jnp.where(x > 0, x, 0.0) is the cleanest route, and it gives a gradient "
        "of exactly 0 at x=0. jnp.maximum(x, 0.0) also works but hands you a "
        "subgradient of 0.5 there — either is defensible, so know which one you "
        "wrote and why."
    ),
    "description": r"""
Implement the **ReLU** activation from scratch.

$$\text{ReLU}(x) = \max(0, x)$$

### Rules
- Do **not** use `jax.nn.relu`, `jnp.clip`, or any built-in activation
- Must be differentiable via `jax.grad`
- Element-wise: works on any shape

### The interesting part
ReLU is not differentiable at exactly `x = 0`, so autodiff has to pick a
subgradient. Different formulations give different answers:

| Implementation | grad at `x = 0` |
|---|---|
| `jnp.where(x > 0, x, 0.0)` | `0.0` |
| `jnp.maximum(x, 0.0)` | `0.5` |

Anything in `[0, 1]` is a valid subgradient. This exact question — *"what is the
derivative of ReLU at zero?"* — is a very common warm-up, and the answer
interviewers want is "it's a subgradient, frameworks pick a convention."
""",
    "stub": '''import jax
import jax.numpy as jnp


def relu(x):
    """ReLU activation, element-wise.

    Args:
        x: array of any shape

    Returns:
        Array of the same shape, with negatives replaced by 0.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def relu(x):
    return jnp.where(x > 0, x, 0.0)
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("input: ", x)
print("output:", relu(x))
print("grad:  ", jax.grad(lambda v: jnp.sum(relu(v)))(x))
''',
    "tests": [
        {
            "name": "Basic values",
            "code": """
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
out = {fn}(x)
expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])

assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert jnp.allclose(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "2-D input",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (4, 8))
out = {fn}(x)

assert out.shape == x.shape, f'Shape mismatch on 2-D input: {out.shape} vs {x.shape}'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert jnp.allclose(out, jnp.maximum(x, 0.0)), 'Value mismatch on random input'
""",
        },
        {
            "name": "Gradient",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([-1.0, 0.0, 1.0, 2.0])
g = jax.grad(lambda v: jnp.sum({fn}(v)))(x)

assert g.shape == x.shape, f'Gradient shape {g.shape} vs {x.shape}'
assert jnp.allclose(g[0], 0.0), f'grad at x=-1 should be 0, got {g[0]}'
assert jnp.allclose(g[2], 1.0), f'grad at x=1 should be 1, got {g[2]}'
assert jnp.allclose(g[3], 1.0), f'grad at x=2 should be 1, got {g[3]}'
assert 0.0 <= float(g[1]) <= 1.0, (
    f'grad at x=0 must be a valid subgradient in [0, 1], got {g[1]}'
)
""",
        },
        {
            "name": "No built-in activation used",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (16,))
assert jnp.allclose({fn}(x), jax.nn.relu(x), atol=1e-6), 'Disagrees with jax.nn.relu'

# 3-D and scalar inputs should behave too.
x3 = jax.random.normal(jax.random.key(2), (2, 3, 4))
assert {fn}(x3).shape == (2, 3, 4), f'{{fn}}(x3).shape'
assert jnp.allclose({fn}(jnp.array(-5.0)), 0.0), 'Scalar negative input'
assert jnp.allclose({fn}(jnp.array(5.0)), 5.0), 'Scalar positive input'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (8, 16))

assert jnp.allclose(jax.jit({fn})(x), {fn}(x), atol=1e-6), 'jit changes the result'
assert jnp.allclose(jax.vmap({fn})(x), {fn}(x), atol=1e-6), 'vmap changes the result'
""",
        },
    ],
}
