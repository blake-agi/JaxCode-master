"""Gradient checkpointing with jax.checkpoint — trading compute for memory."""

TASK = {
    "title": "Gradient Checkpointing with jax.checkpoint",
    "category": "JAX Fundamentals",
    "order": 10,
    "difficulty": "Medium",
    "function_name": "deep_chain",
    "hint": (
        "Wrap the per-block function, not the whole chain: "
        "h = jax.checkpoint(block)(W, h) inside the loop. jax.checkpoint returns a "
        "new function, so apply it to `block` and then call the result. The numbers "
        "must come out identical to the unwrapped version — only the memory "
        "profile changes."
    ),
    "description": r"""
Apply a deep stack of blocks with **gradient checkpointing** (rematerialization).

Each block is
$$h \leftarrow \tanh(h W_i)$$

Given `params` (a list of `(D, D)` weight matrices) and `x` of shape `(B, D)`,
apply every block in order and return the final `(B, D)` activations — with each
block wrapped in `jax.checkpoint`.

### Rules
- Wrap **each block**, not the whole chain
- The output and gradients must be **numerically identical** to the unwrapped version
- Use `jax.checkpoint` (a.k.a. `jax.remat`)

### The tradeoff
Reverse-mode autodiff normally saves every intermediate activation on the
forward pass so the backward pass can use them. For an `L`-layer network that is
`O(L)` memory.

`jax.checkpoint` says: *don't save this block's internals — recompute them during
the backward pass.* Memory drops to `O(1)` per checkpointed block at the cost of
one extra forward evaluation. Checkpointing every layer takes peak memory from
`O(L)` to `O(1)` for about 1.3x the compute.

### Why it matters
This is exactly how large transformers are trained — one `checkpoint` per
transformer layer is standard practice, and it is often the difference between
a model fitting in HBM and not. Expect a follow-up question about where the
extra compute comes from, and about `policy=` for saving only the expensive ops
(like matmuls) while rematerializing the cheap elementwise ones.
""",
    "stub": '''import jax
import jax.numpy as jnp


def block(W, h):
    """One block: a matmul followed by a tanh."""
    return jnp.tanh(h @ W)


def deep_chain(params, x):
    """Apply every block in `params` to x, checkpointing each one.

    Args:
        params: list of (D, D) weight matrices
        x:      (B, D) input activations

    Returns:
        (B, D) final activations.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def block(W, h):
    return jnp.tanh(h @ W)


def deep_chain(params, x):
    h = x
    for W in params:
        # Wrap each block: its internals are recomputed in the backward pass
        # instead of being held in memory from the forward pass.
        h = jax.checkpoint(block)(W, h)
    return h
''',
    "demo": '''import jax
import jax.numpy as jnp

keys = jax.random.split(jax.random.key(0), 4)
params = [jax.random.normal(k, (8, 8)) * 0.5 for k in keys]
x = jax.random.normal(jax.random.key(1), (2, 8))

out = deep_chain(params, x)
print("output shape:", out.shape)

# The remat2 primitive in the gradient's jaxpr is the proof it is checkpointed.
jaxpr = str(jax.make_jaxpr(lambda p, v: jnp.sum(deep_chain(p, v)))(params, x))
print("checkpointed:", "remat" in jaxpr)
''',
    "tests": [
        {
            "name": "Forward matches the unwrapped chain",
            "code": """
import jax
import jax.numpy as jnp

keys = jax.random.split(jax.random.key(0), 5)
params = [jax.random.normal(k, (6, 6)) * 0.5 for k in keys]
x = jax.random.normal(jax.random.key(1), (3, 6))

out = {fn}(params, x)

ref = x
for W in params:
    ref = jnp.tanh(ref @ W)

assert out.shape == (3, 6), f'Shape mismatch: {out.shape} vs (3, 6)'
assert jnp.allclose(out, ref, atol=1e-5), 'Checkpointing must not change the values'
""",
        },
        {
            "name": "Gradients match the unwrapped chain",
            "code": """
import jax
import jax.numpy as jnp

keys = jax.random.split(jax.random.key(2), 4)
params = [jax.random.normal(k, (5, 5)) * 0.5 for k in keys]
x = jax.random.normal(jax.random.key(3), (2, 5))


def plain(ps, v):
    h = v
    for W in ps:
        h = jnp.tanh(h @ W)
    return jnp.sum(h)


g_yours = jax.grad(lambda ps, v: jnp.sum({fn}(ps, v)), argnums=0)(params, x)
g_ref = jax.grad(plain, argnums=0)(params, x)

assert len(g_yours) == len(g_ref), 'Gradient pytree has the wrong length'
for i, (a, b) in enumerate(zip(g_yours, g_ref)):
    assert a.shape == b.shape, f'Layer {i} grad shape {a.shape} vs {b.shape}'
    assert jnp.allclose(a, b, atol=1e-4), f'Layer {i} gradient differs from reference'

gx_yours = jax.grad(lambda v: jnp.sum({fn}(params, v)))(x)
gx_ref = jax.grad(lambda v: plain(params, v))(x)
assert jnp.allclose(gx_yours, gx_ref, atol=1e-4), 'Input gradient differs'
""",
        },
        {
            "name": "jax.checkpoint is actually applied",
            "code": """
import jax
import jax.numpy as jnp

params = [jnp.eye(4) * 0.9 for _ in range(3)]
x = jnp.ones((2, 4))

jaxpr = str(jax.make_jaxpr(lambda p, v: jnp.sum({fn}(p, v)))(params, x))
assert "remat" in jaxpr, (
    'No remat primitive in the jaxpr — jax.checkpoint does not appear to be '
    'applied. Wrap each block: jax.checkpoint(block)(W, h)'
)

grad_jaxpr = str(
    jax.make_jaxpr(jax.grad(lambda p, v: jnp.sum({fn}(p, v)), argnums=1))(params, x)
)
assert "remat" in grad_jaxpr, 'The backward pass is not rematerializing'
""",
        },
        {
            "name": "Scales to a deep stack",
            "code": """
import jax
import jax.numpy as jnp

keys = jax.random.split(jax.random.key(4), 32)
params = [jax.random.normal(k, (16, 16)) * 0.3 for k in keys]
x = jax.random.normal(jax.random.key(5), (4, 16))

out = {fn}(params, x)
assert out.shape == (4, 16), f'{out.shape} vs (4, 16)'
assert jnp.isfinite(out).all(), 'Non-finite activations'
assert (jnp.abs(out) <= 1.0 + 1e-5).all(), 'tanh output must lie in [-1, 1]'

g = jax.grad(lambda p: jnp.sum({fn}(p, x)))(params)
assert len(g) == 32, f'Expected 32 gradient entries, got {len(g)}'
assert all(jnp.isfinite(gi).all() for gi in g), 'Non-finite gradients'
""",
        },
        {
            "name": "Single block and jit",
            "code": """
import jax
import jax.numpy as jnp

W = jnp.eye(3) * 2.0
x = jnp.ones((1, 3))

out = {fn}([W], x)
assert jnp.allclose(out, jnp.tanh(x @ W), atol=1e-6), f'{out}'

jitted = jax.jit({fn})
assert jnp.allclose(jitted([W], x), out, atol=1e-6), 'jit changes the result'
""",
        },
    ],
}
