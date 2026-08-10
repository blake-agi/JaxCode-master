"""Gradient checkpointing with jax.checkpoint — trading compute for memory."""

TASK = {
    "title": "Gradient Checkpointing with jax.checkpoint",
    "category": "JAX Fundamentals",
    "order": 10,
    "number": "b_10",
    "difficulty": "Medium",
    "function_name": "deep_chain",
    "hint": (
        "jax.checkpoint is a transform, not a call: it takes a FUNCTION and hands "
        "back a new function with the same signature, so you apply it to `block` "
        "and then call the result on the arguments. Granularity is the whole "
        "decision — one checkpoint around the entire chain buys you nothing, so "
        "the wrapping belongs inside the loop, once per block. Nothing about the "
        "numbers changes; only what gets stored between forward and backward."
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
Reverse-mode autodiff normally keeps every intermediate from the forward pass
alive until the backward pass consumes it. For `L` layers that each stash `k`
tensors internally, that is $O(kL)$.

`jax.checkpoint` says: *don't keep this block's internals — recompute them when
the backward pass gets here.* Checkpointing every layer stores one tensor per
layer **boundary** instead of `k` per layer, so $O(kL)$ becomes $O(L)$.

Be precise about that: it is a large constant-factor win, **not** an asymptotic
one. The boundary activations still grow with depth. The genuinely sublinear
scheme is Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost*,
which checkpoints every $\sqrt{L}$-th layer for $O(\sqrt{L})$ memory.

The price is one extra forward evaluation of each rematerialized region. A
backward pass already costs roughly 2x a forward pass, so recomputing the
forward once takes the total from about 3x to about 4x — the familiar "~33%
more compute" figure.

### The granularity trap
Wrapping the *whole* chain in one `checkpoint` looks like the aggressive choice
and saves nothing. Remat drops the intermediates during the forward pass, but
the backward pass then has to recompute the entire chain in one go — and that
recomputation materializes all `L` layers' activations anyway. Peak memory is
unchanged and you paid for the extra forward. The memory you get is decided by
how finely you cut the chain.

### Why it matters
This is exactly how large transformers are trained — one `checkpoint` per
transformer layer is standard practice, and it is often the difference between
a model fitting in HBM and not. Expect a follow-up about where the extra compute
comes from, and about `policy=` — e.g.
`jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)`
keeps the expensive matmul outputs and rematerializes only the cheap elementwise
ops, which recovers most of the memory for a fraction of the recompute.
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
