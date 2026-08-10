"""Discounted returns via a reverse lax.scan — carry/xs mechanics."""

TASK = {
    "title": "Discounted Returns with lax.scan",
    "category": "JAX Fundamentals",
    "order": 6,
    "number": "b_06",
    "difficulty": "Medium",
    "function_name": "discounted_returns",
    "hint": (
        "The recurrence runs backwards, so pass reverse=True to jax.lax.scan. "
        "Your step function has signature f(carry, x) -> (new_carry, y). Here the "
        "carry is the running return G, x is one reward, and you emit the new G "
        "as the output: f(G, r) = (r + gamma*G, r + gamma*G). Start the carry at 0."
    ),
    "description": r"""
Compute **discounted returns** for a reward sequence — the backbone of every
policy-gradient algorithm.

$$G_t = r_t + \gamma\, G_{t+1}, \qquad G_{T-1} = r_{T-1}$$

Given `rewards` of shape `(T,)` and a scalar `gamma`, return `(T,)` returns.

### Rules
- Use `jax.lax.scan` — no Python `for` loop over `T`
- The result must be **jittable** and **differentiable**
- Must work when `T` is large (10,000+) without unrolling

### Example
```
rewards = [1.0, 1.0, 1.0], gamma = 0.9
G[2] = 1.0
G[1] = 1.0 + 0.9 * 1.0  = 1.9
G[0] = 1.0 + 0.9 * 1.9  = 2.71
```

### Why it matters
`lax.scan` is how JAX expresses sequential recurrences without unrolling the
graph. A Python loop over 10,000 steps produces a 10,000-node graph that takes
minutes to compile; `scan` compiles **one** step and loops it, so compile time
is constant and reverse-mode autodiff stays memory-efficient.

The exact same pattern is an RNN cell, a KV-cache decode loop, an optimizer
sweep, or a diffusion sampler. If you know `scan`, you know all of them.
""",
    "stub": '''import jax
import jax.numpy as jnp


def discounted_returns(rewards, gamma):
    """Discounted return at every timestep.

    Args:
        rewards: (T,) array of rewards
        gamma:   scalar discount factor

    Returns:
        (T,) array where out[t] = rewards[t] + gamma * out[t + 1].
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def discounted_returns(rewards, gamma):
    def step(carry, r):
        g = r + gamma * carry
        return g, g          # (new_carry, per-step output)

    # reverse=True walks the sequence from the last element to the first.
    _, out = jax.lax.scan(step, jnp.zeros_like(rewards[0]), rewards, reverse=True)
    return out
''',
    "demo": '''import jax.numpy as jnp

rewards = jnp.array([1.0, 1.0, 1.0])
print(discounted_returns(rewards, 0.9))   # [2.71, 1.9, 1.0]

# A single reward at the end, discounted back through time:
sparse = jnp.array([0.0, 0.0, 0.0, 1.0])
print(discounted_returns(sparse, 0.5))    # [0.125, 0.25, 0.5, 1.0]
''',
    "tests": [
        {
            "name": "Hand-computed example",
            "code": """
import jax.numpy as jnp

out = {fn}(jnp.array([1.0, 1.0, 1.0]), 0.9)
expected = jnp.array([2.71, 1.9, 1.0])
assert out.shape == (3,), f'Shape mismatch: {out.shape} vs (3,)'
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "Matches a reference loop",
            "code": """
import jax
import jax.numpy as jnp

rewards = jax.random.normal(jax.random.key(0), (50,))
gamma = 0.95
out = {fn}(rewards, gamma)

ref = [0.0] * 50
running = 0.0
for t in range(49, -1, -1):
    running = float(rewards[t]) + gamma * running
    ref[t] = running

assert jnp.allclose(out, jnp.array(ref), atol=1e-4), 'Disagrees with the reference loop'
""",
        },
        {
            "name": "Edge cases: gamma=0, gamma=1, T=1",
            "code": """
import jax.numpy as jnp

r = jnp.array([1.0, 2.0, 3.0, 4.0])

# gamma = 0 -> returns are just the rewards
assert jnp.allclose({fn}(r, 0.0), r, atol=1e-6), 'gamma=0 should give back the rewards'

# gamma = 1 -> reverse cumulative sum
assert jnp.allclose({fn}(r, 1.0), jnp.array([10.0, 9.0, 7.0, 4.0]), atol=1e-5), (
    'gamma=1 should give the reverse cumulative sum'
)

# T = 1
one = {fn}(jnp.array([5.0]), 0.9)
assert one.shape == (1,) and jnp.allclose(one, 5.0), f'T=1 case: {one}'
""",
        },
        {
            "name": "Jittable and long sequences stay fast",
            "code": """
import time
import jax
import jax.numpy as jnp

f = jax.jit({fn}, static_argnums=())
r = jnp.ones((20000,))

t0 = time.perf_counter()
out = f(r, 0.99)
out.block_until_ready()
elapsed = time.perf_counter() - t0

assert out.shape == (20000,), f'{out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite values'
# Steady-state return of an all-ones reward stream is 1/(1-gamma) = 100.
assert abs(float(out[0]) - 100.0) < 1.0, f'out[0] should approach 100, got {out[0]}'
assert elapsed < 20.0, (
    f'Took {elapsed:.1f}s on T=20000 — this suggests the graph is being '
    'unrolled with a Python loop instead of using lax.scan'
)
""",
        },
        {
            "name": "Differentiable",
            "code": """
import jax
import jax.numpy as jnp

rewards = jnp.array([1.0, 2.0, 3.0])
gamma = 0.5

g = jax.grad(lambda r: jnp.sum({fn}(r, gamma)))(rewards)
assert g.shape == (3,), f'{g.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'

# d(sum_t G_t)/d r_k = sum_{t<=k} gamma^(k-t) = 1 + g + g^2 + ...
expected = jnp.array([1.0, 1.5, 1.75])
assert jnp.allclose(g, expected, atol=1e-5), f'{g} vs {expected}'
""",
        },
        {
            "name": "Composes with vmap for a batch of episodes",
            "code": """
import jax
import jax.numpy as jnp

batch = jax.random.normal(jax.random.key(1), (8, 20))
out = jax.vmap({fn}, in_axes=(0, None))(batch, 0.9)

assert out.shape == (8, 20), f'{out.shape} vs (8, 20)'
for i in range(8):
    assert jnp.allclose(out[i], {fn}(batch[i], 0.9), atol=1e-5), (
        f'vmapped row {i} disagrees with the unbatched call'
    )
""",
        },
    ],
}
