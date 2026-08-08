"""Explicit PRNG key splitting — JAX's most distinctive break from PyTorch."""

TASK = {
    "title": "PRNG Keys and Splitting",
    "category": "JAX Fundamentals",
    "order": 5,
    "difficulty": "Easy",
    "function_name": "init_ensemble",
    "hint": (
        "jax.random.split(key, n) gives you n independent subkeys. Give each "
        "model its own subkey — reusing one key for every draw makes all the "
        "models identical, which is the classic JAX bug. Note that "
        "jax.random.normal(key, (n, *shape)) in ONE call is also valid and "
        "faster, but this exercise wants per-model keys so each member is "
        "reproducible on its own."
    ),
    "description": r"""
Initialise an **ensemble** of `n_models` weight matrices from a single PRNG key.

JAX has no global random state. Every draw takes an explicit `key`, and the
same key always produces the same numbers. To get independent randomness you
must **split**.

### Rules
- Give every model its **own** subkey derived from `key` via `jax.random.split`
- Never reuse the same key for two different draws
- Do not consume the caller's `key` directly for the sample itself
- Return shape `(n_models, *shape)`, standard normal values
- `init_ensemble(key, ...)` called twice with the same key must be identical

### Signature
```python
def init_ensemble(key, n_models, shape):  # -> (n_models, *shape)
    ...
```

### The bug this problem exists to teach
```python
# WRONG — every model gets identical weights
[jax.random.normal(key, shape) for _ in range(n)]

# WRONG — subtle: reuses `key` after splitting from it
key, sub = jax.random.split(key)
a = jax.random.normal(key, shape)
```

### Why it matters
Explicit keys are what make JAX reproducible under `jit`, `vmap`, and multi-host
parallelism — the same program gives the same numbers regardless of how it is
compiled or sharded. Every JAX interview probes whether you understand this.
""",
    "stub": '''import jax
import jax.numpy as jnp


def init_ensemble(key, n_models, shape):
    """Draw n_models independent standard-normal arrays of the given shape.

    Args:
        key:      a jax.random key
        n_models: number of ensemble members
        shape:    tuple, the shape of ONE member

    Returns:
        Array of shape (n_models, *shape).
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def init_ensemble(key, n_models, shape):
    # One independent subkey per member — never reuse a key across draws.
    keys = jax.random.split(key, n_models)
    return jnp.stack([jax.random.normal(k, shape) for k in keys])
''',
    "demo": '''import jax

key = jax.random.key(0)
w = init_ensemble(key, 4, (2, 3))
print("shape:", w.shape)
print("member 0 == member 1?", bool((w[0] == w[1]).all()), "(should be False)")

again = init_ensemble(jax.random.key(0), 4, (2, 3))
print("reproducible?", bool((w == again).all()), "(should be True)")
''',
    "tests": [
        {
            "name": "Output shape",
            "code": """
import jax

out = {fn}(jax.random.key(0), 5, (2, 3))
assert out.shape == (5, 2, 3), f'Shape mismatch: {out.shape} vs (5, 2, 3)'

flat = {fn}(jax.random.key(0), 3, (4,))
assert flat.shape == (3, 4), f'Shape mismatch: {flat.shape} vs (3, 4)'

scalarish = {fn}(jax.random.key(1), 2, ())
assert scalarish.shape == (2,), f'Shape mismatch on empty shape: {scalarish.shape}'
""",
        },
        {
            "name": "Members are independent, not copies",
            "code": """
import jax
import jax.numpy as jnp

out = {fn}(jax.random.key(0), 6, (4, 4))
for i in range(6):
    for j in range(i + 1, 6):
        assert not jnp.allclose(out[i], out[j]), (
            f'Members {i} and {j} are identical — you reused one key. '
            'Use jax.random.split(key, n_models) and give each member its own subkey.'
        )
""",
        },
        {
            "name": "Deterministic for a given key",
            "code": """
import jax
import jax.numpy as jnp

a = {fn}(jax.random.key(42), 4, (3, 3))
b = {fn}(jax.random.key(42), 4, (3, 3))
assert jnp.array_equal(a, b), 'Same key must reproduce exactly the same values'

c = {fn}(jax.random.key(43), 4, (3, 3))
assert not jnp.allclose(a, c), 'Different keys must produce different values'
""",
        },
        {
            "name": "Standard normal statistics",
            "code": """
import jax
import jax.numpy as jnp

out = {fn}(jax.random.key(7), 64, (64, 8))
m, s = float(jnp.mean(out)), float(jnp.std(out))
assert abs(m) < 0.05, f'Mean should be ~0 for a standard normal, got {m:.4f}'
assert abs(s - 1.0) < 0.05, f'Std should be ~1 for a standard normal, got {s:.4f}'
assert jnp.isfinite(out).all(), 'Non-finite values in output'
""",
        },
        {
            "name": "Caller's key is not consumed for the samples",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.key(0)
out = {fn}(key, 4, (8,))

# No member may equal a draw made straight from the un-split key: that would
# mean the parent key was used for data as well as for splitting.
direct = jax.random.normal(key, (8,))
for i in range(4):
    assert not jnp.allclose(out[i], direct), (
        f'Member {i} was drawn from the caller key itself. Split first, '
        'then sample only from the subkeys.'
    )
""",
        },
    ],
}
