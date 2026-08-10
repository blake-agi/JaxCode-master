"""Kaiming/He initialization — variance preservation through a ReLU stack."""

TASK = {
    "title": "Kaiming Initialization",
    "category": "Core Ops & Layers",
    "number": "20",
    "difficulty": "Easy",
    "function_name": "kaiming_init",
    "hint": (
        "fan_in is the number of INPUTS to a unit — for a (din, dout) kernel that "
        "is shape[0], and for a 1-D array there is only one axis to take. The "
        "std is sqrt(2 / fan_in); the 2 is the ReLU correction. JAX has no global "
        "RNG and no in-place fill, so you take a key and return a new array."
    ),
    "description": r"""
Implement **Kaiming (He) initialization**, the standard for ReLU networks.

$$W \sim \mathcal{N}\!\left(0,\ \sigma^2\right), \qquad
\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$

### Signature
```python
def kaiming_init(key, weight):
    ...  # -> a new array shaped like `weight`
```

### Rules
- Draw from a **normal** distribution with mean 0
- `std = sqrt(2 / fan_in)`
- `fan_in` is the input dimension, which is `weight.shape[0]` in both cases:
  a 2-D Flax kernel is `(in_features, out_features)`, and a 1-D array has only
  one axis
- Do not use `nnx.initializers` or `jax.nn.initializers`

### Where the 2 comes from
For a linear layer $y = Wx$ with independent zero-mean weights,
$\mathrm{Var}(y) = \text{fan\_in} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}(x)$.
To keep the variance from shrinking or exploding layer over layer you want that
factor to be 1, giving Xavier's $\mathrm{Var}(W) = 1/\text{fan\_in}$.

But ReLU zeroes half its inputs, which halves the output variance. Kaiming
compensates by doubling: $\mathrm{Var}(W) = 2/\text{fan\_in}$. That single
factor of 2 is the entire difference from Xavier — and it is what made it
possible to train very deep ReLU networks without careful layer-wise
pre-training.

Use Kaiming with ReLU/GELU/SiLU, Xavier with tanh/sigmoid.

### ⚠️ Two things JAX forces here
1. **A key argument.** PyTorch calls `weight.normal_(0, std)`, drawing from a
   hidden global RNG. JAX has no global RNG, so randomness is explicit — the
   key is the first argument, by convention.
2. **A return value.** `normal_` fills in place; JAX arrays are immutable, so
   this returns a new array instead.

`weight` is therefore only used for its **shape and dtype**. Passing the array
rather than the shape keeps the call site looking like the PyTorch original.
""",
    "stub": '''import jax
import jax.numpy as jnp


def kaiming_init(key, weight):
    """Kaiming-normal values shaped like `weight`.

    Args:
        key:    a jax.random key
        weight: array whose shape (and dtype) the result should match

    Returns:
        A NEW array shaped like `weight`, drawn from N(0, sqrt(2/fan_in)).
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def kaiming_init(key, weight):
    # fan_in = number of inputs feeding one unit. For a (din, dout) kernel
    # that is axis 0 — note this is the JAX/Flax layout, the transpose of
    # PyTorch's (out, in).
    fan_in = weight.shape[0]
    std = jnp.sqrt(2.0 / fan_in)
    # No in-place fill and no global RNG: draw fresh values from the key.
    return jax.random.normal(key, weight.shape, dtype=weight.dtype) * std
''',
    "demo": '''import jax
import jax.numpy as jnp

w = jnp.zeros((256, 64))
out = kaiming_init(jax.random.key(0), w)

print("shape:", out.shape)
print("mean :", float(out.mean()), "(~0)")
print("std  :", float(out.std()), "vs target", float(jnp.sqrt(2 / 256)))

# Variance is preserved through a ReLU stack — that is the whole point.
x = jax.random.normal(jax.random.key(1), (1000, 256))
for layer in range(5):
    w = kaiming_init(jax.random.key(layer + 2), jnp.zeros((x.shape[-1], 256)))
    x = jax.nn.relu(x @ w)
    print(f"  after layer {layer}: std {float(x.std()):.3f}")
''',
    "tests": [
        {
            "name": "Correct std and mean",
            "code": """
import jax
import jax.numpy as jnp

w = jnp.zeros((512, 128))
out = {fn}(jax.random.key(0), w)

assert out.shape == w.shape, f'Shape mismatch: {out.shape} vs {w.shape}'

target = jnp.sqrt(2.0 / 512)
assert jnp.allclose(out.std(), target, rtol=0.05), (
    f'std should be sqrt(2/fan_in) = {float(target):.5f}, got {float(out.std()):.5f}. '
    f'Xavier (sqrt(1/fan_in) = {float(jnp.sqrt(1/512)):.5f}) is the usual wrong answer.'
)
assert jnp.allclose(out.mean(), 0.0, atol=0.02), f'mean should be ~0, got {float(out.mean())}'
""",
        },
        {
            "name": "fan_in is the input axis",
            "code": """
import jax
import jax.numpy as jnp

# Very different fan_in on each axis, so picking the wrong one is unmissable.
out = {fn}(jax.random.key(1), jnp.zeros((1024, 16)))
assert jnp.allclose(out.std(), jnp.sqrt(2.0 / 1024), rtol=0.06), (
    f'For a (1024, 16) kernel fan_in is 1024, so std should be '
    f'{float(jnp.sqrt(2/1024)):.5f}, got {float(out.std()):.5f}. '
    f'Using shape[1]=16 would give {float(jnp.sqrt(2/16)):.5f}.'
)
""",
        },
        {
            "name": "Not uniform, and not all one value",
            "code": """
import jax
import jax.numpy as jnp

out = {fn}(jax.random.key(2), jnp.zeros((4096,)))
assert float(out.std()) > 0, 'Output is constant'

# A normal draw puts ~0.3% of mass beyond 3 sigma; a uniform draw puts none.
s = float(out.std())
frac = float(jnp.mean(jnp.abs(out) > 2.5 * s))
assert frac > 0.002, (
    f'Only {frac:.4%} of samples exceed 2.5 sigma — that looks like a uniform '
    'distribution. Kaiming-normal draws from a normal.'
)
""",
        },
        {
            "name": "Deterministic in the key",
            "code": """
import jax
import jax.numpy as jnp

w = jnp.zeros((64, 32))
a = {fn}(jax.random.key(0), w)
b = {fn}(jax.random.key(0), w)
c = {fn}(jax.random.key(1), w)

assert jnp.array_equal(a, b), 'Same key must give the same values'
assert not jnp.array_equal(a, c), 'Different keys must give different values'
""",
        },
        {
            "name": "Does not mutate the input",
            "code": """
import jax
import jax.numpy as jnp

w = jnp.zeros((32, 8))
out = {fn}(jax.random.key(3), w)

assert jnp.allclose(w, 0.0), (
    'The array passed in must be untouched — JAX has no in-place normal_(), '
    'which is why this returns a new array'
)
assert not jnp.allclose(out, 0.0), 'The returned array should be filled'
""",
        },
        {
            "name": "1-D arrays work",
            "code": """
import jax
import jax.numpy as jnp

out = {fn}(jax.random.key(4), jnp.zeros((256,)))
assert out.shape == (256,), f'Shape mismatch: {out.shape}'
assert jnp.allclose(out.std(), jnp.sqrt(2.0 / 256), rtol=0.12), (
    f'1-D fan_in is the only axis, so std should be {float(jnp.sqrt(2/256)):.5f}, '
    f'got {float(out.std()):.5f}'
)
""",
        },
        {
            "name": "Preserves variance through a deep ReLU stack",
            "code": """
import jax
import jax.numpy as jnp

# The reason Kaiming exists: activations should neither vanish nor explode.
x = jax.random.normal(jax.random.key(5), (2000, 128))
start = float(x.std())
for i in range(12):
    w = {fn}(jax.random.key(100 + i), jnp.zeros((128, 128)))
    x = jax.nn.relu(x @ w)

end = float(x.std())
assert jnp.isfinite(x).all(), 'Activations blew up to non-finite values'
assert 0.3 < end / start < 3.0, (
    f'Activation std went {start:.3f} -> {end:.3f} over 12 ReLU layers '
    f'(ratio {end/start:.3f}). Kaiming should hold this near 1; Xavier '
    'would decay it by roughly 2^(-12/2).'
)
""",
        },
    ],
}
