"""Problem 26 without Flax — and freezing becomes trivial."""

_LINEAR = '''class Linear:
    """Given to you, as nnx.Linear is in problem 26."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

TASK = {
    "title": "LoRA without Flax",
    "category": "Training",
    "number": "b_32",
    "difficulty": "Medium",
    "function_name": "LoRALinear",
    "hint": (
        "self.linear = Linear(in_features, out_features), then "
        "self.lora_A = jax.random.normal(key, (in_features, rank)) * 0.01 and "
        "self.lora_B = jnp.zeros((rank, out_features)). B starts at ZERO so "
        "B @ A is zero and the adapter is a no-op at step 0 — that is what "
        "makes LoRA safe to attach to a trained model. self.scaling = alpha / "
        "rank. __call__ is linear(x) + (x @ lora_A @ lora_B) * scaling."
    ),
    "description": r"""
Problem 26's LoRA adapter with no Flax.

### Signature
```python
class LoRALinear:
    def __init__(self, in_features, out_features, rank, alpha=1.0, *, key): ...
    def __call__(self, x): ...        # (..., in_features) -> (..., out_features)
```

| | shape |
|---|---|
| `self.linear` | a `Linear(in_features, out_features)` — the frozen base |
| `self.lora_A` | `(in_features, rank)`, random, scaled by `0.01` |
| `self.lora_B` | `(rank, out_features)`, **zeros** |
| `self.scaling` | `alpha / rank` |

$$y = Wx + b + \frac{\alpha}{r}\,(x A) B$$

### B starts at zero, and that is the whole trick
`B = 0` makes `A @ B` zero, so at step 0 the adapter is an **exact no-op** and
the model behaves exactly as it did before you attached it. Fine-tuning then
moves away from the original smoothly instead of jolting it.

Initialising **both** to zero would be worse than useless — the gradient of a
product where both factors are zero is zero, so nothing would ever learn. One
random, one zero.

### Freezing gets easier, not harder
This is the one place where dropping Flax **simplifies** things. Problem 26
had to demote the base parameters to plain `nnx.Variable` so that
`nnx.state(self, nnx.Param)` — what an optimizer filters on — would see only
the adapter:

```python
self.linear.kernel = nnx.Variable(self.linear.kernel[...])   # problem 26
```

Here there is nothing to demote. You decide what to differentiate by choosing
what to pass to `jax.grad`, so "frozen" just means "not in the argument":

```python
jax.grad(loss)(A, B)      # the base never enters
```

### Why rank matters
The adapter costs `r · (in + out)` parameters instead of `in · out`. For a
4096x4096 projection at `r = 8` that is 65k instead of 16.7M — 0.4%. `alpha /
rank` keeps the update magnitude roughly constant as you change `r`, so you
can retune the rank without retuning the learning rate.

### Why this exists alongside problem 26
Interview sandboxes ship `jax` but not `flax`. Same class name, same argument
names, same `linear`/`lora_A`/`lora_B`/`scaling` attributes.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class LoRALinear:
    """A frozen base projection plus a low-rank trainable adapter."""

    def __init__(self, in_features, out_features, rank, alpha=1.0, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class LoRALinear:
    def __init__(self, in_features, out_features, rank, alpha=1.0, *, key):
        k_base, k_a = jax.random.split(key, 2)
        self.linear = Linear(in_features, out_features, key=k_base)

        # A random, B zero -> A @ B == 0 -> the adapter starts as an exact
        # no-op. Zeroing BOTH would kill the gradient and it would never learn.
        self.lora_A = jax.random.normal(k_a, (in_features, rank)) * 0.01
        self.lora_B = jnp.zeros((rank, out_features))
        self.scaling = alpha / rank

    def __call__(self, x):
        # Keep the two x @ A @ B matmuls separate: never form A @ B, which
        # would be (in, out) and defeat the point.
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
''',
    "demo": '''import jax
import jax.numpy as jnp

lora = LoRALinear(8, 4, rank=2, alpha=4.0, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (3, 8))

print("A", lora.lora_A.shape, " B", lora.lora_B.shape, " scaling", lora.scaling)
print("adapter is a no-op at init:", bool(jnp.allclose(lora(x), lora.linear(x))))

lora.lora_B = jnp.ones((2, 4)) * 0.1
print("after training B:          ", bool(jnp.allclose(lora(x), lora.linear(x))))

# "Frozen" is just a choice of what to differentiate.
def loss(A, B):
    lora.lora_A, lora.lora_B = A, B
    return jnp.sum(lora(x))

full = 8 * 4
adapter = 2 * (8 + 4)
print(f"\\nparams: base {full}, adapter {adapter} ({adapter / full:.0%})")
''',
    "tests": [
        {
            "name": "Attributes and shapes match problem 26",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 4, rank=2, alpha=4.0, key=jax.random.key(0))
for name in ('linear', 'lora_A', 'lora_B', 'scaling'):
    assert hasattr(m, name), f'missing {name} — keep problem 26 names'

assert m.linear.kernel.shape == (8, 4), f'base kernel {m.linear.kernel.shape} vs (8, 4)'
assert m.lora_A.shape == (8, 2), f'lora_A {m.lora_A.shape} vs (in_features, rank)'
assert m.lora_B.shape == (2, 4), f'lora_B {m.lora_B.shape} vs (rank, out_features)'
assert abs(float(m.scaling) - 4.0 / 2) < 1e-9, (
    f'scaling {m.scaling}, expected alpha / rank = {4.0/2}'
)
""",
        },
        {
            "name": "B starts at zero and A does not",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 4, rank=2, alpha=1.0, key=jax.random.key(0))
assert jnp.allclose(m.lora_B, 0.0), (
    'lora_B must start at zeros so the adapter is an exact no-op at step 0'
)
assert not jnp.allclose(m.lora_A, 0.0), (
    'lora_A must be random. Zeroing BOTH factors makes the gradient of their '
    'product zero, so the adapter would never learn anything.'
)
assert float(jnp.std(m.lora_A)) < 0.1, (
    f'lora_A std {float(jnp.std(m.lora_A)):.4f} — it should be small (~0.01), '
    'so the adapter starts near zero even once B moves'
)
""",
        },
        {
            "name": "At init the adapter is exactly the base layer",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 4, rank=2, alpha=4.0, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (5, 8))

assert jnp.allclose(m(x), m.linear(x), atol=1e-7), (
    'at initialisation LoRALinear(x) must equal the base linear(x) exactly — '
    'that is what makes it safe to attach to an already-trained model'
)
for shape in [(8,), (5, 8), (2, 3, 8)]:
    xx = jax.random.normal(jax.random.key(2), shape)
    assert m(xx).shape == shape[:-1] + (4,), f'{shape} -> {m(xx).shape}'
""",
        },
        {
            "name": "The formula, once B is non-zero",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 4, rank=2, alpha=4.0, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (5, 8))
m.lora_B = jax.random.normal(jax.random.key(3), (2, 4))

want = m.linear(x) + (x @ m.lora_A @ m.lora_B) * m.scaling
assert jnp.allclose(m(x), want, atol=1e-5), (
    'should be linear(x) + (x @ A @ B) * scaling'
)
assert not jnp.allclose(m(x), m.linear(x), atol=1e-4), 'the adapter is being ignored'

# scaling must actually scale: doubling alpha doubles the adapter's contribution.
m2 = {fn}(8, 4, rank=2, alpha=8.0, key=jax.random.key(0))
m2.lora_B = m.lora_B
delta1 = m(x) - m.linear(x)
delta2 = m2(x) - m2.linear(x)
assert jnp.allclose(delta2, 2.0 * delta1, atol=1e-5), (
    'doubling alpha should double the adapter term — scaling is alpha / rank'
)
""",
        },
        {
            "name": "The adapter trains; the base is frozen by choice",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 4, rank=2, alpha=4.0, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (16, 8))
y = jax.random.normal(jax.random.key(2), (16, 4))
base_kernel = m.linear.kernel

def loss(A, B):
    pred = m.linear(x) + (x @ A @ B) * m.scaling
    return jnp.mean((pred - y) ** 2)

gA, gB = jax.grad(loss, argnums=(0, 1))(m.lora_A, m.lora_B)
assert gA.shape == m.lora_A.shape and gB.shape == m.lora_B.shape, 'grad shapes'
assert jnp.isfinite(gA).all() and jnp.isfinite(gB).all(), 'non-finite gradient'
assert float(jnp.abs(gB).max()) > 0, 'B got no gradient'

# A gradient step must lower the loss, and the base must be untouched — not
# because it is marked frozen, but because it was never an argument.
l0 = float(loss(m.lora_A, m.lora_B))
A2 = m.lora_A - 0.05 * gA
B2 = m.lora_B - 0.05 * gB
assert float(loss(A2, B2)) < l0, 'a gradient step did not reduce the loss'
assert jnp.array_equal(m.linear.kernel, base_kernel), 'the base weights moved'
""",
        },
    ],
}
