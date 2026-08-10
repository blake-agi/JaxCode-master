"""LoRA — a frozen base layer plus a trainable low-rank detour."""

TASK = {
    "title": "LoRA (Low-Rank Adaptation)",
    "category": "Attention & Transformers",
    "number": "26",
    "difficulty": "Medium",
    "function_name": "LoRALinear",
    "hint": (
        "self.linear is an ordinary nnx.Linear you never train. lora_A is small "
        "random, lora_B is ZEROS — so B @ A is zero and the adapter starts as an "
        "exact no-op. Forward is linear(x) + (x @ A @ B) * scaling with "
        "scaling = alpha / rank. To keep the base out of the optimizer, hold its "
        "kernel and bias as nnx.Variable rather than nnx.Param, so "
        "nnx.state(layer, nnx.Param) returns only the adapter."
    ),
    "description": r"""
Implement **LoRA**: freeze a pretrained linear layer and learn a low-rank
correction beside it.

$$h = W_0 x + \frac{\alpha}{r}\,(B A) x, \qquad
A \in \mathbb{R}^{d_{in} \times r},\; B \in \mathbb{R}^{r \times d_{out}}$$

### Signature
```python
class LoRALinear(nnx.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0,
                 *, rngs: nnx.Rngs): ...
    def __call__(self, x): ...
```

### Requirements
- `self.linear`: an `nnx.Linear(in_features, out_features)` that is **frozen**
- `self.lora_A`: small random, shape `(in_features, rank)`
- `self.lora_B`: **zeros**, shape `(rank, out_features)`
- `self.scaling = alpha / rank`
- Only `lora_A` and `lora_B` are trainable

### Why B starts at zero
At initialization $BA = 0$, so the adapted layer is **exactly** the pretrained
one. Fine-tuning therefore starts from the pretrained function rather than a
perturbed one — no warmup, no risk of destroying the base model on step one.
If both $A$ and $B$ were zero the gradient would also be zero and nothing would
ever learn, so exactly one of them is zeroed: random $A$ gives $B$ something to
receive gradient through.

### The parameter arithmetic
A $4096 \times 4096$ layer holds 16.8M weights. With $r = 8$ the adapter holds
$4096 \times 8 + 8 \times 4096 = 65{,}536$ — about **0.4%**. That is what makes
it possible to keep dozens of task-specific adapters for one base model, and to
train on a single GPU: the optimizer state, which for Adam is 2 floats per
trainable parameter, shrinks by the same factor.

### What $\alpha/r$ is for
It decouples the learning rate from the rank. Without it, doubling $r$ roughly
doubles the update magnitude and you would have to retune the learning rate
every time. With the scaling, $r$ becomes a capacity knob you can sweep freely.

### At inference, LoRA is free
$W_0 + \frac{\alpha}{r}BA$ can be folded into a single matrix once training
ends, so a merged adapter costs exactly nothing at serving time. The detour
only exists while you are training.

### ⚠️ A JAX layout note
PyTorch stores `lora_A` as `(rank, in_features)` and computes `x @ A.T @ B.T`.
Flax kernels are `(in, out)`, so here `A` is `(in_features, rank)` and `B` is
`(rank, out_features)` and the forward pass is a plain `x @ A @ B` — the same
maths, transposed to match the surrounding convention.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class LoRALinear(nnx.Module):
    """Frozen linear layer plus a trainable low-rank adapter."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float = 1.0, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(..., in_features) -> (..., out_features)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class LoRALinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float = 1.0, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

        # Freeze the base: demote its Params to plain Variables so
        # nnx.state(self, nnx.Param) — what an optimizer filters on — sees only
        # the adapter. This is the JAX counterpart of requires_grad_(False).
        self.linear.kernel = nnx.Variable(self.linear.kernel[...])
        self.linear.bias = nnx.Variable(self.linear.bias[...])

        # A random, B zero -> B @ A == 0 -> the adapter starts as a no-op.
        self.lora_A = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, rank)) * 0.01
        )
        self.lora_B = nnx.Param(jnp.zeros((rank, out_features)))
        self.scaling = alpha / rank

    def __call__(self, x):
        return self.linear(x) + (x @ self.lora_A[...] @ self.lora_B[...]) * self.scaling
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

layer = LoRALinear(64, 64, rank=4, alpha=8.0, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(1), (2, 64))

base = layer.linear(x)
print("adapter is a no-op at init:", bool(jnp.allclose(layer(x), base)))

trainable = sum(v.size for v in jax.tree.leaves(nnx.state(layer, nnx.Param)))
print(f"trainable: {trainable} of {64*64 + 64} base weights "
      f"({100*trainable/(64*64+64):.1f}%)")
''',
    "tests": [
        {
            "name": "Structure: frozen base, adapter shapes",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(8, 4, rank=2, alpha=4.0, rngs=nnx.Rngs(params=0))

assert isinstance(layer.linear, nnx.Linear), (
    f'self.linear must be an nnx.Linear, got {type(layer.linear)}'
)
assert layer.lora_A[...].shape == (8, 2), (
    f'lora_A should be (in_features, rank) = (8, 2), got {layer.lora_A[...].shape}'
)
assert layer.lora_B[...].shape == (2, 4), (
    f'lora_B should be (rank, out_features) = (2, 4), got {layer.lora_B[...].shape}'
)
assert abs(layer.scaling - 2.0) < 1e-6, f'scaling should be alpha/rank = 2.0, got {layer.scaling}'

x = jax.random.normal(jax.random.key(1), (3, 8))
assert layer(x).shape == (3, 4), f'{layer(x).shape}'
""",
        },
        {
            "name": "B is zero, so the adapter starts as an exact no-op",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(16, 16, rank=4, alpha=8.0, rngs=nnx.Rngs(params=2))

assert jnp.allclose(layer.lora_B[...], 0.0), (
    'lora_B must be initialised to ZEROS so the adapter contributes nothing '
    'at step 0 and fine-tuning starts from the pretrained function'
)
assert not jnp.allclose(layer.lora_A[...], 0.0), (
    'lora_A must NOT be zero — if both were zero the gradient would be zero '
    'too and the adapter could never learn'
)

x = jax.random.normal(jax.random.key(3), (4, 16))
assert jnp.allclose(layer(x), layer.linear(x), atol=1e-6), (
    'At init the output must equal the frozen base layer exactly'
)
""",
        },
        {
            "name": "Only the adapter is trainable",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(16, 8, rank=2, alpha=2.0, rngs=nnx.Rngs(params=4))
params = nnx.state(layer, nnx.Param)
total = sum(v.size for v in jax.tree.leaves(params))

expected = 16 * 2 + 2 * 8      # lora_A + lora_B
assert total == expected, (
    f'nnx.state(layer, nnx.Param) holds {total} values, expected {expected} '
    '(lora_A + lora_B only). The base layer must be frozen — an optimizer '
    'filtering on nnx.Param would otherwise train it.'
)
""",
        },
        {
            "name": "Matches the reference formula once B is non-zero",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(8, 4, rank=2, alpha=6.0, rngs=nnx.Rngs(params=5))
layer.lora_B[...] = jax.random.normal(jax.random.key(6), (2, 4))

x = jax.random.normal(jax.random.key(7), (3, 8))
ref = layer.linear(x) + (x @ layer.lora_A[...] @ layer.lora_B[...]) * (6.0 / 2)
assert jnp.allclose(layer(x), ref, atol=1e-5), 'Output does not match W0 x + (alpha/r) B A x'
""",
        },
        {
            "name": "scaling is alpha / rank",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

for rank, alpha in ((2, 8.0), (4, 8.0), (8, 16.0)):
    layer = {fn}(8, 8, rank=rank, alpha=alpha, rngs=nnx.Rngs(params=8))
    assert abs(layer.scaling - alpha / rank) < 1e-6, (
        f'rank={rank}, alpha={alpha}: scaling should be {alpha/rank}, got {layer.scaling}'
    )

# Doubling alpha doubles the adapter's contribution.
a = {fn}(8, 8, rank=2, alpha=2.0, rngs=nnx.Rngs(params=9))
b = {fn}(8, 8, rank=2, alpha=4.0, rngs=nnx.Rngs(params=9))
delta = jax.random.normal(jax.random.key(10), (2, 8))
a.lora_B[...] = delta
b.lora_B[...] = delta
x = jax.random.normal(jax.random.key(11), (3, 8))
da = a(x) - a.linear(x)
db = b(x) - b.linear(x)
assert jnp.allclose(db, 2 * da, atol=1e-5), 'Doubling alpha must double the adapter output'
""",
        },
        {
            "name": "Gradients reach the adapter and not the base",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(8, 4, rank=2, alpha=2.0, rngs=nnx.Rngs(params=12))
layer.lora_B[...] = jnp.ones((2, 4)) * 0.1
x = jax.random.normal(jax.random.key(13), (4, 8))

grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(layer)
state = nnx.state(grads)

for name in ("lora_A", "lora_B"):
    g = state[name]
    val = g[...] if isinstance(g, nnx.Variable) else g
    assert jnp.isfinite(val).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(val).sum()) > 0, f'No gradient reached {name}'

flat = jax.tree.leaves(state)
assert len(flat) == 2, (
    f'nnx.grad should differentiate only the two adapter matrices, got '
    f'{len(flat)} gradient leaves — the base layer is not frozen'
)
""",
        },
        {
            "name": "Merging the adapter reproduces the output",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# At inference W0 + (alpha/r) A B folds into one matrix, so LoRA is free.
layer = {fn}(8, 4, rank=2, alpha=4.0, rngs=nnx.Rngs(params=14))
layer.lora_B[...] = jax.random.normal(jax.random.key(15), (2, 4))

x = jax.random.normal(jax.random.key(16), (5, 8))
merged_kernel = layer.linear.kernel[...] + layer.scaling * (
    layer.lora_A[...] @ layer.lora_B[...]
)
merged = x @ merged_kernel + layer.linear.bias[...]

assert jnp.allclose(layer(x), merged, atol=1e-5), (
    'A merged weight must reproduce the LoRA output exactly — if not, the '
    'adapter is not a plain additive low-rank term'
)
""",
        },
    ],
}
