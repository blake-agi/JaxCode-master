"""LoRA — why B must start at zero, and where the parameter savings come from."""

TASK = {
    "title": "LoRA (Low-Rank Adaptation)",
    "category": "Attention & Transformers",
    "number": "26",
    "difficulty": "Medium",
    "function_name": "LoRALinear",
    "hint": (
        "Store the base weight as a plain nnx.Variable so an nnx.Param filter "
        "does not pick it up — that is what 'frozen' means here. A is (din, r) "
        "with a small random init, B is (r, dout) initialised to ZEROS. Forward "
        "is x @ W + (alpha / r) * ((x @ A) @ B); keep those two matmuls "
        "separate, since collapsing to A @ B first builds the full (din, dout) "
        "matrix you were trying to avoid."
    ),
    "description": r"""
Implement a **LoRA**-adapted linear layer as an `nnx.Module`.

$$h = xW + \frac{\alpha}{r}\,(xA)B$$

where $W \in \mathbb{R}^{d_{in}\times d_{out}}$ is **frozen**,
$A \in \mathbb{R}^{d_{in}\times r}$ and $B \in \mathbb{R}^{r\times d_{out}}$ are
trainable, and $r \ll \min(d_{in}, d_{out})$.

### Signature
```python
class LoRALinear(nnx.Module):
    def __init__(self, din, dout, rank, alpha=1.0, *, rngs: nnx.Rngs):
        ...
    def __call__(self, x):
        ...  # (..., din) -> (..., dout)
```

### Rules
- Name the three tensors `self.W` `(din, dout)`, `self.A` `(din, rank)` and
  `self.B` `(rank, dout)` — the tests read and overwrite them by name
- The base weight must **not** be an `nnx.Param` — use a plain `nnx.Variable`,
  so `nnx.split(model, nnx.Param, ...)` sees only the adapter
- `A`: random init (scaled normal is fine). `B`: **zeros**
- Keep `(x @ A) @ B` factored; never form `A @ B`
- Scale by `alpha / rank`
- No bias term

### Why B must be zero
At initialization $BA = 0$, so $h = xW$ exactly — the adapted model is
**identical** to the base model. Training therefore starts from the pretrained
function rather than from a randomly perturbed one, which is what makes LoRA
stable at high learning rates.

Both matrices zero would be broken instead: $\partial \mathcal{L}/\partial A
\propto B^\top = 0$ and $\partial\mathcal{L}/\partial B \propto (xA)^\top = 0$,
so nothing ever moves. You need exactly one of them zero — the product vanishes
but the gradients do not. (By symmetry, random-$B$/zero-$A$ works too;
zero-$B$ is the convention.)

### The arithmetic that makes it worth it
For $d_{in}=d_{out}=4096$: full fine-tuning trains $16.8$M parameters per
matrix. LoRA at $r=8$ trains $2 \cdot 4096 \cdot 8 = 65$K — **0.39%**.

The memory win is bigger than the parameter count suggests, because Adam keeps
*two* fp32 moments per trainable parameter ([[adam]]). Optimizer state drops by
the same 256x, and that — not the parameter count — is usually what decides
whether a model fits on your GPU.

### Why $\alpha/r$ and not just $\alpha$
The scale keeps the update magnitude roughly constant as you change $r$, so
retuning the rank does not force you to retune the learning rate. In practice
people fix $\alpha$ (often $2r$) and sweep $r$.

### The deployment property
After training, $W' = W + \frac{\alpha}{r}AB$ can be folded into the base weight
once, giving a plain linear layer with **zero** added inference latency —
unlike adapter layers, which add depth to the forward pass. And since the base
weight is untouched, many task-specific LoRAs can be swapped against one shared
frozen model.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class LoRALinear(nnx.Module):
    """Linear layer with a frozen base weight and a trainable low-rank adapter."""

    def __init__(self, din: int, dout: int, rank: int, alpha: float = 1.0,
                 *, rngs: nnx.Rngs):
        # Expected attributes: self.W (din, dout) frozen, self.A (din, rank) and
        # self.B (rank, dout) trainable.
        pass  # Replace this

    def __call__(self, x):
        """(..., din) -> (..., dout)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class LoRALinear(nnx.Module):
    """Linear layer with a frozen base weight and a trainable low-rank adapter."""

    def __init__(self, din: int, dout: int, rank: int, alpha: float = 1.0,
                 *, rngs: nnx.Rngs):
        self.din, self.dout, self.rank = din, dout, rank
        self.scaling = alpha / rank

        # Plain nnx.Variable, NOT nnx.Param — this is what "frozen" means here:
        # an nnx.Param filter will not collect it, so no gradient is produced.
        key_w, key_a = jax.random.split(rngs.params(), 2)
        self.W = nnx.Variable(
            jax.random.normal(key_w, (din, dout)) * (1.0 / jnp.sqrt(din))
        )

        # A random, B zero -> the adapter contributes exactly nothing at init,
        # while both still receive gradient on the first step.
        self.A = nnx.Param(jax.random.normal(key_a, (din, rank)) * 0.01)
        self.B = nnx.Param(jnp.zeros((rank, dout)))

    def __call__(self, x):
        base = x @ self.W[...]
        # Kept factored: (..., din) @ (din, r) @ (r, dout). Forming A @ B first
        # would build the full (din, dout) matrix LoRA exists to avoid.
        delta = (x @ self.A[...]) @ self.B[...]
        return base + self.scaling * delta
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

layer = LoRALinear(64, 64, rank=4, alpha=8.0, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(1), (2, 64))

print("adapter output at init:", float(jnp.abs(layer(x) - x @ layer.W[...]).max()))

params = nnx.state(layer, nnx.Param)
n = sum(p.size for p in jax.tree.leaves(params))
print(f"trainable params: {n}  (full weight would be {64 * 64})")
print(f"                  {100 * n / (64 * 64):.1f}% of full fine-tuning")
''',
    "tests": [
        {
            "name": "Identity at initialization",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(32, 16, rank=4, alpha=8.0, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(1), (5, 32))

out = layer(x)
assert out.shape == (5, 16), f'Expected (5, 16), got {out.shape}'

base = x @ layer.W[...]
assert jnp.allclose(out, base, atol=1e-6), (
    f'At init B is zero so the adapter must contribute EXACTLY nothing: '
    f'max deviation {float(jnp.abs(out - base).max()):.2e}. This is the property '
    'that lets LoRA start from the pretrained function.'
)
""",
        },
        {
            "name": "B is zero, A is not",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(24, 12, rank=4, alpha=1.0, rngs=nnx.Rngs(0))

A = layer.A[...]
B = layer.B[...]
assert A.shape == (24, 4), f'A should be (din, rank) = (24, 4), got {A.shape}'
assert B.shape == (4, 12), f'B should be (rank, dout) = (4, 12), got {B.shape}'

assert jnp.allclose(B, 0.0), f'B must be initialised to zeros, got max |B| = {float(jnp.abs(B).max())}'
assert jnp.abs(A).max() > 1e-8, (
    'A must NOT be zero — if both were zero, dL/dA and dL/dB would both vanish '
    'and the adapter could never learn anything.'
)
""",
        },
        {
            "name": "Base weight is frozen, adapter is trainable",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(16, 8, rank=2, alpha=4.0, rngs=nnx.Rngs(0))

params = nnx.state(layer, nnx.Param)
leaves = jax.tree.leaves(params)
n_params = sum(p.size for p in leaves)

expected = 16 * 2 + 2 * 8   # A + B
assert n_params == expected, (
    f'nnx.Param should collect ONLY A and B ({expected} values), found {n_params}. '
    'The base weight must be a plain nnx.Variable so it is excluded from the '
    'trainable state.'
)
assert n_params < 16 * 8, f'The adapter ({n_params}) must be smaller than the full weight ({16*8})'
""",
        },
        {
            "name": "Gradient flows to A and B but not W",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(16, 8, rank=2, alpha=4.0, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(2), (4, 16))
W_before = jnp.array(layer.W[...])


def loss_fn(m):
    return jnp.sum(m(x) ** 2)


grads = nnx.grad(loss_fn)(layer)
g_leaves = jax.tree.leaves(grads)
assert len(g_leaves) > 0, 'No gradients produced'

# Every collected gradient belongs to the adapter.
total = sum(g.size for g in g_leaves)
assert total == 16 * 2 + 2 * 8, (
    f'nnx.grad should differentiate only the Params (A and B), got {total} values'
)

flat = nnx.state(grads)
gA = flat["A"][...]
gB = flat["B"][...]
assert jnp.abs(gB).max() > 1e-8, (
    'dL/dB must be non-zero — it is proportional to (xA)^T, and A is non-zero'
)
assert jnp.allclose(gA, 0.0, atol=1e-9), (
    f'At init dL/dA is proportional to B^T = 0, so it must vanish on the FIRST '
    f'step, got max {float(jnp.abs(gA).max()):.2e}'
)
assert jnp.allclose(layer.W[...], W_before), 'The base weight must not be mutated'
""",
        },
        {
            "name": "alpha / rank scaling",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

x = jax.random.normal(jax.random.key(3), (3, 20))

def with_alpha(alpha, rank=4):
    m = {fn}(20, 10, rank=rank, alpha=alpha, rngs=nnx.Rngs(0))
    # Give B a fixed non-zero value so the adapter actually contributes.
    m.B[...] = jnp.ones_like(m.B[...]) * 0.1
    return m, m(x) - x @ m.W[...]

_, d1 = with_alpha(1.0)
_, d2 = with_alpha(2.0)
assert jnp.allclose(d2, 2 * d1, rtol=1e-4), (
    'Doubling alpha must double the adapter contribution'
)

# Pin the divisor exactly: at each rank the delta must equal
# (alpha / rank) * (x A) B. An implementation that scales by alpha alone, or
# forgets the scaling entirely, fails here.
for rank, alpha in ((2, 4.0), (4, 4.0), (8, 16.0)):
    m = {fn}(20, 10, rank=rank, alpha=alpha, rngs=nnx.Rngs(0))
    m.B[...] = jax.random.normal(jax.random.key(9), m.B[...].shape) * 0.3

    delta = m(x) - x @ m.W[...]
    expected = (alpha / rank) * ((x @ m.A[...]) @ m.B[...])
    assert jnp.allclose(delta, expected, atol=1e-5), (
        f'rank={rank} alpha={alpha}: the adapter must be scaled by '
        f'alpha/rank = {alpha / rank}, max diff '
        f'{float(jnp.abs(delta - expected).max()):.2e}'
    )

    # Scaling by alpha alone would be wrong by exactly a factor of `rank`.
    wrong = alpha * ((x @ m.A[...]) @ m.B[...])
    if rank != 1:
        assert not jnp.allclose(delta, wrong, atol=1e-5), (
            f'rank={rank}: the contribution is not divided by rank'
        )
""",
        },
        {
            "name": "Matches the explicit low-rank formula",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(12, 6, rank=3, alpha=6.0, rngs=nnx.Rngs(0))
layer.B[...] = jax.random.normal(jax.random.key(4), layer.B[...].shape) * 0.3

x = jax.random.normal(jax.random.key(5), (7, 12))
got = layer(x)

W, A, B = layer.W[...], layer.A[...], layer.B[...]
expected = x @ W + (6.0 / 3) * ((x @ A) @ B)
assert jnp.allclose(got, expected, atol=1e-5), (
    f'max diff {float(jnp.abs(got - expected).max()):.2e}'
)

# Folding the adapter into the base weight must give the same function.
merged = W + (6.0 / 3) * (A @ B)
assert jnp.allclose(got, x @ merged, atol=1e-4), (
    'W + (alpha/r) A B should reproduce the layer exactly — this is what makes '
    'LoRA free at inference time.'
)
""",
        },
        {
            "name": "Shapes, batching and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(16, 32, rank=2, alpha=4.0, rngs=nnx.Rngs(0))

for shape in ((16,), (4, 16), (2, 3, 16)):
    x = jax.random.normal(jax.random.key(6), shape)
    out = layer(x)
    assert out.shape == shape[:-1] + (32,), (
        f'Input {shape} should give {shape[:-1] + (32,)}, got {out.shape}'
    )

x = jax.random.normal(jax.random.key(7), (4, 16))

@nnx.jit
def fwd(m, inp):
    return m(inp)

assert jnp.allclose(fwd(layer, x), layer(x), atol=1e-6), 'jit changes the result'

# Per-example vmap agrees with the batched call.
per = jax.vmap(lambda row: layer(row))(x)
assert jnp.allclose(per, layer(x), atol=1e-5), 'vmap disagrees with the batched call'
""",
        },
    ],
}
