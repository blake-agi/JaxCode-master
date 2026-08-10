"""ViT patch embedding — reshape an image into tokens, then project."""

TASK = {
    "title": "Vision Transformer Patch Embedding",
    "category": "Attention & Transformers",
    "number": "27",
    "difficulty": "Medium",
    "function_name": "PatchEmbedding",
    "hint": (
        "Split BOTH spatial axes at once: (B, C, H, W) reshapes to "
        "(B, C, n_h, p, n_w, p). Then transpose so the two grid axes come first "
        "and the (channel, row-in-patch, col-in-patch) axes end up adjacent, and "
        "flatten those three into one feature vector. Getting the transpose "
        "wrong still produces the right SHAPE, so check the values, not the "
        "dimensions."
    ),
    "description": r"""
Implement the **patch embedding** that turns an image into a sequence of tokens
for a Vision Transformer.

$$(B, C, H, W) \;\longrightarrow\; (B, N, D), \qquad
N = \frac{H}{p}\cdot\frac{W}{p}$$

### Signature
```python
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim,
                 *, rngs: nnx.Rngs): ...
    def __call__(self, x): ...
```

### Requirements
- Input is **NCHW** — `(B, C, H, W)`, the PyTorch layout
- `self.patch_size`, `self.num_patches = (img_size // patch_size) ** 2`
- `self.proj`: `nnx.Linear(in_channels * patch_size * patch_size, embed_dim)`
- Patches in **row-major** order, and each flattened as `(C, p, p)`

`nnx.Linear` is an allowed building block; the exercise is the reshape.

### The reshape, step by step
```
(B, C, H, W)                      split each spatial axis
  -> (B, C, n_h, p, n_w, p)       grid index and within-patch index
  -> (B, n_h, n_w, C, p, p)       grid axes first
  -> (B, n_h*n_w, C*p*p)          flatten to tokens
```

### Why this is exactly a strided convolution
A `Conv2d(C, D, kernel_size=p, stride=p)` computes the identical thing: each
output position sees one non-overlapping `p × p` patch and projects it to `D`
channels. Real ViT implementations use the conv because it is one fused kernel,
but the reshape makes it obvious that **no spatial mixing happens here** — every
patch is embedded independently, and all interaction between patches is left to
the attention layers.

### What the reshape throws away, and what gets it back
After this step the model has no idea where any patch came from: the sequence
is permutation-equivariant, so shuffling the patches shuffles the outputs
identically. That is why ViT adds **position embeddings** immediately
afterwards, and why it needs far more data than a CNN — translation
equivariance is baked into a conv, but a transformer has to learn it.

### The trap
The wrong transpose gives the right output *shape*. `(B, n_h*n_w, C*p*p)` comes
out either way, so shape assertions pass while the patch contents are scrambled.
The tests below check values against a hand-built patch.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    """Image (B, C, H, W) -> patch tokens (B, N, embed_dim)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(B, C, H, W) -> (B, num_patches, embed_dim)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, *, rngs: nnx.Rngs):
        assert img_size % patch_size == 0, "img_size must divide by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nnx.Linear(
            in_channels * patch_size * patch_size, embed_dim, rngs=rngs
        )

    def __call__(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        n_h, n_w = H // p, W // p

        # Split both spatial axes into (grid index, index within patch).
        x = x.reshape(B, C, n_h, p, n_w, p)
        # Grid axes to the front; (C, p, p) left adjacent so they flatten
        # into one patch vector.
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * p * p)
        return self.proj(x)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

pe = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64,
                    rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(1), (2, 3, 32, 32))    # NCHW

print("image :", x.shape)
print("tokens:", pe(x).shape, f"(num_patches={pe.num_patches})")
print("patch feature dim:", 3 * 8 * 8, "-> projected to 64")
''',
    "tests": [
        {
            "name": "Shapes and the projection layer",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(32, 8, 3, 64, rngs=nnx.Rngs(params=0))
assert pe.patch_size == 8, f'patch_size {pe.patch_size}'
assert pe.num_patches == 16, f'num_patches should be (32//8)**2 = 16, got {pe.num_patches}'
assert isinstance(pe.proj, nnx.Linear), (
    f'self.proj must be an nnx.Linear, got {type(pe.proj)}'
)
assert pe.proj.kernel.shape == (192, 64), (
    f'proj kernel should be (C*p*p, embed_dim) = (192, 64), got {pe.proj.kernel.shape}'
)

x = jax.random.normal(jax.random.key(1), (2, 3, 32, 32))
assert pe(x).shape == (2, 16, 64), f'{pe(x).shape} vs (2, 16, 64)'
""",
        },
        {
            "name": "Patch CONTENTS are correct, not just the shape",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# Identity projection so tokens are the raw flattened patches.
pe = {fn}(4, 2, 1, 4, rngs=nnx.Rngs(params=2))
pe.proj.kernel[...] = jnp.eye(4)
pe.proj.bias[...] = jnp.zeros(4)

x = jnp.arange(16.0).reshape(1, 1, 4, 4)
out = pe(x)

# Row-major patches of [[0..3],[4..7],[8..11],[12..15]] with p=2:
#   patch0 = [0,1,4,5]  patch1 = [2,3,6,7]
#   patch2 = [8,9,12,13] patch3 = [10,11,14,15]
expected = jnp.array([[[0., 1., 4., 5.],
                       [2., 3., 6., 7.],
                       [8., 9., 12., 13.],
                       [10., 11., 14., 15.]]])
assert jnp.allclose(out, expected), (
    f'Patch contents wrong.\\ngot      {out}\\nexpected {expected}\\n'
    'The shape is right either way — this is the transpose, not the reshape.'
)
""",
        },
        {
            "name": "Channels are interleaved per patch, not concatenated",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(2, 2, 2, 8, rngs=nnx.Rngs(params=3))
pe.proj.kernel[...] = jnp.eye(8)
pe.proj.bias[...] = jnp.zeros(8)

# One patch, two channels: channel 0 is 0..3, channel 1 is 10..13.
x = jnp.stack([jnp.arange(4.0).reshape(2, 2),
               10 + jnp.arange(4.0).reshape(2, 2)])[None]
out = pe(x)

assert out.shape == (1, 1, 8), f'{out.shape}'
assert jnp.allclose(out[0, 0], jnp.array([0., 1., 2., 3., 10., 11., 12., 13.])), (
    f'Got {out[0, 0]}. A patch flattens as (C, p, p), so all of channel 0 comes '
    'before channel 1.'
)
""",
        },
        {
            "name": "Equivalent to a strided convolution",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(8, 4, 3, 5, rngs=nnx.Rngs(params=4))
x = jax.random.normal(jax.random.key(5), (2, 3, 8, 8))
out = pe(x)

# Same weights viewed as an OIHW conv kernel with stride = patch size.
w = pe.proj.kernel[...].T.reshape(5, 3, 4, 4)       # (D, C, p, p)
conv = jax.lax.conv_general_dilated(
    x, w, window_strides=(4, 4), padding=((0, 0), (0, 0)),
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
)
conv = conv.reshape(2, 5, -1).transpose(0, 2, 1) + pe.proj.bias[...]

assert jnp.allclose(out, conv, atol=1e-4), (
    'Patch embedding must equal a Conv2d with kernel_size = stride = patch_size'
)
""",
        },
        {
            "name": "Patches are embedded independently",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(8, 4, 1, 6, rngs=nnx.Rngs(params=6))
x = jax.random.normal(jax.random.key(7), (1, 1, 8, 8))
base = pe(x)

# Perturbing the last patch must leave the other three untouched: no spatial
# mixing happens at this stage.
x2 = x.at[:, :, 4:, 4:].add(100.0)
pert = pe(x2)

assert jnp.allclose(base[:, :3], pert[:, :3], atol=1e-4), (
    'Changing one patch altered the others — patches must be independent here'
)
assert not jnp.allclose(base[:, 3], pert[:, 3], atol=1e-3), 'The changed patch should differ'
""",
        },
        {
            "name": "Non-square grids and divisibility",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(16, 4, 3, 32, rngs=nnx.Rngs(params=8))
x = jax.random.normal(jax.random.key(9), (2, 3, 16, 16))
assert pe(x).shape == (2, 16, 32), f'{pe(x).shape}'

try:
    {fn}(10, 4, 3, 32, rngs=nnx.Rngs(params=10))
except Exception:
    pass
else:
    raise AssertionError('img_size=10 is not divisible by patch_size=4 and must be rejected')
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(8, 4, 3, 16, rngs=nnx.Rngs(params=11))
x = jax.random.normal(jax.random.key(12), (2, 3, 8, 8))

grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(pe)
k = nnx.state(grads)["proj"]["kernel"]
val = k[...] if isinstance(k, nnx.Variable) else k
assert jnp.isfinite(val).all(), 'Non-finite gradient'
assert float(jnp.abs(val).sum()) > 0, 'No gradient reached proj'

graphdef, st = nnx.split(pe)
run = jax.jit(lambda st, a: nnx.merge(graphdef, st)(a))
assert jnp.allclose(run(st, x), pe(x), atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
