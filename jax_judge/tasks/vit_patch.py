"""ViT patch embedding — reshape an image into patch tokens and project them."""

TASK = {
    "title": "ViT Patch Embedding",
    "category": "Attention & Transformers",
    "order": 12,
    "difficulty": "Medium",
    "function_name": "PatchEmbedding",
    "hint": (
        "It is two reshapes with one transpose between them, and no indexing or "
        "loops. Start by splitting each spatial axis in two — H becomes "
        "(grid_rows, P) and W becomes (grid_cols, P) — which is a free reshape "
        "to six axes. Now look at the axis order you have versus the order you "
        "need: for a final reshape to (B, N, P*P*C) to produce squares, the two "
        "grid axes must be adjacent and outermost and the two within-patch axes "
        "must be adjacent and innermost, with C last. One transpose gets you "
        "there. To check yourself, compare token 0 against "
        "x[0, :P, :P, :].reshape(-1) before you touch the projection, which is "
        "just a single (P*P*C, embed_dim) matmul plus a bias."
    ),
    "description": r"""
Implement the **patch embedding** stem of a Vision Transformer as an
`nnx.Module`: cut an image into non-overlapping square patches, flatten each
one, and linearly project it to a token vector.

$$x \in \mathbb{R}^{B \times H \times W \times C}
\;\longrightarrow\;
z \in \mathbb{R}^{B \times N \times E},
\qquad N = \frac{H}{P}\cdot\frac{W}{P}$$

$$z_n = \mathrm{flatten}(\text{patch}_n)\,W + b,
\qquad W \in \mathbb{R}^{(P^2 C) \times E}$$

### Rules
- Subclass `nnx.Module`; signature
  `PatchEmbedding(img_size, patch_size, in_channels, embed_dim, *, rngs)`
- Input is **NHWC** — `(B, H, W, C)` — the JAX/XLA channel-last convention,
  not PyTorch's NCHW
- Output `(B, N, embed_dim)` with patches in **row-major grid order**: patch
  index `n = row * (W // P) + col`
- Within a patch, flatten in `(patch_row, patch_col, channel)` order
- Expose `self.num_patches` (computed from `img_size`), but read the actual
  grid off `x.shape` in `__call__` — the tests feed a non-square image through
  a module configured with a square `img_size`
- Do the reshaping yourself — no `jax.lax.conv_general_dilated`, no
  `einops.rearrange`
- One projection matrix plus a bias, both `nnx.Param`, named `self.w` of shape
  `(P*P*C, embed_dim)` and `self.b` of shape `(embed_dim,)` — the tests write
  to them directly

### This layer is a strided convolution
`patchify + project` is *exactly* a convolution with kernel size `P` and stride
`P`. Reshape your `(P²C, E)` weight to `(P, P, C, E)`, run
`conv_general_dilated(x, kernel, strides=(P, P), padding='VALID')` with
`dimension_numbers=('NHWC', 'HWIO', 'NHWC')`, and you get the same numbers —
one of the tests checks this. That is how the reference ViT implementation is
written, because a single conv kernel is fused and never materialises the
`(B, N, P²C)` intermediate. Knowing the equivalence tells you the "transformers
have no convolutions" claim is only true after layer one.

The equivalence also pins the memory story: the reshape route allocates
`B·N·P²C` floats — the same count as the image, just re-laid-out — so it is
cheap, but it does force a full transpose of the image.

### Why ViT needs a class token and position embeddings
A transformer is **permutation-equivariant**: shuffle the tokens and the outputs
shuffle with them. Patch order carries all the 2-D geometry, and after this
layer nothing in the model knows that patch 3 sits above patch 17. So ViT adds a
learned position embedding to every token right after this step. Sinusoids work
too, but learned embeddings are standard here, and they are why changing the
input resolution requires interpolating the position table.

The `[CLS]` token is a separate trick: you need one vector to classify from.
Mean-pooling the patch tokens works, but a prepended learned token gives
attention a free slot that starts with no spatial identity of its own and can
learn to aggregate whatever the head needs. Either way, this layer emits `N`
tokens and the model that wraps it feeds forward `N + 1`.

### The trap
`x.reshape(B, N, P*P*C)` straight from `(B, H, W, C)` produces the correct output
shape and completely wrong patches — it slices full image rows, not squares. The
shape assertion passes, the model trains, accuracy is mysteriously bad. Get the
transpose right.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    """Split an NHWC image into P x P patches and project each to embed_dim."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Set self.num_patches, plus the projection params self.w
        # (patch_size**2 * in_channels, embed_dim) and self.b (embed_dim,).
        pass  # Replace this

    def __call__(self, x):
        """(B, H, W, C) -> (B, num_patches, embed_dim)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        grid = img_size // patch_size
        self.num_patches = grid * grid

        patch_dim = patch_size * patch_size * in_channels
        self.w = nnx.Param(
            jax.random.normal(rngs.params(), (patch_dim, embed_dim)) / jnp.sqrt(patch_dim)
        )
        self.b = nnx.Param(jnp.zeros((embed_dim,)))

    def __call__(self, x):
        B, H, W, C = x.shape
        P = self.patch_size
        gh, gw = H // P, W // P

        # (B, gh, P, gw, P, C): split each spatial axis into grid x within-patch.
        x = x.reshape(B, gh, P, gw, P, C)
        # Put both grid axes first, both within-patch axes last.
        x = x.transpose(0, 1, 3, 2, 4, 5)          # (B, gh, gw, P, P, C)
        # Row-major patch order, and (prow, pcol, channel) inside each patch.
        patches = x.reshape(B, gh * gw, P * P * C)

        return patches @ self.w + self.b
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

pe = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64,
                    rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 32, 32, 3))
print("image", x.shape, "->", pe(x).shape, " num_patches =", pe.num_patches)

# It is a stride-P convolution. Same weight, reshaped into an HWIO kernel.
P, C, E = 8, 3, 64
kernel = pe.w[...].reshape(P, P, C, E)
conv = jax.lax.conv_general_dilated(
    x, kernel, window_strides=(P, P), padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
conv = conv.reshape(x.shape[0], -1, E) + pe.b[...]
print("max |reshape - conv| =", float(jnp.abs(pe(x) - conv).max()))

# The trap: reshaping without the transpose gives full image ROWS, not squares.
naive = x.reshape(2, 16, 8 * 8 * 3)
correct = x.reshape(2, 4, 8, 4, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(2, 16, 8 * 8 * 3)
print("naive reshape equals the real patches?", bool(jnp.allclose(naive, correct)))
''',
    "tests": [
        {
            "name": "Shapes and num_patches",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64, rngs=nnx.Rngs(0))
assert isinstance(pe, nnx.Module), 'PatchEmbedding must subclass nnx.Module'
assert pe.num_patches == 16, f'num_patches is {pe.num_patches}, expected (32/8)**2 = 16'

x = jax.random.normal(jax.random.key(0), (2, 32, 32, 3))
out = pe(x)
assert out.shape == (2, 16, 64), f'Shape {out.shape} vs (2, 16, 64) = (B, N, E)'
assert jnp.isfinite(out).all(), 'Non-finite output'

big = {fn}(img_size=224, patch_size=16, in_channels=3, embed_dim=768, rngs=nnx.Rngs(1))
assert big.num_patches == 196, f'ViT-B/16 on 224px has 196 patches, got {big.num_patches}'

grey = {fn}(img_size=64, patch_size=16, in_channels=1, embed_dim=32, rngs=nnx.Rngs(2))
g = grey(jax.random.normal(jax.random.key(3), (1, 64, 64, 1)))
assert g.shape == (1, 16, 32), f'Single-channel case: {g.shape} vs (1, 16, 32)'
""",
        },
        {
            "name": "Projection parameters have the flattened-patch layout",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=32, patch_size=4, in_channels=3, embed_dim=16, rngs=nnx.Rngs(0))

params = jax.tree.leaves(nnx.state(pe, nnx.Param))
assert len(params) >= 1, 'No nnx.Param found — the projection must be a learnable Param'

patch_dim = 4 * 4 * 3
shapes = sorted(tuple(p.shape) for p in params)
assert (patch_dim, 16) in shapes, (
    f'Expected a ({patch_dim}, 16) projection weight (JAX layout: din, dout), got {shapes}'
)

total = sum(int(p.size) for p in params)
assert patch_dim * 16 <= total <= patch_dim * 16 + 16, (
    f'{total} params; expected {patch_dim * 16} plus at most a {16}-element bias. '
    'One matrix does the whole projection — do not build a Linear per patch.'
)
""",
        },
        {
            "name": "Patches are real squares, in row-major order",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, H, W, C, P = 1, 8, 12, 2, 4
E = P * P * C
pe = {fn}(img_size=8, patch_size=P, in_channels=C, embed_dim=E, rngs=nnx.Rngs(0))

# Make the projection an identity so the output IS the flattened patch.
pe.w[...] = jnp.eye(E)
pe.b[...] = jnp.zeros((E,))

x = jax.random.normal(jax.random.key(0), (B, H, W, C))
out = pe(x)
gh, gw = H // P, W // P
assert out.shape == (B, gh * gw, E), f'{out.shape} vs {(B, gh * gw, E)}'

for r in range(gh):
    for c in range(gw):
        n = r * gw + c
        want = x[0, r * P:(r + 1) * P, c * P:(c + 1) * P, :].reshape(-1)
        assert jnp.allclose(out[0, n], want, atol=1e-5), (
            f'Patch {n} (row {r}, col {c}) is wrong. Either you reshaped straight to '
            '(B, N, P*P*C) without the transpose — which slices full image rows '
            'instead of squares — or the patch order / within-patch order is off. '
            'Expected order: patch n = row*(W//P)+col, flattened as (prow, pcol, channel).'
        )
""",
        },
        {
            "name": "Equivalent to a stride-P convolution",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

P, C, E = 4, 3, 8
pe = {fn}(img_size=16, patch_size=P, in_channels=C, embed_dim=E, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 16, 16, C))

kernel = pe.w[...].reshape(P, P, C, E)          # (P*P*C, E) -> HWIO
conv = jax.lax.conv_general_dilated(
    x, kernel, window_strides=(P, P), padding='VALID',
    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
)
conv = conv.reshape(2, -1, E) + pe.b[...]

assert jnp.allclose(pe(x), conv, atol=1e-4), (
    'Your patch embedding disagrees with the equivalent stride-P convolution. '
    'The HWIO kernel indexes (patch_row, patch_col, channel, out), so your flattened '
    'patch must use exactly that order.'
)
""",
        },
        {
            "name": "Patches are independent, and the layer is affine",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=16, patch_size=4, in_channels=1, embed_dim=8, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (1, 16, 16, 1))
base = pe(x)

# Perturb exactly one patch (row 2, col 1 -> token 2*4 + 1 = 9).
x2 = x.at[0, 8:12, 4:8, :].add(5.0)
out2 = pe(x2)
changed = jnp.abs(out2 - base).max(axis=-1)[0] > 1e-4
idx = [int(i) for i in jnp.nonzero(changed)[0]]
assert idx == [9], (
    f'Editing the pixels of patch 9 changed tokens {idx}. Each token must depend on '
    'its own patch only — non-overlapping patches, stride == patch size.'
)

# No nonlinearity here: doubling the input doubles the output minus the bias.
zero = pe(jnp.zeros_like(x))
assert jnp.allclose(pe(2 * x) - zero, 2 * (base - zero), atol=1e-4), (
    'The patch embedding is a pure affine map — no activation belongs in it'
)
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

pe = {fn}(img_size=8, patch_size=2, in_channels=3, embed_dim=16, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 8, 8, 3))

grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(pe)
leaves = jax.tree.leaves(grads)
assert len(leaves) >= 1, 'No parameter gradients'
assert all(jnp.isfinite(g).all() for g in leaves), 'Non-finite parameter gradient'
assert max(float(jnp.abs(g).max()) for g in leaves) > 0, 'All parameter gradients are zero'

# Gradient w.r.t. the image: every pixel feeds exactly one patch, so none is dead.
gx = nnx.grad(lambda m, v: jnp.sum(m(v) ** 2), argnums=1)(pe, x)
assert gx.shape == x.shape, f'Input gradient {gx.shape} vs {x.shape}'
assert float(jnp.abs(gx).min()) > 0, (
    'Some pixels received zero gradient — patches must tile the image with no gaps'
)

jitted = nnx.jit(lambda m, v: m(v))
assert jnp.allclose(jitted(pe, x), pe(x), atol=1e-5), 'nnx.jit changed the result'
""",
        },
        {
            "name": "Non-square images and batch consistency",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# The module is configured with img_size, but the reshape logic should read H and W
# off the input rather than hard-coding a square grid.
pe = {fn}(img_size=8, patch_size=4, in_channels=2, embed_dim=6, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (3, 8, 16, 2))
out = pe(x)
assert out.shape == (3, (8 // 4) * (16 // 4), 6), (
    f'{out.shape} — a (8, 16) image has 2*4 = 8 patches; derive the grid from x.shape'
)

# Batching must be pure: row i of a batch equals the same image run alone.
single = pe(x[1:2])
assert jnp.allclose(single[0], out[1], atol=1e-5), (
    'A single image gave a different embedding inside a batch — no cross-sample mixing'
)

vm = jax.vmap(lambda img: pe(img[None])[0])(x)
assert jnp.allclose(vm, out, atol=1e-5), 'vmap over the batch disagrees with the batched call'
""",
        },
    ],
}
