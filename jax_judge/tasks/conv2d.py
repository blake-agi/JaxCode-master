"""2-D convolution from scratch — NHWC/HWIO, stride, and TF-style SAME padding."""

TASK = {
    "title": "Conv2D from Scratch (NHWC)",
    "category": "Core Ops & Layers",
    "order": 12,
    "difficulty": "Hard",
    "function_name": "conv2d",
    "hint": (
        "Two steps. (1) Padding: 'VALID' pads nothing; for 'SAME' the output is "
        "ceil(H / stride), so work out the total padding needed to reach that and "
        "split it between top and bottom — when it is odd, the extra row goes on "
        "the BOTTOM (this is the convention torch and XLA both use, and getting "
        "it backwards shifts your whole feature map by one). (2) Gather all the "
        "windows at once instead of looping over pixels: build an (out_h, KH) "
        "matrix of row indices by broadcasting a strided arange against an arange "
        "over the kernel, do the same for columns, and use them to index x. Then "
        "contract patches against the kernel with a single jnp.einsum. An "
        "equivalent route is to loop over the KH*KW kernel taps — not the pixels "
        "— accumulating strided slices."
    ),
    "description": r"""
Implement a 2-D **convolution** (really a cross-correlation, like every deep
learning framework) from scratch.

$$y[n,p,q,o] = b[o] + \sum_{i=0}^{K_H-1}\sum_{j=0}^{K_W-1}\sum_{c=0}^{C_{in}-1}
x[n,\; p s_h + i,\; q s_w + j,\; c]\; w[i,j,c,o]$$

### Rules
- Signature: `conv2d(x, w, b=None, stride=1, padding="VALID")`
- `x` is **NHWC**: `(N, H, W, C_in)`; `w` is **HWIO**: `(KH, KW, C_in, C_out)`
- `stride` is an int or a `(sh, sw)` pair; `padding` is `"VALID"` or `"SAME"`
- `b` is `(C_out,)` or `None`
- Returns `(N, H_out, W_out, C_out)`
- **Banned**: `jax.lax.conv_general_dilated` (and its `conv`/`conv_with_general_padding`
  wrappers), `jax.scipy.signal.convolve*`, and anything else that does the
  convolution for you
- Must be jittable and differentiable — no Python loop over output pixels, no
  data-dependent control flow

### Output sizes
$$\text{VALID}:\; H_{out} = \left\lfloor\frac{H - K_H}{s_h}\right\rfloor + 1
\qquad
\text{SAME}:\; H_{out} = \left\lceil\frac{H}{s_h}\right\rceil$$

### Why this is the interview question
Three traps live in here.

**1. Layout.** JAX's native layout is NHWC with HWIO kernels — channels last,
output channels last. Frameworks that default to NCHW/OIHW transpose on the way
in, and a silently transposed kernel still produces plausible-looking numbers.
Getting `einsum('nhwijc,ijco->nhwo', ...)` right first try is the actual test.

**2. SAME padding is asymmetric.** If `(out-1)*s + K - H` is odd, XLA puts the
extra row on the **bottom** and the extra column on the **right**. A symmetric
`pad=K//2` gives the same answer only for odd kernels at stride 1 — which is why
the bug survives so long before someone runs a 2-strided even-kernel layer and
gets an off-by-one shift.

**3. im2col is not an optimisation detail.** Materialising the
`(N, H_out, W_out, KH, KW, C_in)` patch tensor blows the input up by $K_H K_W$
and then hands the work to one big matmul — which is exactly what cuDNN does
under the hood for most shapes, because a GEMM on tensor cores beats a bespoke
sliding-window kernel. The memory blow-up is real, though: that tensor is 9x the
input for a 3x3 kernel, so production kernels fuse the gather into the GEMM
rather than writing it out.

Note that this is **cross-correlation** — no kernel flip. Learned kernels make
the distinction irrelevant (the network just learns the flipped filter), so
every framework quietly dropped the flip and kept the name.
""",
    "stub": '''import jax
import jax.numpy as jnp


def conv2d(x, w, b=None, stride=1, padding="VALID"):
    """2-D cross-correlation.

    Args:
        x:       (N, H, W, C_in)  input, channels last
        w:       (KH, KW, C_in, C_out) kernel
        b:       (C_out,) bias or None
        stride:  int or (sh, sw)
        padding: "VALID" or "SAME"

    Returns:
        (N, H_out, W_out, C_out)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def conv2d(x, w, b=None, stride=1, padding="VALID"):
    N, H, W, C = x.shape
    KH, KW, CI, CO = w.shape
    assert CI == C, f"kernel expects {CI} input channels, got {C}"

    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    mode = padding.upper()

    if mode == "SAME":
        out_h = -(-H // sh)                      # ceil(H / sh)
        out_w = -(-W // sw)
        pad_h = max((out_h - 1) * sh + KH - H, 0)
        pad_w = max((out_w - 1) * sw + KW - W, 0)
        top, left = pad_h // 2, pad_w // 2       # the extra goes bottom/right
        x = jnp.pad(x, ((0, 0), (top, pad_h - top), (left, pad_w - left), (0, 0)))
    elif mode == "VALID":
        out_h = (H - KH) // sh + 1
        out_w = (W - KW) // sw + 1
    else:
        raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding!r}")

    # im2col: build the (out_h, KH) and (out_w, KW) index matrices once, then
    # gather every window in a single pass — no Python loop over pixels.
    rows = (jnp.arange(out_h) * sh)[:, None] + jnp.arange(KH)[None, :]
    cols = (jnp.arange(out_w) * sw)[:, None] + jnp.arange(KW)[None, :]

    patches = x[:, rows]                 # (N, out_h, KH, W_pad, C)
    patches = patches[:, :, :, cols]     # (N, out_h, KH, out_w, KW, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)   # (N, out_h, out_w, KH, KW, C)

    out = jnp.einsum("nhwijc,ijco->nhwo", patches, w)
    if b is not None:
        out = out + b                    # broadcasts over (N, H_out, W_out)
    return out
''',
    "demo": '''import jax
import jax.numpy as jnp

# A 3x3 edge detector on a single-channel step image.
img = jnp.concatenate([jnp.zeros((1, 6, 3, 1)), jnp.ones((1, 6, 3, 1))], axis=2)
sobel = jnp.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]]).reshape(3, 3, 1, 1)

print("VALID:", conv2d(img, sobel, padding="VALID").shape)
print("SAME :", conv2d(img, sobel, padding="SAME").shape)
print("SAME, stride 2:", conv2d(img, sobel, stride=2, padding="SAME").shape)
print(conv2d(img, sobel, padding="SAME")[0, :, :, 0])   # spike at the edge column

# Where the asymmetry shows: H=5, K=2, stride=2 needs 1 pad row -> all of it at the bottom.
x = jnp.arange(25.0).reshape(1, 5, 5, 1)
k = jnp.ones((2, 2, 1, 1))
print("SAME 2x2 stride 2 ->", conv2d(x, k, stride=2, padding="SAME").shape)
''',
    "tests": [
        {
            "name": "Hand-computed 3x3 with a 2x2 kernel",
            "code": """
import jax.numpy as jnp

x = jnp.arange(9.0).reshape(1, 3, 3, 1)     # [[0,1,2],[3,4,5],[6,7,8]]
w = jnp.ones((2, 2, 1, 1))                  # sums each 2x2 window
out = {fn}(x, w)

assert out.shape == (1, 2, 2, 1), f'Shape {out.shape} vs (1, 2, 2, 1)'
expected = jnp.array([[8.0, 12.0], [20.0, 24.0]]).reshape(1, 2, 2, 1)
assert jnp.allclose(out, expected), f'{out.reshape(2, 2)} vs {expected.reshape(2, 2)}'

# Cross-correlation, not convolution: an asymmetric kernel must NOT be flipped.
k = jnp.array([[1.0, 0.0], [0.0, 0.0]]).reshape(2, 2, 1, 1)   # picks the top-left pixel
picked = {fn}(x, k)
assert jnp.allclose(picked.reshape(2, 2), jnp.array([[0.0, 1.0], [3.0, 4.0]])), (
    f'Got {picked.reshape(2, 2)} — a flipped kernel would return the bottom-right '
    'pixels instead. Deep learning conv does not flip.'
)
""",
        },
        {
            "name": "Output shapes for VALID and SAME",
            "code": """
import jax.numpy as jnp

x = jnp.ones((2, 8, 8, 3))
w = jnp.ones((3, 3, 3, 5))

got = {fn}(x, w).shape
assert got == (2, 6, 6, 5), f'VALID 3x3 stride 1: {got} vs (2, 6, 6, 5)'
assert {fn}(x, w, padding='VALID').shape == (2, 6, 6, 5), 'VALID must be the default'
got = {fn}(x, w, padding='SAME').shape
assert got == (2, 8, 8, 5), f'SAME must keep H, W at stride 1: {got} vs (2, 8, 8, 5)'
assert {fn}(x, w, stride=2, padding='SAME').shape == (2, 4, 4, 5), 'SAME s2: ceil(8/2)=4'
assert {fn}(x, w, stride=2, padding='VALID').shape == (2, 3, 3, 5), 'VALID s2: (8-3)//2+1=3'
assert {fn}(x, w, stride=3, padding='VALID').shape == (2, 2, 2, 5), 'VALID s3: (8-3)//3+1=2'

# Odd input size, SAME, stride 2 -> ceil(7/2) = 4
odd = jnp.ones((1, 7, 7, 1))
k = jnp.ones((3, 3, 1, 1))
assert {fn}(odd, k, stride=2, padding='SAME').shape == (1, 4, 4, 1), 'SAME with odd H'

# Non-square input and a rectangular stride tuple.
rect = jnp.ones((1, 9, 5, 2))
kr = jnp.ones((3, 2, 2, 4))
assert {fn}(rect, kr, stride=(2, 1), padding='VALID').shape == (1, 4, 4, 4), (
    'stride given as a (sh, sw) tuple is not being honoured'
)
""",
        },
        {
            "name": "Matches lax.conv_general_dilated (VALID, multi-channel, bias)",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 9, 7, 3))
w = jax.random.normal(jax.random.key(1), (3, 3, 3, 4))
b = jax.random.normal(jax.random.key(2), (4,))

DN = ('NHWC', 'HWIO', 'NHWC')
ref = jax.lax.conv_general_dilated(x, w, (1, 1), 'VALID', dimension_numbers=DN) + b
out = {fn}(x, w, b)

assert out.shape == ref.shape, f'Shape {out.shape} vs reference {ref.shape}'
assert jnp.allclose(out, ref, atol=1e-4), (
    f'Max abs diff vs lax.conv_general_dilated: {float(jnp.abs(out - ref).max()):.6f}. '
    'Check the einsum axes — HWIO means w is (KH, KW, C_in, C_out).'
)

# b=None must be the default and must not add anything.
nob = {fn}(x, w)
assert jnp.allclose(nob, ref - b, atol=1e-4), 'b=None should skip the bias entirely'

# A 1x1 kernel is a pure per-pixel channel mix.
w11 = jax.random.normal(jax.random.key(3), (1, 1, 3, 4))
assert jnp.allclose({fn}(x, w11), x @ w11[0, 0], atol=1e-4), (
    '1x1 conv should equal x @ w[0, 0]'
)
""",
        },
        {
            "name": "SAME padding, including the asymmetric cases",
            "code": """
import jax
import jax.numpy as jnp

DN = ('NHWC', 'HWIO', 'NHWC')

cases = [
    ((1, 5, 5, 1), (3, 3, 1, 1), (1, 1)),   # even total pad, symmetric
    ((1, 5, 5, 2), (4, 4, 2, 3), (1, 1)),   # pad_total = 3 -> 1 top, 2 bottom
    ((1, 6, 6, 1), (2, 2, 1, 1), (2, 2)),   # even kernel, strided
    ((1, 7, 5, 2), (3, 3, 2, 2), (2, 2)),   # odd sizes, strided
    ((2, 8, 8, 3), (5, 5, 3, 2), (3, 3)),   # big kernel, stride 3
]

for xs, ws, st in cases:
    x = jax.random.normal(jax.random.key(0), xs)
    w = jax.random.normal(jax.random.key(1), ws)
    ref = jax.lax.conv_general_dilated(x, w, st, 'SAME', dimension_numbers=DN)
    out = {fn}(x, w, stride=st, padding='SAME')
    assert out.shape == ref.shape, f'x={xs} w={ws} stride={st}: {out.shape} vs {ref.shape}'
    assert jnp.allclose(out, ref, atol=1e-4), (
        f'x={xs} w={ws} stride={st}: max diff '
        f'{float(jnp.abs(out - ref).max()):.6f}. SAME padding is asymmetric — the extra '
        'row goes on the BOTTOM and the extra column on the RIGHT.'
    )

# Zero padding, not edge/reflect: a SAME conv of ones with an all-ones 3x3 kernel
# must give 4 in the corner (only 2x2 of the window is inside the image).
ones = jnp.ones((1, 5, 5, 1))
k = jnp.ones((3, 3, 1, 1))
o = {fn}(ones, k, padding='SAME')
assert abs(float(o[0, 0, 0, 0]) - 4.0) < 1e-5, (
    f'Corner value {float(o[0, 0, 0, 0])} should be 4.0 — pad with zeros'
)
assert abs(float(o[0, 2, 2, 0]) - 9.0) < 1e-5, 'Interior value should be 9.0'
""",
        },
        {
            "name": "Gradients w.r.t. both x and w",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 6, 6, 2))
w = jax.random.normal(jax.random.key(1), (3, 3, 2, 3))
DN = ('NHWC', 'HWIO', 'NHWC')

loss = lambda x_, w_: jnp.sum({fn}(x_, w_, stride=2, padding='SAME') ** 2)
ref_loss = lambda x_, w_: jnp.sum(
    jax.lax.conv_general_dilated(x_, w_, (2, 2), 'SAME', dimension_numbers=DN) ** 2
)

gx, gw = jax.grad(loss, argnums=(0, 1))(x, w)
rx, rw = jax.grad(ref_loss, argnums=(0, 1))(x, w)

assert gx.shape == x.shape, f'grad wrt x has shape {gx.shape}, expected {x.shape}'
assert gw.shape == w.shape, f'grad wrt w has shape {gw.shape}, expected {w.shape}'
assert jnp.isfinite(gx).all() and jnp.isfinite(gw).all(), 'Non-finite gradients'
assert jnp.allclose(gx, rx, atol=1e-3), (
    f'grad wrt x differs from the reference by {float(jnp.abs(gx - rx).max()):.5f} — '
    'padded positions must receive zero gradient'
)
assert jnp.allclose(gw, rw, atol=1e-3), (
    f'grad wrt w differs from the reference by {float(jnp.abs(gw - rw).max()):.5f}'
)
""",
        },
        {
            "name": "jit and vmap over a batch of kernels",
            "code": """
import jax
import jax.numpy as jnp
from functools import partial

x = jax.random.normal(jax.random.key(0), (2, 8, 8, 3))
w = jax.random.normal(jax.random.key(1), (3, 3, 3, 4))
eager = {fn}(x, w, stride=2, padding='SAME')

# stride and padding are Python values -> static arguments.
fast = jax.jit({fn}, static_argnums=(3, 4))(x, w, None, 2, 'SAME')
assert jnp.allclose(fast, eager, atol=1e-5), 'jitted result differs from the eager result'

# vmap over a stack of kernels (e.g. an ensemble of filter banks).
ws = jax.random.normal(jax.random.key(2), (5, 3, 3, 3, 4))
f = partial({fn}, stride=2, padding='SAME')
stacked = jax.vmap(f, in_axes=(None, 0))(x, ws)
assert stacked.shape == (5,) + eager.shape, f'{stacked.shape} vs {(5,) + eager.shape}'
for i in range(5):
    assert jnp.allclose(stacked[i], f(x, ws[i]), atol=1e-4), f'vmapped kernel {i} disagrees'

# vmap over the leading batch axis of x (a "video" of frames).
vids = jax.random.normal(jax.random.key(3), (4, 2, 8, 8, 3))
per_frame = jax.vmap(f, in_axes=(0, None))(vids, w)
assert per_frame.shape == (4, 2, 4, 4, 4), f'{per_frame.shape}'
""",
        },
        {
            "name": "Degenerate and single-element cases",
            "code": """
import jax
import jax.numpy as jnp

# Kernel exactly the size of the image -> a single output pixel.
x = jax.random.normal(jax.random.key(0), (1, 4, 4, 2))
w = jax.random.normal(jax.random.key(1), (4, 4, 2, 3))
out = {fn}(x, w)
assert out.shape == (1, 1, 1, 3), f'Full-size kernel should give 1x1, got {out.shape}'
manual = jnp.einsum('ijc,ijco->o', x[0], w)
assert jnp.allclose(out[0, 0, 0], manual, atol=1e-4), 'Full-window sum is wrong'

# Stride larger than the kernel (no overlap) still lines up.
big = jnp.arange(16.0).reshape(1, 4, 4, 1)
k = jnp.ones((2, 2, 1, 1))
o = {fn}(big, k, stride=2)
assert o.shape == (1, 2, 2, 1), f'{o.shape} vs (1, 2, 2, 1)'
assert jnp.allclose(o.reshape(2, 2), jnp.array([[10.0, 18.0], [42.0, 50.0]])), (
    f'Non-overlapping 2x2 sums are wrong: {o.reshape(2, 2)}'
)

# SAME with a kernel bigger than the input: pad_total is large but the output is ceil(H/s).
tiny = jnp.ones((1, 2, 2, 1))
kbig = jnp.ones((5, 5, 1, 1))
ot = {fn}(tiny, kbig, padding='SAME')
assert ot.shape == (1, 2, 2, 1), f'SAME with an oversized kernel: {ot.shape} vs (1, 2, 2, 1)'
assert jnp.allclose(ot, 4.0), f'Every window covers all 4 ones, got {ot.reshape(2, 2)}'
""",
        },
    ],
}
