"""2-D convolution from scratch — NCHW, gathered windows, one einsum."""

TASK = {
    "title": "2D Convolution",
    "category": "Core Ops & Layers",
    "order": 12,
    "number": "22",
    "difficulty": "Medium",
    "function_name": "my_conv2d",
    "hint": (
        "Do not loop over output pixels. Build an (H_out, kH) matrix of row "
        "indices by broadcasting a strided arange against an arange over the "
        "kernel, do the same for columns, and use them to gather every window at "
        "once. Arrange the result as (B, C_in, H_out, W_out, kH, kW) and the "
        "whole contraction is a single einsum against an (C_out, C_in, kH, kW) "
        "kernel. Pad first, then compute the output size from the padded input."
    ),
    "description": r"""
Implement a 2-D convolution (strictly, a cross-correlation) from scratch.

### Signature
```python
def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    ...
```

- `x`: `(B, C_in, H, W)` — **NCHW**, the PyTorch layout
- `weight`: `(C_out, C_in, kH, kW)` — **OIHW**
- `padding`: an **int**, applied symmetrically to all four sides
- output: `(B, C_out, H_out, W_out)` with
  $H_{out} = \lfloor (H + 2p - k_H)/s \rfloor + 1$

### Rules
- Do **not** use `jax.lax.conv_general_dilated` or `jax.scipy.signal`
- No Python loop over output pixels
- `bias` is `(C_out,)` and is added per output channel

### The shape that makes this easy
The trick is to materialise every sliding window at once, so the convolution
becomes a single tensor contraction:

```
patches: (B, C_in, H_out, W_out, kH, kW)
weight:  (C_out, C_in, kH, kW)
        -> einsum('bihwjk,oijk->bohw')
```

Getting that einsum right first try is the actual test. Read it as: for each
batch `b` and output position `(h, w)`, contract over the input channel `i` and
the kernel offsets `(j, k)` to produce output channel `o`.

Build the windows with **index arrays**, not a loop:
`rows = arange(H_out)[:, None] * stride + arange(kH)[None, :]` gives an
`(H_out, kH)` matrix that gathers every row window in one go.

### Layout note
This task follows the PyTorch original's NCHW/OIHW convention so the two
implementations line up line for line. Flax goes the other way — `nnx.Conv`
uses **NHWC** inputs and **HWIO** kernels, because that layout is what XLA
prefers on TPU. Converting is a transpose in each direction; the arithmetic
here is identical either way.

### Why im2col at all
Materialising the patches costs $k_H k_W$ memory blow-up, which sounds
terrible — but it turns the convolution into one big matmul, and matmuls are
what hardware is built to do. Real implementations either do exactly this
(im2col + GEMM) or fuse the gather into the matmul; almost nobody runs the
naive seven-deep loop.
""",
    "stub": '''import jax
import jax.numpy as jnp


def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    """2-D cross-correlation.

    Args:
        x:       (B, C_in, H, W)          NCHW
        weight:  (C_out, C_in, kH, kW)    OIHW
        bias:    (C_out,) or None
        stride:  int
        padding: int, applied to all four sides

    Returns:
        (B, C_out, H_out, W_out)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        # Pad only the spatial axes, symmetrically.
        x = jnp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # Index matrices that name every window position at once — this is the
    # loop-free replacement for torch's x.unfold(2, kH, stride).
    rows = jnp.arange(H_out)[:, None] * stride + jnp.arange(kH)[None, :]  # (H_out, kH)
    cols = jnp.arange(W_out)[:, None] * stride + jnp.arange(kW)[None, :]  # (W_out, kW)

    # (B, C_in, H_out, kH, W) -> (B, C_in, H_out, kH, W_out, kW)
    patches = x[:, :, rows][..., cols]
    # -> (B, C_in, H_out, W_out, kH, kW), so the einsum below reads exactly
    #    like the PyTorch original's.
    patches = patches.transpose(0, 1, 2, 4, 3, 5)

    out = jnp.einsum("bihwjk,oijk->bohw", patches, weight)

    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 3, 8, 8))     # NCHW
w = jax.random.normal(jax.random.key(1), (4, 3, 3, 3))     # OIHW

print("valid, stride 1:", my_conv2d(x, w).shape,                    "(2, 4, 6, 6)")
print("same-ish, pad 1:", my_conv2d(x, w, padding=1).shape,         "(2, 4, 8, 8)")
print("stride 2:       ", my_conv2d(x, w, stride=2).shape,          "(2, 4, 3, 3)")
''',
    "tests": [
        {
            "name": "Matches jax.lax.conv_general_dilated",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 3, 9, 9))
w = jax.random.normal(jax.random.key(1), (4, 3, 3, 3))

out = {fn}(x, w)
ref = jax.lax.conv_general_dilated(
    x, w, window_strides=(1, 1), padding=((0, 0), (0, 0)),
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
)
assert out.shape == ref.shape, f'Shape mismatch: {out.shape} vs {ref.shape}'
assert jnp.allclose(out, ref, atol=1e-4), 'Values differ from the reference convolution'
""",
        },
        {
            "name": "Padding and stride",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (2, 3, 9, 9))
w = jax.random.normal(jax.random.key(3), (4, 3, 3, 3))

for stride, pad in ((1, 1), (2, 0), (2, 1), (3, 2)):
    out = {fn}(x, w, stride=stride, padding=pad)
    ref = jax.lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    assert out.shape == ref.shape, (
        f'stride={stride} pad={pad}: shape {out.shape} vs {ref.shape}'
    )
    assert jnp.allclose(out, ref, atol=1e-4), f'stride={stride} pad={pad}: values differ'
""",
        },
        {
            "name": "Output size formula",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(4), (1, 2, 10, 12))
for (kH, kW), stride, pad in (((3, 3), 1, 0), ((5, 5), 2, 2), ((1, 1), 1, 0), ((3, 5), 2, 1)):
    w = jax.random.normal(jax.random.key(5), (3, 2, kH, kW))
    out = {fn}(x, w, stride=stride, padding=pad)
    H_out = (10 + 2 * pad - kH) // stride + 1
    W_out = (12 + 2 * pad - kW) // stride + 1
    assert out.shape == (1, 3, H_out, W_out), (
        f'k=({kH},{kW}) s={stride} p={pad}: got {out.shape}, expected (1, 3, {H_out}, {W_out})'
    )
""",
        },
        {
            "name": "Hand-computed 1-channel case",
            "code": """
import jax.numpy as jnp

# Identity kernel picks out the top-left element of each 2x2 window.
x = jnp.arange(16.0).reshape(1, 1, 4, 4)
w = jnp.array([[[[1.0, 0.0], [0.0, 0.0]]]])          # (1, 1, 2, 2)
out = {fn}(x, w)

expected = jnp.array([[[[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [8.0, 9.0, 10.0]]]])
assert out.shape == (1, 1, 3, 3), f'{out.shape}'
assert jnp.allclose(out, expected), f'{out} vs {expected}'

# Summing kernel over a 2x2 window.
w2 = jnp.ones((1, 1, 2, 2))
out2 = {fn}(x, w2)
assert jnp.allclose(out2[0, 0, 0, 0], 0 + 1 + 4 + 5), f'{out2[0, 0, 0, 0]}'
""",
        },
        {
            "name": "Cross-correlation, not flipped convolution",
            "code": """
import jax.numpy as jnp

# An asymmetric kernel distinguishes correlation from true convolution.
x = jnp.array([[[[1.0, 2.0, 3.0]]]])                 # (1, 1, 1, 3)
w = jnp.array([[[[1.0, 0.0, 0.0]]]])                 # (1, 1, 1, 3)
out = {fn}(x, w)

assert jnp.allclose(out, 1.0), (
    f'Got {out}. Deep-learning "convolution" is cross-correlation — the kernel '
    'is NOT flipped. A flipped kernel would give 3.0 here.'
)
""",
        },
        {
            "name": "Bias is per output channel",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(6), (2, 3, 6, 6))
w = jax.random.normal(jax.random.key(7), (4, 3, 3, 3))
b = jnp.array([1.0, -2.0, 10.0, 0.0])

no_bias = {fn}(x, w)
with_bias = {fn}(x, w, bias=b)

assert jnp.allclose(with_bias, no_bias + b.reshape(1, -1, 1, 1), atol=1e-4), (
    'bias must add per OUTPUT CHANNEL — broadcast over (B, H_out, W_out)'
)
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(8), (2, 2, 7, 7))
w = jax.random.normal(jax.random.key(9), (3, 2, 3, 3))

gx, gw = jax.grad(lambda a, b: jnp.sum({fn}(a, b) ** 2), argnums=(0, 1))(x, w)
assert gx.shape == x.shape and gw.shape == w.shape, 'Gradient shapes wrong'
assert jnp.isfinite(gx).all() and jnp.isfinite(gw).all(), 'Non-finite gradient'

jitted = jax.jit(functools.partial({fn}, stride=2, padding=1))
assert jnp.allclose(jitted(x, w), {fn}(x, w, stride=2, padding=1), atol=1e-4), (
    'jit changes the result'
)
""",
        },
    ],
}
