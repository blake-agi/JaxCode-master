"""int8 quantization — scale, zero-point, and why outliers ruin everything."""

TASK = {
    "title": "INT8 Quantization (symmetric and asymmetric)",
    "category": "Inference & Decoding",
    "order": 5,
    "difficulty": "Medium",
    "function_name": "quantize_int8",
    "hint": (
        "Both modes are the same three lines with a different (scale, "
        "zero_point). Derive the asymmetric zero point rather than memorising "
        "it: you want min(x) to land on -128 and max(x) on 127, so write down "
        "round(min/s) + z = -128 and solve. Sanity-check your formula on both "
        "endpoints before writing any code. Three mechanical traps: clip to "
        "[-128, 127] yourself instead of trusting the cast to saturate, do the "
        "dequant subtraction in float (q - zero_point wraps if both stay int8), "
        "and guard the zero-range tensor with jnp.where rather than a Python "
        "`if`, so the function still traces under jit."
    ),
    "description": r"""
Implement per-tensor **int8 quantization**, both symmetric and asymmetric.

**Symmetric** — one parameter, zero maps exactly to zero:

$$s = \frac{\max|x|}{127}, \quad z = 0, \quad q = \text{clip}\!\left(\text{round}\!\left(\tfrac{x}{s}\right), -128, 127\right)$$

**Asymmetric** — two parameters, uses the full range on skewed data:

$$s = \frac{\max(x) - \min(x)}{255}, \quad
z = \text{round}\!\left(\tfrac{-\min(x)}{s}\right) - 128, \quad
q = \text{clip}\!\left(\text{round}\!\left(\tfrac{x}{s}\right) + z, -128, 127\right)$$

Dequantization is the same in both cases: $\hat{x} = (q - z)\,s$.

### Signature
```python
def quantize_int8(x, symmetric=True):
    ...  # -> (q, scale, zero_point, x_dequant)
```
`q` must be `int8`. `scale` is a scalar float, `zero_point` a scalar int.

### Rules
- Per-**tensor** (one scale for the whole array), not per-channel
- Clip into `[-128, 127]` before casting. JAX's float→int8 cast happens to
  saturate, but NumPy's and PyTorch's wrap (`200.0` becomes `-56`) — the
  explicit clip is what makes the arithmetic portable
- Handle a constant tensor (zero range) without producing `NaN`/`Inf`
- Round with `jnp.round` — round-half-to-even, the IEEE-754 default, matching
  `np.round` and `torch.round` so a cross-framework diff stays bit-identical

### Symmetric vs asymmetric
Symmetric is cheaper: with $z=0$, a quantized matmul is just an integer matmul
times a scale. Asymmetric needs cross-terms and extra bookkeeping. But on skewed
data — post-ReLU activations are all $\ge 0$ — symmetric throws away half the
range, since nothing ever maps below 0. Rule of thumb: **symmetric for weights**
(roughly zero-centred), **asymmetric for activations**.

### The thing that actually breaks LLM quantization
Per-tensor quantization is hostage to a single number: $\max|x|$. Transformer
activations famously contain a handful of **outlier channels** with magnitudes
20-100x everything else, and they consistently appear in the same feature
dimensions. One outlier inflates the scale, and every ordinary value collapses
into a couple of quantization levels — accuracy falls off a cliff, and it gets
*worse* as models get bigger.

That single observation is what the whole modern literature is built around:
LLM.int8() splits the outlier channels out into fp16 — correct, but a
mixed-precision matmul costs throughput. SmoothQuant keeps one dtype and instead
migrates the difficulty from activations into weights with a per-channel
rescaling, and AWQ searches for per-channel scales that protect the ~1% salient
weight channels *without* holding anything at higher precision. "Why did the
later work move away from splitting outliers into fp16?" is the natural
follow-up question. The tests below reproduce the failure directly so you can
see the magnitude of it.
""",
    "stub": '''import jax
import jax.numpy as jnp


def quantize_int8(x, symmetric=True):
    """Per-tensor int8 quantization.

    Args:
        x:         float array
        symmetric: True for symmetric (zero_point = 0), False for asymmetric

    Returns:
        (q, scale, zero_point, x_dequant)
          q          int8 array, same shape as x
          scale      python/JAX float scalar
          zero_point python/JAX int scalar
          x_dequant  float array, the round-trip reconstruction
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def quantize_int8(x, symmetric=True):
    x = jnp.asarray(x, dtype=jnp.float32)

    if symmetric:
        amax = jnp.max(jnp.abs(x))
        # A constant-zero tensor has no range; fall back to scale 1 so the
        # division is finite and everything quantizes to 0.
        scale = jnp.where(amax > 0, amax / 127.0, 1.0)
        zero_point = jnp.array(0, dtype=jnp.int32)
        q = jnp.round(x / scale)
    else:
        xmin, xmax = jnp.min(x), jnp.max(x)
        rng = xmax - xmin
        scale = jnp.where(rng > 0, rng / 255.0, 1.0)
        zero_point = (jnp.round(-xmin / scale) - 128).astype(jnp.int32)
        q = jnp.round(x / scale) + zero_point

    q = jnp.clip(q, -128, 127).astype(jnp.int8)
    x_dequant = (q.astype(jnp.float32) - zero_point) * scale
    return q, scale, zero_point, x_dequant
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (1000,))
_, s, z, xd = quantize_int8(x, symmetric=True)
print(f"clean:    scale={float(s):.5f}  max err={float(jnp.abs(x - xd).max()):.5f}")

# Now plant a single outlier, as real transformer activations have.
x_out = x.at[0].set(50.0)
_, s2, _, xd2 = quantize_int8(x_out, symmetric=True)
err = float(jnp.abs(x_out[1:] - xd2[1:]).max())
print(f"1 outlier: scale={float(s2):.5f}  max err on the OTHER 999 values={err:.5f}")
print(f"-> one value inflated the error on everything else by {err / float(jnp.abs(x - xd).max()):.0f}x")
''',
    "tests": [
        {
            "name": "Symmetric: dtype, scale and zero point",
            "code": """
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
q, scale, zp, xd = {fn}(x, symmetric=True)

assert q.dtype == jnp.int8, f'q must be int8, got {q.dtype}'
assert q.shape == x.shape, f'q shape {q.shape} vs {x.shape}'
assert xd.shape == x.shape, f'dequant shape {xd.shape} vs {x.shape}'
assert int(zp) == 0, f'Symmetric quantization must have zero_point == 0, got {int(zp)}'
assert jnp.allclose(scale, 2.0 / 127.0, rtol=1e-5), (
    f'scale should be max|x|/127 = {2.0/127.0:.6f}, got {float(scale)}'
)
# The extreme value must land exactly on 127.
assert int(q[-1]) == 127, f'max|x| should map to 127, got {int(q[-1])}'
assert int(q[2]) == 0, f'0.0 must map to 0 under symmetric quantization, got {int(q[2])}'
""",
        },
        {
            "name": "Asymmetric: uses the full range on skewed data",
            "code": """
import jax
import jax.numpy as jnp

# Post-ReLU style: everything non-negative.
x = jax.random.uniform(jax.random.key(0), (500,), minval=0.0, maxval=6.0)

q_s, s_s, z_s, xd_s = {fn}(x, symmetric=True)
q_a, s_a, z_a, xd_a = {fn}(x, symmetric=False)

assert jnp.allclose(s_a, (x.max() - x.min()) / 255.0, rtol=1e-4), (
    f'Asymmetric scale should be (max-min)/255, got {float(s_a)}'
)
assert int(z_a) != 0, 'Asymmetric quantization on non-negative data needs a non-zero zero_point'

# Symmetric wastes the whole negative half of the range, so it must be worse.
err_s = float(jnp.abs(x - xd_s).max())
err_a = float(jnp.abs(x - xd_a).max())
assert err_a < err_s, (
    f'On all-positive data asymmetric should beat symmetric: {err_a} vs {err_s}. '
    'Symmetric maps nothing below zero, throwing away 128 of the 256 levels.'
)

# Asymmetric should span nearly the whole int8 range.
assert int(q_a.max()) - int(q_a.min()) > 250, (
    f'Asymmetric should use ~all 256 levels, spans only {int(q_a.max()) - int(q_a.min())}'
)
""",
        },
        {
            "name": "Round-trip error is bounded by half a step",
            "code": """
import jax
import jax.numpy as jnp

for sym in (True, False):
    for i, scale_factor in enumerate((0.01, 1.0, 100.0)):
        x = jax.random.normal(jax.random.key(i), (400,)) * scale_factor
        q, scale, zp, xd = {fn}(x, symmetric=sym)

        err = jnp.abs(x - xd).max()
        assert err <= scale * 0.5 + 1e-4, (
            f'symmetric={sym} scale_factor={scale_factor}: max round-trip error '
            f'{float(err):.6f} exceeds half a quantization step {float(scale)*0.5:.6f}'
        )
        assert jnp.isfinite(xd).all(), 'Non-finite dequantized values'
        assert int(q.min()) >= -128 and int(q.max()) <= 127, (
            f'q out of int8 range: [{int(q.min())}, {int(q.max())}]'
        )
""",
        },
        {
            "name": "Dequantization is the stated inverse",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (7, 11))

for sym in (True, False):
    q, scale, zp, xd = {fn}(x, symmetric=sym)
    manual = (q.astype(jnp.float32) - zp) * scale
    assert jnp.allclose(xd, manual, atol=1e-5), (
        f'symmetric={sym}: the returned dequant must equal (q - zero_point) * scale'
    )
""",
        },
        {
            "name": "Constant tensor does not divide by zero",
            "code": """
import jax.numpy as jnp

for sym in (True, False):
    for const in (0.0, 3.5, -2.0):
        x = jnp.full((10,), const)
        q, scale, zp, xd = {fn}(x, symmetric=sym)
        assert jnp.isfinite(scale), f'symmetric={sym} const={const}: scale={float(scale)}'
        assert jnp.isfinite(xd).all(), (
            f'symmetric={sym} const={const}: dequant has non-finite values — '
            'a zero range divided by zero'
        )
        assert jnp.isfinite(jnp.asarray(q, dtype=jnp.float32)).all(), 'non-finite q'
""",
        },
        {
            "name": "The outlier failure mode",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(4), (1000,))
_, _, _, xd_clean = {fn}(x, symmetric=True)
err_clean = float(jnp.abs(x[1:] - xd_clean[1:]).max())

# One channel 50x larger, exactly the transformer-activation pathology.
x_out = x.at[0].set(50.0)
_, scale_out, _, xd_out = {fn}(x_out, symmetric=True)
err_out = float(jnp.abs(x_out[1:] - xd_out[1:]).max())

assert jnp.allclose(scale_out, 50.0 / 127.0, rtol=1e-4), (
    f'The scale must be set by the outlier: expected {50.0/127.0:.5f}, got {float(scale_out)}'
)
assert err_out > 10 * err_clean, (
    f'A single outlier should wreck the precision of every other value '
    f'({err_out:.5f} vs {err_clean:.5f}). If it did not, the scale is not '
    'being driven by max|x| as per-tensor quantization requires.'
)
""",
        },
        {
            "name": "Shape preservation and jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(5), (3, 4, 5))
for sym in (True, False):
    q, scale, zp, xd = {fn}(x, symmetric=sym)
    assert q.shape == (3, 4, 5), f'symmetric={sym}: q shape {q.shape}'
    assert xd.shape == (3, 4, 5), f'symmetric={sym}: dequant shape {xd.shape}'
    assert jnp.ndim(scale) == 0, f'scale must be a scalar, got shape {jnp.shape(scale)}'

    jitted = jax.jit(functools.partial({fn}, symmetric=sym))
    qj, sj, zj, xdj = jitted(x)
    assert jnp.allclose(xdj, xd, atol=1e-5), f'symmetric={sym}: jit changes the result'
    assert (qj == q).all(), f'symmetric={sym}: jit changes the codes'
""",
        },
    ],
}
