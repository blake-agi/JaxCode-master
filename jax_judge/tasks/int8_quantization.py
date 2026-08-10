"""INT8 quantized linear — per-output-channel symmetric quantization."""

TASK = {
    "title": "INT8 Quantized Linear",
    "category": "Inference & Decoding",
    "number": "36",
    "difficulty": "Hard",
    "function_name": "Int8Linear",
    "hint": (
        "One scale per OUTPUT channel: take the max absolute value along the "
        "input axis with keepdims, divide by 127. Quantize with round then clip "
        "to [-128, 127] and cast to int8. At forward time dequantize back to "
        "float (int8 * scale) and do the matmul — with a (out, in) weight that "
        "means x @ w.T. Store the int8 weights and the scale as nnx.Variable, "
        "not nnx.Param: they are frozen buffers, not things you train."
    ),
    "description": r"""
Implement an **INT8-quantized linear layer** using symmetric per-channel
quantization.

$$s_o = \frac{\max_i |W_{oi}|}{127}, \qquad
  W^{q}_{oi} = \mathrm{clip}\big(\mathrm{round}(W_{oi}/s_o),\ -128,\ 127\big)$$

and at inference $\hat{W} = W^{q} \cdot s$, then $y = x\hat{W}^\top + b$.

### Signature
```python
class Int8Linear(nnx.Module):
    def __init__(self, weight, bias=None): ...   # weight: (out, in)
    def __call__(self, x): ...
```

### Rules
- **Symmetric**: zero maps to zero, so there is no zero-point
- **Per output channel**: one scale per row of `weight`, shape `(out, 1)`
- Quantize with `round`, then clip to `[-128, 127]`, then cast to `int8`
- Guard the division with `1e-10` so an all-zero row does not produce `NaN`
- Store `weight_int8` and `scale` as `nnx.Variable` (buffers); `bias` stays an
  `nnx.Param`

### Per-tensor vs per-channel
A single scale for the whole matrix is per-*tensor* quantization. It is cheaper
but fragile: one output channel with an unusually large weight sets the scale
for every channel, and all the small ones collapse into a handful of integer
levels. Per-channel gives each row its own scale, costs one float per row, and
is what makes INT8 weight quantization essentially lossless in practice.

### What actually breaks in LLMs
Weight quantization is the easy half. **Activation** quantization is where INT8
falls over, because transformer activations develop systematic outlier
channels — a few dimensions with magnitudes 100× the rest, appearing past
roughly 6.7B parameters. One outlier sets the scale for the whole tensor and
destroys the rest. That observation is exactly what LLM.int8() addresses, by
keeping outlier channels in fp16 and quantizing only the well-behaved ones.

### Why dequantize at all
This implementation stores int8 and computes in float — the win is **memory**
and bandwidth (4× smaller weights), not arithmetic. True int8 matmul with int32
accumulation needs hardware support and a fused kernel; the dequantize-then-
matmul form is what you write when you are showing you understand the numerics.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class Int8Linear(nnx.Module):
    """Linear layer holding INT8 weights plus a per-channel scale."""

    def __init__(self, weight, bias=None):
        """Args:
            weight: (out_features, in_features) float array to quantize
            bias:   (out_features,) or None
        """
        pass  # Replace this

    def __call__(self, x):
        """(..., in_features) -> (..., out_features)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class Int8Linear(nnx.Module):
    def __init__(self, weight, bias=None):
        # One scale per OUTPUT channel: reduce over the input axis.
        scale = jnp.max(jnp.abs(weight), axis=1, keepdims=True) / 127.0

        q = jnp.round(weight / (scale + 1e-10))
        q = jnp.clip(q, -128, 127).astype(jnp.int8)

        # Buffers, not parameters — nnx.Variable keeps them out of nnx.Param
        # so an optimizer never tries to train them.
        self.weight_int8 = nnx.Variable(q)
        self.scale = nnx.Variable(scale)
        self.bias = nnx.Param(bias) if bias is not None else None

    def __call__(self, x):
        # Dequantize, then matmul in float. The saving is memory, not flops.
        w = self.weight_int8[...].astype(x.dtype) * self.scale[...]
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias[...]
        return out
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

w = jax.random.normal(jax.random.key(0), (16, 32))
layer = Int8Linear(w)

x = jax.random.normal(jax.random.key(1), (4, 32))
exact = x @ w.T
approx = layer(x)

print("int8 dtype :", layer.weight_int8[...].dtype)
print("scale shape:", layer.scale[...].shape, "(one per output channel)")
print("rel error  :", float(jnp.abs(approx - exact).max() / jnp.abs(exact).max()))
print("memory     : 4x smaller weights (int8 vs float32)")
''',
    "tests": [
        {
            "name": "Stores int8 weights and a per-channel scale",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

w = jax.random.normal(jax.random.key(0), (8, 16))
layer = {fn}(w)

q = layer.weight_int8[...]
s = layer.scale[...]

assert q.dtype == jnp.int8, f'weight_int8 dtype should be int8, got {q.dtype}'
assert q.shape == (8, 16), f'weight_int8 shape {q.shape} vs (8, 16)'
assert s.shape == (8, 1), (
    f'scale shape {s.shape} vs (8, 1) — one scale per OUTPUT channel, so reduce '
    'over the input axis with keepdims'
)
assert jnp.all(q >= -128) and jnp.all(q <= 127), 'int8 values out of range'
""",
        },
        {
            "name": "Buffers are not trainable parameters",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

w = jax.random.normal(jax.random.key(1), (8, 16))
b = jnp.zeros(8)
layer = {fn}(w, b)

params = nnx.state(layer, nnx.Param)
leaves = jax.tree.leaves(params)
total = sum(x.size for x in leaves)
assert total == 8, (
    f'Only the bias should be an nnx.Param ({8} values), found {total}. '
    'weight_int8 and scale are frozen buffers — use nnx.Variable.'
)
""",
        },
        {
            "name": "Round-trip error is small",
            "code": """
import jax
import jax.numpy as jnp

w = jax.random.normal(jax.random.key(2), (16, 32))
layer = {fn}(w)

deq = layer.weight_int8[...].astype(jnp.float32) * layer.scale[...]
rel = jnp.abs(deq - w).max() / jnp.abs(w).max()
assert rel < 0.02, (
    f'Weight round-trip relative error {float(rel):.4f} is too large for '
    'per-channel int8 — expected well under 1%'
)

x = jax.random.normal(jax.random.key(3), (4, 32))
out = layer(x)
exact = x @ w.T
out_rel = jnp.abs(out - exact).max() / jnp.abs(exact).max()
assert out_rel < 0.05, f'Output relative error {float(out_rel):.4f} too large'
""",
        },
        {
            "name": "Per-channel, not per-tensor",
            "code": """
import jax
import jax.numpy as jnp

# Row 0 is tiny, row 1 is huge. Per-tensor quantization would crush row 0 into
# almost no integer levels; per-channel keeps both accurate.
w = jnp.stack([jnp.full((8,), 0.001), jnp.full((8,), 100.0)])
layer = {fn}(w)

s = layer.scale[...]
assert s.shape == (2, 1), f'scale shape {s.shape}'
assert not jnp.allclose(s[0], s[1]), (
    'Both rows got the same scale — that is per-TENSOR quantization. '
    'Each output channel needs its own.'
)

deq = layer.weight_int8[...].astype(jnp.float32) * s
assert jnp.abs(deq[0] - w[0]).max() / 0.001 < 0.02, (
    'The small row lost precision — its scale is being set by the large row'
)
""",
        },
        {
            "name": "Symmetric: zero maps to zero",
            "code": """
import jax
import jax.numpy as jnp

w = jnp.zeros((4, 8)).at[0, 0].set(5.0)
layer = {fn}(w)
q = layer.weight_int8[...]

assert jnp.all(q[1:] == 0), (
    'Zero weights must quantize to exactly 0 — symmetric quantization has no '
    'zero-point offset'
)
assert jnp.isfinite(layer.scale[...]).all(), (
    'All-zero rows produced a non-finite scale — guard the division with 1e-10'
)
out = layer(jnp.ones((2, 8)))
assert jnp.isfinite(out).all(), 'Non-finite output from an all-zero row'
""",
        },
        {
            "name": "Bias and shapes",
            "code": """
import jax
import jax.numpy as jnp

w = jax.random.normal(jax.random.key(4), (8, 16))
b = jnp.arange(8.0)

no_b = {fn}(w)
with_b = {fn}(w, b)
x = jax.random.normal(jax.random.key(5), (5, 16))

assert no_b(x).shape == (5, 8), f'{no_b(x).shape}'
assert jnp.allclose(with_b(x), no_b(x) + b, atol=1e-4), 'bias must be added per output channel'
assert no_b.bias is None, 'bias should be None when not supplied'

# Leading batch axes pass through.
x3 = jax.random.normal(jax.random.key(6), (2, 5, 16))
assert with_b(x3).shape == (2, 5, 8), f'{with_b(x3).shape}'
""",
        },
        {
            "name": "Clipping at the int8 boundary",
            "code": """
import jax
import jax.numpy as jnp

# The max-magnitude entry should land exactly on +/-127, never overflow.
w = jax.random.normal(jax.random.key(7), (4, 32)) * 10.0
layer = {fn}(w)
q = layer.weight_int8[...]

assert jnp.abs(q).max() == 127, (
    f'The largest-magnitude weight in a row should quantize to 127, got '
    f'{int(jnp.abs(q).max())} — check that scale is max|W|/127'
)
assert q.min() >= -128, f'Value below -128: {int(q.min())}'
""",
        },
    ],
}
