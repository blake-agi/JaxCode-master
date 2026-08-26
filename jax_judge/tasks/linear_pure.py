"""Problem 03 without Flax — same shape of API, no library."""

_WHY = r"""
### Why this exists alongside problem 03
Interview sandboxes (CoderPad and friends) often ship `jax` and nothing else,
which makes every `nnx.Module` problem here unrunnable there.

**The API is deliberately as close to `nnx` as it can be** — same class name,
same argument names, same attribute names, same array layouts. Only the source
of randomness changes:

```python
nnx.Linear(4, 3, rngs=nnx.Rngs(params=0))    # nnx
SimpleLinear(4, 3, key=jax.random.key(0))    # here
```

so practising this reinforces the `nnx` version instead of competing with it.

### What you give up
A plain Python class is **not a pytree**, so `jax.grad(loss)(layer)` does not
work. Differentiate with respect to the input, or keep the arrays outside the
object.

Rebinding `self.kernel` to a tracer inside a traced function looks like a way
around that — it even returns the right gradient once — and then leaks the
tracer into the next call:

```
UnexpectedTracerError: A function transformed by JAX had a side effect ...
```

That leak is exactly the problem Flax and Equinox exist to solve. In an
interview you are almost always asked for the forward pass, so this trade is
usually free.
"""

TASK = {
    "title": "Linear Layer without Flax",
    "category": "Core Ops & Layers",
    "number": "b_23",
    "difficulty": "Easy",
    "function_name": "SimpleLinear",
    "hint": (
        "Identical to nnx.Linear except you build the arrays yourself: "
        "self.kernel = jax.random.normal(key, (in_features, out_features)) / "
        "jnp.sqrt(in_features), and self.bias = jnp.zeros((out_features,)) if "
        "use_bias else None. __call__ is x @ self.kernel plus the bias when it "
        "is not None. Keep the (in, out) layout — that is nnx.Linear's, and it "
        "is why __call__ needs no transpose."
    ),
    "description": r"""
Problem 03's linear layer, written with no Flax.

### Signature
```python
class SimpleLinear:
    def __init__(self, in_features, out_features, *, key, use_bias=True): ...
    def __call__(self, x): ...        # (..., in_features) -> (..., out_features)
```

Same class name, same argument names, same attributes as the `nnx` version:

| | |
|---|---|
| `self.kernel` | `(in_features, out_features)`, scaled by `1/sqrt(in_features)` |
| `self.bias` | `(out_features,)` zeros, or **`None`** when `use_bias=False` |

`kernel` is `(in, out)` — the Flax layout, the transpose of PyTorch's — which
is why `__call__` is `x @ self.kernel` with no transpose anywhere.

### Any leading shape
`(in,)`, `(N, in)` and `(B, T, in)` all work for free, as long as `__call__`
never mentions the batch axes.
""" + _WHY,
    "stub": '''import jax
import jax.numpy as jnp


class SimpleLinear:
    """y = x @ kernel + bias, with the arrays built by hand."""

    def __init__(self, in_features, out_features, *, key, use_bias=True):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


class SimpleLinear:
    def __init__(self, in_features, out_features, *, key, use_bias=True):
        # (in, out), the same layout as nnx.Linear — so __call__ never transposes.
        self.kernel = jax.random.normal(
            key, (in_features, out_features)
        ) / jnp.sqrt(in_features)
        # None rather than zeros: "no bias" should be visibly absent.
        self.bias = jnp.zeros((out_features,)) if use_bias else None

    def __call__(self, x):
        # Never name the leading axes and they look after themselves.
        y = x @ self.kernel
        return y if self.bias is None else y + self.bias
''',
    "demo": '''import jax
import jax.numpy as jnp

layer = SimpleLinear(4, 3, key=jax.random.key(0))
print("kernel", layer.kernel.shape, " bias", layer.bias.shape)

for shape in [(4,), (10, 4), (2, 5, 4)]:
    print(f"  {str(shape):<10} -> {layer(jnp.ones(shape)).shape}")

no_bias = SimpleLinear(4, 3, key=jax.random.key(0), use_bias=False)
print("\\nuse_bias=False -> bias is", no_bias.bias)

# A plain class is not a pytree, so differentiate w.r.t. the INPUT.
g = jax.grad(lambda v: jnp.sum(layer(v)))(jnp.ones((2, 4)))
print("d/dx shape:", g.shape)
''',
    "tests": [
        {
            "name": "Attributes match nnx.Linear's layout",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(4, 3, key=jax.random.key(0))
assert hasattr(m, 'kernel'), "the weight attribute must be called 'kernel', as in nnx.Linear"
assert hasattr(m, 'bias'), "the bias attribute must be called 'bias'"
assert m.kernel.shape == (4, 3), (
    f'kernel {m.kernel.shape} vs (4, 3) — (in, out), the transpose of torch'
)
assert m.bias.shape == (3,), f'bias {m.bias.shape} vs (3,)'
assert jnp.allclose(m.bias, 0.0), 'bias should start at zeros'

nb = {fn}(4, 3, key=jax.random.key(0), use_bias=False)
assert nb.bias is None, (
    f'use_bias=False should leave bias as None, got {type(nb.bias).__name__}. '
    'A zero bias is still a parameter; an absent one is absent.'
)
""",
        },
        {
            "name": "Computes x @ kernel + bias for any leading shape",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(4, 3, key=jax.random.key(0))
for shape in [(4,), (10, 4), (2, 5, 4)]:
    x = jax.random.normal(jax.random.key(1), shape)
    out = m(x)
    assert out.shape == shape[:-1] + (3,), f'{shape} -> {out.shape}'
    assert jnp.allclose(out, x @ m.kernel + m.bias, atol=1e-5), f'wrong values for {shape}'

nb = {fn}(4, 3, key=jax.random.key(0), use_bias=False)
x = jax.random.normal(jax.random.key(2), (6, 4))
assert jnp.allclose(nb(x), x @ nb.kernel, atol=1e-5), 'without a bias it is exactly x @ kernel'
""",
        },
        {
            "name": "Initialisation scale and key behaviour",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(256, 128, key=jax.random.key(0))
std = float(jnp.std(m.kernel))
assert abs(std - 1.0 / 256 ** 0.5) < 0.02, (
    f'kernel std {std:.4f}, expected ~{1/256**0.5:.4f} = 1/sqrt(in_features)'
)
a = {fn}(8, 8, key=jax.random.key(1)).kernel
b = {fn}(8, 8, key=jax.random.key(2)).kernel
c = {fn}(8, 8, key=jax.random.key(1)).kernel
assert not jnp.allclose(a, b), 'different keys gave identical weights'
assert jnp.allclose(a, c), 'the same key must be reproducible'
""",
        },
        {
            "name": "Gradient w.r.t. the input; the arrays are plain jnp",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(4, 3, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (7, 4))

g = jax.grad(lambda v: jnp.sum(m(v)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. x'
assert jnp.allclose(g, jnp.broadcast_to(m.kernel.sum(axis=1), x.shape), atol=1e-5), (
    'd/dx sum(x @ kernel + b) should be the row sums of kernel'
)

assert isinstance(m.kernel, jax.Array), f'kernel is {type(m.kernel).__name__}, not a jax array'
m.kernel = m.kernel - 0.1 * jnp.ones_like(m.kernel)
assert jnp.isfinite(m(x)).all(), 'the layer stopped working after a manual update'
""",
        },
        {
            "name": "jit and vmap through __call__",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(4, 3, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (6, 4))

assert jnp.allclose(jax.jit(lambda v: m(v))(x), m(x), atol=1e-6), 'jit disagrees'
per_row = jax.vmap(lambda v: m(v))(x)
assert per_row.shape == (6, 3), f'{per_row.shape} vs (6, 3)'
assert jnp.allclose(per_row, m(x), atol=1e-5), (
    'vmap disagrees — __call__ must not mention the batch axis'
)
""",
        },
    ],
}
