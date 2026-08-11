"""Multi-head cross-attention — queries from one stream, keys/values from another."""

TASK = {
    "title": "Cross-Attention",
    "category": "Attention & Transformers",
    "number": "23",
    "difficulty": "Medium",
    "function_name": "MultiHeadCrossAttention",
    "hint": (
        "Same four nnx.Linear(d_model, d_model) as self-attention. The only "
        "change is where each projection reads from: W_q sees x_q, while W_k and "
        "W_v both see x_kv. Read seq_q off x_q and seq_kv off x_kv separately — the "
        "score matrix is (seq_q, seq_kv) and is generally not square. The output "
        "length always follows the QUERY."
    ),
    "description": r"""
Implement **multi-head cross-attention**: queries come from one sequence, keys
and values from another.

$$\text{CrossAttn}(x_q, x_{kv}) = \text{softmax}\!\left(
\frac{(x_q W^Q)(x_{kv} W^K)^\top}{\sqrt{d_k}}\right)(x_{kv} W^V)\,W^O$$

### Signature
```python
class MultiHeadCrossAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs): ...
    def __call__(self, x_q, x_kv): ...
```

### Requirements
- Use `nnx.Linear(d_model, d_model)` for `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`
- `self.d_k = d_model // num_heads`
- `x_q` is `(B, seq_q, d_model)`, `x_kv` is `(B, seq_kv, d_model)`
- Output is `(B, seq_q, d_model)` — the **query** length

### Self-attention vs cross-attention
Structurally they are the same computation; the difference is entirely in what
gets projected:

| | Q from | K, V from |
|---|---|---|
| self-attention | `x` | `x` |
| cross-attention | `x_q` | `x_kv` |

So `cross(x, x)` is exactly self-attention. Everything else — the heads, the
scaling, the softmax axis — is unchanged.

### Where it shows up
- **Encoder-decoder** transformers: the decoder queries the encoder's output,
  which is how translation conditions on the source sentence.
- **Diffusion models**: image latents query text embeddings — this is the
  single point where the prompt enters a U-Net.
- **Perceiver / Flamingo**: a small set of learned latent queries attends over a
  large input, decoupling compute from input size.

### Why the output follows the query
Each query position produces exactly one output row, no matter how many keys it
attended over. That is what lets cross-attention consume a context of a
completely different length — a 1000-token document can condition a 10-token
generation, and the cost is $O(seq_q S_{kv})$ rather than anything quadratic in
the larger one alone.

### The trap
Reading a single sequence length off `x_q` and reusing it for `x_kv` passes
every equal-length test and then fails the moment the two differ. Read both.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadCrossAttention(nnx.Module):
    """Queries from x_q attend over keys/values from x_kv."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x_q, x_kv):
        """(B, seq_q, d_model), (B, seq_kv, d_model) -> (B, seq_q, d_model)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadCrossAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def _split(self, t, B, seq):
        return t.reshape(B, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, x_q, x_kv):
        B, seq_q, _ = x_q.shape
        seq_kv = x_kv.shape[1]        # read separately — the two can differ

        # The ONLY difference from self-attention: q reads x_q, k/v read x_kv.
        q = self._split(self.W_q(x_q), B, seq_q)
        k = self._split(self.W_k(x_kv), B, seq_kv)
        v = self._split(self.W_v(x_kv), B, seq_kv)

        # == q @ jnp.swapaxes(k, -1, -2)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, x_q.dtype)
        )
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhqk,bhkd->bhqd", weights, v)     # == weights @ v

        return self.W_o(attn.transpose(0, 2, 1, 3).reshape(B, seq_q, -1))
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

ca = MultiHeadCrossAttention(32, 4, rngs=nnx.Rngs(params=0))

dec = jax.random.normal(jax.random.key(1), (1, 4, 32))    # 4 decoder tokens
enc = jax.random.normal(jax.random.key(2), (1, 20, 32))   # 20 encoder tokens

print("decoder:", dec.shape, " encoder:", enc.shape)
print("output :", ca(dec, enc).shape, "(follows the query length)")
print("self-attention as a special case:", ca(dec, dec).shape)
''',
    "tests": [
        {
            "name": "Shapes and required sub-layers",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 4, rngs=nnx.Rngs(params=0))
assert m.d_k == 8, f'd_k should be 8, got {m.d_k}'

for name in ("W_q", "W_k", "W_v", "W_o"):
    assert hasattr(m, name), f'Missing self.{name}'
    assert isinstance(getattr(m, name), nnx.Linear), (
        f'self.{name} must be an nnx.Linear, got {type(getattr(m, name))}'
    )
    assert getattr(m, name).kernel.shape == (32, 32), f'{name} kernel wrong'

x_q = jax.random.normal(jax.random.key(1), (2, 5, 32))
x_kv = jax.random.normal(jax.random.key(2), (2, 9, 32))
assert m(x_q, x_kv).shape == (2, 5, 32), f'{m(x_q, x_kv).shape}'
""",
        },
        {
            "name": "Output length follows the query",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=3))
for seq_q, seq_kv in ((3, 11), (11, 3), (1, 7), (6, 6)):
    q = jax.random.normal(jax.random.key(4), (2, seq_q, 16))
    kv = jax.random.normal(jax.random.key(5), (2, seq_kv, 16))
    out = m(q, kv)
    assert out.shape == (2, seq_q, 16), (
        f'seq_q={seq_q}, seq_kv={seq_kv}: got {out.shape}, expected (2, {seq_q}, 16). '
        'The output must follow the query, not the context.'
    )
""",
        },
        {
            "name": "Reduces to self-attention when both inputs match",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, rngs=nnx.Rngs(params=6))
x = jax.random.normal(jax.random.key(7), (2, 6, 16))

B, seq, H, d_k = 2, 6, 4, 4
def split(t):
    return t.reshape(B, seq, H, d_k).transpose(0, 2, 1, 3)
q, k, v = split(m.W_q(x)), split(m.W_k(x)), split(m.W_v(x))
s = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
a = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), v)
ref = m.W_o(a.transpose(0, 2, 1, 3).reshape(B, seq, -1))

assert jnp.allclose(m(x, x), ref, atol=1e-5), 'cross(x, x) must equal self-attention'
""",
        },
        {
            "name": "K and V come from the context, Q from the query",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=8))
q = jax.random.normal(jax.random.key(9), (1, 4, 16))
kv = jax.random.normal(jax.random.key(10), (1, 6, 16))
base = m(q, kv)

# Changing the context must change the output...
kv2 = kv.at[:, 0].add(50.0)
assert not jnp.allclose(base, m(q, kv2), atol=1e-3), (
    'Changing x_kv did not change the output — K/V are not reading from x_kv'
)
# ...and changing the query must too.
q2 = q.at[:, 0].add(50.0)
assert not jnp.allclose(base, m(q2, kv), atol=1e-3), (
    'Changing x_q did not change the output — Q is not reading from x_q'
)
# A query position depends only on itself among the queries.
q3 = q.at[:, 3].add(50.0)
out3 = m(q3, kv)
assert jnp.allclose(base[:, :3], out3[:, :3], atol=1e-4), (
    'Perturbing query 3 changed earlier query outputs — queries must not '
    'attend to each other in cross-attention'
)
""",
        },
        {
            "name": "Scaled by 1/sqrt(d_k)",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 1, rngs=nnx.Rngs(params=11))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(4)
    lin.bias[...] = jnp.zeros(4)

q = jnp.array([[[1.0, 0.0, 0.0, 0.0]]])
kv = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
out = m(q, kv)

w = jax.nn.softmax(jnp.array([1.0 / jnp.sqrt(4.0), 0.0]))
expected = w[0] * kv[0, 0] + w[1] * kv[0, 1]
assert jnp.allclose(out[0, 0], expected, atol=1e-5), (
    f'Got {out[0, 0]}, expected {expected} — check the 1/sqrt(d_k) scaling'
)
""",
        },
        {
            "name": "Attention weights are a distribution over the context",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 1, rngs=nnx.Rngs(params=12))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(4)
    lin.bias[...] = jnp.zeros(4)

q = jax.random.normal(jax.random.key(13), (1, 3, 4))
kv = jnp.array([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
out = m(q, kv)

assert (out >= -1e-5).all() and (out <= 1.0 + 1e-5).all(), (
    f'Outputs {out} escape the convex hull of V — the softmax is over the '
    'wrong axis (it must normalise across CONTEXT positions)'
)
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, rngs=nnx.Rngs(params=14))
q = jax.random.normal(jax.random.key(15), (2, 4, 16))
kv = jax.random.normal(jax.random.key(16), (2, 7, 16))

grads = nnx.grad(lambda mod: jnp.sum(mod(q, kv) ** 2))(m)
state = nnx.state(grads)
for name in ("W_q", "W_k", "W_v", "W_o"):
    k = state[name]["kernel"]
    val = k[...] if isinstance(k, nnx.Variable) else k
    assert jnp.isfinite(val).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(val).sum()) > 0, f'No gradient reached {name}'

graphdef, st = nnx.split(m)
run = jax.jit(lambda st, a, b: nnx.merge(graphdef, st)(a, b))
assert jnp.allclose(run(st, q, kv), m(q, kv), atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
