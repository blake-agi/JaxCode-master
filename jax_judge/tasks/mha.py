"""Multi-head attention assembled from nnx.Linear — heads are a reshape."""

TASK = {
    "title": "Multi-Head Attention",
    "category": "Attention & Transformers",
    "number": "06",
    "difficulty": "Hard",
    "function_name": "MultiHeadAttention",
    "hint": (
        "Four nnx.Linear(d_model, d_model) layers. Project, then reshape "
        "(B, S, d_model) -> (B, S, H, d_k) and transpose to (B, H, S, d_k) so "
        "the heads become a batch axis. Q and K/V may have DIFFERENT sequence "
        "lengths — read S_q off Q and S_k off K — which is what makes the same "
        "class work for cross-attention. Merge the heads back with the inverse "
        "transpose-then-reshape before the output projection."
    ),
    "description": r"""
Implement **multi-head attention**.

$$\text{head}_i = \text{softmax}\!\left(\frac{Q W^Q_i (K W^K_i)^\top}{\sqrt{d_k}}\right)V W^V_i$$

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1..\text{head}_H)\,W^O$$

### Signature
```python
class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs): ...
    def __call__(self, Q, K, V): ...
```

### Requirements
- Use `nnx.Linear(d_model, d_model)` for `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`
- `self.d_k = d_model // num_heads`
- `Q` is `(B, seq_q, d_model)`; `K` and `V` are `(B, seq_k, d_model)`
- Must support **cross-attention** — `seq_q != seq_k`
- Do **not** use `nnx.MultiHeadAttention`

`nnx.Linear` is an allowed building block: you are implementing attention, not
the projection. It also brings its own initialization.

### Heads are a reshape, not a loop
The whole trick is that $H$ separate attention computations are one batched
computation. `(B, S, d_model)` reshapes to `(B, S, H, d_k)` and transposes to
`(B, H, S, d_k)`, after which the head axis is just another batch axis and the
same einsum handles all of them. Nothing is looped, and the parameter count is
identical to single-head attention with the same `d_model` — you are
partitioning the projection, not adding to it.

### Why Q, K and V are separate arguments
Passing one `x` would only ever give you self-attention. Taking three inputs
means the identical class does self-attention (`mha(x, x, x)`) and
cross-attention (`mha(decoder, encoder, encoder)`), which is exactly how an
encoder-decoder transformer reuses one implementation.

### The trap
It is tempting to read a single `S` off `Q` and use it for `K` too. That works
for every self-attention test and then fails the moment the sequence lengths
differ — so the score matrix is `(S_q, S_k)`, not square.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    """Multi-head attention over (B, S, d_model)."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, Q, K, V):
        """Q: (B, seq_q, d_model), K/V: (B, seq_k, d_model) -> (B, seq_q, d_model)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def _split(self, t, B, S):
        # (B, S, d_model) -> (B, H, S, d_k): heads become a batch axis.
        return t.reshape(B, S, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]        # NOT S_q — this is what allows cross-attention

        q = self._split(self.W_q(Q), B, S_q)
        k = self._split(self.W_k(K), B, S_k)
        v = self._split(self.W_v(V), B, S_k)

        # == q @ jnp.swapaxes(k, -1, -2)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, Q.dtype)
        )
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhqk,bhkd->bhqd", weights, v)      # == weights @ v

        out = attn.transpose(0, 2, 1, 3).reshape(B, S_q, -1)
        return self.W_o(out)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

mha = MultiHeadAttention(32, 4, rngs=nnx.Rngs(params=0))

x = jax.random.normal(jax.random.key(1), (2, 6, 32))
print("self-attention :", mha(x, x, x).shape)

ctx = jax.random.normal(jax.random.key(2), (2, 10, 32))
print("cross-attention:", mha(x, ctx, ctx).shape, "(query length wins)")
''',
    "tests": [
        {
            "name": "Shapes and required sub-layers",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 4, rngs=nnx.Rngs(params=0))
assert m.d_k == 8, f'd_k should be d_model // num_heads = 8, got {m.d_k}'
assert m.num_heads == 4, f'num_heads {m.num_heads}'

for name in ("W_q", "W_k", "W_v", "W_o"):
    assert hasattr(m, name), f'Missing self.{name}'
    assert isinstance(getattr(m, name), nnx.Linear), (
        f'self.{name} must be an nnx.Linear, got {type(getattr(m, name))}'
    )
    assert getattr(m, name).kernel.shape == (32, 32), (
        f'{name} kernel {getattr(m, name).kernel.shape} vs (32, 32)'
    )

x = jax.random.normal(jax.random.key(1), (2, 6, 32))
assert m(x, x, x).shape == (2, 6, 32), f'{m(x, x, x).shape}'
""",
        },
        {
            "name": "Cross-attention with different sequence lengths",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=2))
q = jax.random.normal(jax.random.key(3), (2, 5, 16))
kv = jax.random.normal(jax.random.key(4), (2, 9, 16))

out = m(q, kv, kv)
assert out.shape == (2, 5, 16), (
    f'Output should follow the QUERY length: got {out.shape}, expected (2, 5, 16). '
    'Reading one sequence length off Q and reusing it for K breaks here.'
)
""",
        },
        {
            "name": "Matches the reference computation",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, rngs=nnx.Rngs(params=5))
Q = jax.random.normal(jax.random.key(6), (2, 5, 16))
K = jax.random.normal(jax.random.key(7), (2, 7, 16))
V = jax.random.normal(jax.random.key(8), (2, 7, 16))

B, S_q, S_k, H, d_k = 2, 5, 7, 4, 4
def split(t, S):
    return t.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
q, k, v = split(m.W_q(Q), S_q), split(m.W_k(K), S_k), split(m.W_v(V), S_k)
scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(d_k, Q.dtype))
attn = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)
ref = m.W_o(attn.transpose(0, 2, 1, 3).reshape(B, S_q, -1))

assert jnp.allclose(m(Q, K, V), ref, atol=1e-5), 'Output does not match the reference'
""",
        },
        {
            "name": "Scaled by 1/sqrt(d_k), not d_k",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 1, rngs=nnx.Rngs(params=9))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(4)
    lin.bias[...] = jnp.zeros(4)

# One query, two keys with dot products 1 and 0 -> softmax([1/2, 0]) with d_k=4.
Q = jnp.array([[[1.0, 0.0, 0.0, 0.0]]])
K = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
V = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
out = m(Q, K, V)

w = jax.nn.softmax(jnp.array([1.0 / jnp.sqrt(4.0), 0.0]))
expected = w[0] * V[0, 0] + w[1] * V[0, 1]
assert jnp.allclose(out[0, 0], expected, atol=1e-5), (
    f'Got {out[0, 0]}, expected {expected}. Dividing by d_k instead of sqrt(d_k) '
    f'would give {jax.nn.softmax(jnp.array([0.25, 0.0]))}.'
)
""",
        },
        {
            "name": "Heads are independent, not mixed",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# With identity projections, running H heads over d_model must equal running
# each d_k slice through single-head attention separately.
m = {fn}(8, 2, rngs=nnx.Rngs(params=10))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(8)
    lin.bias[...] = jnp.zeros(8)

x = jax.random.normal(jax.random.key(11), (1, 4, 8))
out = m(x, x, x)

for h in range(2):
    sl = slice(h * 4, (h + 1) * 4)
    q = k = v = x[..., sl]
    s = jnp.einsum("bqd,bkd->bqk", q, k) / jnp.sqrt(4.0)
    ref = jnp.einsum("bqk,bkd->bqd", jax.nn.softmax(s, axis=-1), v)
    assert jnp.allclose(out[..., sl], ref, atol=1e-5), (
        f'Head {h} does not match an independent single-head computation — '
        'the reshape/transpose is mixing head contents'
    )
""",
        },
        {
            "name": "Rows are probability distributions over keys",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# Output must lie in the convex hull of V rows when W_o and W_v are identity.
m = {fn}(4, 1, rngs=nnx.Rngs(params=12))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(4)
    lin.bias[...] = jnp.zeros(4)

Q = jax.random.normal(jax.random.key(13), (1, 3, 4))
V = jnp.array([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
K = jax.random.normal(jax.random.key(14), (1, 2, 4))
out = m(Q, K, V)

assert (out >= -1e-5).all() and (out <= 1.0 + 1e-5).all(), (
    f'Outputs {out} escape the convex hull of V, so the attention weights do '
    'not form a distribution — check the softmax axis'
)
""",
        },
        {
            "name": "Gradients, jit and head divisibility",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, rngs=nnx.Rngs(params=15))
x = jax.random.normal(jax.random.key(16), (2, 5, 16))

grads = nnx.grad(lambda mod: jnp.sum(mod(x, x, x) ** 2))(m)
state = nnx.state(grads)
for name in ("W_q", "W_k", "W_v", "W_o"):
    k = state[name]["kernel"]
    val = k[...] if isinstance(k, nnx.Variable) else k
    assert jnp.isfinite(val).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(val).sum()) > 0, f'No gradient reached {name}'

graphdef, st = nnx.split(m)
run = jax.jit(lambda st, a: nnx.merge(graphdef, st)(a, a, a))
assert jnp.allclose(run(st, x), m(x, x, x), atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
