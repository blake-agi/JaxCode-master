"""Grouped-Query Attention as an nnx.Module — fewer KV heads than query heads."""

TASK = {
    "title": "Grouped Query Attention",
    "category": "Attention & Transformers",
    "order": 5,
    "difficulty": "Hard",
    "function_name": "GroupQueryAttention",
    "hint": (
        "Only w_q and w_o are square. w_k and w_v project d_model -> "
        "num_kv_heads * head_dim, so after reshaping you get K/V of shape "
        "(B, num_kv_heads, T, head_dim) while Q is (B, num_heads, T, head_dim). "
        "Bridge the gap with jnp.repeat(k, n_rep, axis=1) where "
        "n_rep = num_heads // num_kv_heads — jnp.repeat interleaves "
        "(0,0,1,1), which is what you want; jnp.tile would give (0,1,0,1) and "
        "silently pair every query head with the wrong KV head."
    ),
    "description": r"""
Implement **Grouped-Query Attention** (GQA) as an `nnx.Module`: multi-head
attention where the key/value heads are *fewer* than the query heads and each
KV head is shared by a contiguous group of query heads.

With $H$ query heads, $H_{kv}$ key/value heads and $r = H / H_{kv}$:

$$\text{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_{\lfloor h/r \rfloor}^\top}{\sqrt{d_h}}\right) V_{\lfloor h/r \rfloor}$$

$$\text{out} = \mathrm{concat}(\text{head}_0, \dots, \text{head}_{H-1})\, W_o$$

$H_{kv} = H$ is ordinary MHA; $H_{kv} = 1$ is Multi-Query Attention.

### Rules
- Subclass `nnx.Module`. Do **not** use `nnx.MultiHeadAttention`,
  `nnx.dot_product_attention` or `jax.nn.dot_product_attention`
- Signature: `GroupQueryAttention(d_model, num_heads, num_kv_heads, *, rngs)`
- Parameters — all `nnx.Param`, all `(din, dout)`, **no biases**:
  - `self.w_q`, `self.w_o`: `(d_model, d_model)`
  - `self.w_k`, `self.w_v`: `(d_model, num_kv_heads * head_dim)`
- `head_dim = d_model // num_heads`; `num_heads % num_kv_heads == 0`
- `__call__(x)` maps `(B, T, d_model) -> (B, T, d_model)`, bidirectional
  self-attention (no causal mask)
- Split heads as `(B, T, H, head_dim) -> transpose -> (B, H, T, head_dim)`,
  i.e. head `h` owns channels `[h*head_dim : (h+1)*head_dim]`
- Scale scores by `1 / sqrt(head_dim)` — **not** `1 / sqrt(d_model)`
- Query head `h` reads KV head `h // r`, so the KV heads are repeated
  **interleaved** `(0,0,1,1)`, not tiled `(0,1,0,1)`

### Why GQA exists: the KV cache, not the FLOPs
GQA saves almost no compute. After the repeat, the score and value matmuls are
FLOP-for-FLOP identical to MHA; only the two KV projection matrices shrink, from
$d_{model} \times d_{model}$ to $d_{model} \times H_{kv}d_h$. The win is
entirely **decode-time memory**.

At generation time every past token's K and V must be kept. Per token per layer
the cache costs $2 \cdot H_{kv} \cdot d_h \cdot \text{bytes}$. For a 70B-class
model (80 layers, $H = 64$, $d_h = 128$, fp16) the whole-model totals are:

| | per token, all layers | 4k context | 4k ctx, batch 8 |
|---|---|---|---|
| MHA ($H_{kv}=64$) | 2.6 MB | 10.7 GB | 86 GB |
| GQA ($H_{kv}=8$) | 328 KB | 1.3 GB | 10.7 GB |
| MQA ($H_{kv}=1$) | 41 KB | 168 MB | 1.3 GB |

Autoregressive decoding is memory-**bandwidth** bound: producing one token means
streaming the weights *and* the entire cache out of HBM. Shrinking the cache 8x
shrinks the cache half of that read 8x. How much of a speedup that is depends on
which half dominates — at batch 1 and short context the weights do, and GQA buys
little; at long context or large batch the cache dominates and the gain
approaches the full 8x. The memory saving is unconditional, and it is often the
reason the context fits at all.

MQA takes this to the limit but measurably degrades quality; GQA is the
interpolation that holds roughly MHA quality — and it is cheap to *uptrain* an
existing MHA checkpoint into GQA by mean-pooling the KV heads within each group,
then continuing training for a small fraction of the original budget.

### The trap
`jnp.repeat(k, r, axis=1)` and `jnp.tile(k, (1, r, 1, 1))` produce arrays of
identical shape and different content. Pick the wrong one and every test that
only checks shapes passes, the model trains, and it silently pairs query head 1
with the KV head meant for query head `r`. When you later load real GQA weights,
the output is garbage with a perfectly valid shape.

A production kernel skips the `repeat` entirely: reshape Q to
`(B, H_kv, r, T, d_h)` and `einsum` against the un-repeated K, so the shared KV
is read once instead of materialised `r` times. Same math, less memory traffic.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class GroupQueryAttention(nnx.Module):
    """Multi-head attention with num_kv_heads < num_heads shared KV heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        pass  # Replace this

    def __call__(self, x):
        """(B, T, d_model) -> (B, T, d_model)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class GroupQueryAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        assert d_model % num_heads == 0, "d_model must divide into num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads      # query heads per KV head

        kv_dim = num_kv_heads * self.head_dim
        scale = 1.0 / jnp.sqrt(jnp.float32(d_model))

        # (din, dout) layout — x @ w. Only Q and O are square.
        self.w_q = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)
        self.w_k = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * scale)
        self.w_v = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * scale)
        self.w_o = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)

    def _split(self, x, n_heads):
        B, T, _ = x.shape
        return x.reshape(B, T, n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x):
        B, T, _ = x.shape

        q = self._split(x @ self.w_q[...], self.num_heads)      # (B, H,   T, Dh)
        k = self._split(x @ self.w_k[...], self.num_kv_heads)   # (B, Hkv, T, Dh)
        v = self._split(x @ self.w_v[...], self.num_kv_heads)   # (B, Hkv, T, Dh)

        # Interleaved repeat: KV head g serves query heads [g*r, (g+1)*r).
        # jnp.repeat gives (0,0,1,1); jnp.tile would give (0,1,0,1) — wrong.
        k = jnp.repeat(k, self.n_rep, axis=1)                    # (B, H, T, Dh)
        v = jnp.repeat(v, self.n_rep, axis=1)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(jnp.float32(self.head_dim))
        out = jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(scores, axis=-1), v)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return out @ self.w_o[...]
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

D, H = 512, 8
x = jax.random.normal(jax.random.key(0), (1, 16, D))

for kv in (8, 4, 1):
    m = GroupQueryAttention(D, H, kv, rngs=nnx.Rngs(params=0))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(m, nnx.Param)))
    cache_per_token = 2 * kv * (D // H) * 2          # K and V, fp16 bytes
    print(
        f"kv_heads={kv:>2}  w_k={str(m.w_k.shape):>12}  "
        f"params={n_params:>8,}  KV cache/token/layer={cache_per_token:>6} B  "
        f"out={m(x).shape}"
    )
print("\\n-> output shape never changes; the cache shrinks linearly in kv_heads")
''',
    "tests": [
        {
            "name": "Output and projection shapes",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 8, 2, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 6, 32))
out = m(x)

assert out.shape == (2, 6, 32), f'Output shape {out.shape} vs (2, 6, 32)'
assert jnp.isfinite(out).all(), 'Non-finite values in the output'

head_dim = 32 // 8
assert m.w_q.shape == (32, 32), f'w_q should be (32, 32), got {m.w_q.shape}'
assert m.w_o.shape == (32, 32), f'w_o should be (32, 32), got {m.w_o.shape}'
assert m.w_k.shape == (32, 2 * head_dim), (
    f'w_k should be (d_model, num_kv_heads*head_dim) = (32, {2*head_dim}), '
    f'got {m.w_k.shape} — the KV projections must be NARROWER than w_q'
)
assert m.w_v.shape == (32, 2 * head_dim), f'w_v should be (32, {2*head_dim}), got {m.w_v.shape}'
""",
        },
        {
            "name": "Params are nnx.Param and the KV projections are smaller",
            "code": """
import jax
from flax import nnx

m = {fn}(256, 8, 2, rngs=nnx.Rngs(params=0))
for name in ('w_q', 'w_k', 'w_v', 'w_o'):
    p = getattr(m, name)
    assert isinstance(p, nnx.Param), f'{name} must be an nnx.Param, got {type(p)}'

leaves = jax.tree.leaves(nnx.state(m, nnx.Param))
assert len(leaves) == 4, f'Expected exactly 4 parameter arrays, found {len(leaves)}'

gqa_params = sum(l.size for l in leaves)
mha = {fn}(256, 8, 8, rngs=nnx.Rngs(params=0))
mha_params = sum(l.size for l in jax.tree.leaves(nnx.state(mha, nnx.Param)))
assert gqa_params < mha_params, (
    f'GQA ({gqa_params}) should have fewer params than MHA ({mha_params}) — '
    'w_k/w_v must shrink with num_kv_heads'
)
""",
        },
        {
            "name": "Matches an explicit reference built from the same weights",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T, D, H, KV = 2, 5, 32, 4, 2
Dh, r = D // H, H // KV

m = {fn}(D, H, KV, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (B, T, D))
out = m(x)

def split(y, n):
    return y.reshape(B, T, n, Dh).transpose(0, 2, 1, 3)

q = split(x @ m.w_q[...], H)
k = jnp.repeat(split(x @ m.w_k[...], KV), r, axis=1)
v = jnp.repeat(split(x @ m.w_v[...], KV), r, axis=1)
s = jnp.einsum('bhtd,bhsd->bhts', q, k) / jnp.sqrt(jnp.float32(Dh))
o = jnp.einsum('bhts,bhsd->bhtd', jax.nn.softmax(s, axis=-1), v)
ref = o.transpose(0, 2, 1, 3).reshape(B, T, D) @ m.w_o[...]

assert jnp.allclose(out, ref, atol=1e-5), (
    f'Max diff {float(jnp.abs(out - ref).max()):.2e}. Check the head split '
    '(B,T,H,Dh)->transpose(0,2,1,3), the 1/sqrt(head_dim) scale, and that the '
    'output projection is applied after re-merging the heads.'
)

# The 1/sqrt(head_dim) scale, not 1/sqrt(d_model).
wrong = jnp.einsum('bhtd,bhsd->bhts', q, k) / jnp.sqrt(jnp.float32(D))
wrong_ref = jnp.einsum(
    'bhts,bhsd->bhtd', jax.nn.softmax(wrong, axis=-1), v
).transpose(0, 2, 1, 3).reshape(B, T, D) @ m.w_o[...]
assert not jnp.allclose(out, wrong_ref, atol=1e-5), 'Scores look scaled by 1/sqrt(d_model)'
""",
        },
        {
            "name": "KV heads are repeated interleaved, not tiled",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T, D, H, KV = 1, 6, 32, 4, 2
Dh, r = D // H, H // KV

m = {fn}(D, H, KV, rngs=nnx.Rngs(params=1))
x = jax.random.normal(jax.random.key(3), (B, T, D))
out = m(x)

def split(y, n):
    return y.reshape(B, T, n, Dh).transpose(0, 2, 1, 3)

q = split(x @ m.w_q[...], H)
k0 = split(x @ m.w_k[...], KV)
v0 = split(x @ m.w_v[...], KV)

def attend(k, v):
    s = jnp.einsum('bhtd,bhsd->bhts', q, k) / jnp.sqrt(jnp.float32(Dh))
    o = jnp.einsum('bhts,bhsd->bhtd', jax.nn.softmax(s, axis=-1), v)
    return o.transpose(0, 2, 1, 3).reshape(B, T, D) @ m.w_o[...]

interleaved = attend(jnp.repeat(k0, r, axis=1), jnp.repeat(v0, r, axis=1))   # 0,0,1,1
tiled = attend(jnp.tile(k0, (1, r, 1, 1)), jnp.tile(v0, (1, r, 1, 1)))       # 0,1,0,1

assert not jnp.allclose(interleaved, tiled, atol=1e-5), 'Test setup is degenerate'
assert jnp.allclose(out, interleaved, atol=1e-5), (
    'Query head h must use KV head h // (num_heads // num_kv_heads). '
    'Use jnp.repeat (gives 0,0,1,1), not jnp.tile (gives 0,1,0,1).'
)
""",
        },
        {
            "name": "Degenerates to MHA and to MQA at the extremes",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# num_kv_heads == num_heads: the KV projections become square (plain MHA).
mha = {fn}(16, 4, 4, rngs=nnx.Rngs(params=0))
assert mha.w_k.shape == (16, 16), f'With kv_heads == heads, w_k should be (16,16), got {mha.w_k.shape}'
assert mha(jax.random.normal(jax.random.key(0), (1, 4, 16))).shape == (1, 4, 16)

# num_kv_heads == 1: multi-query attention, one shared K/V for every head.
mqa = {fn}(16, 4, 1, rngs=nnx.Rngs(params=0))
assert mqa.w_v.shape == (16, 4), f'MQA w_v should be (16, 4), got {mqa.w_v.shape}'

x = jax.random.normal(jax.random.key(1), (2, 7, 16))
out = mqa(x)
assert out.shape == (2, 7, 16), f'MQA output shape {out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite MQA output'

# Softmax rows sum to 1, so identical value vectors must pass straight through
# (before w_o): feed x whose V rows are all equal by zeroing w_v and using bias-free
# projections -> the attention output is exactly 0 regardless of the scores.
mqa.w_v[...] = jnp.zeros_like(mqa.w_v[...])
assert jnp.allclose(mqa(x), 0.0, atol=1e-6), (
    'With w_v = 0 every value vector is 0, so the output must be 0 — '
    'a non-zero result means a bias is being added somewhere'
)
""",
        },
        {
            "name": "Gradients reach every projection",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, 2, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 5, 16))

def loss(mod, inp):
    return jnp.sum(mod(inp) ** 2)

grads = nnx.grad(loss)(m, x)
leaves = jax.tree.leaves(grads)
assert len(leaves) == 4, f'Expected 4 parameter gradients, got {len(leaves)}'
for g in leaves:
    assert jnp.isfinite(g).all(), 'Non-finite parameter gradient'
    assert float(jnp.abs(g).sum()) > 0, (
        'A projection received a zero gradient — every one of w_q/w_k/w_v/w_o '
        'must sit on the forward path'
    )

# Gradient w.r.t. the input needs argnums=1 (argnums=0 is the module state).
gx = nnx.grad(loss, argnums=1)(m, x)
assert gx.shape == x.shape, f'Input gradient shape {gx.shape} vs {x.shape}'
assert jnp.isfinite(gx).all() and float(jnp.abs(gx).sum()) > 0, 'Bad input gradient'
""",
        },
        {
            "name": "Jits and handles varying batch/sequence lengths",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 8, 2, rngs=nnx.Rngs(params=0))

@nnx.jit
def fwd(mod, inp):
    return mod(inp)

for B, T in [(1, 1), (3, 4), (2, 17)]:
    x = jax.random.normal(jax.random.key(B * 10 + T), (B, T, 32))
    eager, jitted = m(x), fwd(m, x)
    assert eager.shape == (B, T, 32), f'Shape {eager.shape} for B={B}, T={T}'
    assert jnp.allclose(eager, jitted, atol=1e-5), f'jit disagrees with eager at B={B}, T={T}'

# T = 1 is the decode step: attention over a single position returns V itself,
# so softmax must be taken over the KEY axis (-1), not the query axis.
one = m(jax.random.normal(jax.random.key(9), (1, 1, 32)))
assert jnp.isfinite(one).all() and one.shape == (1, 1, 32), f'T=1 failed: {one.shape}'
""",
        },
    ],
}
