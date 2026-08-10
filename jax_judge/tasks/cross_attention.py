"""Multi-head cross-attention — queries from one stream, keys/values from another."""

TASK = {
    "title": "Multi-Head Cross-Attention",
    "category": "Attention & Transformers",
    "order": 4,
    "difficulty": "Medium",
    "function_name": "MultiHeadCrossAttention",
    "hint": (
        "w_q is (d_model, d_model) but w_k and w_v are (d_context, d_model) — the "
        "context stream may be a different width. Split both sides into heads with "
        "reshape(B, T, H, d_head).transpose(0, 2, 1, 3); T_q and T_kv are simply "
        "different, and the score matrix comes out (B, H, T_q, T_kv). Nothing here "
        "is square, so never write a (T, T) mask or a tril. Scale by sqrt(d_head)."
    ),
    "description": r"""
Implement **multi-head cross-attention** as a `flax.nnx.Module`: the queries come
from one sequence, the keys and values from a different one, and the two have
unrelated lengths.

$$Q = x_q W_q,\quad K = x_{kv} W_k,\quad V = x_{kv} W_v,\qquad
\text{out} = \big[\operatorname{softmax}\!\big(\tfrac{Q_hK_h^\top}{\sqrt{d_h}} + M\big)V_h\big]_{h}W_o$$

with $M$ the additive form of the optional mask ($0$ where `mask` is `True`,
$-\infty$ elsewhere) and $[\cdot]_h$ the concatenation over heads.

### Signature
- `MultiHeadCrossAttention(d_model, num_heads, *, d_context=None, rngs: nnx.Rngs)`
- `d_context` defaults to `d_model`
- `__call__(x_q, x_kv, mask=None)` maps
  `(B, T_q, d_model)`, `(B, T_kv, d_context)` `->` `(B, T_q, d_model)`
- `mask` is boolean, broadcastable to `(B, H, T_q, T_kv)`; `True` = attend.
  A key-padding mask is `(B, 1, 1, T_kv)`

### Rules
- Subclass `nnx.Module`; no `nnx.MultiHeadAttention`, no
  `jax.nn.dot_product_attention`
- Parameters named exactly `w_q`, `w_k`, `w_v`, `w_o`, all `nnx.Param`, no biases:
  - `w_q`: `(d_model, d_model)`
  - `w_k`, `w_v`: `(d_context, d_model)`
  - `w_o`: `(d_model, d_model)`
- Initialise each with `jax.random.normal(rngs.params(), shape) / sqrt(fan_in)`
- **No causal mask** — the whole context is visible to every query
- `T_q` and `T_kv` are independent; nothing may assume they match

### Where this actually shows up
- **Encoder–decoder** (the original transformer, T5, Whisper): decoder tokens
  query the encoder's finished representation. $T_q$ is the tokens generated so
  far, $T_{kv}$ the source sentence or the audio frames.
- **Diffusion** (Stable Diffusion and descendants): the UNet's image latents are
  the queries and CLIP text embeddings are the keys/values. This is the *only*
  place the prompt enters the network — swap the K/V stream and you swap the
  prompt. It is also why `d_context` is a separate number: text encoders are 768
  or 1024 wide, UNet blocks are 320/640/1280.
- **Perceiver / Flamingo / DETR**: a small fixed array of learned queries reads
  an enormous input. Flamingo's Perceiver Resampler uses 64 latent queries;
  Perceiver ingests all $224^2 \approx 50\,000$ raw ImageNet pixels as keys;
  DETR decodes with 100 object queries over a convolutional feature map. Cost is
  $O(T_q T_{kv})$, not $O(T_{kv}^2)$ — cross-attention is how you get a
  fixed-cost bottleneck onto arbitrarily large inputs.

### The structural property worth naming in an interview
Cross-attention **cannot mix information across query positions**. Output $i$
depends on $x_q[i]$ and on all of $x_{kv}$, and on no other query. It is a
per-query lookup into a shared memory — batched, not sequential. That is why a
decoder block is always *self-attention, then cross-attention, then MLP*: the
self-attention layer is what lets query positions talk to each other, and
removing it leaves a model that can never build a representation spanning two
output tokens.

The second consequence is that the two streams get **separate lengths and
separate masks**. The causal mask belongs to self-attention; what cross-attention
needs is a key-padding mask over $T_{kv}$, shaped `(B, 1, 1, T_kv)` so it
broadcasts across heads and queries. Writing a `(T, T)` mask here is a bug that
only shows up when the two sequences happen to differ in length.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadCrossAttention(nnx.Module):
    """Queries from x_q, keys/values from x_kv. (B, T_q, d_model) out."""

    def __init__(self, d_model: int, num_heads: int, *,
                 d_context: int | None = None, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x_q, x_kv, mask=None):
        """x_q: (B, T_q, d_model), x_kv: (B, T_kv, d_context)."""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadCrossAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *,
                 d_context: int | None = None, rngs: nnx.Rngs):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")

        d_context = d_model if d_context is None else d_context
        self.d_model = d_model
        self.d_context = d_context
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        qs = 1.0 / jnp.sqrt(d_model)
        cs = 1.0 / jnp.sqrt(d_context)
        # K and V read the CONTEXT stream, so their fan-in is d_context.
        self.w_q = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * qs)
        self.w_k = nnx.Param(jax.random.normal(rngs.params(), (d_context, d_model)) * cs)
        self.w_v = nnx.Param(jax.random.normal(rngs.params(), (d_context, d_model)) * cs)
        self.w_o = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * qs)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, H, T, d_head)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

    def __call__(self, x_q, x_kv, mask=None):
        B, T_q, _ = x_q.shape

        q = self._split_heads(x_q @ self.w_q)      # (B, H, T_q,  d_head)
        k = self._split_heads(x_kv @ self.w_k)     # (B, H, T_kv, d_head)
        v = self._split_heads(x_kv @ self.w_v)     # (B, H, T_kv, d_head)

        # Rectangular by construction: (B, H, T_q, T_kv).
        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(
            jnp.asarray(self.d_head, q.dtype)
        )
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))

        weights = jax.nn.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        return out @ self.w_o
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

# A diffusion-style bridge: 256 image latents query 77 CLIP text tokens.
attn = MultiHeadCrossAttention(d_model=320, num_heads=8, d_context=768,
                               rngs=nnx.Rngs(params=0))
latents = jax.random.normal(jax.random.key(0), (1, 256, 320))
text = jax.random.normal(jax.random.key(1), (1, 77, 768))
print("w_q", attn.w_q.shape, " w_k", attn.w_k.shape, " out", attn(latents, text).shape)

# Query positions never interact: perturbing one query leaves the others alone.
small = MultiHeadCrossAttention(d_model=16, num_heads=2, rngs=nnx.Rngs(params=0))
xq = jax.random.normal(jax.random.key(2), (1, 4, 16))
xkv = jax.random.normal(jax.random.key(3), (1, 6, 16))
base = small(xq, xkv)
poked = small(xq.at[:, 2].set(99.0), xkv)
print("max change at query 0:", float(jnp.abs(base[:, 0] - poked[:, 0]).max()))
print("max change at query 2:", float(jnp.abs(base[:, 2] - poked[:, 2]).max()))

# A key-padding mask hiding the last two context tokens.
pad = jnp.array([[[[True, True, True, True, False, False]]]])   # (1, 1, 1, 6)
print("masked out:", small(xq, xkv, pad).shape)
''',
    "tests": [
        {
            "name": "Shapes with T_q != T_kv",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

attn = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(params=0))
assert isinstance(attn, nnx.Module), 'Must subclass nnx.Module'

out = attn(jax.random.normal(jax.random.key(0), (2, 6, 64)),
           jax.random.normal(jax.random.key(1), (2, 10, 64)))
assert out.shape == (2, 6, 64), (
    f'{out.shape} vs (2, 6, 64) — the output length follows the QUERY sequence'
)

small = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(params=0))
for t_q, t_kv in [(1, 20), (3, 1), (7, 7), (12, 5)]:
    o = small(jax.random.normal(jax.random.key(t_q), (1, t_q, 32)),
              jax.random.normal(jax.random.key(t_kv + 99), (1, t_kv, 32)))
    assert o.shape == (1, t_q, 32), (
        f'T_q={t_q}, T_kv={t_kv} gave {o.shape} — nothing may assume a square '
        'score matrix'
    )
""",
        },
        {
            "name": "Parameter shapes, including d_context != d_model",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

a = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(params=0))
for name, want in [('w_q', (32, 32)), ('w_k', (32, 32)),
                   ('w_v', (32, 32)), ('w_o', (32, 32))]:
    p = getattr(a, name)
    assert isinstance(p, nnx.Param), f'{name} must be an nnx.Param, got {type(p)}'
    assert p.shape == want, f'{name} shape {p.shape} vs {want}'

# The context stream may be a different width (768-d text -> 320-d UNet).
b = {fn}(d_model=32, num_heads=4, d_context=48, rngs=nnx.Rngs(params=0))
assert b.w_q.shape == (32, 32), f'w_q {b.w_q.shape} vs (32, 32)'
assert b.w_k.shape == (48, 32), (
    f'w_k {b.w_k.shape} vs (48, 32) — K reads the context, so its fan-in is d_context'
)
assert b.w_v.shape == (48, 32), f'w_v {b.w_v.shape} vs (48, 32)'
assert b.w_o.shape == (32, 32), f'w_o {b.w_o.shape} vs (32, 32)'

o = b(jax.random.normal(jax.random.key(0), (2, 5, 32)),
      jax.random.normal(jax.random.key(1), (2, 9, 48)))
assert o.shape == (2, 5, 32), f'Mixed-width output {o.shape} vs (2, 5, 32)'

try:
    {fn}(d_model=30, num_heads=4, rngs=nnx.Rngs(params=0))
except Exception:
    pass
else:
    raise AssertionError('d_model=30 with num_heads=4 should raise')
""",
        },
        {
            "name": "Exact match against the reference computation",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T_q, T_kv, D, C, H = 2, 4, 7, 16, 24, 4
dh = D // H
m = {fn}(d_model=D, num_heads=H, d_context=C, rngs=nnx.Rngs(params=0))
x_q = jax.random.normal(jax.random.key(0), (B, T_q, D))
x_kv = jax.random.normal(jax.random.key(1), (B, T_kv, C))
out = m(x_q, x_kv)


def split(y, t):
    return y.reshape(B, t, H, dh).transpose(0, 2, 1, 3)


q = split(x_q @ m.w_q[...], T_q)
k = split(x_kv @ m.w_k[...], T_kv)
v = split(x_kv @ m.w_v[...], T_kv)
scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(float(dh))
assert scores.shape == (B, H, T_q, T_kv)
attn = jax.nn.softmax(scores, axis=-1) @ v
ref = attn.transpose(0, 2, 1, 3).reshape(B, T_q, D) @ m.w_o[...]

assert jnp.allclose(out, ref, atol=1e-5), (
    'Output does not match the reference. Check: Q comes from x_q and K/V from '
    'x_kv (not the other way round), heads split with '
    'reshape(B, T, H, d_head).transpose(0, 2, 1, 3), and the scale is sqrt(d_head).'
)
""",
        },
        {
            "name": "Query positions are independent; all context is visible",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=32, num_heads=2, rngs=nnx.Rngs(params=0))
x_q = jax.random.normal(jax.random.key(0), (1, 4, 32))
x_kv = jax.random.normal(jax.random.key(1), (1, 6, 32))
base = m(x_q, x_kv)

# Cross-attention has no path between query positions.
poked = m(x_q.at[:, 2].set(jax.random.normal(jax.random.key(2), (32,))), x_kv)
for i in (0, 1, 3):
    assert jnp.allclose(base[:, i], poked[:, i], atol=1e-5), (
        f'Changing query 2 changed output {i}. Cross-attention must not mix '
        'information across query positions — that is what self-attention is for.'
    )
assert not jnp.allclose(base[:, 2], poked[:, 2], atol=1e-5), 'Query 2 was ignored'

# No causal mask: every query sees every context token, including the last.
kv2 = x_kv.at[:, -1].set(jax.random.normal(jax.random.key(3), (32,)))
assert not jnp.allclose(base[:, 0], m(x_q, kv2)[:, 0], atol=1e-5), (
    'Changing the LAST context token left query 0 unchanged — you applied a '
    'causal mask, which does not belong in cross-attention'
)
kv3 = x_kv.at[:, 0].set(jax.random.normal(jax.random.key(4), (32,)))
assert not jnp.allclose(base[:, 3], m(x_q, kv3)[:, 3], atol=1e-5), (
    'Changing the FIRST context token left query 3 unchanged'
)
""",
        },
        {
            "name": "Key-padding mask over the context",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T_q, T_kv, D, H = 1, 4, 6, 32, 2
m = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(params=0))
x_q = jax.random.normal(jax.random.key(0), (B, T_q, D))
x_kv = jax.random.normal(jax.random.key(1), (B, T_kv, D))

keep = jnp.array([True, True, True, False, False, False])
mask = keep[None, None, None, :]          # (B, 1, 1, T_kv), broadcasts over heads
masked = m(x_q, x_kv, mask)
assert masked.shape == (B, T_q, D), f'Masked output {masked.shape} vs {(B, T_q, D)}'

# Masking the tail must equal simply not passing it.
truncated = m(x_q, x_kv[:, :3])
assert jnp.allclose(masked, truncated, atol=1e-5), (
    'Masked attention over 6 context tokens (last 3 blocked) must equal '
    'attention over the first 3 alone. If it does not, the mask is being '
    'applied after the softmax instead of to the scores before it.'
)

# Padded slots may hold arbitrary garbage without changing the answer.
junk = x_kv.at[:, 3:].set(1e3)
assert jnp.allclose(m(x_q, junk, mask), masked, atol=1e-4), (
    'Garbage in the padded context slots leaked into the output'
)
assert not jnp.allclose(masked, m(x_q, x_kv), atol=1e-5), 'The mask is being ignored'
""",
        },
        {
            "name": "Gradients reach both streams and all projections",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=16, num_heads=2, d_context=12, rngs=nnx.Rngs(params=0))
x_q = jax.random.normal(jax.random.key(0), (2, 4, 16))
x_kv = jax.random.normal(jax.random.key(1), (2, 6, 12))

grads = nnx.grad(lambda mod: jnp.sum(mod(x_q, x_kv) ** 2))(m)
leaves = [jnp.asarray(l) for l in jax.tree.leaves(grads)]
assert len(leaves) == 4, f'Expected 4 parameter gradients, got {len(leaves)}'
for i, g in enumerate(leaves):
    assert jnp.isfinite(g).all(), f'Non-finite gradient in leaf {i}'
    assert float(jnp.abs(g).sum()) > 0.0, f'Leaf {i} has an all-zero gradient'

# Both input streams must receive gradient.
gq, gkv = nnx.grad(lambda mod, a, b: jnp.sum(mod(a, b) ** 2),
                   argnums=(1, 2))(m, x_q, x_kv)
assert gq.shape == x_q.shape and gkv.shape == x_kv.shape, f'{gq.shape} {gkv.shape}'
assert float(jnp.abs(gq).sum()) > 0.0, 'No gradient flows to x_q'
assert float(jnp.abs(gkv).sum()) > 0.0, (
    'No gradient flows to x_kv — in an encoder-decoder this is the only path '
    'that trains the encoder'
)
""",
        },
        {
            "name": "Composes with nnx.jit and varying lengths",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=16, num_heads=4, rngs=nnx.Rngs(params=0))
x_q = jax.random.normal(jax.random.key(0), (3, 5, 16))
x_kv = jax.random.normal(jax.random.key(1), (3, 8, 16))
eager = m(x_q, x_kv)


@nnx.jit
def fwd(mod, a, b):
    return mod(a, b)


assert jnp.allclose(fwd(m, x_q, x_kv), eager, atol=1e-5), 'nnx.jit changed the result'

# Batch items stay independent.
assert jnp.allclose(m(x_q[1:2], x_kv[1:2])[0], eager[1], atol=1e-5), (
    'Running example 1 alone differs from the batched call'
)

# Growing the query length one step at a time, as a decoder does.
for t in (1, 2, 5, 9):
    o = m(jax.random.normal(jax.random.key(t), (1, t, 16)), x_kv[:1])
    assert o.shape == (1, t, 16), f'T_q={t} gave {o.shape}'
""",
        },
    ],
}
