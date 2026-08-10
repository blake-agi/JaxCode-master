"""Multi-head self-attention as an nnx.Module — the reshape/transpose dance."""

TASK = {
    "title": "Multi-Head Attention",
    "category": "Attention & Transformers",
    "number": "06",
    "difficulty": "Hard",
    "function_name": "MultiHeadAttention",
    "hint": (
        "Four (d_model, d_model) nnx.Param matrices: w_q, w_k, w_v, w_o. "
        "Split heads with x.reshape(B, T, H, d_head).transpose(0, 2, 1, 3) — "
        "reshape FIRST, then transpose; going straight to reshape(B, H, T, d_head) "
        "gives the right shape and the wrong tokens. Scale by sqrt(d_head), not "
        "sqrt(d_model). Merge back with "
        "out.transpose(0, 2, 1, 3).reshape(B, T, d_model) before the w_o projection."
    ),
    "description": r"""
Implement **multi-head self-attention** as a `flax.nnx.Module`.

$$\text{head}_h = \operatorname{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right)V_h,
\qquad
\text{out} = \big[\text{head}_1 \,\|\, \cdots \,\|\, \text{head}_H\big]\,W_o$$

with $Q = xW_q$, $K = xW_k$, $V = xW_v$ and $d_h = d_{model}/H$.

### Signature
- `MultiHeadAttention(d_model, num_heads, *, rngs: nnx.Rngs)`
- `__call__(x, mask=None)` maps `(B, T, d_model) -> (B, T, d_model)`
- `mask` is boolean, broadcastable to `(B, H, T, T)`; `True` = attend

### Rules
- Subclass `nnx.Module`; no `nnx.MultiHeadAttention`, no `jax.nn.dot_product_attention`
- Parameters named exactly `w_q`, `w_k`, `w_v`, `w_o`, each an `nnx.Param` of
  shape `(d_model, d_model)` — JAX layout is `(din, dout)` and you compute `x @ W`
- No biases on the projections
- Draw each matrix with `jax.random.normal(rngs.params(), ...) / sqrt(d_model)`
- Scale scores by $\sqrt{d_h}$, the **per-head** dim
- Raise or assert if `d_model % num_heads != 0`

### The reshape trap
The projected tensor is `(B, T, d_model)` and you want `(B, H, T, d_h)`. There is
exactly one correct route:

```python
x.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)     # correct
x.reshape(B, H, T, d_head)                            # silently wrong
```

`reshape` reads memory in row-major order, so the last axis `d_model` splits
naturally into `(H, d_h)` — head $h$ owns the contiguous slice
`[h*d_h : (h+1)*d_h]` of the feature vector. Reshaping straight to `(B, H, T, d_h)`
instead slices along the **token** axis, so "head 0" ends up holding the first
`T/H` tokens' full feature vectors. The shapes typecheck, the loss goes down a
bit, and the model is quietly broken. Coming back the other way you must
`transpose` before `reshape` for the same reason.

### Why heads are free
Count parameters: $4d_{model}^2$, with no $H$ anywhere. Count FLOPs for the score
matrix: $B \cdot H \cdot T^2 \cdot d_h = B \cdot T^2 \cdot d_{model}$ — again no
$H$. Splitting into heads is a *reinterpretation* of the same matrix, not extra
work. What you buy is $H$ independent similarity subspaces: one head can track
syntactic agreement while another tracks positional offset, instead of a single
softmax being forced to average those signals into one distribution. The cost is
that each head sees a $d_h$-dimensional space, so very large $H$ makes each head
too narrow to represent anything. That tension is why $d_h$ does *not* grow with
model size: it is pinned at 64 or 128 in most production models (256 at the
outside), and $H$ is what scales with $d_{model}$ instead.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention: (B, T, d_model) -> (B, T, d_model)."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x, mask=None):
        """x: (B, T, d_model), mask: broadcastable to (B, H, T, T) or None."""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        scale = 1.0 / jnp.sqrt(d_model)
        self.w_q = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)
        self.w_k = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)
        self.w_v = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)
        self.w_o = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_model)) * scale)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, H, T, d_head)"""
        B, T, _ = x.shape
        # reshape splits the LAST axis into (H, d_head), then transpose moves
        # the head axis in front of the token axis.
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(B, H, T, d_head) -> (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

    def __call__(self, x, mask=None):
        q = self._split_heads(x @ self.w_q)
        k = self._split_heads(x @ self.w_k)
        v = self._split_heads(x @ self.w_v)

        # (B, H, T, d_head) @ (B, H, d_head, T) -> (B, H, T, T)
        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(
            jnp.asarray(self.d_head, q.dtype)
        )
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))

        weights = jax.nn.softmax(scores, axis=-1)
        out = self._merge_heads(weights @ v)
        return out @ self.w_o
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

mha = MultiHeadAttention(d_model=64, num_heads=8, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 10, 64))
print("out:", mha(x).shape)

# Parameter count does not depend on the number of heads.
for h in (1, 2, 4, 8, 16):
    m = MultiHeadAttention(d_model=64, num_heads=h, rngs=nnx.Rngs(params=0))
    n = sum(int(jnp.size(leaf)) for leaf in jax.tree.leaves(nnx.state(m, nnx.Param)))
    print(f"H={h:2d}  d_head={64 // h:3d}  params={n}")

# A causal mask is just a boolean array broadcast over the head axis.
T = 10
causal = jnp.tril(jnp.ones((T, T), dtype=bool))
print("masked out:", mha(x, causal).shape)
''',
    "tests": [
        {
            "name": "Shapes and parameter layout",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(params=0))
for name in ('w_q', 'w_k', 'w_v', 'w_o'):
    p = getattr(m, name)
    assert isinstance(p, nnx.Param), f'{name} must be an nnx.Param, got {type(p)}'
    assert p.shape == (32, 32), (
        f'{name} shape {p.shape} vs (32, 32) — JAX Linear layout is '
        '(din, dout) with x @ W'
    )

x = jax.random.normal(jax.random.key(0), (2, 6, 32))
out = m(x)
assert out.shape == (2, 6, 32), f'Output shape {out.shape} vs (2, 6, 32)'
assert jnp.isfinite(out).all(), 'Non-finite output'

# Indivisible d_model must be rejected rather than silently truncating.
try:
    {fn}(d_model=30, num_heads=4, rngs=nnx.Rngs(params=0))
except Exception:
    pass
else:
    raise AssertionError('d_model=30, num_heads=4 should raise: 30 is not divisible by 4')
""",
        },
        {
            "name": "Exact match against the reference computation",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T, D, H = 2, 5, 16, 4
dh = D // H
m = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (B, T, D))
out = m(x)


def split(y):
    return y.reshape(B, T, H, dh).transpose(0, 2, 1, 3)


q = split(x @ m.w_q[...])
k = split(x @ m.w_k[...])
v = split(x @ m.w_v[...])
scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(float(dh))
attn = jax.nn.softmax(scores, axis=-1) @ v
ref = attn.transpose(0, 2, 1, 3).reshape(B, T, D) @ m.w_o[...]

assert jnp.allclose(out, ref, atol=1e-5), (
    'Output does not match the reference. Check three things: heads are split '
    'with reshape(B, T, H, d_head).transpose(0, 2, 1, 3), the scale is '
    'sqrt(d_head) and not sqrt(d_model), and w_o is applied after merging.'
)

# sqrt(d_model) instead of sqrt(d_head) is the classic near-miss.
wrong = jax.nn.softmax(scores * jnp.sqrt(float(dh)) / jnp.sqrt(float(D)), axis=-1) @ v
wrong = wrong.transpose(0, 2, 1, 3).reshape(B, T, D) @ m.w_o[...]
assert not jnp.allclose(out, wrong, atol=1e-6), 'You are scaling by sqrt(d_model)'
""",
        },
        {
            "name": "Head splitting slices features, not tokens",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T, D, H = 1, 4, 8, 2
dh = D // H
m = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(params=0))

# Make every projection the identity so the module reduces to pure attention
# on x itself, and w_o is the identity too.
eye = jnp.eye(D)
m.w_q[...] = eye
m.w_k[...] = eye
m.w_v[...] = eye
m.w_o[...] = eye

x = jax.random.normal(jax.random.key(0), (B, T, D))
out = m(x)

# Head h must attend using ONLY feature columns [h*dh : (h+1)*dh], and it must
# write its result back into exactly those columns.
for h in range(H):
    sl = slice(h * dh, (h + 1) * dh)
    qh, kh, vh = x[:, :, sl], x[:, :, sl], x[:, :, sl]
    s = (qh @ jnp.swapaxes(kh, -1, -2)) / jnp.sqrt(float(dh))
    ref_h = jax.nn.softmax(s, axis=-1) @ vh
    assert jnp.allclose(out[:, :, sl], ref_h, atol=1e-5), (
        f'Head {h} is wrong. Heads partition the FEATURE axis: reshape to '
        '(B, T, H, d_head) and then transpose. reshape(B, H, T, d_head) has '
        'the right shape but hands each head a block of tokens instead.'
    )
""",
        },
        {
            "name": "Parameter count is independent of num_heads",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

counts = {}
for h in (1, 2, 4, 8):
    m = {fn}(d_model=32, num_heads=h, rngs=nnx.Rngs(params=0))
    leaves = jax.tree.leaves(nnx.state(m, nnx.Param))
    counts[h] = sum(int(jnp.size(jnp.asarray(l))) for l in leaves)

assert len(set(counts.values())) == 1, (
    f'Parameter count changed with the head count: {counts}. Heads reinterpret '
    'the same (d_model, d_model) matrices; they must not add parameters.'
)
assert counts[1] == 4 * 32 * 32, (
    f'Expected 4 * d_model^2 = {4 * 32 * 32} parameters, got {counts[1]} — '
    'the projections must have no bias'
)

# H=1 must reduce to plain single-head attention on the full d_model.
m1 = {fn}(d_model=16, num_heads=1, rngs=nnx.Rngs(params=1))
x = jax.random.normal(jax.random.key(0), (1, 4, 16))
q = x @ m1.w_q[...]
k = x @ m1.w_k[...]
v = x @ m1.w_v[...]
s = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(16.0)
ref = (jax.nn.softmax(s, axis=-1) @ v) @ m1.w_o[...]
assert jnp.allclose(m1(x), ref, atol=1e-5), 'num_heads=1 must be plain attention'
""",
        },
        {
            "name": "Mask is honoured and broadcasts over heads",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, T, D, H = 2, 6, 16, 4
m = {fn}(d_model=D, num_heads=H, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (B, T, D))

causal = jnp.tril(jnp.ones((T, T), dtype=bool))     # (T, T), broadcasts to (B, H, T, T)
out = m(x, causal)
assert out.shape == (B, T, D), f'Masked output shape {out.shape} vs {(B, T, D)}'

# Under a causal mask, later tokens cannot influence earlier outputs.
x2 = x.at[:, 3:].set(jax.random.normal(jax.random.key(1), (B, 3, D)))
out2 = m(x2, causal)
assert jnp.allclose(out[:, :3], out2[:, :3], atol=1e-5), (
    'Rewriting tokens 3.. changed the outputs at positions 0..2, so the mask '
    'is not blocking future keys'
)
assert not jnp.allclose(out, m(x), atol=1e-5), 'The mask argument is being ignored'

# An explicit (B, 1, T, T) mask must give the same answer.
out3 = m(x, jnp.broadcast_to(causal, (B, 1, T, T)))
assert jnp.allclose(out, out3, atol=1e-5), 'Mask does not broadcast over the head axis'
""",
        },
        {
            "name": "Gradients reach all four projections",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=16, num_heads=2, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (2, 5, 16))

grads = nnx.grad(lambda mod: jnp.sum(mod(x) ** 2))(m)
leaves = [jnp.asarray(l) for l in jax.tree.leaves(grads)]
assert len(leaves) == 4, f'Expected 4 parameter gradients (w_q/w_k/w_v/w_o), got {len(leaves)}'
for i, g in enumerate(leaves):
    assert jnp.isfinite(g).all(), f'Non-finite gradient in leaf {i}'
    assert float(jnp.abs(g).sum()) > 0.0, (
        f'Leaf {i} has an all-zero gradient — one projection is unused'
    )

# One gradient step must reduce a simple reconstruction loss.
target = jax.random.normal(jax.random.key(2), (2, 5, 16))


def loss_fn(mod):
    return jnp.mean((mod(x) - target) ** 2)


before = float(loss_fn(m))
for _ in range(20):
    g = nnx.grad(loss_fn)(m)
    nnx.update(m, jax.tree.map(lambda p, d: p - 0.1 * d, nnx.state(m, nnx.Param), g))
after = float(loss_fn(m))
assert after < before, f'Loss did not decrease under nnx.grad: {before} -> {after}'
""",
        },
        {
            "name": "Composes with nnx.jit and vmap",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(d_model=16, num_heads=4, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(0), (3, 7, 16))
eager = m(x)


@nnx.jit
def fwd(mod, inp):
    return mod(inp)


assert jnp.allclose(fwd(m, x), eager, atol=1e-5), 'nnx.jit changed the result'

# Batch items are independent: running one example alone must agree.
alone = m(x[1:2])
assert jnp.allclose(alone[0], eager[1], atol=1e-5), (
    'Example 1 alone differs from the batched call — something is reducing '
    'across the batch axis'
)

# Sequence length is not baked into the parameters.
for t in (1, 2, 13):
    o = m(jax.random.normal(jax.random.key(t), (2, t, 16)))
    assert o.shape == (2, t, 16), f'T={t} gave {o.shape}'
""",
        },
    ],
}
