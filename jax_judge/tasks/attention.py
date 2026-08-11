"""Scaled dot-product attention as a pure function — the scaling and the mask."""

TASK = {
    "title": "Softmax Attention",
    "category": "Attention & Transformers",
    "number": "05",
    "difficulty": "Hard",
    "function_name": "scaled_dot_product_attention",
    "hint": (
        "scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(Q.shape[-1]) — the "
        "swapaxes only touches the last two axes so the same code works for "
        "(B, seq, D) and (B, H, seq, Dh). Apply the mask with "
        "jnp.where(mask, scores, -1e9) BEFORE jax.nn.softmax(scores, axis=-1), "
        "then weights @ V. Note d_k comes from Q.shape[-1] (the key dim), not "
        "from V.shape[-1] — those two are allowed to differ."
    ),
    "description": r"""
Implement **scaled dot-product attention** as a plain function.

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### Signature
`scaled_dot_product_attention(Q, K, V, mask=None)`

| tensor | shape | notes |
|---|---|---|
| `Q` | `(B, seq_q, d_k)` | queries |
| `K` | `(B, seq_k, d_k)` | keys — same feature dim as `Q` |
| `V` | `(B, seq_k, d_v)` | values — `d_v` may differ from `d_k` |
| `mask` | broadcastable to `(..., seq_q, seq_k)` | boolean, `True` = **attend**, `False` = **block** |
| returns | `(B, seq_q, d_v)` | |

### Rules
- No `jax.nn.dot_product_attention`, no `nnx.MultiHeadAttention`
- `jax.nn.softmax` is allowed and encouraged (it subtracts the row max for you)
- `seq_q` and `seq_k` are independent — do not assume a square score matrix
- Only the **last two** axes are contracted, so the identical code must also
  accept `(B, H, seq, D_h)` inputs. Use `jnp.swapaxes(K, -1, -2)`, never a
  hard-coded `transpose(0, 2, 1)`
- Masked positions are removed **before** the softmax, not zeroed after it

### Why the $1/\sqrt{d_k}$ is not cosmetic
Take $Q, K$ with i.i.d. zero-mean unit-variance entries. Their dot product is a
sum of $d_k$ independent unit-variance terms, so

$$\operatorname{Var}(Q \cdot K) = d_k, \qquad \operatorname{std}(Q \cdot K) = \sqrt{d_k}$$

At $d_k = 64$ the raw logits have standard deviation $8$, so across a few hundred
keys the gap between the largest logit and the mean runs to about $23$ (measured
over 300 trials: 10th–90th percentile $19$–$28$).

Concretely, at that scale one key takes roughly **59%** of the attention mass;
after dividing by $\sqrt{64}$ it takes **3%**. So the unscaled version is not
literally one-hot, but it is sharply peaked — and the Jacobian of softmax,
$\operatorname{diag}(p) - pp^\top$, shrinks toward zero as $p$ concentrates, so
the layer passes progressively less gradient the sharper it gets. Dividing by
$\sqrt{d_k}$ pulls the logit variance back to $1$ regardless of head width,
which is exactly why you can widen heads without retuning the initialisation.

Interviewers probe two things here. First that the scale is present at all.
Second — the near-miss that actually separates candidates — that it is
$\sqrt{d_k}$, the **per-head key** dim, and not $\sqrt{d_{model}}$. With $H$
heads those dims differ by a factor of $H$, so the two scale factors differ by
$\sqrt{H}$: at $H = 16$ you would be shrinking every logit by an extra $4\times$
and the attention would come out close to uniform.
""",
    "stub": '''import jax
import jax.numpy as jnp


def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q:    (..., seq_q, d_k)
        K:    (..., seq_k, d_k)
        V:    (..., seq_k, d_v)
        mask: optional boolean array broadcastable to (..., seq_q, seq_k);
              True means "attend here", False means "block".

    Returns:
        (..., seq_q, d_v)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # swapaxes(-1, -2) transposes only the last two axes, so this works for
    # (B, seq, D) and for (B, H, seq, Dh) without changing a line.
    scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, Q.dtype))

    if mask is not None:
        # Blocked entries get a large negative logit BEFORE the softmax, so the
        # normaliser never sees them at all.
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))

    weights = jax.nn.softmax(scores, axis=-1)   # subtracts the row max for us
    return weights @ V
''',
    "demo": '''import jax
import jax.numpy as jnp

# What the scaling actually buys you: logit spread vs head width.
# 64 queries over 512 keys, so the sample statistics settle near their limits.
for d_k in (8, 64, 512):
    kq, kk = jax.random.split(jax.random.key(d_k))
    Q = jax.random.normal(kq, (1, 64, d_k))
    K = jax.random.normal(kk, (1, 512, d_k))
    raw = (Q @ jnp.swapaxes(K, -1, -2))[0]              # (64, 512)
    scaled = raw / jnp.sqrt(float(d_k))
    p_raw = jax.nn.softmax(raw, axis=-1)
    p_scaled = jax.nn.softmax(scaled, axis=-1)
    print(f"d_k={d_k:4d}  sqrt(d_k)={float(d_k) ** 0.5:6.2f}"
          f"  logit std raw={raw.std():6.2f} scaled={scaled.std():4.2f}"
          f"  mean max prob raw={p_raw.max(-1).mean():.3f}"
          f" scaled={p_scaled.max(-1).mean():.3f}")

# Cross shapes: 3 queries attending over 5 keys, values 8-dim.
Q = jax.random.normal(jax.random.key(1), (2, 3, 16))
K = jax.random.normal(jax.random.key(2), (2, 5, 16))
V = jax.random.normal(jax.random.key(3), (2, 5, 8))
print("out:", scaled_dot_product_attention(Q, K, V).shape)   # (2, 3, 8)
''',
    "tests": [
        {
            "name": "Hand-computed 1x2 case",
            "code": """
import jax.numpy as jnp

Q = jnp.array([[[1.0, 0.0]]])                    # (1, 1, 2)
K = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])        # (1, 2, 2)
V = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])        # (1, 2, 2)

out = {fn}(Q, K, V)
assert out.shape == (1, 1, 2), f'Shape mismatch: {out.shape} vs (1, 1, 2)'

# scores = [1, 0] / sqrt(2) = [0.7071, 0]
# softmax -> [0.669762, 0.330238];  out = w0*v0 + w1*v1
expected = jnp.array([[[0.669762, 0.330238]]])
assert jnp.allclose(out, expected, atol=1e-5), (
    f'{out} vs {expected} — if you got [0.7311, 0.2689] you forgot to divide '
    'the scores by sqrt(d_k)'
)
""",
        },
        {
            "name": "The 1/sqrt(d_k) scaling is present",
            "code": """
import jax.numpy as jnp

# d_k = 4 so sqrt(d_k) = 2 exactly. Raw score is 4, scaled score is 2.
Q = jnp.array([[[2.0, 0.0, 0.0, 0.0]]])
K = jnp.array([[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
V = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])

out = {fn}(Q, K, V)
scaled = jnp.array([[[0.880797, 0.119203]]])     # softmax([2, 0])
unscaled = jnp.array([[[0.982014, 0.017986]]])   # softmax([4, 0])

assert not jnp.allclose(out, unscaled, atol=1e-4), (
    'Scores are not being divided by sqrt(d_k) at all'
)
assert jnp.allclose(out, scaled, atol=1e-5), (
    f'{out} vs {scaled} — divide by sqrt(Q.shape[-1]), not by d_k itself '
    'and not by sqrt(d_v)'
)
""",
        },
        {
            "name": "Rectangular shapes and 4-D head axis",
            "code": """
import jax
import jax.numpy as jnp

# seq_q != seq_k and d_v != d_k must both be fine.
Q = jax.random.normal(jax.random.key(0), (2, 3, 16))
K = jax.random.normal(jax.random.key(1), (2, 5, 16))
V = jax.random.normal(jax.random.key(2), (2, 5, 8))
out = {fn}(Q, K, V)
assert out.shape == (2, 3, 8), (
    f'{out.shape} vs (2, 3, 8) — output length comes from Q, output width from V'
)

# The same function must accept a head axis: (B, H, seq, Dh).
q4 = jax.random.normal(jax.random.key(3), (2, 4, 6, 8))
k4 = jax.random.normal(jax.random.key(4), (2, 4, 7, 8))
v4 = jax.random.normal(jax.random.key(5), (2, 4, 7, 8))
o4 = {fn}(q4, k4, v4)
assert o4.shape == (2, 4, 6, 8), (
    f'{o4.shape} vs (2, 4, 6, 8) — use jnp.swapaxes(K, -1, -2), not a '
    'hard-coded 3-axis transpose'
)
# Each head must be computed independently of the others.
assert jnp.allclose(o4[:, 0], {fn}(q4[:, 0], k4[:, 0], v4[:, 0]), atol=1e-5), (
    'Head 0 of the batched call disagrees with computing head 0 alone'
)
""",
        },
        {
            "name": "Output is a convex combination of the values",
            "code": """
import jax
import jax.numpy as jnp

Q = jax.random.normal(jax.random.key(0), (2, 4, 8)) * 3.0
K = jax.random.normal(jax.random.key(1), (2, 6, 8)) * 3.0
V = jax.random.normal(jax.random.key(2), (2, 6, 5))
out = {fn}(Q, K, V)

lo = jnp.min(V, axis=1, keepdims=True)
hi = jnp.max(V, axis=1, keepdims=True)
assert jnp.all(out >= lo - 1e-4) and jnp.all(out <= hi + 1e-4), (
    'Every output must lie inside the elementwise range of V, because the '
    'attention weights are non-negative and sum to 1. Yours does not — the '
    'softmax is probably along the wrong axis (it must be axis=-1, over keys).'
)

# Identical keys with identical values => a uniform average of V.
kk = jnp.zeros((1, 4, 8))
vv = jnp.array([[[0.0], [2.0], [4.0], [6.0]]])
avg = {fn}(jnp.zeros((1, 1, 8)), kk, vv)
assert jnp.allclose(avg, 3.0, atol=1e-5), (
    f'All-equal scores should give the mean of V (3.0), got {avg}'
)
""",
        },
        {
            "name": "Mask blocks keys before the softmax",
            "code": """
import jax
import jax.numpy as jnp

Q = jax.random.normal(jax.random.key(0), (1, 2, 4))
K = jax.random.normal(jax.random.key(1), (1, 5, 4))
V = jax.random.normal(jax.random.key(2), (1, 5, 3))

mask = jnp.array([[[True, True, False, False, False],
                   [True, True, True, False, False]]])   # (1, 2, 5)
out = {fn}(Q, K, V, mask)
assert out.shape == (1, 2, 3), f'Masked output shape {out.shape} vs (1, 2, 3)'

# Row 0 may only see keys 0..1, row 1 only keys 0..2. Renormalised over the
# visible keys, so it must equal attention run on that slice alone.
ref0 = {fn}(Q[:, 0:1], K[:, :2], V[:, :2])[:, 0]
ref1 = {fn}(Q[:, 1:2], K[:, :3], V[:, :3])[:, 0]
assert jnp.allclose(out[:, 0], ref0, atol=1e-5), (
    'Masked row 0 does not match attention over its visible keys — the mask '
    'must be applied to the scores BEFORE softmax so the denominator excludes '
    'the blocked keys'
)
assert jnp.allclose(out[:, 1], ref1, atol=1e-5), 'Masked row 1 is wrong'

# Blocked keys must have exactly zero influence.
v2 = V.at[:, 3:].set(1e3)
assert jnp.allclose({fn}(Q, K, v2, mask), out, atol=1e-4), (
    'Changing V at masked positions changed the output'
)
""",
        },
        {
            "name": "Numerically stable at large logits",
            "code": """
import jax
import jax.numpy as jnp

# Scores of order 1e3: a naive jnp.exp(scores) overflows to inf and then inf/inf.
Q = jax.random.normal(jax.random.key(0), (1, 3, 16)) * 60.0
K = jax.random.normal(jax.random.key(1), (1, 3, 16)) * 60.0
V = jax.random.normal(jax.random.key(2), (1, 3, 4))

out = {fn}(Q, K, V)
assert jnp.isfinite(out).all(), (
    f'Non-finite output at large logits: {out} — use jax.nn.softmax, which '
    'subtracts the row max, instead of exp/sum by hand'
)

# All-equal-and-huge scores must still give the plain average of V.
big = jnp.full((1, 1, 4), 100.0)
kb = jnp.full((1, 3, 4), 100.0)
vb = jnp.array([[[1.0], [2.0], [3.0]]])
assert jnp.allclose({fn}(big, kb, vb), 2.0, atol=1e-4), 'Overflowed on equal large scores'
""",
        },
        {
            "name": "Gradients, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

Q = jax.random.normal(jax.random.key(0), (2, 4, 8))
K = jax.random.normal(jax.random.key(1), (2, 6, 8))
V = jax.random.normal(jax.random.key(2), (2, 6, 8))

gq, gk, gv = jax.grad(lambda a, b, c: jnp.sum({fn}(a, b, c) ** 2),
                      argnums=(0, 1, 2))(Q, K, V)
for name, g, ref in [('Q', gq, Q), ('K', gk, K), ('V', gv, V)]:
    assert g.shape == ref.shape, f'Gradient w.r.t. {name} has shape {g.shape} vs {ref.shape}'
    assert jnp.isfinite(g).all(), f'Non-finite gradient w.r.t. {name}'
    assert float(jnp.abs(g).sum()) > 0.0, f'Gradient w.r.t. {name} is identically zero'

jitted = jax.jit({fn})
assert jnp.allclose(jitted(Q, K, V), {fn}(Q, K, V), atol=1e-5), 'jit changed the result'

# vmap over the batch axis: the per-example call takes (seq, D) inputs.
vmapped = jax.vmap({fn})(Q, K, V)
assert vmapped.shape == (2, 4, 8), f'{vmapped.shape}'
assert jnp.allclose(vmapped, {fn}(Q, K, V), atol=1e-5), (
    'vmap over the batch disagrees with the batched call — your code probably '
    'assumes a fixed number of leading axes'
)
""",
        },
    ],
}
