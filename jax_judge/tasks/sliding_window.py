"""Sliding-window (local causal) attention — a banded mask and why it works."""

TASK = {
    "title": "Sliding Window Attention",
    "category": "Attention & Transformers",
    "number": "11",
    "difficulty": "Hard",
    "function_name": "sliding_window_attention",
    "hint": (
        "Build the band with two broadcast aranges: i = jnp.arange(T)[:, None], "
        "j = jnp.arange(T)[None, :], mask = (j <= i) & (i - j < window_size). "
        "That single (T, T) boolean broadcasts against scores of shape "
        "(..., T, T) whatever the leading batch/head axes are. Apply it with "
        "jnp.where(mask, scores, -jnp.inf) BEFORE the softmax, not after."
    ),
    "description": r"""
Implement **sliding-window attention**: causal attention where query $i$ may
only look at the `window_size` most recent positions, itself included.

$$\text{out}_i = \sum_{j \in \mathcal{W}(i)} \alpha_{ij} v_j,
\qquad \mathcal{W}(i) = \{\, j : i - W < j \le i \,\}$$

$$\alpha_{i\cdot} = \mathrm{softmax}_{j \in \mathcal{W}(i)}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)$$

So `window_size=1` means "attend to yourself only", `window_size=3` means
"yourself plus the two previous tokens", and any `window_size >= T` is ordinary
causal attention.

### Rules
- Plain function — no `nnx.Module`, no parameters
- Banned: `jax.nn.dot_product_attention` (its `local_window_size` /
  `is_causal` flags do exactly this), `nnx.MultiHeadAttention`
- Signature: `sliding_window_attention(q, k, v, window_size)`
- Shapes: `q, k` are `(..., T, d)` and `v` is `(..., T, d_v)`, with **any**
  number of leading axes — `(T, d)`, `(B, T, D)` and `(B, H, T, Dh)` must all
  work through broadcasting
- Scale by `1 / sqrt(d)` where `d = q.shape[-1]`
- `window_size >= 1` is a Python `int`; masked positions get `-inf` before the
  softmax, never a post-softmax zeroing
- Return shape `(..., T, d_v)`

### Why a window is enough: receptive field through depth
One layer of sliding-window attention only mixes information across $W$
positions. Stack $L$ of them and the receptive field is
$\approx L \cdot (W - 1) + 1$, because layer 2 reads tokens that already
absorbed their own windows at layer 1 — the same argument as stacked
convolutions. Mistral-7B ships $W = 4096$ over 32 layers, giving a theoretical
reach past 130k tokens while every layer only ever computes an
$O(T \cdot W)$ band instead of an $O(T^2)$ square.

The decode-time consequence is bigger than the FLOP saving: keys and values
older than $W$ can never be read again, so the KV cache becomes a **rolling
buffer of fixed size $W$**. Cache memory stops growing with context length —
that is what makes a 100k-token conversation fit on one GPU.

What the receptive-field argument does *not* say is that a distant token has
the same *influence* as a nearby one. Information reaches far, but it is
re-mixed and diluted at every hop, which is why long-context models interleave a
few full-attention layers among the windowed ones instead of trusting depth
alone.

### The traps
- **Post-softmax masking is wrong.** Zeroing weights after the softmax leaves
  the denominator polluted by out-of-window scores, so the surviving weights no
  longer sum to 1. Mask the *scores*.
- **A fully masked row produces NaN.** `softmax([-inf, -inf, ...])` is
  $0/0$. Here the diagonal is always inside the window so every row has at
  least one live entry — which is exactly why `window_size` counts *inclusive*
  of the query position and must be at least 1.
- Prefer `jnp.where(mask, scores, -inf)` over `scores + (1 - mask) * -1e9`: at
  bf16 the magic constant saturates, and adding it to already-large scores can
  overflow.
""",
    "stub": '''import jax
import jax.numpy as jnp


def sliding_window_attention(q, k, v, window_size):
    """Causal attention restricted to the last `window_size` positions.

    Args:
        q, k:        (..., T, d)
        v:           (..., T, d_v)
        window_size: int >= 1; query i attends to j with i - window_size < j <= i

    Returns:
        (..., T, d_v)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def sliding_window_attention(q, k, v, window_size):
    T, d = q.shape[-2], q.shape[-1]

    # (..., T, T) — the leading batch/head axes ride along via broadcasting.
    scores = jnp.einsum("...td,...sd->...ts", q, k) / jnp.sqrt(jnp.float32(d))

    i = jnp.arange(T)[:, None]      # query position
    j = jnp.arange(T)[None, :]      # key position
    band = (j <= i) & (i - j < window_size)     # causal AND within the window

    # Mask the SCORES, so the softmax denominator only sees live positions.
    scores = jnp.where(band, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)

    return jnp.einsum("...ts,...se->...te", weights, v)
''',
    "demo": '''import jax.numpy as jnp

T = 6
# q = 0 makes every in-window score equal, so the weights are uniform and the
# output at position i is just the mean of v over its window.
q = jnp.zeros((T, 4))
k = jnp.zeros((T, 4))
v = jnp.arange(T, dtype=jnp.float32)[:, None]

for w in (1, 2, 3, T):
    out = sliding_window_attention(q, k, v, w)
    print(f"W={w}: {out[:, 0]}")
print("\\n-> row i is the running mean of the last W values: the band in action")
''',
    "tests": [
        {
            "name": "window_size=1 returns V unchanged",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (2, 8, 16))
k = jax.random.normal(jax.random.key(1), (2, 8, 16))
v = jax.random.normal(jax.random.key(2), (2, 8, 16))

out = {fn}(q, k, v, 1)
assert out.shape == (2, 8, 16), f'Shape {out.shape} vs (2, 8, 16)'
assert jnp.allclose(out, v, atol=1e-6), (
    'With window_size=1 each query sees only itself, so softmax over a single '
    'live entry is 1 and the output must equal V exactly. Off by one? The window '
    'is INCLUSIVE of the query position.'
)
""",
        },
        {
            "name": "Uniform weights: exact running-mean check",
            "code": """
import jax.numpy as jnp

# q = k = 0 -> every score is 0 -> softmax is uniform over the live band,
# so out[i] is the plain mean of v over positions max(0, i-W+1)..i.
T = 6
q = jnp.zeros((T, 4))
k = jnp.zeros((T, 4))
v = jnp.arange(T, dtype=jnp.float32)[:, None]

out = {fn}(q, k, v, 3)[:, 0]
expected = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
assert jnp.allclose(out, expected, atol=1e-5), (
    f'{out} vs {expected} — out[i] must be the mean of the last 3 values '
    '(clipped at the start of the sequence)'
)

out1 = {fn}(q, k, v, 2)[:, 0]
expected1 = jnp.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
assert jnp.allclose(out1, expected1, atol=1e-5), f'window_size=2: {out1} vs {expected1}'
""",
        },
        {
            "name": "Large window equals full causal attention",
            "code": """
import jax
import jax.numpy as jnp

B, T, D = 2, 7, 8
q = jax.random.normal(jax.random.key(0), (B, T, D))
k = jax.random.normal(jax.random.key(1), (B, T, D))
v = jax.random.normal(jax.random.key(2), (B, T, D))

scores = jnp.einsum('btd,bsd->bts', q, k) / jnp.sqrt(jnp.float32(D))
causal = jnp.tril(jnp.ones((T, T), dtype=bool))
ref = jnp.einsum(
    'bts,bse->bte', jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1), v
)

for w in (T, T + 5, 1000):
    out = {fn}(q, k, v, w)
    assert jnp.allclose(out, ref, atol=1e-5), (
        f'window_size={w} >= T should reduce to plain causal attention '
        f'(max diff {float(jnp.abs(out - ref).max()):.2e})'
    )

# It must NOT equal bidirectional attention — the causal half of the mask matters.
full = jnp.einsum('bts,bse->bte', jax.nn.softmax(scores, axis=-1), v)
assert not jnp.allclose(ref, full, atol=1e-4), 'Test setup degenerate'
assert not jnp.allclose({fn}(q, k, v, T), full, atol=1e-4), (
    'Output matches bidirectional attention — the mask is missing the causal j <= i half'
)
""",
        },
        {
            "name": "Locality: only the last W positions can matter",
            "code": """
import jax
import jax.numpy as jnp

B, T, D, W = 1, 10, 8, 3
q = jax.random.normal(jax.random.key(0), (B, T, D))
k = jax.random.normal(jax.random.key(1), (B, T, D))
v = jax.random.normal(jax.random.key(2), (B, T, D))

out = {fn}(q, k, v, W)

# Rewrite everything strictly before position 5 - W + 1 = 3: out[:, 5] must not move.
noise = jax.random.normal(jax.random.key(3), (B, 3, D))
k2 = k.at[:, :3].set(noise)
v2 = v.at[:, :3].set(noise)
out2 = {fn}(q, k2, v2, W)
assert jnp.allclose(out[:, 5], out2[:, 5], atol=1e-6), (
    'Positions further back than window_size still influence the output — '
    'the i - j < window_size half of the mask is missing or off by one'
)

# Rewrite the future: earlier outputs must not move either.
k3 = k.at[:, 6:].set(jax.random.normal(jax.random.key(4), (B, 4, D)))
v3 = v.at[:, 6:].set(jax.random.normal(jax.random.key(5), (B, 4, D)))
out3 = {fn}(q, k3, v3, W)
assert jnp.allclose(out[:, :6], out3[:, :6], atol=1e-6), 'Future tokens leak into the past'

# But position 5 MUST depend on position 4 (which is inside its window).
k4 = k.at[:, 4].set(jax.random.normal(jax.random.key(6), (B, D)))
assert not jnp.allclose(out[:, 5], {fn}(q, k4, v, W)[:, 5], atol=1e-5), (
    'Position 5 ignores position 4, which is inside its window — window too narrow'
)
""",
        },
        {
            "name": "Arbitrary leading axes and d_v != d",
            "code": """
import jax
import jax.numpy as jnp

# Unbatched, batched and (B, H, T, Dh) must all work by broadcasting.
for shape in [(6, 8), (2, 6, 8), (2, 4, 6, 8)]:
    q = jax.random.normal(jax.random.key(0), shape)
    k = jax.random.normal(jax.random.key(1), shape)
    v = jax.random.normal(jax.random.key(2), shape)
    out = {fn}(q, k, v, 2)
    assert out.shape == shape, f'Input {shape} gave {out.shape}'
    assert jnp.isfinite(out).all(), f'Non-finite output for {shape}'

# Values may have their own width.
q = jax.random.normal(jax.random.key(0), (2, 3, 5, 8))
k = jax.random.normal(jax.random.key(1), (2, 3, 5, 8))
v = jax.random.normal(jax.random.key(2), (2, 3, 5, 12))
out = {fn}(q, k, v, 2)
assert out.shape == (2, 3, 5, 12), f'd_v != d_k gave {out.shape} vs (2, 3, 5, 12)'

# A (B, H, T, Dh) call must equal H independent (B, T, Dh) calls.
per_head = jnp.stack([{fn}(q[:, h], k[:, h], v[:, h], 2) for h in range(3)], axis=1)
assert jnp.allclose(out, per_head, atol=1e-5), 'Heads are not independent'
""",
        },
        {
            "name": "Stable under huge scores, and differentiable",
            "code": """
import jax
import jax.numpy as jnp

# Scores near 1e3: an implementation that exponentiates before subtracting the
# row max, or that adds a -1e9 constant, blows up here.
q = jax.random.normal(jax.random.key(0), (1, 12, 16)) * 40.0
k = jax.random.normal(jax.random.key(1), (1, 12, 16)) * 40.0
v = jax.random.normal(jax.random.key(2), (1, 12, 16))

out = {fn}(q, k, v, 4)
assert jnp.isfinite(out).all(), f'Non-finite output with large scores: {out}'
lo, hi = v.min(axis=1), v.max(axis=1)
assert bool(jnp.all(out >= lo - 1e-4)) and bool(jnp.all(out <= hi + 1e-4)), (
    'Outputs escaped the convex hull of V — the attention weights are not a '
    'valid probability distribution (post-softmax masking does this)'
)

g_q, g_v = jax.grad(lambda a, b: jnp.sum({fn}(a, k, b, 4) ** 2), argnums=(0, 1))(q, v)
assert jnp.isfinite(g_q).all(), 'NaN/Inf gradient w.r.t. q — check how -inf enters the scores'
assert jnp.isfinite(g_v).all(), 'NaN/Inf gradient w.r.t. v'
assert float(jnp.abs(g_v).sum()) > 0, 'Zero gradient w.r.t. v'
""",
        },
        {
            "name": "Composes with jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

f = jax.jit({fn}, static_argnums=(3,))
q = jax.random.normal(jax.random.key(0), (2, 4, 9, 8))
k = jax.random.normal(jax.random.key(1), (2, 4, 9, 8))
v = jax.random.normal(jax.random.key(2), (2, 4, 9, 8))

eager, jitted = {fn}(q, k, v, 3), f(q, k, v, 3)
assert jnp.allclose(eager, jitted, atol=1e-6), 'jit result differs from eager'

# vmap over the batch axis of an unbatched (T, d) implementation.
vm = jax.vmap(lambda a, b, c: {fn}(a, b, c, 3))
out = vm(q[:, 0], k[:, 0], v[:, 0])
assert out.shape == (2, 9, 8), f'vmap gave {out.shape} vs (2, 9, 8)'
assert jnp.allclose(out, eager[:, 0], atol=1e-5), 'vmapped result differs from the batched call'
""",
        },
    ],
}
