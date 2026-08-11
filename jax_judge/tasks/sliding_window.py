"""Sliding-window (local causal) attention — a banded mask and why it works."""

TASK = {
    "title": "Sliding Window Attention",
    "category": "Attention & Transformers",
    "number": "11",
    "difficulty": "Hard",
    "function_name": "sliding_window_attention",
    "hint": (
        "Build the band with two broadcast aranges: i = jnp.arange(seq)[:, None], "
        "j = jnp.arange(seq)[None, :], mask = (j <= i) & (i - j < window_size). "
        "That single (seq, seq) boolean broadcasts against scores of shape "
        "(..., seq, seq) whatever the leading batch/head axes are. Apply it with "
        "jnp.where(mask, scores, -jnp.inf) BEFORE the softmax, not after."
    ),
    "description": r"""
Implement **sliding-window attention**: causal attention where query $i$ may
only look at the `window_size` most recent positions, itself included.

$$\text{out}_i = \sum_{j \in \mathcal{W}(i)} \alpha_{ij} v_j,
\qquad \mathcal{W}(i) = \{\, j : i - W < j \le i \,\}$$

$$\alpha_{i\cdot} = \mathrm{softmax}_{j \in \mathcal{W}(i)}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)$$

So `window_size=1` means "attend to yourself only", `window_size=3` means
"yourself plus the two previous tokens", and any `window_size >= seq` is ordinary
causal attention.

### Rules
- Plain function — no `nnx.Module`, no parameters
- Banned: `jax.nn.dot_product_attention` (its `local_window_size` /
  `is_causal` flags do exactly this), `nnx.MultiHeadAttention`
- Signature: `sliding_window_attention(Q, K, V, window_size)`
- Shapes: `Q, K` are `(..., seq, d)` and `V` is `(..., seq, d_v)`, with **any**
  number of leading axes — `(seq, d)`, `(B, seq, D)` and `(B, H, seq, Dh)` must all
  work through broadcasting
- Scale by `1 / sqrt(d)` where `d = Q.shape[-1]`
- `window_size >= 1` is a Python `int`; masked positions get `-inf` before the
  softmax, never a post-softmax zeroing
- Return shape `(..., seq, d_v)`

### Why a window is enough: receptive field through depth
One layer of sliding-window attention only mixes information across $W$
positions. Stack $L$ of them and the receptive field is
$\approx L \cdot (W - 1) + 1$, because layer 2 reads tokens that already
absorbed their own windows at layer 1 — the same argument as stacked
convolutions. Mistral-7B ships $W = 4096$ over 32 layers, giving a theoretical
reach past 130k tokens while every layer only ever computes an
$O(seq \cdot W)$ band instead of an $O(seq^2)$ square.

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


def sliding_window_attention(Q, K, V, window_size):
    """Causal attention restricted to the last `window_size` positions.

    Args:
        Q, K:        (..., seq, d)
        V:           (..., seq, d_v)
        window_size: int >= 1; query i attends to j with i - window_size < j <= i

    Returns:
        (..., seq, d_v)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def sliding_window_attention(Q, K, V, window_size):
    seq, d = Q.shape[-2], Q.shape[-1]

    # (..., seq, seq) — the leading batch/head axes ride along via broadcasting.
    scores = jnp.einsum("...td,...sd->...ts", Q, K) / jnp.sqrt(jnp.float32(d))

    i = jnp.arange(seq)[:, None]      # query position
    j = jnp.arange(seq)[None, :]      # key position
    band = (j <= i) & (i - j < window_size)     # causal AND within the window

    # Mask the SCORES, so the softmax denominator only sees live positions.
    scores = jnp.where(band, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)

    return jnp.einsum("...ts,...se->...te", weights, V)
''',
    "demo": '''import jax.numpy as jnp

seq = 6
# Q = 0 makes every in-window score equal, so the weights are uniform and the
# output at position i is just the mean of V over its window.
Q = jnp.zeros((seq, 4))
K = jnp.zeros((seq, 4))
V = jnp.arange(seq, dtype=jnp.float32)[:, None]

for w in (1, 2, 3, seq):
    out = sliding_window_attention(Q, K, V, w)
    print(f"W={w}: {out[:, 0]}")
print("\\n-> row i is the running mean of the last W values: the band in action")
''',
    "tests": [
        {
            "name": "window_size=1 returns V unchanged",
            "code": """
import jax
import jax.numpy as jnp

Q = jax.random.normal(jax.random.key(0), (2, 8, 16))
K = jax.random.normal(jax.random.key(1), (2, 8, 16))
V = jax.random.normal(jax.random.key(2), (2, 8, 16))

out = {fn}(Q, K, V, 1)
assert out.shape == (2, 8, 16), f'Shape {out.shape} vs (2, 8, 16)'
assert jnp.allclose(out, V, atol=1e-6), (
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

# Q = K = 0 -> every score is 0 -> softmax is uniform over the live band,
# so out[i] is the plain mean of V over positions max(0, i-W+1)..i.
seq = 6
Q = jnp.zeros((seq, 4))
K = jnp.zeros((seq, 4))
V = jnp.arange(seq, dtype=jnp.float32)[:, None]

out = {fn}(Q, K, V, 3)[:, 0]
expected = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
assert jnp.allclose(out, expected, atol=1e-5), (
    f'{out} vs {expected} — out[i] must be the mean of the last 3 values '
    '(clipped at the start of the sequence)'
)

out1 = {fn}(Q, K, V, 2)[:, 0]
expected1 = jnp.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
assert jnp.allclose(out1, expected1, atol=1e-5), f'window_size=2: {out1} vs {expected1}'
""",
        },
        {
            "name": "Large window equals full causal attention",
            "code": """
import jax
import jax.numpy as jnp

B, seq, D = 2, 7, 8
Q = jax.random.normal(jax.random.key(0), (B, seq, D))
K = jax.random.normal(jax.random.key(1), (B, seq, D))
V = jax.random.normal(jax.random.key(2), (B, seq, D))

scores = jnp.einsum('btd,bsd->bts', Q, K) / jnp.sqrt(jnp.float32(D))
causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
ref = jnp.einsum(
    'bts,bse->bte', jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1), V
)

for w in (seq, seq + 5, 1000):
    out = {fn}(Q, K, V, w)
    assert jnp.allclose(out, ref, atol=1e-5), (
        f'window_size={w} >= seq should reduce to plain causal attention '
        f'(max diff {float(jnp.abs(out - ref).max()):.2e})'
    )

# It must NOT equal bidirectional attention — the causal half of the mask matters.
full = jnp.einsum('bts,bse->bte', jax.nn.softmax(scores, axis=-1), V)
assert not jnp.allclose(ref, full, atol=1e-4), 'Test setup degenerate'
assert not jnp.allclose({fn}(Q, K, V, seq), full, atol=1e-4), (
    'Output matches bidirectional attention — the mask is missing the causal j <= i half'
)
""",
        },
        {
            "name": "Locality: only the last W positions can matter",
            "code": """
import jax
import jax.numpy as jnp

B, seq, D, W = 1, 10, 8, 3
Q = jax.random.normal(jax.random.key(0), (B, seq, D))
K = jax.random.normal(jax.random.key(1), (B, seq, D))
V = jax.random.normal(jax.random.key(2), (B, seq, D))

out = {fn}(Q, K, V, W)

# Rewrite everything strictly before position 5 - W + 1 = 3: out[:, 5] must not move.
noise = jax.random.normal(jax.random.key(3), (B, 3, D))
k2 = K.at[:, :3].set(noise)
v2 = V.at[:, :3].set(noise)
out2 = {fn}(Q, k2, v2, W)
assert jnp.allclose(out[:, 5], out2[:, 5], atol=1e-6), (
    'Positions further back than window_size still influence the output — '
    'the i - j < window_size half of the mask is missing or off by one'
)

# Rewrite the future: earlier outputs must not move either.
k3 = K.at[:, 6:].set(jax.random.normal(jax.random.key(4), (B, 4, D)))
v3 = V.at[:, 6:].set(jax.random.normal(jax.random.key(5), (B, 4, D)))
out3 = {fn}(Q, k3, v3, W)
assert jnp.allclose(out[:, :6], out3[:, :6], atol=1e-6), 'Future tokens leak into the past'

# But position 5 MUST depend on position 4 (which is inside its window).
k4 = K.at[:, 4].set(jax.random.normal(jax.random.key(6), (B, D)))
assert not jnp.allclose(out[:, 5], {fn}(Q, k4, V, W)[:, 5], atol=1e-5), (
    'Position 5 ignores position 4, which is inside its window — window too narrow'
)
""",
        },
        {
            "name": "Arbitrary leading axes and d_v != d",
            "code": """
import jax
import jax.numpy as jnp

# Unbatched, batched and (B, H, seq, Dh) must all work by broadcasting.
for shape in [(6, 8), (2, 6, 8), (2, 4, 6, 8)]:
    Q = jax.random.normal(jax.random.key(0), shape)
    K = jax.random.normal(jax.random.key(1), shape)
    V = jax.random.normal(jax.random.key(2), shape)
    out = {fn}(Q, K, V, 2)
    assert out.shape == shape, f'Input {shape} gave {out.shape}'
    assert jnp.isfinite(out).all(), f'Non-finite output for {shape}'

# Values may have their own width.
Q = jax.random.normal(jax.random.key(0), (2, 3, 5, 8))
K = jax.random.normal(jax.random.key(1), (2, 3, 5, 8))
V = jax.random.normal(jax.random.key(2), (2, 3, 5, 12))
out = {fn}(Q, K, V, 2)
assert out.shape == (2, 3, 5, 12), f'd_v != d_k gave {out.shape} vs (2, 3, 5, 12)'

# A (B, H, seq, Dh) call must equal H independent (B, seq, Dh) calls.
per_head = jnp.stack([{fn}(Q[:, h], K[:, h], V[:, h], 2) for h in range(3)], axis=1)
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
Q = jax.random.normal(jax.random.key(0), (1, 12, 16)) * 40.0
K = jax.random.normal(jax.random.key(1), (1, 12, 16)) * 40.0
V = jax.random.normal(jax.random.key(2), (1, 12, 16))

out = {fn}(Q, K, V, 4)
assert jnp.isfinite(out).all(), f'Non-finite output with large scores: {out}'
lo, hi = V.min(axis=1), V.max(axis=1)
assert bool(jnp.all(out >= lo - 1e-4)) and bool(jnp.all(out <= hi + 1e-4)), (
    'Outputs escaped the convex hull of V — the attention weights are not a '
    'valid probability distribution (post-softmax masking does this)'
)

g_q, g_v = jax.grad(lambda a, b: jnp.sum({fn}(a, K, b, 4) ** 2), argnums=(0, 1))(Q, V)
assert jnp.isfinite(g_q).all(), 'NaN/Inf gradient w.r.t. Q — check how -inf enters the scores'
assert jnp.isfinite(g_v).all(), 'NaN/Inf gradient w.r.t. V'
assert float(jnp.abs(g_v).sum()) > 0, 'Zero gradient w.r.t. V'
""",
        },
        {
            "name": "Composes with jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

f = jax.jit({fn}, static_argnums=(3,))
Q = jax.random.normal(jax.random.key(0), (2, 4, 9, 8))
K = jax.random.normal(jax.random.key(1), (2, 4, 9, 8))
V = jax.random.normal(jax.random.key(2), (2, 4, 9, 8))

eager, jitted = {fn}(Q, K, V, 3), f(Q, K, V, 3)
assert jnp.allclose(eager, jitted, atol=1e-6), 'jit result differs from eager'

# vmap over the batch axis of an unbatched (seq, d) implementation.
vm = jax.vmap(lambda a, b, c: {fn}(a, b, c, 3))
out = vm(Q[:, 0], K[:, 0], V[:, 0])
assert out.shape == (2, 9, 8), f'vmap gave {out.shape} vs (2, 9, 8)'
assert jnp.allclose(out, eager[:, 0], atol=1e-5), 'vmapped result differs from the batched call'
""",
        },
    ],
}
