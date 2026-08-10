"""Rotary Position Embeddings — the inverse-frequency table and the pairwise rotation."""

TASK = {
    "title": "Rotary Position Embedding (RoPE)",
    "category": "Attention & Transformers",
    "order": 9,
    "number": "24",
    "difficulty": "Hard",
    "function_name": "apply_rope",
    "hint": (
        "There are only D/2 distinct angles per position, so the cos/sin table is "
        "(T, D/2), not (T, D) — build it once and reuse it for both q and k. The "
        "shape puzzle is getting channel 2j next to channel 2j+1: a reshape that "
        "splits the last axis into (D//2, 2) does exactly that, and the inverse "
        "reshape puts them back. Note that a (T, D/2) table broadcasts against "
        "(..., T, D/2) for any number of leading axes, so nothing about the batch "
        "or head axes needs special handling. Sanity checks while you debug: "
        "position 0 must come out unchanged, and every output vector must have "
        "the same norm as its input."
    ),
    "description": r"""
Implement **RoPE**: rotate every adjacent pair of head channels by an angle
proportional to the token's position, for both `q` and `k`.

Split the head dimension into $D/2$ pairs. Pair $j$ at position $t$ rotates by

$$\theta_{t,j} = \frac{t}{\text{base}^{\,2j/D}}, \qquad j = 0, \dots, \tfrac{D}{2}-1$$

$$\begin{pmatrix} x'_{2j} \\ x'_{2j+1} \end{pmatrix} =
\begin{pmatrix} \cos\theta_{t,j} & -\sin\theta_{t,j} \\
                \sin\theta_{t,j} & \phantom{-}\cos\theta_{t,j} \end{pmatrix}
\begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}$$

### Rules
- Signature: `apply_rope(q, k, base=10000.0) -> (q_rot, k_rot)`
- Input shape `(..., T, D)`: positions run along axis `-2`, channels along axis `-1`,
  `D` even. It must work unchanged for `(B, T, D)` **and** `(B, H, T, Dh)` — build
  `cos`/`sin` as `(T, D/2)` and let broadcasting handle the leading axes
- Positions are `0, 1, ..., T-1`; `q` and `k` get the **same** table
- Use the **interleaved** pairing `(2j, 2j+1)`, not the split-halves variant
- No learned parameters, no lookup table, no `nnx` layers — this is pure math
- Must be jittable and differentiable

### The property that makes it work
Rotation matrices compose: $R_m^\top R_n = R_{n-m}$. So the attention logit

$$\langle R_m q,\; R_n k \rangle = \langle q,\; R_{n-m}\, k \rangle$$

depends **only on the offset** $n - m$, never on where the pair sits in the
window. You inject absolute positions into `q` and `k` and get relative
positions out of the dot product for free — no $O(T^2)$ bias matrix, no extra
parameters, nothing added to `v`.

Two corollaries worth saying out loud in an interview. Each $R$ is orthogonal,
so `‖q_rot‖ = ‖q‖` — RoPE cannot inflate or shrink logits the way an *additive*
position embedding can. And $\theta$ is a closed-form function of $t$, not a row
of a learned table, so position 100,000 is perfectly well defined even if
training stopped at 4,096.

### Why that is *not* free length extrapolation
The usual claim is "RoPE extrapolates." It half does. Any position is
*representable*, but the model has only ever been trained on offsets inside its
window: the low-frequency channels ($j$ near $D/2$, where $\theta \approx
t/10000$) complete only a small fraction of a turn across the whole training
context, so at $10\times$ the length they land at angles the attention heads
have never seen and quality collapses. That gap is the entire reason position
interpolation, NTK-aware base scaling and YaRN exist — they rescale $\theta$ so
inference-time offsets fall back inside the trained range.

### The convention trap
The RoPE paper pairs channels `(0,1), (2,3), ...` — what you are implementing.
Most HuggingFace code instead uses `rotate_half`, pairing `(j, j + D/2)`. Both
give the same relative-offset property; they differ by a fixed permutation of
the head dimension, which a learned `W_q`/`W_k` can absorb. So either is fine to
*train* with, and mixing them is silent corruption at *load* time — the weights
still fit, the loss just quietly goes to garbage.
""",
    "stub": '''import jax
import jax.numpy as jnp


def apply_rope(q, k, base=10000.0):
    """Apply rotary position embeddings to queries and keys.

    Args:
        q, k: (..., T, D) arrays. Positions along axis -2, channels along -1, D even.
        base: geometric base for the inverse-frequency table.

    Returns:
        (q_rot, k_rot), each the same shape as its input.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def apply_rope(q, k, base=10000.0):
    T, D = q.shape[-2], q.shape[-1]
    half = D // 2

    # theta_{t,j} = t / base^(2j/D)  ->  D/2 frequencies, geometrically spaced.
    inv_freq = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / D))
    pos = jnp.arange(T, dtype=jnp.float32)
    theta = pos[:, None] * inv_freq[None, :]          # (T, D/2)
    cos, sin = jnp.cos(theta), jnp.sin(theta)

    def rotate(x):
        # (..., T, D) -> (..., T, D/2, 2): last axis is one (even, odd) pair.
        pairs = x.reshape(*x.shape[:-1], half, 2)
        x_even, x_odd = pairs[..., 0], pairs[..., 1]  # (..., T, D/2)
        # cos/sin are (T, D/2) and broadcast against any leading axes.
        out = jnp.stack(
            [x_even * cos - x_odd * sin,
             x_even * sin + x_odd * cos],
            axis=-1,
        )
        return out.reshape(x.shape)

    return rotate(q), rotate(k)
''',
    "demo": '''import jax
import jax.numpy as jnp

# The frequency table for D=8, base=10000 is exactly [1, 0.1, 0.01, 0.001]:
# pair 0 spins a radian per token, pair 3 barely moves across a whole document.
print("inv_freq:", 1.0 / (10000.0 ** (jnp.arange(4) * 2.0 / 8)))

# Put the SAME vector at every position, then look at the score matrix.
a = jax.random.normal(jax.random.key(0), (16,))
b = jax.random.normal(jax.random.key(1), (16,))
q = jnp.broadcast_to(a, (1, 6, 16))
k = jnp.broadcast_to(b, (1, 6, 16))
qr, kr = apply_rope(q, k)

scores = jnp.einsum("btd,bsd->bts", qr, kr)[0]
print(jnp.round(scores, 3))
print("-> constant along every diagonal: the logit is a function of (n - m) alone")
print("norms preserved:", jnp.allclose(jnp.linalg.norm(q, axis=-1),
                                       jnp.linalg.norm(qr, axis=-1), atol=1e-4))
''',
    "tests": [
        {
            "name": "Shapes, norm preservation, position 0 is identity",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (2, 8, 64))
k = jax.random.normal(jax.random.key(1), (2, 8, 64))
qr, kr = {fn}(q, k)

assert qr.shape == q.shape, f'Q shape mismatch: {qr.shape} vs {q.shape}'
assert kr.shape == k.shape, f'K shape mismatch: {kr.shape} vs {k.shape}'

# Every block is a 2-D rotation, so the head-vector norm cannot change.
assert jnp.allclose(jnp.linalg.norm(q, axis=-1), jnp.linalg.norm(qr, axis=-1), atol=1e-4), (
    'RoPE must preserve ||q|| — a rotation is orthogonal. Non-preserved norms mean '
    'you added/scaled instead of rotating, or mismatched cos/sin per pair.'
)
assert jnp.allclose(jnp.linalg.norm(k, axis=-1), jnp.linalg.norm(kr, axis=-1), atol=1e-4), (
    'RoPE must preserve ||k||'
)

# theta = 0 at t = 0, so position 0 passes through untouched.
assert jnp.allclose(qr[:, 0], q[:, 0], atol=1e-6), (
    'Position 0 must be unchanged (theta=0 -> cos=1, sin=0). '
    'Check that positions start at 0, not 1.'
)
assert not jnp.allclose(qr[:, 1], q[:, 1], atol=1e-3), 'Position 1 should be rotated'
""",
        },
        {
            "name": "Exact inverse-frequency table and pairing",
            "code": """
import jax.numpy as jnp

# D=8, base=10000 -> 2j/D = 0, .25, .5, .75 -> inv_freq = [1, 0.1, 0.01, 0.001] exactly.
inv_freq = jnp.array([1.0, 0.1, 0.01, 0.001])
T, D = 4, 8
x = jnp.ones((1, T, D))
xr, _ = {fn}(x, x)

for t in range(T):
    theta = t * inv_freq                       # (4,)
    # Each pair is (1, 1) rotated by theta_j.
    want_even = jnp.cos(theta) - jnp.sin(theta)
    want_odd = jnp.sin(theta) + jnp.cos(theta)
    got_even = xr[0, t, 0::2]
    got_odd = xr[0, t, 1::2]
    assert jnp.allclose(got_even, want_even, atol=1e-5), (
        f't={t}: even channels {got_even} vs {want_even}. Check inv_freq = '
        'base ** -(2j/D) (not j/D), pair (2j, 2j+1), and x_e*cos - x_o*sin.'
    )
    assert jnp.allclose(got_odd, want_odd, atol=1e-5), (
        f't={t}: odd channels {got_odd} vs {want_odd} — expected x_e*sin + x_o*cos'
    )

# The frequency schedule must DECREASE with channel index: pair 0 spins fastest.
big = jnp.ones((1, 64, 8))
br, _ = {fn}(big, big)
moved_first = float(jnp.abs(br[0, 63, 0:2] - big[0, 63, 0:2]).sum())
moved_last = float(jnp.abs(br[0, 63, 6:8] - big[0, 63, 6:8]).sum())
assert moved_first > moved_last, (
    f'Low channels must rotate FASTER than high ones ({moved_first} vs {moved_last}) — '
    'your frequency table looks reversed'
)
""",
        },
        {
            "name": "Logits depend only on the relative offset",
            "code": """
import jax
import jax.numpy as jnp

# Put the SAME q vector and the SAME k vector at every position: any variation
# in the score matrix is then purely positional.
a = jax.random.normal(jax.random.key(2), (16,))
b = jax.random.normal(jax.random.key(3), (16,))
T = 12
q = jnp.broadcast_to(a, (1, T, 16))
k = jnp.broadcast_to(b, (1, T, 16))
qr, kr = {fn}(q, k)

scores = jnp.einsum('btd,bsd->bts', qr, kr)[0]     # (T, T), scores[m, n] = <R_m a, R_n b>
assert scores.shape == (T, T), f'{scores.shape}'

# <R_m a, R_n b> = <a, R_{n-m} b>: the matrix must be constant on every diagonal.
for d in range(-T + 1, T):
    diag = jnp.diagonal(scores, offset=d)
    assert jnp.allclose(diag, diag[0], atol=1e-4), (
        f'Offset {d}: scores along the diagonal are {diag} but must all be equal. '
        'The logit has to be a function of (n - m) alone.'
    )

# Offset 0 is the un-rotated dot product, since R_m^T R_m = I.
assert jnp.allclose(jnp.diagonal(scores), jnp.dot(a, b), atol=1e-4), (
    'Same-position dot product must equal the un-rotated one — q and k must be '
    'rotated by the SAME angle table'
)

# Shifting the whole window right by 3 leaves the pairwise logits untouched.
q2 = jnp.concatenate([jnp.zeros((1, 3, 16)), q], axis=1)
k2 = jnp.concatenate([jnp.zeros((1, 3, 16)), k], axis=1)
q2r, k2r = {fn}(q2, k2)
assert jnp.allclose(
    jnp.sum(qr[:, 5] * kr[:, 2], -1), jnp.sum(q2r[:, 8] * k2r[:, 5], -1), atol=1e-4
), 'Translating the sequence changed the logit — offset invariance is broken'
""",
        },
        {
            "name": "base actually changes the frequencies",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(4), (1, 6, 16))
k = jax.random.normal(jax.random.key(5), (1, 6, 16))

default_q, _ = {fn}(q, k)
small_q, small_k = {fn}(q, k, base=100.0)
assert not jnp.allclose(default_q, small_q, atol=1e-3), (
    'base is being ignored — it must appear as base ** (2j/D) in the table'
)

# Recompute pair 1 (j=1, channels 2 and 3) by hand with base=100, D=16.
freq = 1.0 / (100.0 ** (2 * 1 / 16))
for t in [1, 5]:
    th = t * freq
    e, o = q[0, t, 2], q[0, t, 3]
    assert jnp.allclose(small_q[0, t, 2], e * jnp.cos(th) - o * jnp.sin(th), atol=1e-5), (
        f'base=100, t={t}, pair 1 even channel disagrees with the closed form'
    )
    assert jnp.allclose(small_q[0, t, 3], e * jnp.sin(th) + o * jnp.cos(th), atol=1e-5), (
        f'base=100, t={t}, pair 1 odd channel disagrees with the closed form'
    )

# A smaller base spins everything faster, but the relative property must survive.
sc = jnp.einsum('btd,bsd->bts', small_q, small_k)[0]
assert jnp.isfinite(sc).all(), 'Non-finite scores'
""",
        },
        {
            "name": "Edge cases: D=2 and T=1",
            "code": """
import jax
import jax.numpy as jnp

# D=2 -> one pair, inv_freq = [1], so position t is a rotation by t radians.
x = jnp.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])   # (1, 3, 2)
xr, _ = {fn}(x, x)
expected = jnp.array([[[1.0, 0.0],
                       [jnp.cos(1.0), jnp.sin(1.0)],
                       [jnp.cos(2.0), jnp.sin(2.0)]]])
assert jnp.allclose(xr, expected, atol=1e-5), f'D=2 case: {xr} vs {expected}'

# T=1 -> only position 0 exists, so nothing rotates.
one = jax.random.normal(jax.random.key(6), (2, 1, 32))
onr, onk = {fn}(one, one)
assert onr.shape == (2, 1, 32), f'{onr.shape}'
assert jnp.allclose(onr, one, atol=1e-6), 'T=1 must be an identity (position 0 only)'
""",
        },
        {
            "name": "Differentiable through q and k",
            "code": """
import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(7), (1, 4, 8))
k = jax.random.normal(jax.random.key(8), (1, 4, 8))

gq, gk = jax.grad(
    lambda a, b: jnp.sum({fn}(a, b)[0] ** 2) + jnp.sum({fn}(a, b)[1] ** 2),
    argnums=(0, 1),
)(q, k)

assert gq.shape == q.shape and gk.shape == k.shape, f'{gq.shape}, {gk.shape}'
assert jnp.isfinite(gq).all() and jnp.isfinite(gk).all(), 'Non-finite gradients'

# Rotations preserve norms, so sum(rot(x)**2) == sum(x**2) and the grad is exactly 2x.
assert jnp.allclose(gq, 2.0 * q, atol=1e-4), (
    'd/dq sum(q_rot**2) must be 2q because the rotation is norm-preserving'
)
assert jnp.allclose(gk, 2.0 * k, atol=1e-4), 'Same for k'
""",
        },
        {
            "name": "jit, vmap and a (B, H, T, Dh) head axis",
            "code": """
import jax
import jax.numpy as jnp

# Leading axes are arbitrary: cos/sin are (T, D/2) and must broadcast.
q4 = jax.random.normal(jax.random.key(9), (2, 3, 10, 16))
k4 = jax.random.normal(jax.random.key(10), (2, 3, 10, 16))
qr4, kr4 = {fn}(q4, k4)
assert qr4.shape == (2, 3, 10, 16), f'4-D output shape {qr4.shape}'

# Head (0, 1) alone must give the same answer as inside the batch: positions live
# on axis -2, so no leading axis may leak into the angle table.
solo, _ = {fn}(q4[0, 1][None], k4[0, 1][None])
assert jnp.allclose(solo[0], qr4[0, 1], atol=1e-5), (
    'Per-head result changed inside a batch — positions must be read from axis -2'
)

jitted = jax.jit({fn})
jq, jk = jitted(q4, k4)
assert jnp.allclose(jq, qr4, atol=1e-5), 'jit changed the result'

vq = jax.random.normal(jax.random.key(11), (4, 2, 6, 16))
vout = jax.vmap({fn})(vq, vq)[0]
assert vout.shape == (4, 2, 6, 16), f'vmap output {vout.shape}'
assert jnp.allclose(vout[1], {fn}(vq[1], vq[1])[0], atol=1e-5), (
    'vmapped row disagrees with the unbatched call'
)
""",
        },
    ],
}
