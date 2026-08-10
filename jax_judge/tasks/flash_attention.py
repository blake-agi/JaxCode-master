"""FlashAttention — online softmax, tiled, and numerically exact."""

TASK = {
    "title": "Flash Attention (Tiled)",
    "category": "Attention & Transformers",
    "order": 8,
    "difficulty": "Hard",
    "function_name": "flash_attention",
    "hint": (
        "Carry three things across key tiles: the running max m, the running "
        "denominator l, and the running unnormalised output acc. For each tile "
        "compute its scores, take m_new = max(m, tile_max), then rescale what "
        "you already have by exp(m - m_new) before adding the tile's "
        "contribution. Both l and acc get the same rescale factor. Normalise "
        "once at the very end by dividing acc by l."
    ),
    "description": r"""
Implement attention the **FlashAttention** way: stream over key/value tiles,
never materializing the full $T_q \times T_k$ score matrix, and produce output
that is *mathematically exact* — the same function as standard attention, not an
approximation of it.

### Signature
```python
def flash_attention(q, k, v, block_size=16):
    # q: (T_q, d), k: (T_k, d), v: (T_k, d_v)
    ...  # -> (T_q, d_v)
```

### The online softmax recurrence
For each key block, with scores $s$ and current state $(m, \ell, \text{acc})$:

$$m^{\text{new}} = \max(m, \max s) \qquad
\alpha = e^{\,m - m^{\text{new}}}$$

$$\ell^{\text{new}} = \alpha\ell + \sum e^{\,s - m^{\text{new}}} \qquad
\text{acc}^{\text{new}} = \alpha\,\text{acc} + e^{\,s - m^{\text{new}}}V_{\text{block}}$$

Start at $m = -\infty$, $\ell = 0$, $\text{acc} = 0$; finish with
$\text{acc}/\ell$.

### Rules
- Process keys in tiles of `block_size`; never build the full score matrix
- Scale by $1/\sqrt{d}$
- Must agree with standard attention to floating-point tolerance, for **every**
  `block_size` — an approximation that is merely close is a fail
- Handle a `T_k` that is not a multiple of `block_size`
- Do not use `jax.nn.softmax` on the whole matrix

### Why the rescaling is what makes tiling *exact*
Softmax needs a global max for stability, but a streaming algorithm has not seen
the future when it processes block 1. The fix is to keep the max *so far*, and
when a later block raises it, retroactively correct everything already
accumulated by $e^{m_{\text{old}} - m_{\text{new}}}$.

Because $e^{s-m_{\text{old}}} \cdot e^{m_{\text{old}}-m_{\text{new}}} =
e^{s-m_{\text{new}}}$ exactly, the correction is not an approximation — it is an
algebraic identity. In exact arithmetic Flash and naive attention compute the
same number; in float32 they differ only by the rounding of a different
summation order (expect $\sim10^{-7}$, and the demo shows it). That is a
completely different kind of "different" from [[linear_attention]], which drops
the softmax and changes the answer at any precision. Be precise about this in an
interview: FlashAttention is **exact, not bit-identical**.

### It is an IO win, not a FLOP win
FlashAttention does the **same** number of floating-point operations as standard
attention — slightly more, in fact, because of the rescaling. It is faster
because attention is **memory-bandwidth bound**: the naive version writes an
$O(T^2)$ score matrix out to HBM and reads it back for the softmax and again for
the $V$ multiply. Flash keeps each tile in SRAM and never writes the scores at
all, taking HBM traffic from $\Theta(T^2 + Td)$ words down to
$\Theta(T^2 d^2 / M)$, where $M$ is the SRAM capacity in words (Dao et al.,
2022, Thm. 2). Since $M \gg d^2$ on real accelerators, that is a large
constant-factor cut — it is still quadratic in $T$.

The memory consequence is the bigger deal in practice: peak activation memory
for attention drops from $O(T^2)$ to $O(T)$, which is what made long context
affordable at all. Note that this Python/XLA version demonstrates the
*algorithm* — the actual speedup requires a fused kernel (Pallas/Triton/CUDA)
that controls SRAM residency directly.
""",
    "stub": '''import jax
import jax.numpy as jnp


def flash_attention(q, k, v, block_size=16):
    """Tiled attention with an online softmax.

    Args:
        q:          (T_q, d)
        k:          (T_k, d)
        v:          (T_k, d_v)
        block_size: number of keys processed per tile

    Returns:
        (T_q, d_v) — the same function as standard attention, for any block_size.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def flash_attention(q, k, v, block_size=16):
    T_q, d = q.shape
    T_k, d_v = v.shape
    scale = 1.0 / jnp.sqrt(jnp.asarray(d, dtype=q.dtype))

    # Running state: max so far, denominator so far, unnormalised output so far.
    m = jnp.full((T_q, 1), -jnp.inf, dtype=q.dtype)
    ell = jnp.zeros((T_q, 1), dtype=q.dtype)
    acc = jnp.zeros((T_q, d_v), dtype=q.dtype)

    for start in range(0, T_k, block_size):
        stop = min(start + block_size, T_k)          # handles a ragged last tile
        k_blk = k[start:stop]
        v_blk = v[start:stop]

        s = (q @ k_blk.T) * scale                    # (T_q, blk)

        m_new = jnp.maximum(m, jnp.max(s, axis=-1, keepdims=True))
        # Retroactively correct everything accumulated under the OLD max.
        alpha = jnp.exp(m - m_new)
        p = jnp.exp(s - m_new)

        ell = alpha * ell + jnp.sum(p, axis=-1, keepdims=True)
        acc = alpha * acc + p @ v_blk
        m = m_new

    return acc / ell
''',
    "demo": '''import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (8, 16))
k = jax.random.normal(jax.random.key(1), (40, 16))
v = jax.random.normal(jax.random.key(2), (40, 4))

ref = jax.nn.softmax(q @ k.T / jnp.sqrt(16.0), axis=-1) @ v

for bs in (4, 7, 16, 64):
    out = flash_attention(q, k, v, block_size=bs)
    print(f"block_size={bs:>3}: max |diff| vs standard = {float(jnp.abs(out - ref).max()):.2e}")
# ~1e-7 for every tiling: that is float32 round-off from a different summation
# order, not approximation error. The rescaling is an algebraic identity.
''',
    "tests": [
        {
            "name": "Exactly matches standard attention",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(k_[0], (12, 16))
k = jax.random.normal(k_[1], (40, 16))
v = jax.random.normal(k_[2], (40, 8))

ref = jax.nn.softmax(q @ k.T / jnp.sqrt(16.0), axis=-1) @ v
out = {fn}(q, k, v, block_size=16)

assert out.shape == (12, 8), f'Expected (12, 8), got {out.shape}'
assert jnp.allclose(out, ref, atol=1e-5), (
    f'Tiling must be EXACT, not approximate. Max diff '
    f'{float(jnp.abs(out - ref).max()):.2e}'
)
""",
        },
        {
            "name": "Result is independent of block_size",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(1), 3)
q = jax.random.normal(k_[0], (10, 8))
k = jax.random.normal(k_[1], (37, 8))     # deliberately prime-ish
v = jax.random.normal(k_[2], (37, 5))

ref = jax.nn.softmax(q @ k.T / jnp.sqrt(8.0), axis=-1) @ v

for bs in (1, 3, 8, 16, 37, 64):
    out = {fn}(q, k, v, block_size=bs)
    assert out.shape == (10, 5), f'block_size={bs}: shape {out.shape}'
    assert jnp.allclose(out, ref, atol=1e-5), (
        f'block_size={bs} gave a different answer (max diff '
        f'{float(jnp.abs(out - ref).max()):.2e}). Tiling must not change the result — '
        'and a ragged final tile (37 is not a multiple of most of these) must '
        'be handled.'
    )
""",
        },
        {
            "name": "The 1/sqrt(d) scaling is applied",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(2), 3)
d = 64
q = jax.random.normal(k_[0], (6, d))
k = jax.random.normal(k_[1], (20, d))
v = jax.random.normal(k_[2], (20, 4))

out = {fn}(q, k, v, block_size=8)

scaled = jax.nn.softmax(q @ k.T / jnp.sqrt(float(d)), axis=-1) @ v
unscaled = jax.nn.softmax(q @ k.T, axis=-1) @ v

assert not jnp.allclose(scaled, unscaled, atol=1e-3), 'test setup failed to separate the two'
assert jnp.allclose(out, scaled, atol=1e-5), (
    f'Scores must be divided by sqrt(d)={jnp.sqrt(float(d)):.2f}. '
    f'Distance to scaled: {float(jnp.abs(out - scaled).max()):.2e}, '
    f'to unscaled: {float(jnp.abs(out - unscaled).max()):.2e}'
)
""",
        },
        {
            "name": "Numerically stable on large scores",
            "code": """
import jax
import jax.numpy as jnp

# Without the running max, exp() of these overflows to inf and the result is nan.
d = 4
q = jnp.full((3, d), 60.0)
k = jnp.full((32, d), 60.0)
k = k.at[7].set(90.0)
v = jax.random.normal(jax.random.key(3), (32, 2))

out = {fn}(q, k, v, block_size=8)
assert jnp.isfinite(out).all(), (
    f'Got {out} — scores here reach ~10^4, so exp() without subtracting a '
    'running max overflows. Track m and rescale.'
)

ref = jax.nn.softmax(q @ k.T / jnp.sqrt(float(d)), axis=-1) @ v
assert jnp.allclose(out, ref, atol=1e-4), f'max diff {float(jnp.abs(out - ref).max()):.2e}'

# The dominant key should win essentially all the mass.
assert jnp.allclose(out, v[7], atol=1e-2), (
    f'One key scores far above the rest, so the output should be ~v[7]={v[7]}, got {out[0]}'
)
""",
        },
        {
            "name": "Rescaling triggers when a later tile raises the max",
            "code": """
import jax
import jax.numpy as jnp

# The largest score sits in the LAST tile, so a correct implementation must
# retroactively rescale everything accumulated before it.
d = 4
q = jnp.ones((2, d))
k = jnp.concatenate([jnp.full((24, d), 0.1), jnp.full((8, d), 10.0)])
v = jnp.concatenate([jnp.zeros((24, 2)), jnp.ones((8, 2))])

out = {fn}(q, k, v, block_size=8)
ref = jax.nn.softmax(q @ k.T / jnp.sqrt(float(d)), axis=-1) @ v

assert jnp.allclose(out, ref, atol=1e-5), (
    f'Got {out[0]}, expected {ref[0]}. The running max only rises at the final '
    'tile here — the earlier accumulator and denominator must both be rescaled '
    'by exp(m_old - m_new).'
)
assert jnp.allclose(out, 1.0, atol=1e-3), (
    f'The high-scoring block carries v=1, so the output should be ~1, got {out[0]}'
)

# And the mirror case: max in the FIRST tile.
k2 = jnp.concatenate([jnp.full((8, d), 10.0), jnp.full((24, d), 0.1)])
v2 = jnp.concatenate([jnp.ones((8, 2)), jnp.zeros((24, 2))])
out2 = {fn}(q, k2, v2, block_size=8)
ref2 = jax.nn.softmax(q @ k2.T / jnp.sqrt(float(d)), axis=-1) @ v2
assert jnp.allclose(out2, ref2, atol=1e-5), f'Max-in-first-tile case: {out2[0]} vs {ref2[0]}'
""",
        },
        {
            "name": "Rows are a convex combination of v",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(4), 3)
q = jax.random.normal(k_[0], (9, 8))
k = jax.random.normal(k_[1], (25, 8))
v = jax.random.normal(k_[2], (25, 3))

out = {fn}(q, k, v, block_size=8)

# Softmax weights sum to 1 and are non-negative, so every output row lies
# inside the bounding box of the value rows.
assert (out >= v.min(axis=0) - 1e-4).all(), 'Output below the range of v'
assert (out <= v.max(axis=0) + 1e-4).all(), 'Output above the range of v'

# Constant v => that constant back, which pins the denominator.
v_const = jnp.tile(jnp.array([2.0, -1.0, 0.5]), (25, 1))
o = {fn}(q, k, v_const, block_size=8)
assert jnp.allclose(o, jnp.array([2.0, -1.0, 0.5]), atol=1e-5), (
    f'With identical value rows the output must be that row, got {o[0]} — '
    'the final division by l is missing or wrong'
)
""",
        },
        {
            "name": "Differentiable and jittable",
            "code": """
import functools
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(5), 3)
q = jax.random.normal(k_[0], (6, 8))
k = jax.random.normal(k_[1], (20, 8))
v = jax.random.normal(k_[2], (20, 4))

jitted = jax.jit(functools.partial({fn}, block_size=8))
assert jnp.allclose(jitted(q, k, v), {fn}(q, k, v, 8), atol=1e-5), 'jit changes the result'

loss = lambda a, b, c: jnp.sum({fn}(a, b, c, 8) ** 2)
gq, gk, gv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

for name, g, ref in (("q", gq, q), ("k", gk, k), ("v", gv, v)):
    assert g.shape == ref.shape, f'grad {name}: {g.shape} vs {ref.shape}'
    assert jnp.isfinite(g).all(), f'Non-finite gradient w.r.t. {name}'

# Gradients must match those of the standard formulation too.
ref_loss = lambda a, b, c: jnp.sum(
    (jax.nn.softmax(a @ b.T / jnp.sqrt(8.0), axis=-1) @ c) ** 2
)
rq = jax.grad(ref_loss, argnums=0)(q, k, v)
assert jnp.allclose(gq, rq, atol=1e-4), (
    f'Gradient differs from standard attention: max diff {float(jnp.abs(gq - rq).max()):.2e}'
)
""",
        },
    ],
}
