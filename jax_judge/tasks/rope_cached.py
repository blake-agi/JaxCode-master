"""RoPE where the positions are not 0..T-1 — the decode-time case."""

TASK = {
    "title": "RoPE with a KV Cache",
    "category": "Inference & Decoding",
    "number": "b_21",
    "difficulty": "Hard",
    "function_name": "rope_cached_attention",
    "hint": (
        "seq_past = 0 if cache is None else cache[0].shape[-2]. Then "
        "positions = seq_past + jnp.arange(seq_q) — and the SAME positions go "
        "to q and to k_new, because they are the same absolute slots in the "
        "sequence. The cached keys were rotated when they were written, so "
        "concatenate them as they are; rotating them again is the bug this "
        "problem is about. The causal mask is the usual offset one: "
        "tril(ones((seq_q, seq_k)), k=seq_k - seq_q)."
    ),
    "description": r"""
RoPE, but the tokens you are rotating do not start at position 0.

Problem 24 hardcodes `pos = jnp.arange(T)` and rotates `q` and `k` with the
same table, which quietly requires `seq_q == seq_k` and a sequence that starts
at the beginning. Neither holds once there is a KV cache — and that is the
only situation where RoPE actually gets used at inference.

### Signature
```python
def rope_cached_attention(q, k_new, v_new, cache=None, base=10000.0):
    ...
```

| | shape | |
|---|---|---|
| `q`, `k_new`, `v_new` | `(B, H, seq_q, D)` | `D` even |
| `cache` | `None` or `(k_cached, v_cached)`, each `(B, H, seq_past, D)` | **`k_cached` is already rotated** |
| returns | `(out, (k_all, v_all))` | `out` is `(B, H, seq_q, D)` |

### Absolute positions, not 0..seq_q-1
RoPE encodes position by *rotating* each `(even, odd)` pair of the last axis by
an angle proportional to the token's index. Attention then recovers *relative*
position from the difference of the two angles — which only works if both sides
were rotated by their **absolute** slot in the sequence.

With `seq_past` tokens already cached, the incoming `seq_q` tokens occupy slots
`seq_past … seq_past + seq_q - 1`:

```python
seq_past = 0 if cache is None else cache[0].shape[-2]
positions = seq_past + jnp.arange(seq_q)     # 100 cached, 3 new -> [100, 101, 102]
```

The same vector goes to `q` and to `k_new`: they are the same slots.

### The cache is already rotated
A key is rotated once, when it is created, and stored that way. So
`k_cached` comes back rotated and you concatenate it **untouched**. Rotating
it a second time applies the angle twice and silently corrupts every past
position — no error, just a model that attends to the wrong places.

### The test that catches all of it
Run a sequence two ways: all at once, versus prefill-then-decode-one-token-at-a-time.
They must agree to floating-point tolerance. Get the position offset wrong, or
re-rotate the cache, and they will not.

The causal mask is the offset one you have already met:
`jnp.tril(jnp.ones((seq_q, seq_k), dtype=bool), k=seq_k - seq_q)`.

### RoPE itself, for reference
$$\theta_{t,j} = \frac{t}{\text{base}^{2j/D}}, \qquad j = 0 \dots D/2-1$$

Pair the last axis as `(0,1), (2,3), …` and rotate each pair by its angle:
$$(x_{2j}, x_{2j+1}) \mapsto
(x_{2j}\cos\theta - x_{2j+1}\sin\theta,\;
 x_{2j}\sin\theta + x_{2j+1}\cos\theta)$$

This is the same convention as problem 24, so with `cache=None` your rotation
must agree with `apply_rope` exactly.
""",
    "stub": '''import jax
import jax.numpy as jnp


def rope_cached_attention(q, k_new, v_new, cache=None, base=10000.0):
    """Causal attention with RoPE applied at absolute positions.

    Args:
        q, k_new, v_new: (B, H, seq_q, D), D even
        cache: None, or (k_cached, v_cached) each (B, H, seq_past, D).
               k_cached is ALREADY rotated — do not rotate it again.
        base: RoPE frequency base

    Returns:
        (out, (k_all, v_all)) with out of shape (B, H, seq_q, D)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def _rope(x, positions, base):
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / D))
    theta = positions[:, None].astype(jnp.float32) * inv_freq[None, :]  # (seq, D/2)
    cos, sin = jnp.cos(theta), jnp.sin(theta)

    pairs = x.reshape(*x.shape[:-1], half, 2)      # (..., seq, D/2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    out = jnp.stack([even * cos - odd * sin,
                     even * sin + odd * cos], axis=-1)
    return out.reshape(x.shape)


def rope_cached_attention(q, k_new, v_new, cache=None, base=10000.0):
    seq_q = q.shape[-2]
    seq_past = 0 if cache is None else cache[0].shape[-2]

    # The incoming tokens occupy absolute slots seq_past .. seq_past+seq_q-1,
    # and q and k_new are the SAME slots, so they share one position vector.
    positions = seq_past + jnp.arange(seq_q)
    q_rot = _rope(q, positions, base)
    k_rot = _rope(k_new, positions, base)

    if cache is not None:
        # k_cached was rotated when it was written. Rotating it again would
        # double every past angle, silently.
        k_all = jnp.concatenate([cache[0], k_rot], axis=-2)
        v_all = jnp.concatenate([cache[1], v_new], axis=-2)
    else:
        k_all, v_all = k_rot, v_new

    seq_k = k_all.shape[-2]
    D = q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_rot, k_all) / jnp.sqrt(
        jnp.asarray(D, q.dtype)
    )
    allowed = jnp.tril(jnp.ones((seq_q, seq_k), dtype=bool), k=seq_k - seq_q)
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_all)
    return out, (k_all, v_all)
''',
    "demo": '''import jax
import jax.numpy as jnp

B, H, T, D = 1, 2, 8, 8
k = jax.random.split(jax.random.key(0), 3)
Q = jax.random.normal(k[0], (B, H, T, D))
K = jax.random.normal(k[1], (B, H, T, D))
V = jax.random.normal(k[2], (B, H, T, D))

full, _ = rope_cached_attention(Q, K, V)

out, cache = rope_cached_attention(Q[:, :, :5], K[:, :, :5], V[:, :, :5])
print("prefill 5:", out.shape, "cache:", cache[0].shape)
pieces = [out]
for t in range(5, T):
    print(f"  decoding slot {t} -> positions = [{t}]")
    out, cache = rope_cached_attention(
        Q[:, :, t:t + 1], K[:, :, t:t + 1], V[:, :, t:t + 1], cache)
    pieces.append(out)

stepwise = jnp.concatenate(pieces, axis=-2)
print("\\nstepwise == all-at-once?",
      bool(jnp.allclose(stepwise, full, atol=1e-5)),
      f"(max diff {float(jnp.max(jnp.abs(stepwise - full))):.2e})")
''',
    "tests": [
        {
            "name": "Shapes and cache growth",
            "code": """
import jax
import jax.numpy as jnp

B, H, D = 2, 3, 8
k = jax.random.split(jax.random.key(0), 3)
Q = jax.random.normal(k[0], (B, H, 5, D))
K = jax.random.normal(k[1], (B, H, 5, D))
V = jax.random.normal(k[2], (B, H, 5, D))

out, cache = {fn}(Q, K, V)
assert out.shape == (B, H, 5, D), f'{out.shape} vs {(B, H, 5, D)}'
assert isinstance(cache, tuple) and len(cache) == 2, 'Return (out, (k_all, v_all))'
assert cache[0].shape == (B, H, 5, D), f'k_all {cache[0].shape}'
assert cache[1].shape == (B, H, 5, D), f'v_all {cache[1].shape}'

q1 = jax.random.normal(jax.random.key(9), (B, H, 1, D))
out2, cache2 = {fn}(q1, q1, q1, cache)
assert out2.shape == (B, H, 1, D), f'{out2.shape}'
assert cache2[0].shape == (B, H, 6, D), (
    f'cache should grow to 6, got {cache2[0].shape[-2]}'
)
assert jnp.isfinite(out).all() and jnp.isfinite(out2).all(), 'Non-finite output'
""",
        },
        {
            "name": "Stepwise decode equals computing the whole sequence at once",
            "code": """
import jax
import jax.numpy as jnp

B, H, T, D = 2, 2, 8, 8
k = jax.random.split(jax.random.key(1), 3)
Q = jax.random.normal(k[0], (B, H, T, D))
K = jax.random.normal(k[1], (B, H, T, D))
V = jax.random.normal(k[2], (B, H, T, D))

full, _ = {fn}(Q, K, V)

out, cache = {fn}(Q[:, :, :5], K[:, :, :5], V[:, :, :5])
pieces = [out]
for t in range(5, T):
    out, cache = {fn}(
        Q[:, :, t:t + 1], K[:, :, t:t + 1], V[:, :, t:t + 1], cache)
    pieces.append(out)
stepwise = jnp.concatenate(pieces, axis=-2)

assert cache[0].shape[-2] == T, f'cache ended at {cache[0].shape[-2]}, expected {T}'
assert jnp.allclose(stepwise, full, atol=1e-4), (
    f'Stepwise decode disagrees with the one-shot run by up to '
    f'{float(jnp.max(jnp.abs(stepwise - full))):.3e}. Either the new tokens are '
    'being rotated at positions 0..seq_q-1 instead of seq_past+arange(seq_q), '
    'or the cached keys are being rotated a second time.'
)
""",
        },
        {
            "name": "Positions are absolute, not restarted at 0",
            "code": """
import jax
import jax.numpy as jnp

B, H, T, D = 1, 2, 8, 8
k = jax.random.split(jax.random.key(2), 3)
Q = jax.random.normal(k[0], (B, H, T, D))
K = jax.random.normal(k[1], (B, H, T, D))
V = jax.random.normal(k[2], (B, H, T, D))

full, _ = {fn}(Q, K, V)
_, cache5 = {fn}(Q[:, :, :5], K[:, :, :5], V[:, :, :5])
out6, _ = {fn}(Q[:, :, 5:6], K[:, :, 5:6], V[:, :, 5:6], cache5)

assert jnp.allclose(out6, full[:, :, 5:6], atol=1e-4), (
    'Decoding slot 5 on top of a 5-token cache must match row 5 of the '
    'one-shot run. It does not, so the token is not being rotated at '
    'position 5.'
)

# Two caches of different length must rotate the SAME token differently,
# because its absolute slot differs.
_, cache3 = {fn}(Q[:, :, :3], K[:, :, :3], V[:, :, :3])
tok = jax.random.normal(jax.random.key(7), (B, H, 1, D))
a, _ = {fn}(tok, tok, tok, cache3)      # slot 3
b, _ = {fn}(tok, tok, tok, cache5)      # slot 5
assert not jnp.allclose(a, b, atol=1e-4), (
    'The same token gave the same result at slot 3 and slot 5, so the '
    'position offset is being ignored'
)
""",
        },
        {
            "name": "Cached keys are concatenated untouched, not rotated again",
            "code": """
import jax
import jax.numpy as jnp

B, H, D = 1, 2, 8
k = jax.random.split(jax.random.key(3), 3)
Q = jax.random.normal(k[0], (B, H, 4, D))
K = jax.random.normal(k[1], (B, H, 4, D))
V = jax.random.normal(k[2], (B, H, 4, D))

_, (k4, v4) = {fn}(Q, K, V)

tok = jax.random.normal(jax.random.key(8), (B, H, 1, D))
_, (k5, v5) = {fn}(tok, tok, tok, (k4, v4))

assert jnp.array_equal(k5[:, :, :4], k4), (
    'The first 4 cached keys changed. They were already rotated when written, '
    'so they must be concatenated as-is — rotating them again doubles every '
    'past angle and corrupts the whole history.'
)
assert jnp.array_equal(v5[:, :, :4], v4), 'Cached values changed; v is never rotated'
""",
        },
        {
            "name": "Rotation matches problem 24 when there is no cache",
            "code": """
import jax
import jax.numpy as jnp

B, H, T, D = 1, 1, 6, 8
k = jax.random.split(jax.random.key(4), 3)
Q = jax.random.normal(k[0], (B, H, T, D))
K = jax.random.normal(k[1], (B, H, T, D))
V = jax.random.normal(k[2], (B, H, T, D))

# Reference rotation, same (even, odd) pairing as problem 24.
half = D // 2
inv = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / D))
th = jnp.arange(T, dtype=jnp.float32)[:, None] * inv[None, :]
cos, sin = jnp.cos(th), jnp.sin(th)
def rope(x):
    p = x.reshape(*x.shape[:-1], half, 2)
    e, o = p[..., 0], p[..., 1]
    return jnp.stack([e * cos - o * sin, e * sin + o * cos], -1).reshape(x.shape)

out, (k_all, _) = {fn}(Q, K, V)
assert jnp.allclose(k_all, rope(K), atol=1e-5), (
    'With cache=None the returned keys must be K rotated at positions 0..T-1 '
    'with the (even, odd) pairing of problem 24'
)

scores = jnp.einsum('bhqd,bhkd->bhqk', rope(Q), rope(K)) / jnp.sqrt(
    jnp.asarray(D, Q.dtype))
tri = jnp.tril(jnp.ones((T, T), dtype=bool))
ref = jnp.einsum('bhqk,bhkd->bhqd',
                 jax.nn.softmax(jnp.where(tri, scores, -jnp.inf), -1), V)
assert jnp.allclose(out, ref, atol=1e-5), 'Disagrees with plain RoPE causal attention'
""",
        },
        {
            "name": "base is honoured, and jit/grad work",
            "code": """
import jax
import jax.numpy as jnp

B, H, T, D = 1, 2, 4, 8
k = jax.random.split(jax.random.key(5), 3)
Q = jax.random.normal(k[0], (B, H, T, D))
K = jax.random.normal(k[1], (B, H, T, D))
V = jax.random.normal(k[2], (B, H, T, D))

a, _ = {fn}(Q, K, V, None, 10000.0)
b, _ = {fn}(Q, K, V, None, 500.0)
assert not jnp.allclose(a, b, atol=1e-5), 'Changing base changed nothing'

out, _ = {fn}(Q, K, V)
jf = jax.jit({fn})
assert jnp.allclose(jf(Q, K, V)[0], out, atol=1e-6), 'jit disagrees'

g = jax.grad(lambda x: jnp.sum({fn}(x, K, V)[0]))(Q)
assert g.shape == Q.shape and jnp.isfinite(g).all(), 'Bad gradient w.r.t. q'
""",
        },
    ],
}
