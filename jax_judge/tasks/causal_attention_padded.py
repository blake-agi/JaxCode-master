"""Causal attention that also has to survive a key padding mask."""

TASK = {
    "title": "Causal Attention with Padding",
    "category": "Attention & Transformers",
    "number": "b_20",
    "difficulty": "Hard",
    "function_name": "causal_attention_padded",
    "hint": (
        "Three things have to line up. (1) The causal mask is NOT tril of a "
        "square: with seq_k >= seq_q the queries are the LAST seq_q positions, "
        "so query i sits at absolute position past + i where past = seq_k - "
        "seq_q; allowed is `j - i <= past`, i.e. jnp.tril(ones((seq_q, seq_k)), "
        "k=past). (2) The padding mask is (B, seq_k) — per batch item, not per "
        "position — so it needs [:, None, None, :] to meet a (B, H, seq_q, "
        "seq_k) score array. Combine with `&`. (3) A row where the two masks "
        "leave nothing visible cannot be fixed by the fill value; zero those "
        "rows explicitly after the softmax."
    ),
    "description": r"""
Causal attention, but the keys are a padded batch — so some of them are not
real tokens, and some query rows end up with **nothing to attend to at all**.

### Signature
```python
def causal_attention_padded(Q, K, V, key_padding_mask=None):
    ...
```

| | shape | |
|---|---|---|
| `Q` | `(B, H, seq_q, d_k)` | |
| `K` | `(B, H, seq_k, d_k)` | `seq_k >= seq_q` |
| `V` | `(B, H, seq_k, d_v)` | `d_v` need not equal `d_k` |
| `key_padding_mask` | `(B, seq_k)` bool or `None` | `True` = real token |
| returns | `(B, H, seq_q, d_v)` | |

### 1. The causal mask is not `tril` of a square
The queries are the **last** `seq_q` positions of the sequence — that is the
decode-time convention. With `past = seq_k - seq_q`, query `i` sits at absolute
position `past + i`, so it may attend to keys `j <= past + i`:

$$j - i \le \text{past}
\quad\Longrightarrow\quad
\texttt{jnp.tril(jnp.ones((seq\_q, seq\_k)), k=past)}$$

The `k=` argument of `tril`/`triu` is exactly this threshold — `tril(m, k)`
keeps `(i, j)` iff `j - i <= k`. Drop it and you get the top-left triangle,
which hides every past key and does so **silently**:

```
seq_q=2  seq_k=5  past=3

k=past (right)        k omitted (wrong)
  q0  1 1 1 1 0         q0  1 0 0 0 0
  q1  1 1 1 1 1         q1  1 1 0 0 0
```

### 2. The padding mask lives on a different axis
Causal is `(seq_q, seq_k)` — a property of *positions*. Padding is
`(B, seq_k)` — a property of *batch items*. They meet on a `(B, H, seq_q,
seq_k)` score array, so the padding mask needs its head and query axes
inserted before the two can be combined with `&`.

### 3. Some rows have nothing left, and that is the real problem
A query whose whole causal window is padding has **no visible key**. Softmax
over an all-blocked row cannot produce a distribution, and neither fill value
saves you:

```
fill = -inf   ->  [nan nan nan nan nan]                     poisons everything
fill = -1e9   ->  [0.237, 0.299, -0.355, 0.149, -0.047]     finite, plausible, garbage
```

The `-1e9` case is the dangerous one: softmax of equal logits is **uniform**,
so the row returns the average of the padding vectors. No error, no NaN, and a
number that looks perfectly reasonable.

Return **exactly zero** for such rows. That means finding them —
`jnp.any(allowed, axis=-1)` — and zeroing after the softmax; a fill value alone
cannot express "no answer".

### Everything else
Scale by $\sqrt{d_k}$ (not $d_v$). With `key_padding_mask=None` and
`seq_q == seq_k` this must reduce exactly to ordinary causal self-attention.
""",
    "stub": '''import jax
import jax.numpy as jnp


def causal_attention_padded(Q, K, V, key_padding_mask=None):
    """Causal attention over a padded batch of keys.

    Args:
        Q: (B, H, seq_q, d_k)
        K: (B, H, seq_k, d_k)   seq_k >= seq_q
        V: (B, H, seq_k, d_v)
        key_padding_mask: (B, seq_k) bool, True = real token, or None

    Returns:
        (B, H, seq_q, d_v). Rows with no visible key are exactly zero.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def causal_attention_padded(Q, K, V, key_padding_mask=None):
    d_k = Q.shape[-1]
    seq_q, seq_k = Q.shape[-2], K.shape[-2]

    scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, Q.dtype))

    # Queries are the LAST seq_q positions, so query i is at absolute position
    # past + i and may see j <= past + i, i.e. j - i <= past. That threshold is
    # exactly what tril's k= means.
    past = seq_k - seq_q
    allowed = jnp.tril(jnp.ones((seq_q, seq_k), dtype=bool), k=past)

    if key_padding_mask is not None:
        # (B, seq_k) is per batch item; the scores are (B, H, seq_q, seq_k).
        allowed = allowed & key_padding_mask[:, None, None, :]

    # Finite fill, so the softmax never sees an all -inf row and never makes a
    # NaN. A dead row becomes uniform here, which is wrong but harmless — the
    # next step is what actually handles it.
    scores = jnp.where(allowed, scores, jnp.asarray(-1e9, scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)

    # A row with nothing visible has no answer, and no fill value can say so.
    # Zero it explicitly.
    alive = jnp.any(allowed, axis=-1, keepdims=True)
    weights = jnp.where(alive, weights, 0.0)

    return weights @ V
''',
    "demo": '''import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d_k, d_v = 1, 1, 2, 5, 4, 3
k = jax.random.split(jax.random.key(0), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d_k))
K = jax.random.normal(k[1], (B, H, seq_k, d_k))
V = jax.random.normal(k[2], (B, H, seq_k, d_v))

past = seq_k - seq_q
print(f"past = seq_k - seq_q = {past}, so the causal mask is tril(k={past}):")
print(jnp.tril(jnp.ones((seq_q, seq_k), dtype=int), k=past))

print("\\nno padding:")
print(causal_attention_padded(Q, K, V))

# Pad away everything query 0 could have seen (keys 0..past).
mask = jnp.ones((B, seq_k), dtype=bool).at[0, : past + 1].set(False)
out = causal_attention_padded(Q, K, V, mask)
print(f"\\nkey_padding_mask = {mask[0]}")
print("query 0 now has no visible key, so its row must be exactly zero:")
print(out)
''',
    "tests": [
        {
            "name": "Shapes: seq_q != seq_k and d_v != d_k",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d_k, d_v = 2, 3, 4, 7, 8, 5
k = jax.random.split(jax.random.key(0), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d_k))
K = jax.random.normal(k[1], (B, H, seq_k, d_k))
V = jax.random.normal(k[2], (B, H, seq_k, d_v))

out = {fn}(Q, K, V)
assert out.shape == (B, H, seq_q, d_v), (
    f'{out.shape} vs {(B, H, seq_q, d_v)} — length comes from Q, width from V'
)
assert jnp.isfinite(out).all(), 'Non-finite output with no padding at all'

# Scaling must use d_k. Doubling d_v (by widening V) cannot change the weights,
# so the output must stay a fixed linear map of V.
V2 = jnp.concatenate([V, V], axis=-1)
out2 = {fn}(Q, K, V2)
assert out2.shape == (B, H, seq_q, 2 * d_v), f'{out2.shape}'
assert jnp.allclose(out2[..., :d_v], out, atol=1e-5), (
    'Widening V changed the attention weights — the 1/sqrt scaling is using '
    'd_v somewhere instead of d_k'
)
""",
        },
        {
            "name": "Reduces to plain causal self-attention when square and unpadded",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq, d_k, d_v = 2, 2, 5, 6, 6
k = jax.random.split(jax.random.key(1), 3)
Q = jax.random.normal(k[0], (B, H, seq, d_k))
K = jax.random.normal(k[1], (B, H, seq, d_k))
V = jax.random.normal(k[2], (B, H, seq, d_v))

scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, Q.dtype))
tri = jnp.tril(jnp.ones((seq, seq), dtype=bool))
ref = jax.nn.softmax(jnp.where(tri, scores, -1e9), axis=-1) @ V

out = {fn}(Q, K, V)
assert jnp.allclose(out, ref, atol=1e-5), (
    'With seq_q == seq_k and no padding this must be ordinary causal '
    'self-attention'
)

# An all-True padding mask must change nothing.
allo = jnp.ones((B, seq), dtype=bool)
assert jnp.allclose({fn}(Q, K, V, allo), out, atol=1e-6), (
    'An all-True key_padding_mask should be a no-op'
)
""",
        },
        {
            "name": "Causal window is offset by past = seq_k - seq_q",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d = 1, 1, 3, 6, 4
past = seq_k - seq_q          # 3
k = jax.random.split(jax.random.key(2), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d))
K = jax.random.normal(k[1], (B, H, seq_k, d))
V = jax.random.normal(k[2], (B, H, seq_k, d))
base = {fn}(Q, K, V)

# Query i may see keys 0..past+i. Perturbing V at a key BEYOND that window must
# leave row i untouched; perturbing one INSIDE it must change row i.
for i in range(seq_q):
    edge = past + i
    if edge + 1 < seq_k:
        Vb = V.at[:, :, edge + 1, :].add(100.0)
        assert jnp.allclose({fn}(Q, K, Vb)[:, :, i, :], base[:, :, i, :], atol=1e-5), (
            f'Query {i} reacted to key {edge + 1}, which is in its future. With '
            f'seq_q={seq_q} and seq_k={seq_k} the window is j <= past + i = '
            f'{edge}; the mask needs tril(..., k=past), not plain tril.'
        )
    Vi = V.at[:, :, edge, :].add(100.0)
    assert not jnp.allclose({fn}(Q, K, Vi)[:, :, i, :], base[:, :, i, :], atol=1e-5), (
        f'Query {i} ignored key {edge}, which it should be able to see — the '
        'causal window is too narrow (k= is missing or too small)'
    )
""",
        },
        {
            "name": "Padded keys contribute exactly nothing",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d = 2, 2, 3, 6, 4
k = jax.random.split(jax.random.key(3), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d))
K = jax.random.normal(k[1], (B, H, seq_k, d))
V = jax.random.normal(k[2], (B, H, seq_k, d))

# Batch 0 has its last two keys padded; batch 1 is fully real.
mask = jnp.ones((B, seq_k), dtype=bool).at[0, -2:].set(False)
base = {fn}(Q, K, V, mask)

# Changing V at a padded key must not move batch 0 at all.
Vp = V.at[0, :, -2:, :].add(1000.0)
assert jnp.allclose({fn}(Q, K, Vp, mask), base, atol=1e-4), (
    'A padded key changed the output, so it still carries attention weight'
)

# Changing K at a padded key must not move batch 0 either (it must not compete
# in the softmax denominator).
Kp = K.at[0, :, -2:, :].add(1000.0)
assert jnp.allclose({fn}(Q, Kp, V, mask)[0], base[0], atol=1e-4), (
    'A padded key changed the softmax denominator — mask before the softmax, '
    'not after'
)

# Batch 1 is untouched by batch 0's mask.
assert jnp.allclose(base[1], {fn}(Q, K, V)[1], atol=1e-5), (
    'The padding mask leaked across batch items — it is (B, seq_k) and needs '
    'its head and query axes inserted, e.g. mask[:, None, None, :]'
)
""",
        },
        {
            "name": "A row with no visible key returns exactly zero, never NaN",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d = 2, 2, 3, 6, 4
past = seq_k - seq_q          # 3
k = jax.random.split(jax.random.key(4), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d))
K = jax.random.normal(k[1], (B, H, seq_k, d))
V = jax.random.normal(k[2], (B, H, seq_k, d))

# Pad away every key query 0 could see (0..past). Query 0 is now blind.
mask = jnp.ones((B, seq_k), dtype=bool).at[0, : past + 1].set(False)
out = {fn}(Q, K, V, mask)

assert jnp.isfinite(out).all(), (
    'NaN or Inf in the output. A fully masked row makes softmax(-inf ...) '
    'undefined — use a finite fill, then zero the row.'
)
dead = out[0, :, 0, :]
assert float(jnp.abs(dead).max()) == 0.0, (
    f'Row with no visible key returned {dead[0]} instead of exactly 0. With a '
    '-1e9 fill the softmax goes UNIFORM over the blocked keys, so the row '
    'silently returns the mean of the padding vectors — finite and plausible '
    'and wrong. Find those rows with jnp.any(allowed, axis=-1) and zero them.'
)

# Query 1 can still see key past+1, which is real, so it must NOT be zeroed.
assert float(jnp.abs(out[0, :, 1, :]).max()) > 0.0, (
    'Query 1 still has a visible key and must not be zeroed out'
)
# Batch 1 has no padding at all and must be untouched.
assert jnp.allclose(out[1], {fn}(Q, K, V)[1], atol=1e-5), 'Batch 1 was affected'
""",
        },
        {
            "name": "Gradients stay finite through the dead rows, and jit/vmap work",
            "code": """
import jax
import jax.numpy as jnp

B, H, seq_q, seq_k, d = 2, 2, 3, 6, 4
past = seq_k - seq_q
k = jax.random.split(jax.random.key(5), 3)
Q = jax.random.normal(k[0], (B, H, seq_q, d))
K = jax.random.normal(k[1], (B, H, seq_k, d))
V = jax.random.normal(k[2], (B, H, seq_k, d))
mask = jnp.ones((B, seq_k), dtype=bool).at[0, : past + 1].set(False)

g = jax.grad(lambda q: jnp.sum({fn}(q, K, V, mask)))(Q)
assert jnp.isfinite(g).all(), (
    'Non-finite gradient. A -inf fill produces NaN in the forward pass and the '
    'NaN survives into the backward pass even if you clean the output later.'
)

out = {fn}(Q, K, V, mask)
assert jnp.allclose(jax.jit({fn})(Q, K, V, mask), out, atol=1e-6), 'jit disagrees'

# One batch item at a time, mask included.
vm = jax.vmap(lambda q, kk, vv, m: {fn}(q[None], kk[None], vv[None], m[None])[0])
assert jnp.allclose(vm(Q, K, V, mask), out, atol=1e-5), 'vmap disagrees'
""",
        },
    ],
}
