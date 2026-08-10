"""Linear attention — the associativity trick that turns O(T^2) into O(T)."""

TASK = {
    "title": "Linear Self-Attention",
    "category": "Attention & Transformers",
    "order": 7,
    "number": "12",
    "difficulty": "Hard",
    "function_name": "linear_attention",
    "hint": (
        "Map q and k through phi(x) = elu(x) + 1 so both stay positive. Then "
        "never build the (T, T) matrix: contract keys and values FIRST with "
        "jnp.einsum('...td,...te->...de', phi_k, v) to get a (d, d_v) summary, "
        "and multiply the queries into that. The denominator is the same trick "
        "applied to a vector of ones: phi_q @ sum_t phi_k[t]."
    ),
    "description": r"""
Implement **linear attention**: replace $\text{softmax}(QK^\top)V$ with a
kernel feature map so the sequence length drops out of the complexity.

Standard attention computes
$$\text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d}}\right)V$$
which materializes a $T \times T$ matrix. Linear attention replaces the softmax
kernel with $\phi(q)^\top\phi(k)$ for a feature map $\phi$, giving

$$O_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}
             {\phi(q_i)^\top \sum_j \phi(k_j)}$$

Use $\phi(x) = \text{elu}(x) + 1$, which is strictly positive — required, since
it plays the role of an unnormalised probability.

### Signature
```python
def linear_attention(q, k, v):
    # q: (..., T_q, d), k: (..., T_k, d), v: (..., T_k, d_v)
    ...  # -> (..., T_q, d_v)
```

### Rules
- Never form a `(T_q, T_k)` matrix — that defeats the entire point
- Contract $K$ with $V$ **first**
- Include the denominator; without it this is not an average and the output
  scale drifts with sequence length
- Add a small epsilon to the denominator
- Do not use `jax.nn.softmax`

### The whole trick is associativity
$(QK^\top)V$ and $Q(K^\top V)$ are the same product, but:

| Order | Intermediate | Cost |
|---|---|---|
| $(QK^\top)V$ | $T \times T$ | $O(T^2 d)$ |
| $Q(K^\top V)$ | $d \times d_v$ | $O(T d\, d_v)$ |

Softmax is what forbids the second grouping — it is a nonlinearity *between*
$QK^\top$ and the multiplication by $V$. Remove it, and matrix multiplication
becomes reassociable. Every linear-attention variant (Performer, RFA, and the
linear-attention view of state-space models) is a different choice of $\phi$
around that one observation.

### What you give up
The $d \times d_v$ summary is a **fixed-size** state, no matter how long the
sequence. So this is lossy in a way softmax attention is not: softmax can
sharply retrieve a single token out of a million (its "state" grows with $T$),
while linear attention compresses everything into the same matrix and cannot.
That is precisely why linear-attention models underperform on retrieval-heavy
tasks like needle-in-a-haystack, and why the strongest recent designs interleave
a few full-attention layers among the linear ones.

The flip side is the reason to care: because the state is fixed-size and updates
additively, the **causal** version runs as an RNN at inference — $O(1)$ memory
per token instead of a KV cache that grows without bound ([[kv_cache]]).
""",
    "stub": '''import jax
import jax.numpy as jnp


def linear_attention(q, k, v):
    """Linear attention with the phi(x) = elu(x) + 1 feature map.

    Args:
        q: (..., T_q, d)
        k: (..., T_k, d)
        v: (..., T_k, d_v)

    Returns:
        (..., T_q, d_v)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def linear_attention(q, k, v):
    # Strictly positive feature map — it stands in for an unnormalised
    # probability, so it must never be negative.
    phi_q = jax.nn.elu(q) + 1.0
    phi_k = jax.nn.elu(k) + 1.0

    # Contract keys with values FIRST: (..., d, d_v). This is the whole trick —
    # the T axis is summed away before the queries ever get involved, so no
    # (T_q, T_k) matrix is ever built.
    kv = jnp.einsum("...td,...te->...de", phi_k, v)

    numerator = jnp.einsum("...td,...de->...te", phi_q, kv)

    # Same contraction against a vector of ones gives the normaliser.
    k_sum = jnp.sum(phi_k, axis=-2)                       # (..., d)
    denominator = jnp.einsum("...td,...d->...t", phi_q, k_sum)

    return numerator / (denominator[..., None] + 1e-6)
''',
    "demo": '''import jax
import jax.numpy as jnp

q = jax.random.normal(jax.random.key(0), (2, 16, 8))
k = jax.random.normal(jax.random.key(1), (2, 16, 8))
v = jax.random.normal(jax.random.key(2), (2, 16, 4))

out = linear_attention(q, k, v)
print("q:", q.shape, " v:", v.shape, " -> out:", out.shape)

# Cost grows linearly, not quadratically, in T.
for T in (64, 256, 1024):
    qq = jax.random.normal(jax.random.key(3), (1, T, 8))
    vv = jax.random.normal(jax.random.key(4), (1, T, 4))
    print(f"  T={T:>5}: out {linear_attention(qq, qq, vv).shape}")
''',
    "tests": [
        {
            "name": "Shapes, including a head axis",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(k_[0], (2, 10, 8))
k = jax.random.normal(k_[1], (2, 10, 8))
v = jax.random.normal(k_[2], (2, 10, 5))

out = {fn}(q, k, v)
assert out.shape == (2, 10, 5), f'Expected (2, 10, 5), got {out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite output'

# Extra leading axes (batch, heads) must just work.
q4 = jax.random.normal(k_[0], (2, 4, 10, 8))
k4 = jax.random.normal(k_[1], (2, 4, 10, 8))
v4 = jax.random.normal(k_[2], (2, 4, 10, 5))
assert {fn}(q4, k4, v4).shape == (2, 4, 10, 5), f'4-D case: {{}}'.format({fn}(q4, k4, v4).shape)

# T_q need not equal T_k.
q3 = jax.random.normal(k_[0], (2, 7, 8))
assert {fn}(q3, k, v).shape == (2, 7, 5), 'T_q != T_k must be supported'
""",
        },
        {
            "name": "Matches the explicit quadratic form",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(1), 3)
q = jax.random.normal(k_[0], (2, 6, 4))
k = jax.random.normal(k_[1], (2, 6, 4))
v = jax.random.normal(k_[2], (2, 6, 3))

got = {fn}(q, k, v)

# The same maths, written the slow O(T^2) way.
pq = jax.nn.elu(q) + 1.0
pk = jax.nn.elu(k) + 1.0
scores = jnp.einsum('btd,bsd->bts', pq, pk)
expected = jnp.einsum('bts,bse->bte', scores, v) / (
    jnp.sum(scores, axis=-1, keepdims=True) + 1e-6
)

assert jnp.allclose(got, expected, atol=1e-3), (
    f'Reassociating must not change the value.\\nmax diff: '
    f'{float(jnp.abs(got - expected).max()):.6f}'
)
""",
        },
        {
            "name": "Feature map is elu + 1 and stays positive",
            "code": """
import jax
import jax.numpy as jnp

# Strongly negative inputs: relu-based or raw-dot features would give
# zero/negative weights and a degenerate or sign-flipped output.
q = jnp.full((1, 4, 3), -5.0)
k = jnp.full((1, 4, 3), -5.0)
v = jnp.abs(jax.random.normal(jax.random.key(2), (1, 4, 2))) + 1.0

out = {fn}(q, k, v)
assert jnp.isfinite(out).all(), 'Non-finite output on strongly negative inputs'
assert (out > 0).all(), (
    f'With all-positive v the output must stay positive — phi must be strictly '
    f'positive (elu(x)+1), got {out}'
)

# Output is a weighted AVERAGE of v rows, so it lies within their range.
v2 = jax.random.normal(jax.random.key(3), (1, 5, 2))
q2 = jax.random.normal(jax.random.key(4), (1, 5, 3))
k2 = jax.random.normal(jax.random.key(5), (1, 5, 3))
o2 = {fn}(q2, k2, v2)
assert (o2 >= v2.min(axis=1, keepdims=True) - 1e-3).all(), 'Output below the v range'
assert (o2 <= v2.max(axis=1, keepdims=True) + 1e-3).all(), 'Output above the v range'
""",
        },
        {
            "name": "Normalisation: constant v is reproduced exactly",
            "code": """
import jax
import jax.numpy as jnp

# Every value row identical => any correctly normalised weighted average
# returns that row, whatever the queries and keys are.
q = jax.random.normal(jax.random.key(6), (2, 9, 4))
k = jax.random.normal(jax.random.key(7), (2, 9, 4))
v = jnp.tile(jnp.array([1.0, -2.0, 3.0]), (2, 9, 1))

out = {fn}(q, k, v)
assert jnp.allclose(out, jnp.array([1.0, -2.0, 3.0]), atol=1e-3), (
    f'With identical value rows the output must equal that row. Got {out[0, 0]}. '
    'This fails when the denominator is missing — the output then scales with T.'
)

# And the result must not drift as the sequence gets longer.
for T in (4, 32, 128):
    qq = jax.random.normal(jax.random.key(8), (1, T, 4))
    kk = jax.random.normal(jax.random.key(9), (1, T, 4))
    vv = jnp.tile(jnp.array([1.0, -2.0, 3.0]), (1, T, 1))
    o = {fn}(qq, kk, vv)
    assert jnp.allclose(o, jnp.array([1.0, -2.0, 3.0]), atol=1e-2), (
        f'T={T}: output drifted to {o[0, 0]} — the normaliser is not being applied'
    )
""",
        },
        {
            "name": "Linear, not quadratic, in sequence length",
            "code": """
import jax
import jax.numpy as jnp

# A T x T score matrix at T = 20000 would be 4e8 floats (1.6 GB) and blow up
# here. Contracting K with V first keeps the intermediate at (d, d_v).
T, d, dv = 20000, 8, 4
q = jax.random.normal(jax.random.key(10), (1, T, d))
k = jax.random.normal(jax.random.key(11), (1, T, d))
v = jax.random.normal(jax.random.key(12), (1, T, dv))

out = jax.jit({fn})(q, k, v)
out.block_until_ready()

assert out.shape == (1, T, dv), f'Expected (1, {T}, {dv}), got {out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite output at long sequence length'
""",
        },
        {
            "name": "Permuting keys and values leaves the result unchanged",
            "code": """
import jax
import jax.numpy as jnp

# Without a causal mask the summary is an order-independent sum, so shuffling
# the (k, v) pairs together must not change anything.
k_ = jax.random.split(jax.random.key(13), 3)
q = jax.random.normal(k_[0], (1, 8, 4))
k = jax.random.normal(k_[1], (1, 8, 4))
v = jax.random.normal(k_[2], (1, 8, 3))

perm = jnp.array([5, 0, 7, 2, 1, 6, 3, 4])
a = {fn}(q, k, v)
b = {fn}(q, k[:, perm], v[:, perm])

assert jnp.allclose(a, b, atol=1e-4), (
    f'Shuffling key/value pairs together changed the output by '
    f'{float(jnp.abs(a - b).max()):.6f} — the K-V contraction should be a plain sum'
)
""",
        },
        {
            "name": "Gradients, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

k_ = jax.random.split(jax.random.key(14), 3)
q = jax.random.normal(k_[0], (2, 6, 4))
k = jax.random.normal(k_[1], (2, 6, 4))
v = jax.random.normal(k_[2], (2, 6, 3))

assert jnp.allclose(jax.jit({fn})(q, k, v), {fn}(q, k, v), atol=1e-5), 'jit changes the result'

g = jax.grad(lambda a, b, c: jnp.sum({fn}(a, b, c) ** 2))(q, k, v)
assert g.shape == q.shape, f'Gradient shape {g.shape} vs {q.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'
assert jnp.abs(g).max() > 1e-8, 'Zero gradient w.r.t. q'

per_example = jax.vmap({fn})(q, k, v)
assert jnp.allclose(per_example, {fn}(q, k, v), atol=1e-4), (
    'vmap over the batch axis should agree with the batched call'
)
""",
        },
    ],
}
