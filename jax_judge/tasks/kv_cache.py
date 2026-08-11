"""KV-cache attention — the cache as a value threaded through, not hidden state."""

TASK = {
    "title": "KV Cache Attention",
    "category": "Attention & Transformers",
    "number": "14",
    "difficulty": "Hard",
    "function_name": "KVCacheAttention",
    "hint": (
        "Project q/k/v from the NEW tokens only, then concatenate the cached k/v "
        "onto the front along the sequence axis before computing scores. Queries "
        "are always just the new positions, keys and values are the whole "
        "history — so the score matrix is (seq_new, seq_total), not square. During "
        "prefill (seq_new > 1) you still need a causal mask, and it has to be "
        "offset by seq_past = seq_total - seq_new. Single-token decode needs no mask "
        "at all: one query can see the entire past legitimately."
    ),
    "description": r"""
Implement multi-head attention with a **KV cache** for incremental decoding.

### Signature
```python
class KVCacheAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs): ...
    def __call__(self, x, cache=None): ...   # -> (out, new_cache)
```

- `x`: `(B, seq_new, d_model)` — only the **new** tokens
- `cache`: `None`, or `(k, v)` each `(B, H, seq_past, d_k)`
- returns `out` of shape `(B, seq_new, d_model)` and the updated `(k, v)`

### Requirements
- Four `nnx.Linear(d_model, d_model)` layers: `W_q`, `W_k`, `W_v`, `W_o`
- Concatenate the cached `k`/`v` along the **sequence** axis
- Scale by $1/\sqrt{d_k}$
- Causal mask when `seq_new > 1`, offset by `seq_past`

`nnx.Linear` is an allowed building block — the exercise is the cache, not the
projection.

### Why the cache exists
Generating token $n$ re-attends over all $n-1$ previous tokens. Without a cache
you recompute every past key and value at every step, making generation
$O(n^2)$ per token and $O(n^3)$ overall. With a cache each step projects only
the new token and appends, so generation is $O(n)$ per step.

### The mask offset — the part people get wrong
During **decode** (`seq_new == 1`) there is nothing to mask: the single query is
the newest position and may see everything before it.

During **prefill** (`seq_new > 1`) the score matrix is `(seq_new, seq_total)` and is
**not square**. Query $i$ sits at absolute position $S_{past} + i$, so it may
attend to key $j$ only when $j \le S_{past} + i$. That is a triangular mask
shifted right by `seq_past` — using an unshifted `triu` silently blocks the
cached history and is the classic bug here.

### The memory arithmetic
Cache size is
$2 \times L \times B \times H_{kv} \times seq \times d_k \times \text{bytes}$
— the leading 2 is K and V, and $L$ is the layer count, which is easy to drop
and worth a factor of 80.

For Llama-3-70B in bf16 ($L=80$, $H_{kv}=8$, $d_k=128$, 2 bytes) that is
$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes = **320 KiB per
token** — roughly 10 GiB at 32k context, per sequence. At batch 32 the
cache dwarfs the weights, which is why GQA, int8 KV and PagedAttention all
exist.

### A JAX note on this design
The cache here is a **value** passed in and returned, not mutable state hidden
in the module — which is exactly how JAX prefers it. The cost is that `k` grows
by one each step, so a jitted decode loop retraces on every new shape.
Production code preallocates a fixed-size buffer and writes into it with
`jax.lax.dynamic_update_slice`, trading wasted flops on empty slots for a
single compilation. This task keeps the concat version to match the original;
the preallocated variant is the natural follow-up question.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class KVCacheAttention(nnx.Module):
    """Multi-head attention with an incremental key/value cache."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x, cache=None):
        """(B, seq_new, d_model) + optional (k, v) -> (out, (k, v))"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class KVCacheAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def _heads(self, t, B, seq):
        # (B, seq, d_model) -> (B, H, seq, d_k)
        return t.reshape(B, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, x, cache=None):
        B, seq_new, _ = x.shape

        q = self._heads(self.W_q(x), B, seq_new)
        k = self._heads(self.W_k(x), B, seq_new)
        v = self._heads(self.W_v(x), B, seq_new)

        # The cache is a VALUE we extend, not mutable state.
        if cache is not None:
            k = jnp.concatenate([cache[0], k], axis=2)
            v = jnp.concatenate([cache[1], v], axis=2)

        new_cache = (k, v)
        seq_total = k.shape[2]

        # == q @ jnp.swapaxes(k, -1, -2), written as a contraction.
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, x.dtype)
        )

        if seq_new > 1:
            # Query i is at absolute position seq_past + i, so the triangle is
            # shifted right by seq_past. Unshifted triu would hide the cache.
            seq_past = seq_total - seq_new
            blocked = jnp.triu(
                jnp.ones((seq_new, seq_total), dtype=bool), k=seq_past + 1
            )
            scores = jnp.where(blocked, -jnp.inf, scores)

        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhts,bhsd->bhtd", weights, v)   # == weights @ v

        merged = attn.transpose(0, 2, 1, 3).reshape(B, seq_new, -1)
        return self.W_o(merged), new_cache
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

m = KVCacheAttention(32, 4, rngs=nnx.Rngs(params=0))

# Prefill 5 tokens, then decode 3 more one at a time.
x = jax.random.normal(jax.random.key(1), (1, 5, 32))
out, cache = m(x)
print("prefill:", out.shape, "cache k:", cache[0].shape)

for step in range(3):
    tok = jax.random.normal(jax.random.key(10 + step), (1, 1, 32))
    out, cache = m(tok, cache)
    print(f"  step {step}: out {out.shape}  cache grew to {cache[0].shape[2]}")
''',
    "tests": [
        {
            "name": "Shapes, sub-layers, and cache growth",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 4, rngs=nnx.Rngs(params=0))

for name in ("W_q", "W_k", "W_v", "W_o"):
    assert hasattr(m, name), f'Missing self.{name}'
    assert isinstance(getattr(m, name), nnx.Linear), (
        f'self.{name} must be an nnx.Linear, got {type(getattr(m, name))}'
    )

x = jax.random.normal(jax.random.key(1), (2, 5, 32))
out, cache = m(x)
assert out.shape == (2, 5, 32), f'Output shape {out.shape} vs (2, 5, 32)'
assert cache[0].shape == (2, 4, 5, 8), f'cache k shape {cache[0].shape} vs (2, 4, 5, 8)'
assert cache[1].shape == (2, 4, 5, 8), f'cache v shape {cache[1].shape}'

tok = jax.random.normal(jax.random.key(2), (2, 1, 32))
out2, cache2 = m(tok, cache)
assert out2.shape == (2, 1, 32), f'Decode output {out2.shape}'
assert cache2[0].shape == (2, 4, 6, 8), (
    f'Cache should grow by 1 to (2, 4, 6, 8), got {cache2[0].shape}'
)
""",
        },
        {
            "name": "Cached decode equals full recomputation",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# The whole point: incremental decoding must be numerically identical to
# re-running attention over the whole prefix.
m = {fn}(16, 2, rngs=nnx.Rngs(params=3))
full = jax.random.normal(jax.random.key(4), (1, 6, 16))

ref, _ = m(full)                       # one shot over all 6 tokens

out, cache = m(full[:, :3])            # prefill 3
outs = [out]
for i in range(3, 6):
    o, cache = m(full[:, i:i+1], cache)
    outs.append(o)
inc = jnp.concatenate(outs, axis=1)

assert inc.shape == ref.shape, f'{inc.shape} vs {ref.shape}'
assert jnp.allclose(inc, ref, atol=1e-4), (
    'Incremental decoding disagrees with the full forward pass. The usual cause '
    'is the causal mask offset during prefill.'
)
""",
        },
        {
            "name": "Prefill mask is offset by seq_past",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=5))
full = jax.random.normal(jax.random.key(6), (1, 8, 16))
ref, _ = m(full)

# Prefill 4, then prefill 4 MORE at once (seq_new=4 with seq_past=4). An unshifted
# triangular mask would wrongly hide the first four cached positions.
out1, cache = m(full[:, :4])
out2, _ = m(full[:, 4:], cache)
joined = jnp.concatenate([out1, out2], axis=1)

assert jnp.allclose(joined, ref, atol=1e-4), (
    'Chunked prefill disagrees with the one-shot pass. The mask for a '
    '(seq_new, seq_total) score matrix must start its diagonal at seq_past + 1.'
)
""",
        },
        {
            "name": "Causality: a token cannot see the future",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=7))
x = jax.random.normal(jax.random.key(8), (1, 6, 16))
base, _ = m(x)

# Perturb the LAST token; earlier outputs must be untouched.
x2 = x.at[:, 5].add(100.0)
pert, _ = m(x2)

assert jnp.allclose(base[:, :5], pert[:, :5], atol=1e-4), (
    'Changing the last token altered earlier outputs — prefill is not causal'
)
assert not jnp.allclose(base[:, 5], pert[:, 5], atol=1e-3), 'Last output should change'
""",
        },
        {
            "name": "Single-token decode needs no mask",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=9))
x = jax.random.normal(jax.random.key(10), (1, 4, 16))
_, cache = m(x)

tok = jax.random.normal(jax.random.key(11), (1, 1, 16))
out, _ = m(tok, cache)

assert jnp.isfinite(out).all(), (
    'Non-finite output on single-token decode — a mask was applied when seq_new=1, '
    'and masking the only query row leaves softmax with all -inf'
)
""",
        },
        {
            "name": "Scaled by 1/sqrt(d_k)",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 1, rngs=nnx.Rngs(params=12))
# Identity projections so we can predict the scores exactly.
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(8)
    lin.bias[...] = jnp.zeros(8)

x = jnp.zeros((1, 2, 8)).at[0, 0, 0].set(1.0).at[0, 1, 0].set(1.0)
out, _ = m(x)

scores = jnp.array([[1.0 / jnp.sqrt(8.0), 1.0 / jnp.sqrt(8.0)]])
w = jax.nn.softmax(scores, axis=-1)
expected_row1 = w[0, 0] * x[0, 0] + w[0, 1] * x[0, 1]
assert jnp.allclose(out[0, 1], expected_row1, atol=1e-4), (
    f'Row 1 does not match attention scaled by 1/sqrt(d_k)=1/sqrt(8). '
    f'Got {out[0, 1][:3]}, expected {expected_row1[:3]}'
)
""",
        },
        {
            "name": "Gradients reach all four projections",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 2, rngs=nnx.Rngs(params=13))
x = jax.random.normal(jax.random.key(14), (1, 4, 16))

grads = nnx.grad(lambda mod: jnp.sum(mod(x)[0] ** 2))(m)
state = nnx.state(grads)
for name in ("W_q", "W_k", "W_v", "W_o"):
    k = state[name]["kernel"]
    val = k[...] if isinstance(k, nnx.Variable) else k
    assert jnp.isfinite(val).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(val).sum()) > 0, f'No gradient reached {name}'
""",
        },
    ],
}
