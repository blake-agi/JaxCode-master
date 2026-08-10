"""Preallocated KV cache — static shapes, in-place writes, mutable nnx state."""

TASK = {
    "title": "KV Cache for Incremental Decoding (nnx.Module)",
    "category": "Attention & Transformers",
    "order": 10,
    "difficulty": "Medium",
    "function_name": "KVCache",
    "hint": (
        "Allocate the full (B, H, max_len, Dh) buffers once in __init__ with "
        "jnp.zeros and wrap them in nnx.Cache (any non-Param nnx.Variable works) — "
        "plus a scalar int32 write position. In update(), the number of new tokens "
        "S = k_new.shape[2] is static but the write offset is a traced value, which "
        "is exactly what jax.lax.dynamic_update_slice_in_dim(buf, k_new, pos, axis=2) "
        "is for. Then advance pos by S and build the validity mask with "
        "jnp.arange(max_len) < pos. Never jnp.concatenate onto the buffer and never "
        "slice it with [:, :, :pos] — both make the output shape depend on a traced "
        "value, which jit cannot express."
    ),
    "description": r"""
Implement a **preallocated key/value cache** for autoregressive decoding, as an
`nnx.Module` that owns mutable state.

### Rules
- Signature: `KVCache(batch_size, num_heads, max_len, head_dim, *, rngs=None)`
- `self.k_cache`, `self.v_cache`: `(batch_size, num_heads, max_len, head_dim)` zeros,
  wrapped in `nnx.Cache` (or any `nnx.Variable` that is **not** `nnx.Param` — the
  optimizer must never see them)
- `self.pos`: a scalar `int32` write position, starting at `0`, also cached state
- `update(k_new, v_new)` takes `(B, H, S, Dh)` — `S = 1` for a decode step, `S > 1`
  for prefill — writes them at `pos`, advances `pos` by `S`, and returns
  `(keys, values, mask)`
- `keys`/`values` are always the **full** `(B, H, max_len, Dh)` buffers; `mask` is
  a `(max_len,)` boolean marking the written positions
- Write with `jax.lax.dynamic_update_slice_in_dim`. No `jnp.concatenate`, no
  Python-level growth, no `nnx.MultiHeadAttention`

### Why the return value is a mask and not a slice
"Return the valid slice" is the obvious API and it is unimplementable under
`jit`. In JAX a shape is part of the type, so `keys[:, :, :self.pos]` with a
traced `pos` is a shape that depends on a value — an error, not a slow path. The
jit-compatible spelling of "the first `pos` entries" is a fixed-size buffer plus
a boolean mask, which the caller folds into the scores:

```python
scores = jnp.where(mask, scores, -jnp.inf)   # then softmax
```

The masked-out slots are still multiplied, so you pay for `max_len` columns from
step one. That is the deliberate trade: a *constant* amount of wasted flops in
exchange for one compilation. Concatenating instead gives `(B, H, 1, Dh)`,
`(B, H, 2, Dh)`, ... — a **new shape, therefore a new XLA program, on every
decode step**. Generating 4,096 tokens means 4,096 compilations, each of which
costs far more than the whole forward pass. Preallocation is not an optimization
here; it is the difference between working and not.

The buffers are `nnx.Cache` rather than `nnx.Param` for the same reason
BatchNorm's running stats are `nnx.BatchStat`: `nnx.state(model, nnx.Param)`
must hand the optimizer parameters only. And because a write mutates that state,
`jax.grad` over `update` raises `TraceContextError` — use `nnx.grad`, with
`argnums=` to differentiate w.r.t. the incoming tensors.

### The memory arithmetic
A cache holds two tensors per layer for every token generated so far:

$$\text{bytes} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot \text{sizeof(dtype)}$$

Llama-3-70B in bf16: $L = 80$, $H_{kv} = 8$ (grouped-query), $d_h = 128$, so
$2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 327{,}680$ bytes $= 320$ KiB **per
token**. At 32k context that is ~10.7 GB for a *single* sequence; a batch of 32
needs ~344 GB, well past the 140 GB the weights themselves occupy. Past a few
thousand tokens the cache, not the model, is what caps your batch size — and
batch size is throughput. This is why GQA exists
(64 query heads sharing 8 KV heads cuts that 320 KiB by 8x), why people quantize
the cache to int8, and why vLLM's PagedAttention allocates fixed-size blocks
instead of `max_len` per sequence: preallocating for the worst case wastes
everything a short sequence never uses.

The cache also changes the compute asymptotics. Re-running full attention over
the whole prefix at every step costs $O(T^2)$ per step and $O(T^3)$ per
sequence; with a cache each step is $O(T)$ and the sequence is $O(T^2)$.

### Gotcha
`dynamic_update_slice` **clamps** an out-of-range start index instead of raising.
Write past `max_len` and it silently overwrites the last slots — no error, just
wrong answers. Bound the write position yourself in production code.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class KVCache(nnx.Module):
    """Fixed-size key/value buffers plus a write cursor."""

    def __init__(self, batch_size: int, num_heads: int, max_len: int,
                 head_dim: int, *, rngs: nnx.Rngs = None):
        pass  # Replace this

    def update(self, k_new, v_new):
        """Write S new tokens and return the whole cache.

        Args:
            k_new, v_new: (B, H, S, Dh)

        Returns:
            keys:   (B, H, max_len, Dh)
            values: (B, H, max_len, Dh)
            mask:   (max_len,) bool — True for positions written so far
        """
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class KVCache(nnx.Module):
    def __init__(self, batch_size: int, num_heads: int, max_len: int,
                 head_dim: int, *, rngs: nnx.Rngs = None):
        shape = (batch_size, num_heads, max_len, head_dim)
        # nnx.Cache is a non-Param Variable: mutable state the optimizer ignores.
        self.k_cache = nnx.Cache(jnp.zeros(shape))
        self.v_cache = nnx.Cache(jnp.zeros(shape))
        self.pos = nnx.Cache(jnp.array(0, dtype=jnp.int32))
        # Static python ints — safe to use in shapes.
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim

    def update(self, k_new, v_new):
        s = k_new.shape[2]          # static: it comes from a shape
        p = self.pos[...]          # traced: it comes from state

        # In-place write at a dynamic offset, keeping the buffer shape constant.
        self.k_cache[...] = jax.lax.dynamic_update_slice_in_dim(
            self.k_cache[...], k_new, p, axis=2
        )
        self.v_cache[...] = jax.lax.dynamic_update_slice_in_dim(
            self.v_cache[...], v_new, p, axis=2
        )
        self.pos[...] = p + s

        # "The valid slice", expressed so the shape stays static.
        mask = jnp.arange(self.max_len) < self.pos[...]
        return self.k_cache[...], self.v_cache[...], mask
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

B, H, Dh, MAX = 1, 2, 4, 8
cache = KVCache(B, H, MAX, Dh)
key = jax.random.key(0)

print("pos:", cache.pos[...], " buffer:", cache.k_cache[...].shape)

# Prefill 3 tokens, then decode 2 more one at a time.
k, v = jax.random.normal(key, (2, B, H, 3, Dh))
keys, values, mask = cache.update(k, v)
print("after prefill  pos=", cache.pos[...], "mask=", mask)

for step in range(2):
    k1, v1 = jax.random.normal(jax.random.key(step + 1), (2, B, H, 1, Dh))
    keys, values, mask = cache.update(k1, v1)
    # Shape is identical every step — that is the whole point.
    print(f"after decode {step}  keys.shape={keys.shape}  valid={int(mask.sum())}")

# 320 KiB/token for Llama-3-70B in bf16:
per_tok = 2 * 80 * 8 * 128 * 2
print(f"\\nLlama-3-70B cache: {per_tok / 1024:.0f} KiB/token "
      f"-> {per_tok * 32768 / 1e9:.1f} GB at 32k context")
''',
    "tests": [
        {
            "name": "Buffers are preallocated and are not parameters",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

cache = {fn}(2, 4, 16, 8)

assert cache.k_cache[...].shape == (2, 4, 16, 8), (
    f'k_cache is {cache.k_cache[...].shape}, expected the FULL (2, 4, 16, 8) buffer '
    'allocated up front'
)
assert cache.v_cache[...].shape == (2, 4, 16, 8), f'v_cache {cache.v_cache[...].shape}'
assert jnp.allclose(cache.k_cache[...], 0.0), 'k_cache must start as zeros'
assert jnp.allclose(cache.v_cache[...], 0.0), 'v_cache must start as zeros'
assert int(cache.pos[...]) == 0, f'pos must start at 0, got {cache.pos[...]}'

for name in ('k_cache', 'v_cache', 'pos'):
    var = getattr(cache, name)
    assert isinstance(var, nnx.Variable), f'{name} must be an nnx.Variable, got {type(var)}'
    assert not isinstance(var, nnx.Param), (
        f'{name} must NOT be an nnx.Param — cache state is not learned, and '
        'nnx.state(model, nnx.Param) has to hand the optimizer parameters only'
    )

assert len(jax.tree.leaves(nnx.state(cache, nnx.Param))) == 0, (
    'The cache must contribute zero Param leaves'
)
""",
        },
        {
            "name": "Single-token writes land in the right slot",
            "code": """
import jax
import jax.numpy as jnp

B, H, Dh, MAX = 1, 2, 4, 8
cache = {fn}(B, H, MAX, Dh)
k_all = jax.random.normal(jax.random.key(0), (B, H, MAX, Dh))
v_all = jax.random.normal(jax.random.key(1), (B, H, MAX, Dh))

for t in range(3):
    keys, values, mask = cache.update(k_all[:, :, t:t + 1], v_all[:, :, t:t + 1])
    assert int(cache.pos[...]) == t + 1, f'After {t + 1} writes pos={cache.pos[...]}'
    assert mask.shape == (MAX,), f'mask shape {mask.shape} vs ({MAX},)'
    assert int(mask.sum()) == t + 1, f'mask marks {int(mask.sum())} valid, expected {t + 1}'
    assert bool(mask[t]) and not bool(mask[t + 1]), f'mask boundary wrong at t={t}: {mask}'

assert jnp.allclose(keys[:, :, :3], k_all[:, :, :3], atol=1e-6), (
    'Tokens are not in slots 0..2 — check dynamic_update_slice_in_dim on axis 2 '
    'and that pos advances by S'
)
assert jnp.allclose(values[:, :, :3], v_all[:, :, :3], atol=1e-6), 'v slots are wrong'
assert jnp.allclose(keys[:, :, 3:], 0.0), 'Unwritten slots must still be zero'
""",
        },
        {
            "name": "Shapes stay static as the cache fills",
            "code": """
import jax
import jax.numpy as jnp

B, H, Dh, MAX = 2, 3, 4, 12
cache = {fn}(B, H, MAX, Dh)
k = jax.random.normal(jax.random.key(2), (B, H, 1, Dh))
v = jax.random.normal(jax.random.key(3), (B, H, 1, Dh))

shapes = []
for _ in range(5):
    keys, values, mask = cache.update(k, v)
    shapes.append((keys.shape, values.shape, mask.shape))

assert all(s == shapes[0] for s in shapes), (
    f'Returned shapes changed across decode steps: {shapes}. A growing shape is a '
    'new XLA program every step — return the full preallocated buffer plus a mask.'
)
assert shapes[0][0] == (B, H, MAX, Dh), f'{shapes[0][0]} vs {(B, H, MAX, Dh)}'
assert int(cache.pos[...]) == 5, f'pos={cache.pos[...]} after 5 single-token writes'
""",
        },
        {
            "name": "Prefill (S > 1) then incremental decode",
            "code": """
import jax
import jax.numpy as jnp

B, H, Dh, MAX = 1, 2, 4, 10
cache = {fn}(B, H, MAX, Dh)
k_all = jax.random.normal(jax.random.key(4), (B, H, 6, Dh))
v_all = jax.random.normal(jax.random.key(5), (B, H, 6, Dh))

# Prefill 4 tokens at once.
keys, values, mask = cache.update(k_all[:, :, :4], v_all[:, :, :4])
assert int(cache.pos[...]) == 4, f'Prefill must advance pos by S=4, got {cache.pos[...]}'
assert int(mask.sum()) == 4, f'{int(mask.sum())} valid after prefill, expected 4'

# Then two single-token decode steps.
for t in (4, 5):
    keys, values, mask = cache.update(k_all[:, :, t:t + 1], v_all[:, :, t:t + 1])

assert int(cache.pos[...]) == 6, f'pos={cache.pos[...]}, expected 6'
assert jnp.allclose(keys[:, :, :6], k_all, atol=1e-6), (
    'Prefill + decode must reconstruct the full key sequence in order'
)
assert jnp.allclose(values[:, :, :6], v_all, atol=1e-6), 'Value order is wrong'
assert jnp.allclose(keys[:, :, 6:], 0.0), 'Slots past pos must remain zero'
""",
        },
        {
            "name": "Cached attention matches full attention",
            "code": """
import jax
import jax.numpy as jnp

B, H, Dh, MAX, T = 1, 2, 8, 16, 5
k_all = jax.random.normal(jax.random.key(6), (B, H, T, Dh))
v_all = jax.random.normal(jax.random.key(7), (B, H, T, Dh))
q = jax.random.normal(jax.random.key(8), (B, H, 1, Dh))
scale = 1.0 / jnp.sqrt(Dh)

cache = {fn}(B, H, MAX, Dh)
cache.update(k_all[:, :, :3], v_all[:, :, :3])          # prefill
cache.update(k_all[:, :, 3:4], v_all[:, :, 3:4])        # decode
keys, values, mask = cache.update(k_all[:, :, 4:5], v_all[:, :, 4:5])

scores = jnp.einsum('bhqd,bhkd->bhqk', q, keys) * scale
scores = jnp.where(mask, scores, -1e30)                 # mask the empty slots
out = jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(scores, axis=-1), values)

# Reference: plain attention over the true prefix, no cache involved.
ref_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_all) * scale
ref = jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(ref_scores, axis=-1), v_all)

assert out.shape == (B, H, 1, Dh), f'{out.shape}'
assert jnp.allclose(out, ref, atol=1e-5), (
    f'Cached attention disagrees with full attention: max diff '
    f'{float(jnp.max(jnp.abs(out - ref)))}. Either the tokens are in the wrong slots '
    'or the mask does not cover exactly positions 0..pos-1.'
)

# Sanity: without the mask the zero-filled slots would contribute and change the answer.
unmasked = jnp.einsum(
    'bhqk,bhkd->bhqd',
    jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, keys) * scale, axis=-1),
    values,
)
assert not jnp.allclose(unmasked, ref, atol=1e-4), (
    'Empty slots must actually matter — the mask is doing real work here'
)
""",
        },
        {
            "name": "Compiles once under nnx.jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, H, Dh, MAX = 1, 2, 4, 8
cache = {fn}(B, H, MAX, Dh)
traces = {'n': 0}

@nnx.jit
def decode_step(c, k_new, v_new):
    traces['n'] += 1          # only runs while tracing
    return c.update(k_new, v_new)

for i in range(6):
    k = jax.random.normal(jax.random.key(i), (B, H, 1, Dh))
    v = jax.random.normal(jax.random.key(100 + i), (B, H, 1, Dh))
    keys, values, mask = decode_step(cache, k, v)

assert traces['n'] == 1, (
    f'The decode step was traced {traces["n"]} times for 6 tokens — it must compile '
    'exactly once. A shape that grows with the sequence forces a recompile per step.'
)
assert int(cache.pos[...]) == 6, (
    f'pos={cache.pos[...]} after 6 jitted steps — nnx.jit must propagate the '
    'mutated cache state back out'
)
assert int(mask.sum()) == 6, f'{int(mask.sum())} valid positions, expected 6'
assert keys.shape == (B, H, MAX, Dh), f'{keys.shape}'
""",
        },
        {
            "name": "Gradients flow through the write with nnx.grad",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

B, H, Dh, MAX = 1, 2, 4, 8
cache = {fn}(B, H, MAX, Dh)
k_new = jax.random.normal(jax.random.key(9), (B, H, 3, Dh))
v_new = jax.random.normal(jax.random.key(10), (B, H, 3, Dh))

def loss(c, kn, vn):
    keys, values, mask = c.update(kn, vn)
    return jnp.sum(keys ** 2) + jnp.sum(values ** 2)

# The module mutates state, so plain jax.grad is off the table — argnums picks
# out the non-module argument.
gk = nnx.grad(loss, argnums=1)(cache, k_new, v_new)

assert gk.shape == k_new.shape, f'Gradient shape {gk.shape} vs {k_new.shape}'
assert jnp.isfinite(gk).all(), 'Non-finite gradient'
assert jnp.allclose(gk, 2.0 * k_new, atol=1e-5), (
    'd/dk_new sum(keys**2) must be exactly 2*k_new — the written slice has to be a '
    'straight copy of the input, with no scaling or reordering'
)
""",
        },
    ],
}
