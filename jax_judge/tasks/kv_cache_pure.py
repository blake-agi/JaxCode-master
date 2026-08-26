"""Problem 14 without Flax — the cache was never state to begin with."""

_LINEAR = '''class Linear:
    """Given to you, exactly as nnx.Linear is given to you in problem 14."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

TASK = {
    "title": "KV Cache Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_28",
    "difficulty": "Hard",
    "function_name": "KVCacheAttention",
    "hint": (
        "Same four projections as b_26. The cache is a VALUE: take (k, v) in, "
        "concatenate the new keys/values along axis -2, and hand the extended "
        "pair back. The causal mask is the offset one — "
        "jnp.tril(jnp.ones((seq_new, seq_total), dtype=bool), k=seq_total - "
        "seq_new). Plain tril hides the whole cache and raises nothing."
    ),
    "description": r"""
Problem 14 with no Flax — and a cache that was always a value, not state.

### Signature
```python
class KVCacheAttention:
    def __init__(self, d_model, num_heads, *, key): ...
    def __call__(self, x, cache=None): ...     # -> (out, (k_all, v_all))
```

| | shape |
|---|---|
| `x` | `(B, seq_new, d_model)` |
| `cache` | `None`, or `(k, v)` each `(B, H, seq_past, d_k)` |
| `out` | `(B, seq_new, d_model)` |
| returned cache | `(k, v)` each `(B, H, seq_past + seq_new, d_k)` |

Same four projections as `b_26`, from `jax.random.split(key, 4)`.

### The cache never needed a module
In problem 14 it already had to be returned rather than mutated, because JAX
arrays are immutable — so `nnx.Module` was buying you nothing there. Written as
a plain class that becomes obvious: the cache goes in as an argument and comes
back as a return value.

### The mask is the hard part, and it is silent
Query `i` of this chunk sits at absolute position `seq_past + i`, so it may
attend to keys `0 … seq_past + i`:

$$j - i \le \text{seq\_past}
\quad\Longrightarrow\quad
\texttt{tril(ones((seq\_new, seq\_total)), k=seq\_total - seq\_new)}$$

Forget the `k=` and you get the top-left triangle, which hides every cached
key. Nothing errors — the model just stops seeing its own history. During
single-token decode `seq_new == 1` and the mask is all-True, so the bug only
appears when you prefill more than one token at a time.

### The test that catches everything
Run a sequence two ways — all at once, versus prefill-then-decode-one-at-a-time
— and require they agree. A wrong mask offset, a wrong concat axis, or
re-projecting the cache all fail it.

### Why this exists alongside problem 14
Interview sandboxes often ship `jax` alone. Same class name, same arguments,
same `W_q`/`W_k`/`W_v`/`W_o` attributes as the `nnx` version, so practising it
reinforces problem 14 rather than competing with it. `Linear` is handed to you
the way `nnx.Linear` is.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class KVCacheAttention:
    """Causal self-attention with an incremental key/value cache."""

    def __init__(self, d_model, num_heads, *, key):
        pass  # Replace this

    def __call__(self, x, cache=None):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class KVCacheAttention:
    def __init__(self, d_model, num_heads, *, key):
        self.h = num_heads
        self.d_k = d_model // num_heads
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.W_q = Linear(d_model, d_model, key=kq)
        self.W_k = Linear(d_model, d_model, key=kk)
        self.W_v = Linear(d_model, d_model, key=kv)
        self.W_o = Linear(d_model, d_model, key=ko)

    def __call__(self, x, cache=None):
        split = lambda t: t.reshape(*t.shape[:-1], self.h, self.d_k).swapaxes(-3, -2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))

        # A value we extend, not state we mutate. The cached keys are already
        # projected, so they go in untouched.
        if cache is not None:
            k = jnp.concatenate([cache[0], k], axis=-2)
            v = jnp.concatenate([cache[1], v], axis=-2)

        s = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, q.dtype)
        )
        seq_new, seq_total = s.shape[-2], s.shape[-1]
        # Query i is at absolute position seq_past + i, so j - i <= seq_past.
        # Plain tril would hide the entire cache.
        allowed = jnp.tril(
            jnp.ones((seq_new, seq_total), dtype=bool), k=seq_total - seq_new
        )
        o = jnp.einsum(
            "...hqk,...hkd->...hqd", jax.nn.softmax(jnp.where(allowed, s, -jnp.inf), axis=-1), v
        )

        o = o.swapaxes(-3, -2)
        return self.W_o(o.reshape(*o.shape[:-2], self.h * self.d_k)), (k, v)
''',
    "demo": '''import jax
import jax.numpy as jnp

attn = KVCacheAttention(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 8, 8))

full, _ = attn(x)

out, cache = attn(x[:, :5])
print("prefill 5:", out.shape, " cache:", cache[0].shape)
pieces = [out]
for t in range(5, 8):
    out, cache = attn(x[:, t:t + 1], cache)
    print(f"  decode {t}: cache grew to {cache[0].shape[-2]}")
    pieces.append(out)

step = jnp.concatenate(pieces, axis=1)
print("\\nstepwise == all-at-once?", bool(jnp.allclose(step, full, atol=1e-5)),
      f"(max diff {float(jnp.max(jnp.abs(step - full))):.2e})")
''',
    "tests": [
        {
            "name": "Sub-layers, output shapes and cache growth",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert hasattr(m, name), f'missing sub-layer {name}'
    assert getattr(m, name).kernel.shape == (8, 8), f'{name}.kernel wrong shape'

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
out, cache = m(x)
assert out.shape == (2, 5, 8), f'{out.shape} vs (2, 5, 8)'
assert isinstance(cache, tuple) and len(cache) == 2, 'return (out, (k, v))'
assert cache[0].shape == (2, 2, 5, 4), (
    f'cached k {cache[0].shape} vs (B, H, seq, d_k) = (2, 2, 5, 4)'
)

tok = jax.random.normal(jax.random.key(2), (2, 1, 8))
out2, cache2 = m(tok, cache)
assert out2.shape == (2, 1, 8), f'{out2.shape}'
assert cache2[0].shape == (2, 2, 6, 4), (
    f'cache should grow to 6 along axis -2, got {cache2[0].shape}'
)
""",
        },
        {
            "name": "Stepwise decode equals the one-shot run",
            "code": """
import jax
import jax.numpy as jnp

T = 8
m = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, T, 8))

full, _ = m(x)
out, cache = m(x[:, :5])
pieces = [out]
for t in range(5, T):
    out, cache = m(x[:, t:t + 1], cache)
    pieces.append(out)
step = jnp.concatenate(pieces, axis=1)

assert cache[0].shape[-2] == T, f'cache ended at {cache[0].shape[-2]}, expected {T}'
assert jnp.allclose(step, full, atol=1e-4), (
    f'stepwise decode differs from the one-shot run by up to '
    f'{float(jnp.max(jnp.abs(step - full))):.3e}. Usual causes: the causal mask '
    'is missing its k=seq_total-seq_new offset, or the cache is concatenated on '
    'the wrong axis.'
)
""",
        },
        {
            "name": "The causal mask is offset, not top-left",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 6, 8))

# Prefill 4, then feed 2 more. Those queries sit at absolute positions 4 and 5,
# so both must see the whole 4-token history.
_, cache = m(x[:, :4])
out2, _ = m(x[:, 4:6], cache)
full, _ = m(x)
assert jnp.allclose(out2, full[:, 4:6], atol=1e-4), (
    'a 2-token chunk on top of a 4-token cache disagrees with the one-shot run. '
    'A plain tril() would give query 4 only key 0 and query 5 only keys 0-1, '
    'hiding the cache entirely — and it raises nothing.'
)

# Causality still holds inside the chunk.
x_alt = x.at[:, 5].add(50.0)
_, c2 = m(x_alt[:, :4])
alt, _ = m(x_alt[:, 4:6], c2)
assert jnp.allclose(alt[:, 0], out2[:, 0], atol=1e-4), (
    'changing token 5 changed the output at position 4 — a query is seeing its future'
)
""",
        },
        {
            "name": "Single-token decode, and the cache is not re-projected",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 4, 8))
_, c4 = m(x)

tok = jax.random.normal(jax.random.key(2), (1, 1, 8))
out, c5 = m(tok, c4)
assert out.shape == (1, 1, 8), f'{out.shape}'
assert jnp.isfinite(out).all(), 'non-finite output for a single-token decode'

assert jnp.array_equal(c5[0][:, :, :4], c4[0]), (
    'the first 4 cached keys changed. They are already projected — concatenate '
    'them as they are rather than pushing them through W_k again.'
)
assert jnp.array_equal(c5[1][:, :, :4], c4[1]), 'cached values changed'
""",
        },
        {
            "name": "Gradient w.r.t. the input, and jit",
            "code": """
import jax
import jax.numpy as jnp

m = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 4, 8))
out, _ = m(x)

g = jax.grad(lambda v: jnp.sum(m(v)[0]))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. the input'

assert jnp.allclose(jax.jit(lambda v: m(v)[0])(x), out, atol=1e-5), 'jit disagrees'
""",
        },
    ],
}
