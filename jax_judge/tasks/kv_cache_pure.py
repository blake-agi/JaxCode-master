"""Problem 14 without Flax — the cache was never state to begin with."""

TASK = {
    "title": "KV Cache Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_28",
    "difficulty": "Hard",
    "function_name": "init_kv_attention",
    "extra_names": ["apply_kv_attention"],
    "hint": (
        "Same four-projection pytree as b_26. The cache is a VALUE you take in "
        "and hand back — (k, v) each (B, H, seq_total, d_k) — so concatenate "
        "along axis -2 and return the extended pair. seq_past = 0 when cache "
        "is None, and the causal mask is the offset one: "
        "jnp.tril(jnp.ones((seq_new, seq_total), dtype=bool), k=seq_total - "
        "seq_new). Plain tril hides the whole cache and does it silently."
    ),
    "description": r"""
Problem 14 with an explicit parameter pytree — and a cache that was always a
value, not state.

### Signature
```python
def init_kv_attention(key, d_model, num_heads):
    ...   # -> the same four-projection pytree as b_26

def apply_kv_attention(params, x, num_heads, cache=None):
    ...   # -> (out, (k_all, v_all))
```

| | shape |
|---|---|
| `x` | `(B, seq_new, d_model)` |
| `cache` | `None`, or `(k, v)` each `(B, H, seq_past, d_k)` |
| `out` | `(B, seq_new, d_model)` |
| returned cache | `(k, v)` each `(B, H, seq_past + seq_new, d_k)` |

### Two things are now explicit that a module hid
**The parameters.** Same pytree as `b_26`: four `(d_model, d_model)` kernels,
key split four ways.

**The cache.** In problem 14 it already had to be returned rather than mutated,
because JAX arrays are immutable — so `nnx.Module` was buying you nothing there.
Written as a plain function that becomes obvious: the cache goes in as an
argument and comes back as a return value, exactly like the parameters.

### The mask is the hard part, and it is silent
Query `i` of this chunk sits at absolute position `seq_past + i`, so it may
attend to keys `0 … seq_past + i`:

$$j - i \le \text{seq\_past} \quad\Longrightarrow\quad
\texttt{tril(ones((seq\_new, seq\_total)), k=seq\_total - seq\_new)}$$

Forget the `k=` and you get the top-left triangle, which hides every cached
key. Nothing errors — the model just stops seeing its own history.

During single-token decode `seq_new == 1`, the mask is all-True and does
nothing; the bug only shows up when you prefill more than one token at a time.

### The test that catches everything
Run a sequence two ways — all at once, versus prefill-then-decode-one-at-a-time —
and require they agree. Wrong mask offset, wrong concat axis, or re-projecting
the cache all fail it.
""",
    "stub": '''import jax
import jax.numpy as jnp


def init_kv_attention(key, d_model, num_heads):
    """Parameter pytree: W_q, W_k, W_v, W_o, each a kernel and a bias."""
    pass  # Replace this


def apply_kv_attention(params, x, num_heads, cache=None):
    """(B, seq_new, d_model) + optional (k, v) -> (out, (k_all, v_all))."""
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def init_kv_attention(key, d_model, num_heads):
    keys = jax.random.split(key, 4)
    return {
        name: {
            "kernel": jax.random.normal(k, (d_model, d_model)) / jnp.sqrt(d_model),
            "bias": jnp.zeros((d_model,)),
        }
        for name, k in zip(("W_q", "W_k", "W_v", "W_o"), keys)
    }


def _dense(p, x):
    return x @ p["kernel"] + p["bias"]


def apply_kv_attention(params, x, num_heads, cache=None):
    d_model = x.shape[-1]
    d_k = d_model // num_heads

    def heads(t):
        return t.reshape(*t.shape[:-1], num_heads, d_k).swapaxes(-3, -2)

    q = heads(_dense(params["W_q"], x))
    k = heads(_dense(params["W_k"], x))
    v = heads(_dense(params["W_v"], x))

    # The cache is a value we extend, not state we mutate.
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=-2)
        v = jnp.concatenate([cache[1], v], axis=-2)
    new_cache = (k, v)

    seq_new, seq_total = q.shape[-2], k.shape[-2]
    scores = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
        jnp.asarray(d_k, x.dtype)
    )
    # Query i is at absolute position seq_past + i, so allowed is
    # j - i <= seq_past. Plain tril would hide the entire cache.
    allowed = jnp.tril(
        jnp.ones((seq_new, seq_total), dtype=bool), k=seq_total - seq_new
    )
    scores = jnp.where(allowed, scores, -jnp.inf)

    o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(scores, axis=-1), v)
    o = o.swapaxes(-3, -2)
    o = o.reshape(*o.shape[:-2], d_model)
    return _dense(params["W_o"], o), new_cache
''',
    "demo": '''import jax
import jax.numpy as jnp

params = init_kv_attention(jax.random.key(0), d_model=8, num_heads=2)
x = jax.random.normal(jax.random.key(1), (1, 8, 8))

full, _ = apply_kv_attention(params, x, 2)

out, cache = apply_kv_attention(params, x[:, :5], 2)
print("prefill 5:", out.shape, "cache:", cache[0].shape)
pieces = [out]
for t in range(5, 8):
    out, cache = apply_kv_attention(params, x[:, t:t + 1], 2, cache)
    print(f"  decode {t}: cache grew to {cache[0].shape[-2]}")
    pieces.append(out)

step = jnp.concatenate(pieces, axis=1)
print("\\nstepwise == all-at-once?", bool(jnp.allclose(step, full, atol=1e-5)),
      f"(max diff {float(jnp.max(jnp.abs(step - full))):.2e})")
''',
    "tests": [
        {
            "name": "Pytree, output shapes and cache growth",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
assert set(p) == {'W_q', 'W_k', 'W_v', 'W_o'}, f'keys {sorted(p)}'
assert len(jax.tree.leaves(p)) == 8, f'{len(jax.tree.leaves(p))} leaves, expected 8'

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
out, cache = apply_kv_attention(p, x, 2)
assert out.shape == (2, 5, 8), f'{out.shape} vs (2, 5, 8)'
assert isinstance(cache, tuple) and len(cache) == 2, 'Return (out, (k, v))'
assert cache[0].shape == (2, 2, 5, 4), (
    f'cached k {cache[0].shape} vs (B, H, seq, d_k) = (2, 2, 5, 4)'
)

tok = jax.random.normal(jax.random.key(2), (2, 1, 8))
out2, cache2 = apply_kv_attention(p, tok, 2, cache)
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
p = {fn}(jax.random.key(0), 8, 2)
x = jax.random.normal(jax.random.key(1), (2, T, 8))

full, _ = apply_kv_attention(p, x, 2)
out, cache = apply_kv_attention(p, x[:, :5], 2)
pieces = [out]
for t in range(5, T):
    out, cache = apply_kv_attention(p, x[:, t:t + 1], 2, cache)
    pieces.append(out)
step = jnp.concatenate(pieces, axis=1)

assert cache[0].shape[-2] == T, f'cache ended at {cache[0].shape[-2]}, expected {T}'
assert jnp.allclose(step, full, atol=1e-4), (
    f'Stepwise decode differs from the one-shot run by up to '
    f'{float(jnp.max(jnp.abs(step - full))):.3e}. Usual causes: the causal mask '
    'is missing its k=seq_total-seq_new offset, or the cache is concatenated '
    'on the wrong axis.'
)
""",
        },
        {
            "name": "The causal mask is offset, not top-left",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
x = jax.random.normal(jax.random.key(1), (1, 6, 8))

# Prefill 4, then feed 2 more. Those 2 queries sit at absolute positions 4 and
# 5, so both must see the whole 4-token history.
_, cache = apply_kv_attention(p, x[:, :4], 2)
out2, _ = apply_kv_attention(p, x[:, 4:6], 2, cache)
full, _ = apply_kv_attention(p, x, 2)
assert jnp.allclose(out2, full[:, 4:6], atol=1e-4), (
    'A 2-token chunk on top of a 4-token cache disagrees with the one-shot '
    'run. A plain tril() here would give query 4 only key 0 and query 5 only '
    'keys 0-1, hiding the cache entirely — and it raises nothing.'
)

# Causality still holds inside the chunk: query 4 must not see position 5.
x_alt = x.at[:, 5].add(50.0)
_, c2 = apply_kv_attention(p, x_alt[:, :4], 2)
alt, _ = apply_kv_attention(p, x_alt[:, 4:6], 2, c2)
assert jnp.allclose(alt[:, 0], out2[:, 0], atol=1e-4), (
    'Changing token 5 changed the output at position 4 — the mask lets a query '
    'see its own future'
)
""",
        },
        {
            "name": "Single-token decode, and the cache is not re-projected",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
x = jax.random.normal(jax.random.key(1), (1, 4, 8))
_, c4 = apply_kv_attention(p, x, 2)

tok = jax.random.normal(jax.random.key(2), (1, 1, 8))
out, c5 = apply_kv_attention(p, tok, 2, c4)
assert out.shape == (1, 1, 8), f'{out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite output for a single-token decode'

assert jnp.array_equal(c5[0][:, :, :4], c4[0]), (
    'The first 4 cached keys changed. They are already projected — concatenate '
    'them as they are rather than pushing them through W_k again.'
)
assert jnp.array_equal(c5[1][:, :, :4], c4[1]), 'Cached values changed'
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp

p = {fn}(jax.random.key(0), 8, 2)
x = jax.random.normal(jax.random.key(1), (2, 4, 8))

g = jax.grad(lambda q: jnp.sum(apply_kv_attention(q, x, 2)[0]))(p)
assert len(jax.tree.leaves(g)) == 8, f'{len(jax.tree.leaves(g))} grad leaves, expected 8'
assert all(jnp.isfinite(l).all() for l in jax.tree.leaves(g)), 'Non-finite gradient'
for name in ('W_q', 'W_k', 'W_v', 'W_o'):
    assert float(jnp.abs(g[name]['kernel']).max()) > 0, f'{name} got no gradient'

out, _ = apply_kv_attention(p, x, 2)
jf = jax.jit(apply_kv_attention, static_argnums=(2,))
assert jnp.allclose(jf(p, x, 2)[0], out, atol=1e-5), 'jit disagrees'
""",
        },
        {
            "name": "No Flax anywhere",
            "code": """
import sys

assert 'flax' not in sys.modules, (
    'flax got imported — and note the cache never needed a module anyway, '
    'since JAX arrays cannot be mutated in place'
)
""",
        },
    ],
}
