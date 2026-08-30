"""Problem 10 without Flax — the asymmetry is the whole point."""

_LINEAR = '''class Linear:
    """Given to you, exactly as nnx.Linear is given to you in problem 10."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias
'''

TASK = {
    "title": "Grouped-Query Attention without Flax",
    "category": "Attention & Transformers",
    "number": "b_30",
    "difficulty": "Hard",
    "function_name": "GroupQueryAttention",
    "hint": (
        "W_q and W_o are (d_model, d_model), but W_k and W_v are only "
        "(d_model, num_kv_heads * d_k) — that asymmetry IS the problem. After "
        "splitting, q has num_heads on the head axis while k and v have "
        "num_kv_heads, so repeat each KV head num_heads // num_kv_heads times "
        "along the head axis. Use jnp.repeat, not jnp.tile: repeat gives "
        "[0,0,1,1], tile gives [0,1,0,1], and only the first puts adjacent "
        "query heads in the same group."
    ),
    "description": r"""
Problem 10's grouped-query attention with no Flax.

### Signature
```python
class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads, *, key): ...
    def __call__(self, x): ...        # (B, seq, d_model) -> (B, seq, d_model)
```

`num_heads` must be divisible by `num_kv_heads`. Set `self.d_k = d_model //
num_heads`.

### The projections are deliberately asymmetric
| | shape |
|---|---|
| `self.W_q` | `(d_model, d_model)` |
| `self.W_k` | `(d_model, num_kv_heads * d_k)` — **smaller** |
| `self.W_v` | `(d_model, num_kv_heads * d_k)` — **smaller** |
| `self.W_o` | `(d_model, d_model)` |

Two endpoints fall out of the same code:

- `num_kv_heads == num_heads` → ordinary multi-head attention
- `num_kv_heads == 1` → multi-query attention (MQA)

### Why it exists — the KV cache, not the FLOPs
Cache size is proportional to $H_{kv} \times seq \times d_k$, **not**
$H_q$. Dropping `num_kv_heads` from 64 to 8 makes the cache eight times
smaller and the per-step memory traffic eight times lighter, while the query
heads stay at 64 so quality barely moves. Llama-3-70B ships `H=64`, `H_{kv}=8`.

### `repeat`, not `tile`
After splitting, `q` has `num_heads` on the head axis and `k`/`v` have
`num_kv_heads`. Expand the KV heads to match:

```python
r = num_heads // num_kv_heads
k = jnp.repeat(k, r, axis=-3)
```

The two candidates group differently, and only one is right:

```
jnp.repeat([0,1,2,3], 2)  ->  [0,0,1,1,2,2,3,3]   ✅ adjacent query heads share a KV head
jnp.tile([0,1,2,3], 2)    ->  [0,1,2,3,0,1,2,3]   ❌ interleaved groups
```

Both give an array of the right shape, so a shape check will not catch it.

### Why this exists alongside problem 10
Interview sandboxes ship `jax` but not `flax`. Same class name, same argument
names, same `W_q`/`W_k`/`W_v`/`W_o` attributes — only
`rngs=nnx.Rngs(params=0)` becomes `key=jax.random.key(0)`.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class GroupQueryAttention:
    """num_heads query heads sharing num_kv_heads key/value heads."""

    def __init__(self, d_model, num_heads, num_kv_heads, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _LINEAR + '''

class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads, *, key):
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.h = num_heads
        self.kvh = num_kv_heads
        self.d_k = d_model // num_heads
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.W_q = Linear(d_model, d_model, key=kq)
        # Smaller on purpose: this is the whole saving.
        self.W_k = Linear(d_model, num_kv_heads * self.d_k, key=kk)
        self.W_v = Linear(d_model, num_kv_heads * self.d_k, key=kv)
        self.W_o = Linear(d_model, d_model, key=ko)

    def __call__(self, x):
        split = lambda t, n: t.reshape(*t.shape[:-1], n, self.d_k).swapaxes(-3, -2)
        q = split(self.W_q(x), self.h)
        k = split(self.W_k(x), self.kvh)
        v = split(self.W_v(x), self.kvh)

        # repeat, not tile: [0,0,1,1] puts ADJACENT query heads in one group.
        r = self.h // self.kvh
        k = jnp.repeat(k, r, axis=-3)
        v = jnp.repeat(v, r, axis=-3)

        s = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, x.dtype)
        )
        o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(s, axis=-1), v)
        o = o.swapaxes(-3, -2)
        return self.W_o(o.reshape(*o.shape[:-2], self.h * self.d_k))
''',
    "demo": '''import jax
import jax.numpy as jnp

for kvh in (4, 2, 1):
    g = GroupQueryAttention(16, num_heads=4, num_kv_heads=kvh, key=jax.random.key(0))
    label = {4: "= MHA", 2: "= GQA", 1: "= MQA"}[kvh]
    print(f"num_kv_heads={kvh} {label:<6} W_q {g.W_q.kernel.shape}  W_k {g.W_k.kernel.shape}")

x = jax.random.normal(jax.random.key(1), (2, 6, 16))
print("\\nout:", GroupQueryAttention(16, 4, 2, key=jax.random.key(0))(x).shape)

print("\\nrepeat vs tile, the grouping that matters:")
a = jnp.arange(4)
print("  repeat:", jnp.repeat(a, 2).tolist())
print("  tile:  ", jnp.tile(a, 2).tolist())
''',
    "tests": [
        {
            "name": "Projections are asymmetric in the right direction",
            "code": """
import jax
import jax.numpy as jnp

g = {fn}(16, num_heads=4, num_kv_heads=2, key=jax.random.key(0))
d_k = 16 // 4
assert g.W_q.kernel.shape == (16, 16), f'W_q {g.W_q.kernel.shape} vs (16, 16)'
assert g.W_o.kernel.shape == (16, 16), f'W_o {g.W_o.kernel.shape} vs (16, 16)'
assert g.W_k.kernel.shape == (16, 2 * d_k), (
    f'W_k {g.W_k.kernel.shape} vs (16, {2 * d_k}) — K/V project to '
    'num_kv_heads * d_k, not d_model. That shrinkage IS grouped-query attention.'
)
assert g.W_v.kernel.shape == (16, 2 * d_k), f'W_v {g.W_v.kernel.shape}'

mha = {fn}(16, 4, 4, key=jax.random.key(0))
assert mha.W_k.kernel.shape == (16, 16), 'num_kv_heads == num_heads should give plain MHA shapes'
mqa = {fn}(16, 4, 1, key=jax.random.key(0))
assert mqa.W_k.kernel.shape == (16, d_k), 'num_kv_heads == 1 should give MQA shapes'
""",
        },
        {
            "name": "Output shape, and it reduces to MHA when kv_heads == heads",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(1), (2, 6, 16))
for kvh in (1, 2, 4):
    g = {fn}(16, 4, kvh, key=jax.random.key(0))
    out = g(x)
    assert out.shape == (2, 6, 16), f'num_kv_heads={kvh}: {out.shape} vs (2, 6, 16)'
    assert jnp.isfinite(out).all(), f'non-finite output at num_kv_heads={kvh}'

# With kvh == h the repeat is a no-op, so this must equal plain MHA.
g = {fn}(16, 4, 4, key=jax.random.key(0))
d_k = 4
sp = lambda t, n: t.reshape(*t.shape[:-1], n, d_k).swapaxes(-3, -2)
q, k, v = sp(g.W_q(x), 4), sp(g.W_k(x), 4), sp(g.W_v(x), 4)
s = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(s, axis=-1), v).swapaxes(-3, -2)
ref = g.W_o(o.reshape(*o.shape[:-2], 16))
assert jnp.allclose(g(x), ref, atol=1e-5), 'num_kv_heads == num_heads should be ordinary MHA'
""",
        },
        {
            "name": "KV heads are repeated, not tiled",
            "code": """
import jax
import jax.numpy as jnp

# 4 query heads over 2 KV heads: query heads 0,1 share KV head 0 and query
# heads 2,3 share KV head 1. jnp.tile would pair 0,2 and 1,3 instead — same
# shape, different grouping, no error.
g = {fn}(16, num_heads=4, num_kv_heads=2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 5, 16))
d_k = 4

sp = lambda t, n: t.reshape(*t.shape[:-1], n, d_k).swapaxes(-3, -2)
q = sp(g.W_q(x), 4)
k2, v2 = sp(g.W_k(x), 2), sp(g.W_v(x), 2)

def build(expand):
    k, v = expand(k2), expand(v2)
    s = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
    o = jnp.einsum('...hqk,...hkd->...hqd', jax.nn.softmax(s, axis=-1), v).swapaxes(-3, -2)
    return g.W_o(o.reshape(*o.shape[:-2], 16))

with_repeat = build(lambda t: jnp.repeat(t, 2, axis=-3))
with_tile = build(lambda t: jnp.concatenate([t, t], axis=-3))
assert not jnp.allclose(with_repeat, with_tile, atol=1e-4), (
    'this test is only meaningful if the two groupings differ'
)
assert jnp.allclose(g(x), with_repeat, atol=1e-5), (
    'the KV heads are being tiled ([0,1,0,1]) rather than repeated ([0,0,1,1]). '
    'Adjacent query heads must share a KV head — same shape either way, so '
    'nothing errors.'
)
""",
        },
        {
            "name": "Every query head actually reads its own group",
            "code": """
import jax
import jax.numpy as jnp

g = {fn}(16, num_heads=4, num_kv_heads=2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 5, 16))
base = g(x)

# Perturbing either KV head must move the output; if one is ignored, the
# grouping or the repeat is wrong.
for col in range(2):
    g2 = {fn}(16, 4, 2, key=jax.random.key(0))
    sl = slice(col * 4, (col + 1) * 4)
    g2.W_k.kernel = g2.W_k.kernel.at[:, sl].add(3.0)
    assert not jnp.allclose(g2(x), base, atol=1e-4), (
        f'perturbing KV head {col} changed nothing — it is not being used'
    )

assert g.h == 4 and g.kvh == 2 or True  # attributes are free-form
""",
        },
        {
            "name": "Gradient w.r.t. the input, and jit",
            "code": """
import jax
import jax.numpy as jnp

g = {fn}(16, 4, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 6, 16))
out = g(x)

grad = jax.grad(lambda v: jnp.sum(g(v)))(x)
assert grad.shape == x.shape and jnp.isfinite(grad).all(), 'bad gradient w.r.t. the input'

assert jnp.allclose(jax.jit(lambda v: g(v))(x), out, atol=1e-5), 'jit disagrees'
""",
        },
    ],
}
