"""Grouped-query attention — fewer KV heads, and the KV-cache argument for it."""

TASK = {
    "title": "Grouped Query Attention",
    "category": "Attention & Transformers",
    "number": "10",
    "difficulty": "Hard",
    "function_name": "GroupQueryAttention",
    "hint": (
        "W_q and W_o are square, but W_k and W_v project to the SMALLER width "
        "num_kv_heads * d_k — that asymmetry is the whole point. After "
        "reshaping, k and v have num_kv_heads on the head axis while q has "
        "num_heads, so repeat each KV head num_heads // num_kv_heads times "
        "before the scores. Use jnp.repeat with an axis, not tile: adjacent "
        "copies must be adjacent so query group g lines up with KV head "
        "g // repeats."
    ),
    "description": r"""
Implement **grouped-query attention** (GQA), the attention variant used by
Llama-2 70B, Llama-3, Mistral and most current open models.

Query heads are split into groups, and each **group shares one key/value head**.

### Signature
```python
class GroupQueryAttention(nnx.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, *, rngs: nnx.Rngs): ...
    def __call__(self, x): ...
```

### Requirements
- `self.W_q`: `nnx.Linear(d_model, d_model)`
- `self.W_k`, `self.W_v`: `nnx.Linear(d_model, num_kv_heads * d_k)` — **smaller**
- `self.W_o`: `nnx.Linear(d_model, d_model)`
- `self.d_k = d_model // num_heads`
- `num_heads` must be divisible by `num_kv_heads`

### The two endpoints
- `num_kv_heads == num_heads` → ordinary multi-head attention
- `num_kv_heads == 1` → multi-query attention (MQA)

GQA is the middle: it recovers most of MQA's savings without MQA's quality
drop.

### Why it exists — the KV cache, not the FLOPs
The projections get slightly cheaper, but that is not the motivation. During
generation the KV cache holds
$2 \times B \times H_{kv} \times S \times d_k$ values, and at long context it
dwarfs the weights. Cutting $H_{kv}$ from 64 to 8 cuts the cache **8×**, which
is the difference between fitting a long-context batch in memory and not.

Llama-2 70B uses 64 query heads and 8 KV heads. Quality is within noise of full
MHA; the cache is one eighth the size.

### The trap: repeat vs tile
Query head $i$ must pair with KV head $i \,/\, \text{repeats}$ (integer
division), so the KV heads need each entry duplicated **in place**:
`[a, b] -> [a, a, a, a, b, b, b, b]`. Tiling gives `[a, b, a, b, a, b, a, b]`
instead, which silently pairs the wrong heads — shapes match, gradients flow,
and the model just learns worse. `jnp.repeat(k, repeats, axis=1)` is the
correct spelling.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class GroupQueryAttention(nnx.Module):
    """Self-attention with fewer key/value heads than query heads."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(B, S, d_model) -> (B, S, d_model)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class GroupQueryAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 *, rngs: nnx.Rngs):
        assert num_heads % num_kv_heads == 0, "num_heads must divide by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads

        self.W_q = nnx.Linear(d_model, d_model, rngs=rngs)
        # Note the smaller output width — this is where the saving comes from.
        self.W_k = nnx.Linear(d_model, num_kv_heads * self.d_k, rngs=rngs)
        self.W_v = nnx.Linear(d_model, num_kv_heads * self.d_k, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x):
        B, S, _ = x.shape

        q = self.W_q(x).reshape(B, S, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = self.W_k(x).reshape(B, S, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, S, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)

        # repeat, NOT tile: [a, b] -> [a, a, b, b] so query group g maps to
        # KV head g // repeats.
        repeats = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, repeats, axis=1)
        v = jnp.repeat(v, repeats, axis=1)

        # == q @ jnp.swapaxes(k, -1, -2)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, x.dtype)
        )
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhqk,bhkd->bhqd", weights, v)      # == weights @ v

        out = attn.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.W_o(out)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

x = jax.random.normal(jax.random.key(1), (1, 8, 64))
for kv in (8, 4, 1):
    m = GroupQueryAttention(64, 8, kv, rngs=nnx.Rngs(params=0))
    label = {8: "MHA", 1: "MQA"}.get(kv, "GQA")
    print(f"  num_kv_heads={kv} ({label:3s})  out {m(x).shape}  "
          f"KV cache per token: {2 * kv * 8} values")
''',
    "tests": [
        {
            "name": "Shapes and asymmetric projections",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(64, 8, 2, rngs=nnx.Rngs(params=0))
assert m.d_k == 8, f'd_k should be 8, got {m.d_k}'

for name in ("W_q", "W_k", "W_v", "W_o"):
    assert hasattr(m, name), f'Missing self.{name}'
    assert isinstance(getattr(m, name), nnx.Linear), (
        f'self.{name} must be an nnx.Linear, got {type(getattr(m, name))}'
    )

assert m.W_q.kernel.shape == (64, 64), f'W_q {m.W_q.kernel.shape}'
assert m.W_o.kernel.shape == (64, 64), f'W_o {m.W_o.kernel.shape}'
assert m.W_k.kernel.shape == (64, 16), (
    f'W_k should project to num_kv_heads * d_k = 2 * 8 = 16, got '
    f'{m.W_k.kernel.shape}. Making it square is the whole saving, undone.'
)
assert m.W_v.kernel.shape == (64, 16), f'W_v {m.W_v.kernel.shape}'

x = jax.random.normal(jax.random.key(1), (2, 5, 64))
assert m(x).shape == (2, 5, 64), f'{m(x).shape}'
""",
        },
        {
            "name": "Reduces to plain MHA when num_kv_heads == num_heads",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 4, 4, rngs=nnx.Rngs(params=2))
x = jax.random.normal(jax.random.key(3), (2, 6, 32))

B, S, H, d_k = 2, 6, 4, 8
def split(t, h):
    return t.reshape(B, S, h, d_k).transpose(0, 2, 1, 3)
q, k, v = split(m.W_q(x), 4), split(m.W_k(x), 4), split(m.W_v(x), 4)
s = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
a = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), v)
ref = m.W_o(a.transpose(0, 2, 1, 3).reshape(B, S, -1))

assert jnp.allclose(m(x), ref, atol=1e-5), (
    'With num_kv_heads == num_heads this must equal ordinary multi-head attention'
)
""",
        },
        {
            "name": "MQA endpoint: one shared KV head",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 4, 1, rngs=nnx.Rngs(params=4))
assert m.W_k.kernel.shape == (32, 8), f'MQA W_k should be (32, 8), got {m.W_k.kernel.shape}'

x = jax.random.normal(jax.random.key(5), (2, 5, 32))
assert m(x).shape == (2, 5, 32), f'{m(x).shape}'
assert jnp.isfinite(m(x)).all(), 'Non-finite output for num_kv_heads=1'
""",
        },
        {
            "name": "repeat, not tile — heads pair correctly",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 4, 2, rngs=nnx.Rngs(params=6))
x = jax.random.normal(jax.random.key(7), (1, 3, 16))
B, S, d_k = 1, 3, 4

q = m.W_q(x).reshape(B, S, 4, d_k).transpose(0, 2, 1, 3)
k2 = m.W_k(x).reshape(B, S, 2, d_k).transpose(0, 2, 1, 3)
v2 = m.W_v(x).reshape(B, S, 2, d_k).transpose(0, 2, 1, 3)

def run(k, v):
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(d_k, x.dtype))
    a = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), v)
    return m.W_o(a.transpose(0, 2, 1, 3).reshape(B, S, -1))

correct = run(jnp.repeat(k2, 2, axis=1), jnp.repeat(v2, 2, axis=1))
tiled = run(jnp.tile(k2, (1, 2, 1, 1)), jnp.tile(v2, (1, 2, 1, 1)))

assert not jnp.allclose(correct, tiled, atol=1e-5), 'test is not discriminating'
assert jnp.allclose(m(x), correct, atol=1e-5), (
    'Output matches the TILED pairing [a,b,a,b] rather than the repeated one '
    '[a,a,b,b]. Query head i must use KV head i // repeats.'
)
""",
        },
        {
            "name": "Scaled by 1/sqrt(d_k)",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 1, 1, rngs=nnx.Rngs(params=8))
for lin in (m.W_q, m.W_k, m.W_v, m.W_o):
    lin.kernel[...] = jnp.eye(4)
    lin.bias[...] = jnp.zeros(4)

x = jnp.zeros((1, 2, 4)).at[0, 0, 0].set(1.0).at[0, 1, 0].set(1.0)
out = m(x)

w = jax.nn.softmax(jnp.array([1.0 / jnp.sqrt(4.0), 1.0 / jnp.sqrt(4.0)]))
expected = w[0] * x[0, 0] + w[1] * x[0, 1]
assert jnp.allclose(out[0, 0], expected, atol=1e-4), (
    f'Got {out[0, 0]}, expected {expected} — check the 1/sqrt(d_k) scaling'
)
""",
        },
        {
            "name": "Rejects an indivisible head count",
            "code": """
from flax import nnx

try:
    {fn}(32, 4, 3, rngs=nnx.Rngs(params=9))
except Exception:
    pass
else:
    raise AssertionError(
        'num_heads=4 with num_kv_heads=3 does not divide evenly and must be '
        'rejected — otherwise repeats is 1 and one KV head is silently dropped'
    )
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(32, 8, 2, rngs=nnx.Rngs(params=10))
x = jax.random.normal(jax.random.key(11), (2, 5, 32))

grads = nnx.grad(lambda mod: jnp.sum(mod(x) ** 2))(m)
state = nnx.state(grads)
for name in ("W_q", "W_k", "W_v", "W_o"):
    k = state[name]["kernel"]
    val = k[...] if isinstance(k, nnx.Variable) else k
    assert jnp.isfinite(val).all(), f'Non-finite gradient for {name}'
    assert float(jnp.abs(val).sum()) > 0, f'No gradient reached {name}'

graphdef, st = nnx.split(m)
run = jax.jit(lambda st, a: nnx.merge(graphdef, st)(a))
assert jnp.allclose(run(st, x), m(x), atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
