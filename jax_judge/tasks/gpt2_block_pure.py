"""Problem 13 without Flax — the whole block, wired by hand."""

_GIVEN = '''class Linear:
    """Given to you, as nnx.Linear is in problem 13."""

    def __init__(self, d_in, d_out, *, key):
        self.kernel = jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.kernel + self.bias


class LayerNorm:
    """Given to you, as nnx.LayerNorm is in problem 13."""

    def __init__(self, d_model, eps=1e-6):
        self.scale = jnp.ones((d_model,))
        self.bias = jnp.zeros((d_model,))
        self.eps = eps

    def __call__(self, x):
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mu) / jnp.sqrt(var + self.eps) * self.scale + self.bias
'''

TASK = {
    "title": "GPT-2 Block without Flax",
    "category": "Attention & Transformers",
    "number": "b_31",
    "difficulty": "Hard",
    "function_name": "GPT2Block",
    "extra_names": ["CausalSelfAttention", "MLP"],
    "hint": (
        "Three classes. CausalSelfAttention: one fused qkv = Linear(d_model, "
        "3*d_model), jnp.split(..., 3, axis=-1), split heads, causal mask on "
        "the LOGITS with -inf before softmax, merge, out projection. MLP: "
        "Linear(d, 4d) -> gelu -> Linear(4d, d). GPT2Block wires them "
        "pre-norm: x = x + attn(ln1(x)) then x = x + mlp(ln2(x)) — the norm "
        "goes INSIDE each branch, never on the residual stream."
    ),
    "description": r"""
Problem 13's GPT-2 block with no Flax — attention, MLP and the residual wiring.

### Signature
```python
class CausalSelfAttention:
    def __init__(self, d_model, num_heads, *, key): ...
    def __call__(self, x): ...

class MLP:
    def __init__(self, d_model, *, key): ...
    def __call__(self, x): ...

class GPT2Block:
    def __init__(self, d_model, num_heads, *, key): ...
    def __call__(self, x): ...        # (B, T, d_model) -> (B, T, d_model)
```

`GPT2Block` holds `self.ln1`, `self.attn`, `self.ln2`, `self.mlp` — the same
attribute names as problem 13. `Linear` and `LayerNorm` are given to you.

### Attention: one fused QKV projection
GPT-2 projects all three at once, then splits:

```python
self.qkv = Linear(d_model, 3 * d_model, key=...)
q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
```

One matmul instead of three. Mask the **logits** with `-inf` before the
softmax, not the weights after it.

### MLP: 4x wide, gelu
`Linear(d_model, 4*d_model)` → `gelu` → `Linear(4*d_model, d_model)`. The
`4x` is GPT-2's convention and is where most of the parameters live.

### Pre-norm is the part people get wrong
```python
x = x + self.attn(self.ln1(x))     # norm INSIDE the branch
x = x + self.mlp(self.ln2(x))
```

Not `x = self.ln1(x + self.attn(x))`. The residual stream must stay unnormalised
end to end — that clean path is what lets gradients reach the bottom of a deep
stack. Post-norm (the original 2017 transformer) needs a warmup schedule to
train at all; pre-norm is why modern models do not.

Both forms produce the right shape and both run, so only a numerical
comparison catches a swap.

### Why this exists alongside problem 13
Interview sandboxes ship `jax` but not `flax`. Same class names, same
attribute names — only `rngs=nnx.Rngs(params=0)` becomes
`key=jax.random.key(0)`.
""",
    "stub": '''import jax
import jax.numpy as jnp


''' + _GIVEN + '''

class CausalSelfAttention:
    def __init__(self, d_model, num_heads, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this


class MLP:
    def __init__(self, d_model, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this


class GPT2Block:
    def __init__(self, d_model, num_heads, *, key):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


''' + _GIVEN + '''

class CausalSelfAttention:
    def __init__(self, d_model, num_heads, *, key):
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        k1, k2 = jax.random.split(key, 2)
        # One fused projection, GPT-2 style: one matmul instead of three.
        self.qkv = Linear(d_model, 3 * d_model, key=k1)
        self.out = Linear(d_model, d_model, key=k2)

    def __call__(self, x):
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        split = lambda t: t.reshape(*t.shape[:-1], self.h, self.d_head).swapaxes(-3, -2)
        q, k, v = split(q), split(k), split(v)

        s = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_head, x.dtype)
        )
        T = s.shape[-1]
        # Mask the LOGITS, before the softmax.
        s = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), s, -jnp.inf)

        o = jnp.einsum("...hqk,...hkd->...hqd", jax.nn.softmax(s, axis=-1), v)
        o = o.swapaxes(-3, -2)
        return self.out(o.reshape(*o.shape[:-2], self.h * self.d_head))


class MLP:
    def __init__(self, d_model, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.fc = Linear(d_model, 4 * d_model, key=k1)
        self.proj = Linear(4 * d_model, d_model, key=k2)

    def __call__(self, x):
        return self.proj(jax.nn.gelu(self.fc(x)))


class GPT2Block:
    def __init__(self, d_model, num_heads, *, key):
        ka, km = jax.random.split(key, 2)
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, key=ka)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, key=km)

    def __call__(self, x):
        # Pre-norm: the norm sits INSIDE each branch, never on the residual.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
''',
    "demo": '''import jax
import jax.numpy as jnp

block = GPT2Block(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))
print("out:", block(x).shape)

print("qkv is fused:", block.attn.qkv.kernel.shape, "= (d_model, 3*d_model)")
print("mlp is 4x:   ", block.mlp.fc.kernel.shape)

# Causality: perturbing the last token must not move the first.
alt = x.at[:, 4].add(50.0)
print("\\ncausal?", bool(jnp.allclose(block(alt)[:, 0], block(x)[:, 0], atol=1e-4)))

# Pre-norm keeps the residual stream unnormalised.
print("residual present?", not bool(jnp.allclose(block(x), block(x) - x, atol=1e-4)))
''',
    "tests": [
        {
            "name": "Sub-modules are named and shaped like problem 13",
            "code": """
import jax
import jax.numpy as jnp

b = {fn}(8, 2, key=jax.random.key(0))
for name in ('ln1', 'attn', 'ln2', 'mlp'):
    assert hasattr(b, name), f'missing {name} — keep problem 13 names'

assert b.attn.qkv.kernel.shape == (8, 24), (
    f'attn.qkv.kernel {b.attn.qkv.kernel.shape} vs (8, 24) — GPT-2 fuses Q, K '
    'and V into one (d_model, 3*d_model) projection'
)
assert b.attn.out.kernel.shape == (8, 8), f'attn.out {b.attn.out.kernel.shape}'
assert b.mlp.fc.kernel.shape == (8, 32), (
    f'mlp.fc {b.mlp.fc.kernel.shape} vs (8, 32) — the MLP is 4x wide'
)
assert b.mlp.proj.kernel.shape == (32, 8), f'mlp.proj {b.mlp.proj.kernel.shape}'

x = jax.random.normal(jax.random.key(1), (2, 5, 8))
assert b(x).shape == (2, 5, 8), f'{b(x).shape} vs (2, 5, 8)'
assert jnp.isfinite(b(x)).all(), 'non-finite output'
""",
        },
        {
            "name": "Attention is causal",
            "code": """
import jax
import jax.numpy as jnp

a = CausalSelfAttention(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (1, 6, 8))
base = a(x)

for t in range(1, 6):
    alt = x.at[:, t].add(50.0)
    got = a(alt)
    assert jnp.allclose(got[:, :t], base[:, :t], atol=1e-4), (
        f'changing token {t} moved an earlier position — the mask lets a query '
        'see its future. Mask the LOGITS with -inf before the softmax.'
    )
    assert not jnp.allclose(got[:, t], base[:, t], atol=1e-4), (
        f'changing token {t} did not move position {t} itself'
    )
""",
        },
        {
            "name": "MLP is 4x with gelu",
            "code": """
import jax
import jax.numpy as jnp

m = MLP(8, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))
assert m(x).shape == (2, 5, 8), f'{m(x).shape}'
assert jnp.allclose(m(x), m.proj(jax.nn.gelu(m.fc(x))), atol=1e-5), (
    'should be proj(gelu(fc(x)))'
)
assert not jnp.allclose(m(x), m.proj(jax.nn.relu(m.fc(x))), atol=1e-3), 'looks like relu'
""",
        },
        {
            "name": "Pre-norm wiring: the residual stream is never normalised",
            "code": """
import jax
import jax.numpy as jnp

b = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8)) * 3.0 + 7.0

pre = x + b.attn(b.ln1(x))
pre = pre + b.mlp(b.ln2(pre))
post = b.ln1(x + b.attn(x))
post = b.ln2(post + b.mlp(post))

assert not jnp.allclose(pre, post, atol=1e-3), 'this test needs the two forms to differ'
assert jnp.allclose(b(x), pre, atol=1e-5), (
    'the block is wired post-norm. It must be x = x + attn(ln1(x)) then '
    'x = x + mlp(ln2(x)) — the norm goes INSIDE each branch, so the residual '
    'stream stays unnormalised end to end.'
)

# A large constant offset should survive on the residual path.
big = jnp.ones((1, 4, 8)) * 100.0
assert float(jnp.mean(b(big))) > 50.0, (
    'a large input vanished — the residual stream is being normalised'
)
""",
        },
        {
            "name": "Gradient w.r.t. the input, jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

b = {fn}(8, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (2, 5, 8))
out = b(x)

g = jax.grad(lambda v: jnp.sum(b(v)))(x)
assert g.shape == x.shape and jnp.isfinite(g).all(), 'bad gradient w.r.t. the input'

assert jnp.allclose(jax.jit(lambda v: b(v))(x), out, atol=1e-5), 'jit disagrees'
vm = jax.vmap(lambda v: b(v))(x)
assert jnp.allclose(vm, out, atol=1e-5), 'vmap disagrees — do not name the batch axis'
""",
        },
    ],
}
