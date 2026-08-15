"""Assemble a whole decoder from blocks — where the bugs are in the wiring."""

TASK = {
    "title": "Mini GPT — assemble the whole model",
    "category": "Attention & Transformers",
    "number": "b_13",
    "difficulty": "Hard",
    "function_name": "MiniGPT",
    "hint": (
        "Four things decide whether this works, and none of them is a formula. "
        "(1) RoPE rotates q and k INSIDE attention — never add position to the "
        "token embedding; V must stay unrotated. (2) Pre-norm means no block "
        "normalises its own output, so the stack needs a final norm before the "
        "head. (3) The head is the embedding table transposed — nnx.Embed gives "
        "you .attend() for exactly this; a second Linear is a different model "
        "with vocab×d_model extra parameters. (4) Build the causal mask once in "
        "__call__ and pass it down; rebuilding it per block is wasted work."
    ),
    "description": r"""
Assemble a small LLaMA-style decoder from parts you have already built. Every
piece here is simple; **the marks are all in the wiring**.

### Signature
```python
class MiniGPT(nnx.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, *, rngs): ...
    def __call__(self, ids):    # (B, T) int32 -> (B, T, vocab_size)
```

Expose `self.tok_emb`, `self.blocks` (length `num_layers`), and `self.norm_f`.

### Architecture — modern defaults
- **RMSNorm**, not LayerNorm (`nnx.RMSNorm` is allowed)
- **RoPE** for position — `apply_rope` is given to you in the starter cell
- **SwiGLU** MLP: `w_down(silu(w_gate(x)) * w_up(x))`, hidden width `4 * d_model`
- Pre-norm residuals, exactly as in a GPT-2 block:
  `h = x + Attn(Norm(x))`, then `y = h + MLP(Norm(h))`
- Causal: position $i$ attends to $j \le i$ only
- **Weight tying**: the output head is the token embedding transposed

`nnx.Linear`, `nnx.RMSNorm` and `nnx.Embed` are allowed. `nnx.MultiHeadAttention`
is not — the attention is yours. Use the **provided `apply_rope`** rather than
rolling your own; the point of this problem is where it goes, not what it does.

---

## The four traps

Each one produces a model that runs, trains, and is wrong.

### 1. RoPE goes on `q` and `k`, not on the embedding
The intuition from *learned* positional embeddings — "add position to the input"
— does not transfer. RoPE is applied inside attention, to `q` and `k` only,
after the head split. `v` is never rotated.

Add it to the token embedding instead and the model still runs, still learns
something, and quietly encodes position into the *values* it copies around. The
token embedding must come out of `tok_emb` carrying no positional information at
all; position enters only via attention scores.

### 2. Pre-norm needs a final norm
In pre-norm, each block normalises its **input** and adds an unnormalised
residual. Nothing normalises the output. After `num_layers` blocks the residual
stream has grown, and feeding it straight to the head is a real bug — it is why
every pre-norm model has a `norm_f` between the last block and the logits.

Post-norm architectures don't need one, which is exactly why this gets dropped
when porting.

### 3. The head is the embedding transposed
Tying means the logits are $x E^\top$ using the *same* table you embedded with.
`nnx.Embed.attend(x)` does this. Building a `nnx.Linear(d_model, vocab_size)`
gives an untied model — it runs, but it has `vocab_size × d_model` more
parameters and no longer shares representations between input and output.

At `vocab_size = 50257, d_model = 768` that is 38M parameters — about a third
of GPT-2 small.

### 4. Build the mask once
The causal mask depends only on `T`, so it is the same for every layer. Build it
in `__call__` and pass it down. Rebuilding `jnp.tril(...)` inside each block is
not wrong, just wasteful — and it is the kind of thing an interviewer notices.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


# ---- GIVEN — do not change. This is problem 24, provided so you can focus
# ---- on WHERE it gets applied rather than re-deriving the rotation.
def apply_rope(x, positions):
    """Rotate the feature axis of x by angles set by `positions`.

    Args:
        x:         (..., T, d_head), d_head even
        positions: (T,) integer positions

    Returns:
        Array shaped like x.
    """
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class MiniGPT(nnx.Module):
    """A small LLaMA-style decoder: RMSNorm + RoPE + SwiGLU, weights tied."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, ids):
        """(B, T) int32 -> (B, T, vocab_size)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


def apply_rope(x, positions):
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class _Block(nnx.Module):
    def __init__(self, d_model, num_heads, *, rngs):
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)
        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)

        lin = lambda a, b: nnx.Linear(a, b, use_bias=False, rngs=rngs)
        self.wq, self.wk, self.wv, self.wo = (lin(d_model, d_model) for _ in range(4))

        hidden = 4 * d_model
        self.w_gate, self.w_up = lin(d_model, hidden), lin(d_model, hidden)
        self.w_down = lin(hidden, d_model)

    def __call__(self, x, mask, positions):
        B, T, D = x.shape
        h = self.norm1(x)

        def split(t):
            return t.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        q, k, v = split(self.wq(h)), split(self.wk(h)), split(self.wv(h))
        # Position enters HERE, on q and k only. v stays unrotated: rotating it
        # would smuggle position into the values the model copies around.
        q, k = apply_rope(q, positions), apply_rope(k, positions)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            jnp.asarray(self.d_head, x.dtype)
        )
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))
        attn = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)
        x = x + self.wo(attn.transpose(0, 2, 1, 3).reshape(B, T, D))

        h = self.norm2(x)
        return x + self.w_down(jax.nn.silu(self.w_gate(h)) * self.w_up(h))


class MiniGPT(nnx.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, *, rngs: nnx.Rngs):
        self.tok_emb = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.blocks = nnx.List([
            _Block(d_model, num_heads, rngs=rngs) for _ in range(num_layers)
        ])
        # Pre-norm blocks never normalise their output, so the stack needs this
        # before the head — without it the residual stream just keeps growing.
        self.norm_f = nnx.RMSNorm(d_model, rngs=rngs)

    def __call__(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids)                      # no position added here
        positions = jnp.arange(T)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))   # built once, shared

        for blk in self.blocks:
            x = blk(x, mask, positions)

        x = self.norm_f(x)
        # Tied head: the same table, transposed. A second Linear here would add
        # vocab_size * d_model parameters and untie the representations.
        return self.tok_emb.attend(x)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

m = MiniGPT(vocab_size=64, d_model=32, num_heads=4, num_layers=3,
            rngs=nnx.Rngs(params=0))
ids = jnp.array([[1, 2, 3, 4, 5]])
print("logits:", m(ids).shape, "-> (B, T, vocab)")

# Trap 1 — position must NOT be in the embedding, but MUST reach the output.
same = jnp.array([[7, 7]])
emb = m.tok_emb(same)
print("\\nsame token, two positions — embeddings identical?",
      bool(jnp.allclose(emb[0, 0], emb[0, 1])), "(must be True)")
out = m(same)
print("                            ...but logits differ?",
      not bool(jnp.allclose(out[0, 0], out[0, 1])), "(must be True)")

# Trap 3 — tied head: exactly one vocab-sized matrix in the whole model.
shapes = [v.shape for v in jax.tree.leaves(nnx.state(m))]
print("\\nvocab-sized matrices:", sum(1 for s in shapes if 64 in s), "(must be 1)")
print("total params:", sum(int(jnp.prod(jnp.array(s))) for s in shapes))

# Trap 2 — the final norm is really in the path.
before = m(ids)
m.norm_f.scale[...] = m.norm_f.scale[...] * 3.0
print("\\nscaling norm_f changed the logits?",
      not bool(jnp.allclose(before, m(ids))), "(must be True)")
''',
    "tests": [
        {
            "name": "Shapes and required parts",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=3, rngs=nnx.Rngs(params=0))

for name in ("tok_emb", "blocks", "norm_f"):
    assert hasattr(m, name), f'Missing self.{name}'

assert isinstance(m.tok_emb, nnx.Embed), f'tok_emb must be an nnx.Embed, got {type(m.tok_emb)}'
assert len(m.blocks) == 3, f'blocks should have num_layers=3 entries, got {len(m.blocks)}'
assert isinstance(m.norm_f, nnx.RMSNorm), (
    f'norm_f must be an nnx.RMSNorm (this is a LLaMA-style model), got {type(m.norm_f)}'
)

ids = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
out = m(ids)
assert out.shape == (2, 4, 32), f'(B, T) -> (B, T, vocab) expected (2, 4, 32), got {out.shape}'
assert jnp.isfinite(out).all(), 'Non-finite logits'
""",
        },
        {
            "name": "TRAP 1: RoPE goes on q and k, after the head split",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=1))

# Watch where apply_rope is actually used. Correct placement is inside
# attention, on q and k only, after they have been split into heads — so every
# call sees a 4-D (B, H, T, d_head) tensor, twice per layer.
g = type(m).__call__.__globals__
assert "apply_rope" in g, 'Use the provided apply_rope — the tests locate it by name'
original, seen = g["apply_rope"], []

def _spy(x, positions):
    seen.append(tuple(x.shape))
    return original(x, positions)

g["apply_rope"] = _spy
try:
    m(jnp.array([[1, 2, 3, 4]]))
finally:
    g["apply_rope"] = original

assert seen, (
    'apply_rope was never called — the model has no positional information at all'
)
bad = [s for s in seen if len(s) != 4]
assert not bad, (
    f'apply_rope was called on rank-{len(bad[0])} input {bad[0]}. A (B, T, D) '
    'tensor is the token embedding or the residual stream — rotating that puts '
    'position into the VALUES the model copies around. Rotate q and k only, '
    'after splitting them into (B, H, T, d_head).'
)
assert len(seen) == 4, (
    f'apply_rope was called {len(seen)} times for 2 layers; expected 4 — once '
    'for q and once for k in each. Three per layer usually means v was rotated '
    'too, which it must not be.'
)
""",
        },
        {
            "name": "Position reaches attention (not permutation-invariant)",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=8))

# Attention with no positional signal is permutation-invariant over the visible
# prefix: same final token, same context, different order -> identical logits.
a = m(jnp.array([[3, 9, 5]]))
b = m(jnp.array([[9, 3, 5]]))
assert not jnp.allclose(a[0, 2], b[0, 2], atol=1e-5), (
    'Reordering the context left the final position\\'s logits unchanged — '
    'attention is permutation-invariant, so no positional information reached it.'
)
""",
        },
        {
            "name": "TRAP 2: the final norm is in the path",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=3))
ids = jnp.array([[1, 2, 3]])

before = m(ids)
m.norm_f.scale[...] = m.norm_f.scale[...] * 3.0
after = m(ids)

assert not jnp.allclose(before, after, atol=1e-5), (
    'Scaling norm_f.scale changed nothing — self.norm_f exists but is never '
    'applied. Pre-norm blocks do not normalise their output, so the stack needs '
    'a final norm between the last block and the head.'
)
""",
        },
        {
            "name": "TRAP 3: the head is tied to the embedding",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

VOCAB, D = 32, 16
m = {fn}(vocab_size=VOCAB, d_model=D, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=4))

shapes = [tuple(v.shape) for v in jax.tree.leaves(nnx.state(m))]
vocab_sized = [s for s in shapes if VOCAB in s and D in s]
assert len(vocab_sized) == 1, (
    f'Found {len(vocab_sized)} vocab-by-d_model matrices {vocab_sized}; a tied '
    'model has exactly one — the embedding table, reused as the head via '
    'tok_emb.attend(). A separate output Linear adds vocab_size * d_model '
    'parameters and unties the representations.'
)

# Changing the table must move the logits, since it IS the head.
ids = jnp.array([[1, 2, 3]])
before = m(ids)
m.tok_emb.embedding[...] = m.tok_emb.embedding[...] * 2.0
assert not jnp.allclose(before, m(ids), atol=1e-5), (
    'Scaling the embedding table left the logits unchanged — the head is not '
    'reading from it'
)
""",
        },
        {
            "name": "Causal: later tokens cannot change earlier logits",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=5))

a = jnp.array([[3, 1, 4, 1, 5]])
b = a.at[0, 3].set(29)          # change position 3 only

out_a, out_b = m(a), m(b)
assert jnp.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5), (
    'Changing the token at position 3 moved the logits at positions 0-2 — '
    'attention is not causal'
)
assert not jnp.allclose(out_a[:, 3], out_b[:, 3], atol=1e-5), (
    'Changing position 3 did not change its own logits'
)
""",
        },
        {
            "name": "Every block is applied",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=3, rngs=nnx.Rngs(params=6))
ids = jnp.array([[1, 2, 3]])
base = m(ids)

# Perturb something in each block in turn; a stack that drops a layer (or
# applies the same one repeatedly) will not respond to all of them.
for i, blk in enumerate(m.blocks):
    saved = jax.tree.map(lambda v: v, nnx.state(blk))          # snapshot values
    nnx.update(blk, jax.tree.map(lambda v: v + 0.5, saved))
    changed = not jnp.allclose(base, m(ids), atol=1e-6)
    nnx.update(blk, saved)                                     # always restore
    assert changed, (
        f'Perturbing block {i} changed nothing — it is not in the forward pass'
    )
""",
        },
        {
            "name": "Gradients and jit",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(vocab_size=32, d_model=16, num_heads=4, num_layers=2, rngs=nnx.Rngs(params=7))
ids = jnp.array([[1, 2, 3, 4]])

grads = nnx.grad(lambda mod: jnp.sum(mod(ids) ** 2))(m)
flat = [v for v in jax.tree.leaves(nnx.state(grads))]
assert flat, 'No gradients produced'
for g in flat:
    val = g[...] if isinstance(g, nnx.Variable) else g
    assert jnp.isfinite(val).all(), 'Non-finite gradient'
assert any(float(jnp.abs(g[...] if isinstance(g, nnx.Variable) else g).sum()) > 0
           for g in flat), 'All gradients are zero'

graphdef, state = nnx.split(m)
run = jax.jit(lambda s, i: nnx.merge(graphdef, s)(i))
assert jnp.allclose(run(state, ids), m(ids), atol=1e-5), 'jit changes the result'
""",
        },
    ],
}
