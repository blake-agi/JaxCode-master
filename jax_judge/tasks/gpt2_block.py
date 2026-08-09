"""A full pre-norm GPT-2 transformer block — residual wiring, causal attention, 4x MLP."""

TASK = {
    "title": "GPT-2 Transformer Block",
    "category": "Attention & Transformers",
    "order": 11,
    "difficulty": "Hard",
    "function_name": "GPT2Block",
    "hint": (
        "Two residual adds, each with the norm INSIDE the branch: "
        "h = x + self.attn(self.ln1(x)); out = h + self.mlp(self.ln2(h)). "
        "For the attention, project to q/k/v of shape (B, T, D), reshape each to "
        "(B, T, H, D//H) and swap to (B, H, T, D//H) so the heads ride along as a "
        "batch axis. Scores are q @ swapaxes(k, -1, -2) / sqrt(d_head), masked with "
        "jnp.tril(jnp.ones((T, T), bool)) via jnp.where(mask, scores, -jnp.inf), then "
        "softmax over the LAST axis. The MLP is Linear(d, 4d) -> gelu -> Linear(4d, d)."
    ),
    "description": r"""
Implement one **GPT-2 transformer block** as an `nnx.Module`: pre-norm residual
attention followed by pre-norm residual MLP.

$$h = x + \mathrm{Attn}(\mathrm{LN}_1(x)), \qquad y = h + \mathrm{MLP}(\mathrm{LN}_2(h))$$

with causal multi-head self-attention

$$\mathrm{Attn}(x) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V W_O,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

and a position-wise MLP $\;d \to 4d \to \mathrm{GELU} \to d$.

### Rules
- Subclass `nnx.Module`; signature `GPT2Block(d_model, num_heads, *, rngs)`
- `__call__(x)` maps `(B, T, D) -> (B, T, D)`, deterministic (no dropout)
- Expose the four sub-parts as attributes: `self.ln1`, `self.attn`, `self.ln2`,
  `self.mlp` — each callable `(B, T, D) -> (B, T, D)`
- `nnx.Linear` and `nnx.LayerNorm` are allowed building blocks;
  `nnx.MultiHeadAttention` and `nnx.dot_product_attention` are **banned**
- Attention must be genuinely multi-head: split `D` into `num_heads` heads of
  size `d_head = D // num_heads` and scale scores by $1/\sqrt{d_h}$
- Attention must be **causal**: position $i$ may attend to $j \le i$ only
- MLP hidden width is exactly `4 * d_model`, activation GELU

### Pre-norm vs post-norm — the whole point of this problem
The 2017 "Attention Is All You Need" block was **post-norm**:

$$x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$$

GPT-2 moved the norm inside the branch. That one move is why you can stack 96
layers.

Write the network as a chain of blocks and look at the backward path. Pre-norm
gives $y = x + f(\mathrm{LN}(x))$, so $\partial y/\partial x = I + \partial
f/\partial x$: there is an **untouched additive highway** from the embedding to
the logits, and the gradient at layer 0 is the sum of a clean identity term plus
each block's contribution. Post-norm puts a LayerNorm *on* that highway. LN's
Jacobian rescales by $1/\sigma$ and projects out the mean direction, so the
signal is multiplied by $L$ such factors on the way down. At depth the product
drifts — which is exactly why the original Transformer needed learning-rate
**warmup** and careful init, and why pre-norm nets train from step 0 with a flat
schedule.

The price: nothing bounds the residual stream any more. Each block adds an
$O(1)$ correction, so $\mathrm{Var}(x_\ell)$ grows roughly linearly in depth,
and the last block's output is *not* normalized. That is why GPT-2 has a final
`ln_f` before the unembedding — a detail people forget when they hand-roll the
stack, and a good follow-up question.

### Traps hiding in the shapes
- Mask **before** the softmax, on the logits, not on the probabilities.
  Zeroing probabilities afterwards leaves the rows un-normalized.
- Reshape to `(B, T, H, d_h)` then `swapaxes(1, 2)`. Reshaping straight to
  `(B, H, T, d_h)` silently interleaves the heads with the time axis and still
  produces the right output *shape* — a bug tests based on shape alone never catch.
- Divide by $\sqrt{d_h}$, the **per-head** dimension, not by $\sqrt{D}$.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class CausalSelfAttention(nnx.Module):
    """Multi-head self-attention with a causal mask. (B, T, D) -> (B, T, D)"""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this


class MLP(nnx.Module):
    """Position-wise d -> 4d -> GELU -> d."""

    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        pass  # Replace this


class GPT2Block(nnx.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        # Must expose self.ln1, self.attn, self.ln2, self.mlp
        pass  # Replace this

    def __call__(self, x):
        """(B, T, D) -> (B, T, D)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # One fused projection for q, k, v — same params as three separate ones.
        self.qkv = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.out = nnx.Linear(d_model, d_model, rngs=rngs)

    def _split_heads(self, t, B, T):
        # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh): heads become a batch axis.
        return t.reshape(B, T, self.num_heads, self.d_head).swapaxes(1, 2)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = self._split_heads(q, B, T)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        # (B, H, T, T); scale by the PER-HEAD dim.
        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.float32(self.d_head))

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)   # mask the LOGITS
        weights = jax.nn.softmax(scores, axis=-1)

        y = weights @ v                                # (B, H, T, Dh)
        y = y.swapaxes(1, 2).reshape(B, T, D)          # merge heads back
        return self.out(y)


class MLP(nnx.Module):
    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.proj = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x):
        return self.proj(jax.nn.gelu(self.fc(x)))


class GPT2Block(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, num_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp = MLP(d_model, rngs=rngs)

    def __call__(self, x):
        # Pre-norm: the norm sits INSIDE each branch, never on the residual stream.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

block = GPT2Block(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 8, 64))
print("out:", block(x).shape)

# The residual highway: a big input passes through almost untouched, because each
# branch sees a NORMALISED copy and writes back only an O(1) correction.
big = x * 50.0
delta = block(big) - big
print(f"input std {float(jnp.std(big)):.2f} -> block wrote a delta of std {float(jnp.std(delta)):.2f}")

# Parameter budget: 12*d^2 + 13*d for a bias-everywhere GPT-2 block.
total = sum(int(p.size) for p in jax.tree.leaves(nnx.state(block, nnx.Param)))
print("params:", total, "vs 12*d^2 + 13*d =", 12 * 64 ** 2 + 13 * 64)

# Causality: rewriting the future leaves the past bit-identical.
y1 = block(x)
y2 = block(x.at[:, 4:].set(0.0))
print("max change in positions 0..3:", float(jnp.abs(y1[:, :4] - y2[:, :4]).max()))
''',
    "tests": [
        {
            "name": "Shape and required sub-modules",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

block = {fn}(d_model=64, num_heads=4, rngs=nnx.Rngs(0))
assert isinstance(block, nnx.Module), 'GPT2Block must subclass nnx.Module'

for name in ['ln1', 'attn', 'ln2', 'mlp']:
    assert hasattr(block, name), f'Missing self.{name} — the tests call the sub-parts directly'

x = jax.random.normal(jax.random.key(0), (2, 8, 64))
out = block(x)
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape} vs (2, 8, 64)'
assert jnp.isfinite(out).all(), 'Non-finite output — check the -inf mask does not produce NaNs'

# Each sub-part is itself shape-preserving.
assert block.ln1(x).shape == (2, 8, 64), f'ln1 shape {block.ln1(x).shape}'
assert block.attn(x).shape == (2, 8, 64), f'attn shape {block.attn(x).shape}'
assert block.mlp(x).shape == (2, 8, 64), f'mlp shape {block.mlp(x).shape}'

# Odd sequence length and a single head must also work.
solo = {fn}(d_model=32, num_heads=1, rngs=nnx.Rngs(1))
assert solo(jax.random.normal(jax.random.key(2), (1, 5, 32))).shape == (1, 5, 32)
""",
        },
        {
            "name": "Pre-norm wiring is exact",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 6, 32))

h = x + block.attn(block.ln1(x))
expected = h + block.mlp(block.ln2(h))
got = block(x)

assert jnp.allclose(got, expected, atol=1e-4), (
    'Block output != x + attn(ln1(x)) then + mlp(ln2(...)). Common wrong wirings: '
    'post-norm ln(x + attn(x)), norms applied to the residual stream, or the second '
    'branch reading the pre-attention x instead of the updated h.'
)

# Post-norm would renormalise the stream; the identity path must be exact.
zero_in = jnp.zeros((1, 4, 32))
assert jnp.isfinite(block(zero_in)).all(), 'All-zero input must stay finite'
""",
        },
        {
            "name": "Attention is causal",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (1, 8, 32))

out1 = block(x)
x2 = x.at[:, 4:].set(jax.random.normal(jax.random.key(1), (1, 4, 32)))
out2 = block(x2)

assert jnp.allclose(out1[:, :4], out2[:, :4], atol=1e-5), (
    'Rewriting tokens 4..7 changed the output at tokens 0..3 — the mask is missing '
    'or is the wrong triangle. Use jnp.tril (keep j <= i), not jnp.triu.'
)
assert not jnp.allclose(out1[:, 4:], out2[:, 4:], atol=1e-3), (
    'Changing the later tokens did not change their own outputs at all'
)

# Prefix invariance: running a truncated prefix gives the same answer.
pref = block(x[:, :5])
assert jnp.allclose(pref, out1[:, :5], atol=1e-4), (
    'A length-5 prefix must produce the same outputs as the first 5 positions of '
    'the length-8 run — this is what makes autoregressive caching legal'
)
""",
        },
        {
            "name": "Residual highway survives a large input",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

block = {fn}(d_model=64, num_heads=8, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 8, 64)) * 50.0

out = block(x)
delta = out - x

assert float(jnp.std(delta)) < 0.2 * float(jnp.std(x)), (
    f'Input std {float(jnp.std(x)):.2f} but the block changed it by std '
    f'{float(jnp.std(delta)):.2f}. In a PRE-norm block each branch sees a '
    'normalised copy and writes back an O(1) correction, so a large residual '
    'passes through nearly untouched. A post-norm block (ln applied AFTER the '
    'add) would squash the output to unit scale instead.'
)
assert float(jnp.std(out)) > 0.8 * float(jnp.std(x)), (
    'The output scale collapsed — the residual stream is being normalised'
)
""",
        },
        {
            "name": "MLP is a 4x position-wise expansion",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

d = 32
block = {fn}(d_model=d, num_heads=4, rngs=nnx.Rngs(0))

n_mlp = sum(int(p.size) for p in jax.tree.leaves(nnx.state(block.mlp, nnx.Param)))
lo, hi = 8 * d * d, 8 * d * d + 5 * d       # d*4d + 4d*d, plus biases if present
assert lo <= n_mlp <= hi, (
    f'MLP has {n_mlp} params; a 4x block should have {lo} (no bias) to {hi} (with bias). '
    'Hidden width must be exactly 4 * d_model.'
)

# Position-wise: no mixing across the time axis, so it commutes with a permutation.
x = jax.random.normal(jax.random.key(0), (2, 6, d))
perm = jnp.array([3, 0, 5, 1, 4, 2])
assert jnp.allclose(block.mlp(x[:, perm]), block.mlp(x)[:, perm], atol=1e-5), (
    'The MLP must act on each position independently — it should commute with a '
    'permutation of the tokens. Only attention mixes across time.'
)
""",
        },
        {
            "name": "num_heads actually splits the head axis",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# Same seed => identical parameter values; only the head split differs. If heads
# are ignored (one big head), the two blocks compute exactly the same function.
one = {fn}(d_model=32, num_heads=1, rngs=nnx.Rngs(0))
many = {fn}(d_model=32, num_heads=8, rngs=nnx.Rngs(0))

x = jax.random.normal(jax.random.key(0), (2, 6, 32))
a, b = one(x), many(x)

assert a.shape == b.shape == (2, 6, 32)
assert not jnp.allclose(a, b, atol=1e-4), (
    'num_heads=1 and num_heads=8 gave the same output. The head axis is being '
    'ignored: reshape to (B, T, H, D//H) and swap axes 1 and 2 so each head '
    'attends over its own subspace, and scale by 1/sqrt(D//H).'
)

# The head split must not change the parameter count.
n1 = sum(int(p.size) for p in jax.tree.leaves(nnx.state(one, nnx.Param)))
n8 = sum(int(p.size) for p in jax.tree.leaves(nnx.state(many, nnx.Param)))
assert n1 == n8, f'Head count changed the parameter count: {n1} vs {n8}'
""",
        },
        {
            "name": "Gradients reach every parameter, and nnx.jit works",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

block = {fn}(d_model=32, num_heads=4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (2, 6, 32))

grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(block)
leaves = jax.tree.leaves(grads)
assert len(leaves) >= 8, (
    f'Only {len(leaves)} gradient leaves — expected >= 8 (2 layernorms, qkv, out proj, '
    'and the two MLP layers)'
)
assert all(jnp.isfinite(g).all() for g in leaves), (
    'Non-finite gradient. A -inf mask that ends up in a fully-masked softmax row '
    'yields NaN; with a causal mask every row keeps at least the diagonal.'
)
n_dead = sum(1 for g in leaves if float(jnp.abs(g).max()) == 0.0)
assert n_dead == 0, f'{n_dead} parameter tensors got an exactly-zero gradient'

jitted = nnx.jit(lambda m, v: m(v))
assert jnp.allclose(jitted(block, x), block(x), atol=1e-5), 'nnx.jit changed the result'
""",
        },
    ],
}
