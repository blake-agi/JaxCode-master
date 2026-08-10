"""Mixture of Experts — top-k routing and the load-balancing loss that saves it."""

TASK = {
    "title": "Mixture of Experts (top-k routing)",
    "category": "Attention & Transformers",
    "order": 13,
    "difficulty": "Hard",
    "function_name": "MoELayer",
    "hint": (
        "Router logits are (N, E); take jax.lax.top_k(logits, k) and softmax "
        "over ONLY the k selected logits, so the kept weights sum to 1. For the "
        "dense-but-simple version, run every expert on every token to get "
        "(E, N, dout) and gather the k you need — correctness first. The aux "
        "loss is E * sum_e (fraction of tokens routed to e) * (mean router "
        "probability for e), where the fraction uses the hard top-k assignment "
        "and the probability uses the full softmax over all experts."
    ),
    "description": r"""
Implement a **top-k mixture-of-experts** layer as an `nnx.Module`, returning
both the output and the load-balancing auxiliary loss.

### Signature
```python
class MoELayer(nnx.Module):
    def __init__(self, d_model, d_hidden, num_experts, top_k=2, *, rngs):
        ...
    def __call__(self, x):
        ...  # (B, T, d_model) -> (output, aux_loss)
```

Each expert is an independent 2-layer MLP `d_model -> d_hidden -> d_model` with
a ReLU. The router is a single `(d_model, num_experts)` matrix.

### The routing
1. `logits = x @ W_router` → `(N, E)` for `N = B*T` tokens
2. `probs = softmax(logits)` over **all** experts — used by the aux loss
3. Select the top-$k$ experts per token
4. Renormalise **over the selected $k$ only**, so their weights sum to 1
5. Output is the weighted sum of those $k$ experts' outputs

### The auxiliary loss
$$\mathcal{L}_{\text{aux}} = E \sum_{e=1}^{E} f_e \cdot P_e,
\qquad
f_e = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\!\left[e \in \mathrm{top}\text{-}k(n)\right],
\qquad
P_e = \frac{1}{N}\sum_{n=1}^{N} \mathrm{softmax}(\text{logits}_n)_e$$

$f_e$ is the **fraction of tokens** routed to expert $e$ (hard, from top-$k$)
and $P_e$ is the **mean router probability** for expert $e$ (soft, from the full
softmax over all $E$).

Watch the normalisation, because it is a classic follow-up. Each token is
counted once per selected expert, so $\sum_e f_e = k$ while $\sum_e P_e = 1$.
That gives, for this (Switch/Mixtral) convention:

| routing | $\mathcal{L}_{\text{aux}}$ |
|---|---|
| perfectly uniform ($f_e = k/E$, $P_e = 1/E$) | $E \cdot E \cdot \tfrac{k}{E}\cdot\tfrac{1}{E} = k$ |
| total collapse (one expert takes everything) | $E$ |

The famous "uniform gives exactly 1" is the $k = 1$ Switch Transformer case.
DeepSeek-style implementations divide $f_e$ by $k$ so the uniform value is 1 for
any $k$; this task uses the un-divided form, which is what
`transformers`' Mixtral loss computes.

### Rules
- Renormalise over the selected experts only — this is the step people miss
- Return `(output, aux_loss)`; output shape matches the input
- The aux loss must be differentiable **through $P_e$** (the hard counts $f_e$
  are not differentiable, and that is fine — the gradient flows via $P$)
- Store the parameters under these names, since the tests read and overwrite
  them: `self.w_router` `(d_model, num_experts)`, and the experts **stacked on a
  leading expert axis** as `self.w1` `(num_experts, d_model, d_hidden)` and
  `self.w2` `(num_experts, d_hidden, d_model)`. Also keep `self.top_k`.

### Why the aux loss is not optional
Routing is a positive feedback loop: an expert that is slightly better early
gets more tokens, so it trains faster, so it gets picked more. Left alone this
collapses — a handful of experts take nearly all traffic and the rest are dead
weight, so you have paid for $E$ experts and are effectively running two.

The $f_e \cdot P_e$ product is a neat piece of design. $f$ is what you actually
care about but has no gradient (it comes from an argmax). $P$ is differentiable
but does not directly measure load. Multiplying them gives a term whose gradient
pushes down the router probability for experts that are *currently* overloaded,
with the load entering as a constant multiplier.

### Parameters vs FLOPs — the whole point
An MoE layer holds $E$ experts' worth of parameters but activates only $k$ per
token. With $E=64, k=2$ you get 32x the parameters at ~2 experts' compute.
Since capability scales with parameter count while cost scales with *active*
parameters, MoE buys capacity for cheap.

What it costs is memory and communication: all $E$ experts must be resident even
though most are idle for any given token, and at scale the experts are sharded
across devices so routing becomes an all-to-all — which is why real
implementations obsess over expert *capacity* and token dropping, and why the
naive dense-compute version below is correct but not fast.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MoELayer(nnx.Module):
    """Top-k mixture of experts. Returns (output, aux_loss)."""

    def __init__(self, d_model: int, d_hidden: int, num_experts: int,
                 top_k: int = 2, *, rngs: nnx.Rngs):
        # Expected attributes: self.top_k, self.w_router (d_model, num_experts),
        # self.w1 (num_experts, d_model, d_hidden), self.w2 (num_experts,
        # d_hidden, d_model).
        pass  # Replace this

    def __call__(self, x):
        """(B, T, d_model) -> ((B, T, d_model), scalar aux_loss)"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MoELayer(nnx.Module):
    """Top-k mixture of experts. Returns (output, aux_loss)."""

    def __init__(self, d_model: int, d_hidden: int, num_experts: int,
                 top_k: int = 2, *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        key = rngs.params()
        k_router, k_w1, k_w2 = jax.random.split(key, 3)

        self.w_router = nnx.Param(
            jax.random.normal(k_router, (d_model, num_experts)) / jnp.sqrt(d_model)
        )
        # Experts stacked on a leading axis so they can be applied in one einsum.
        self.w1 = nnx.Param(
            jax.random.normal(k_w1, (num_experts, d_model, d_hidden)) / jnp.sqrt(d_model)
        )
        self.w2 = nnx.Param(
            jax.random.normal(k_w2, (num_experts, d_hidden, d_model)) / jnp.sqrt(d_hidden)
        )

    def __call__(self, x):
        B, T, D = x.shape
        E, k = self.num_experts, self.top_k
        flat = x.reshape(-1, D)                             # (N, D)
        N = flat.shape[0]

        logits = flat @ self.w_router[...]                 # (N, E)
        probs = jax.nn.softmax(logits, axis=-1)             # full soft routing

        top_vals, top_idx = jax.lax.top_k(logits, k)        # (N, k)
        # Renormalise over the SELECTED experts only, so their weights sum to 1.
        top_w = jax.nn.softmax(top_vals, axis=-1)

        # Dense compute: every expert on every token, then gather. Correct and
        # simple; a production kernel would dispatch instead.
        h = jax.nn.relu(jnp.einsum("nd,edh->enh", flat, self.w1[...]))
        all_out = jnp.einsum("enh,ehd->ned", h, self.w2[...])   # (N, E, D)

        # picked[n, j] = all_out[n, top_idx[n, j]]
        picked = jnp.take_along_axis(all_out, top_idx[:, :, None], axis=1)

        out = jnp.sum(picked * top_w[:, :, None], axis=1)    # (N, D)

        # Load balancing: hard fraction f_e times mean soft probability P_e.
        one_hot = jax.nn.one_hot(top_idx, E).sum(axis=1)     # (N, E) counts
        f = jnp.mean(one_hot, axis=0)                        # tokens per expert
        P = jnp.mean(probs, axis=0)                          # mean router prob
        aux_loss = E * jnp.sum(f * P)

        return out.reshape(B, T, D), aux_loss
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

layer = MoELayer(d_model=32, d_hidden=64, num_experts=8, top_k=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(1), (2, 16, 32))

out, aux = layer(x)
print("out:", out.shape, " aux_loss:", float(aux))
print(f"(perfectly uniform routing -> top_k = {layer.top_k}; "
      f"total collapse -> num_experts = {layer.num_experts})")

# How many parameters are there vs how many run per token?
p = nnx.state(layer, nnx.Param)
total = sum(v.size for v in jax.tree.leaves(p))
per_token = 32 * 64 * 2 * 2      # top_k experts, two matrices each
print(f"total expert params: {total}, active per token: ~{per_token}")
''',
    "tests": [
        {
            "name": "Shapes and return signature",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(d_model=16, d_hidden=32, num_experts=4, top_k=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(1), (2, 8, 16))

result = layer(x)
assert isinstance(result, tuple) and len(result) == 2, (
    f'__call__ must return (output, aux_loss), got {type(result).__name__}'
)
out, aux = result
assert out.shape == (2, 8, 16), f'Output shape {out.shape} vs (2, 8, 16)'
assert jnp.ndim(aux) == 0, f'aux_loss must be a scalar, got shape {jnp.shape(aux)}'
assert jnp.isfinite(out).all(), 'Non-finite output'
assert jnp.isfinite(aux), 'Non-finite aux loss'
""",
        },
        {
            "name": "Routing weights sum to 1 over the selected experts",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# One expert per token (top_k = 1) with all experts made identical: the output
# must then equal that single expert's output exactly, with weight 1.0.
layer = {fn}(d_model=8, d_hidden=16, num_experts=4, top_k=1, rngs=nnx.Rngs(0))

w1 = layer.w1[...] if hasattr(layer, 'w1') else None
assert w1 is not None, 'Expected stacked expert weights on self.w1'

# Force every expert to be the same function.
layer.w1[...] = jnp.tile(layer.w1[...][:1], (4, 1, 1))
layer.w2[...] = jnp.tile(layer.w2[...][:1], (4, 1, 1))

x = jax.random.normal(jax.random.key(2), (1, 6, 8))
out, _ = layer(x)

flat = x.reshape(-1, 8)
expected = jax.nn.relu(flat @ layer.w1[...][0]) @ layer.w2[...][0]
assert jnp.allclose(out.reshape(-1, 8), expected, atol=1e-4), (
    f'With identical experts and top_k=1 the weight must be exactly 1.0. '
    f'Max diff {float(jnp.abs(out.reshape(-1, 8) - expected).max()):.2e} — this '
    'fails if the weights are not renormalised over the selected experts.'
)
""",
        },
        {
            "name": "Renormalised over top-k, not over all experts",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# All experts identical again, but now top_k = 2. If the weights are
# renormalised over the selected 2 they sum to 1 and the output equals one
# expert's output. If softmax over ALL 8 is used instead, the two kept weights
# sum to well under 1 and the output is scaled down.
E, k = 8, 2
layer = {fn}(d_model=8, d_hidden=16, num_experts=E, top_k=k, rngs=nnx.Rngs(0))
layer.w1[...] = jnp.tile(layer.w1[...][:1], (E, 1, 1))
layer.w2[...] = jnp.tile(layer.w2[...][:1], (E, 1, 1))

x = jax.random.normal(jax.random.key(3), (1, 5, 8))
out, _ = layer(x)

flat = x.reshape(-1, 8)
expected = jax.nn.relu(flat @ layer.w1[...][0]) @ layer.w2[...][0]

assert jnp.allclose(out.reshape(-1, 8), expected, atol=1e-4), (
    f'Expected the kept weights to sum to 1 (giving one expert back exactly). '
    f'Got a max deviation of {float(jnp.abs(out.reshape(-1, 8) - expected).max()):.4f}. '
    'Softmax must be taken over the top-k logits only, after selection.'
)
""",
        },
        {
            "name": "Only top_k experts contribute",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

E, k = 6, 2
layer = {fn}(d_model=8, d_hidden=12, num_experts=E, top_k=k, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(4), (1, 4, 8))

out_before, _ = layer(x)

# Find which experts the router picks, then corrupt an expert it did NOT pick.
logits = x.reshape(-1, 8) @ layer.w_router[...]
_, idx = jax.lax.top_k(logits, k)
used = set(int(i) for i in idx.reshape(-1))
unused = [e for e in range(E) if e not in used]
assert unused, 'test setup: expected at least one unused expert'

victim = unused[0]
layer.w2[...] = layer.w2[...].at[victim].set(layer.w2[...][victim] + 100.0)
out_after, _ = layer(x)

assert jnp.allclose(out_before, out_after, atol=1e-4), (
    f'Perturbing expert {victim}, which no token routed to, changed the output. '
    'Only the top-k experts may contribute.'
)

# And perturbing a USED expert must change it.
target = list(used)[0]
layer.w2[...] = layer.w2[...].at[target].set(layer.w2[...][target] + 100.0)
out_used, _ = layer(x)
assert not jnp.allclose(out_before, out_used, atol=1e-3), (
    f'Perturbing expert {target}, which IS selected, left the output unchanged'
)
""",
        },
        {
            "name": "Aux loss hits its floor of top_k when the router is uniform",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

# A zero router makes every logit 0, so P_e = 1/E exactly. The hard counts then
# drop out of the sum: aux = E * sum_e f_e * (1/E) = sum_e f_e = k, because each
# token is counted once for each of its k selected experts. Same answer for any
# top_k, which pins BOTH the E multiplier and the sum_e f_e = k normalisation.
for E, k in ((4, 1), (4, 2), (8, 4)):
    layer = {fn}(d_model=8, d_hidden=16, num_experts=E, top_k=k, rngs=nnx.Rngs(0))
    layer.w_router[...] = jnp.zeros_like(layer.w_router[...])
    x = jax.random.normal(jax.random.key(5), (4, 32, 8))
    _, aux = layer(x)

    assert jnp.allclose(aux, float(k), atol=1e-4), (
        f'E={E}, top_k={k}: with a uniform router P_e = 1/E and sum_e f_e = k, so '
        f'aux = E * sum_e f_e * (1/E) = k = {k}. Got {float(aux)}. If you got '
        f'{k / E:.3f} you dropped the leading E; if you got 1.0 you normalised '
        'f_e by k, which is a different (DeepSeek-style) convention.'
    )
    assert aux > 0, 'Aux loss must be positive'
""",
        },
        {
            "name": "Aux loss penalises collapse",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

E = 8
# All-positive features, so a single positive router column really does win for
# every token and the collapse below is total rather than approximate.
x = jnp.abs(jax.random.normal(jax.random.key(6), (4, 32, 8))) + 0.1

balanced = {fn}(d_model=8, d_hidden=16, num_experts=E, top_k=1, rngs=nnx.Rngs(0))
balanced.w_router[...] = jnp.zeros_like(balanced.w_router[...])
_, aux_bal = balanced(x)

# Collapsed router: expert 0 outscores every other expert for every token, and
# takes essentially all of the softmax mass too.
collapsed = {fn}(d_model=8, d_hidden=16, num_experts=E, top_k=1, rngs=nnx.Rngs(0))
w = jnp.zeros_like(collapsed.w_router[...])
collapsed.w_router[...] = w.at[:, 0].set(50.0)
_, aux_col = collapsed(x)

assert aux_col > aux_bal, (
    f'Routing every token to one expert must cost MORE than uniform routing: '
    f'collapsed {float(aux_col):.3f} vs uniform {float(aux_bal):.3f}. '
    'That penalty is the only thing preventing router collapse.'
)
assert aux_col > 0.9 * E, (
    f'Total collapse means f_0 = 1 and P_0 = 1, so aux -> E * 1 * 1 = {E}. '
    f'Got {float(aux_col):.3f} — either the E multiplier is missing or f_e is '
    'not the fraction of tokens routed to expert e.'
)
""",
        },
        {
            "name": "Gradients flow, including through the aux loss",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

layer = {fn}(d_model=8, d_hidden=16, num_experts=4, top_k=2, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(7), (2, 6, 8))

def loss_fn(m):
    out, aux = m(x)
    return jnp.sum(out ** 2) + 0.01 * aux

grads = nnx.grad(loss_fn)(layer)
flat = nnx.state(grads)

g_router = flat["w_router"][...]
assert jnp.isfinite(g_router).all(), 'Non-finite router gradient'
assert jnp.abs(g_router).max() > 1e-8, (
    'The router received no gradient — routing weights must multiply the '
    'expert outputs so gradient reaches w_router.'
)

# The aux loss alone must also produce a router gradient, via P_e.
g_aux = nnx.state(nnx.grad(lambda m: m(x)[1])(layer))["w_router"][...]
assert jnp.abs(g_aux).max() > 1e-10, (
    'The aux loss produced no router gradient. f_e is non-differentiable (it '
    'comes from top_k), so the gradient must flow through the soft P_e term.'
)

@nnx.jit
def fwd(m, inp):
    return m(inp)

o1, a1 = fwd(layer, x)
o2, a2 = layer(x)
assert jnp.allclose(o1, o2, atol=1e-5), 'jit changes the output'
assert jnp.allclose(a1, a2, atol=1e-5), 'jit changes the aux loss'
""",
        },
    ],
}
