"""Mixture of Experts — top-k routing over a list of expert MLPs."""

TASK = {
    "title": "Mixture of Experts (MoE)",
    "category": "Attention & Transformers",
    "number": "28",
    "difficulty": "Hard",
    "function_name": "MixtureOfExperts",
    "hint": (
        "self.router is one nnx.Linear(d_model, num_experts); self.experts is a "
        "plain Python list of nnx.Sequential(Linear, relu, Linear). Flatten "
        "(B, S, D) to (N, D) first so routing is per TOKEN. Take top_k over the "
        "router logits, softmax ONLY those k values so the kept weights sum to 1, "
        "then accumulate: for each expert, its weight per token is the sum of "
        "the selected weights where that expert was chosen — jnp.where over "
        "(top_idx == e) gives you that without any boolean indexing."
    ),
    "description": r"""
Implement a **top-k Mixture of Experts** layer.

A router scores every expert per token, the top $k$ win, and their outputs are
combined with softmax weights over just those $k$ scores.

### Signature
```python
class MixtureOfExperts(nnx.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, *, rngs: nnx.Rngs): ...
    def __call__(self, x): ...
```

### Requirements
- `self.router`: `nnx.Linear(d_model, num_experts)`
- `self.experts`: an **`nnx.List`** of `num_experts` MLPs, each
  `nnx.Sequential(nnx.Linear(d_model, d_ff), jax.nn.relu, nnx.Linear(d_ff, d_model))`
  (`nnx.List` is Flax's `nn.ModuleList` — a bare Python list is rejected)
- `self.top_k`
- Accepts `(B, S, D)` or `(N, D)`, and returns the same shape
- Routing is **per token**, not per sequence
- Softmax over the top-k logits **only**

### The point: parameters and FLOPs come apart
A dense layer uses every parameter for every token. An MoE with $E$ experts
holds $E\times$ the parameters but activates only $k$ of them per token, so
capacity grows while per-token compute stays fixed. Mixtral-8x7B has ~47B
parameters and the forward cost of a ~13B model, because $k=2$ of $8$ experts
run per token.

### Softmax over the top-k, not all E
Softmax first and then truncate, and the kept weights no longer sum to 1 — the
layer's output magnitude then depends on how confident the router happened to
be, which destabilises training. Select first, softmax second.

### Why real implementations need a load-balancing loss
Routing is a winner-take-all feedback loop: an expert that is slightly better
early gets more tokens, trains faster, and gets picked even more, until a few
experts do everything and the rest are dead weight. Production MoEs add an
auxiliary loss pushing the router toward uniform expert usage. This task leaves
it out to stay close to the original — but "what stops the router collapsing?"
is the follow-up question this problem exists to set up.

### ⚠️ Why the JAX version has no boolean-mask scatter
The PyTorch original writes `output[mask] += ...` with a boolean mask. JAX has
no in-place scatter and cannot handle a data-dependent output shape under
`jit`, so instead every expert runs on every token and its contribution is
multiplied by a per-token weight that is **zero** where the expert was not
selected. Same result; it is dense rather than sparse, which is fine at this
scale and is exactly why real sparse MoE needs custom kernels to actually
realise the FLOP saving.
""",
    "stub": '''import jax
import jax.numpy as jnp
from flax import nnx


class MixtureOfExperts(nnx.Module):
    """Top-k routed mixture of expert MLPs."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 top_k: int = 2, *, rngs: nnx.Rngs):
        pass  # Replace this

    def __call__(self, x):
        """(B, S, d_model) or (N, d_model) -> same shape"""
        pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from flax import nnx


class MixtureOfExperts(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 top_k: int = 2, *, rngs: nnx.Rngs):
        self.top_k = top_k
        self.router = nnx.Linear(d_model, num_experts, rngs=rngs)
        # nnx.List is the counterpart of torch's nn.ModuleList: a plain Python
        # list of submodules is rejected as a static attribute holding data.
        self.experts = nnx.List([
            nnx.Sequential(
                nnx.Linear(d_model, d_ff, rngs=rngs),
                jax.nn.relu,
                nnx.Linear(d_ff, d_model, rngs=rngs),
            )
            for _ in range(num_experts)
        ])

    def __call__(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])      # route per TOKEN

        logits = self.router(x_flat)                        # (N, E)
        top_vals, top_idx = jax.lax.top_k(logits, self.top_k)
        # Softmax over the SELECTED logits only, so the kept weights sum to 1.
        weights = jax.nn.softmax(top_vals, axis=-1)         # (N, k)

        out = jnp.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            # This expert's weight per token: the selected weight where it was
            # chosen, 0 otherwise. Replaces PyTorch's boolean-mask scatter,
            # which JAX cannot express with a data-dependent shape.
            w = jnp.sum(jnp.where(top_idx == e, weights, 0.0), axis=-1)  # (N,)
            out = out + w[:, None] * expert(x_flat)

        return out.reshape(orig_shape)
''',
    "demo": '''import jax
import jax.numpy as jnp
from flax import nnx

moe = MixtureOfExperts(d_model=16, d_ff=32, num_experts=4, top_k=2,
                       rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.key(1), (2, 5, 16))
print("out:", moe(x).shape)

logits = moe.router(x.reshape(-1, 16))
_, idx = jax.lax.top_k(logits, 2)
counts = jnp.bincount(idx.ravel(), length=4)
print("tokens routed to each expert:", counts, "(uneven — hence the aux loss)")
''',
    "tests": [
        {
            "name": "Structure: router and expert list",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(16, 32, 4, top_k=2, rngs=nnx.Rngs(params=0))

assert isinstance(m.router, nnx.Linear), (
    f'self.router must be an nnx.Linear, got {type(m.router)}'
)
assert m.router.kernel.shape == (16, 4), (
    f'router should map d_model -> num_experts = (16, 4), got {m.router.kernel.shape}'
)
assert len(m.experts) == 4, f'expected 4 experts, got {len(m.experts)}'
assert m.top_k == 2, f'top_k {m.top_k}'

x = jax.random.normal(jax.random.key(1), (2, 5, 16))
assert m(x).shape == (2, 5, 16), f'{m(x).shape}'
assert m(x.reshape(-1, 16)).shape == (10, 16), 'must also accept (N, D)'
""",
        },
        {
            "name": "Matches the reference routing",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, 4, top_k=2, rngs=nnx.Rngs(params=2))
x = jax.random.normal(jax.random.key(3), (6, 8))

logits = m.router(x)
top_vals, top_idx = jax.lax.top_k(logits, 2)
w = jax.nn.softmax(top_vals, axis=-1)
ref = jnp.zeros_like(x)
for e, ex in enumerate(m.experts):
    we = jnp.sum(jnp.where(top_idx == e, w, 0.0), axis=-1)
    ref = ref + we[:, None] * ex(x)

assert jnp.allclose(m(x), ref, atol=1e-5), 'Output does not match top-k routed mixture'
""",
        },
        {
            "name": "Softmax is over the top-k only",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 8, 4, top_k=2, rngs=nnx.Rngs(params=4))
# Close logits, so the discarded mass is large: softmax over all four leaves
# the top-2 summing to only ~0.55, while softmax over the top-2 sums to 1.
m.router.kernel[...] = jnp.zeros((4, 4))
m.router.bias[...] = jnp.array([1.0, 0.9, 0.8, 0.7])

x = jax.random.normal(jax.random.key(5), (3, 4))
w_full = jax.nn.softmax(jnp.array([1.0, 0.9, 0.8, 0.7]))
w_topk = jax.nn.softmax(jnp.array([1.0, 0.9]))
assert abs(float(w_topk.sum()) - 1.0) < 1e-6
assert float(w_full[:2].sum()) < 0.6, 'setup: discarded mass should be large'

ref_topk = w_topk[0] * m.experts[0](x) + w_topk[1] * m.experts[1](x)
ref_full = w_full[0] * m.experts[0](x) + w_full[1] * m.experts[1](x)

assert jnp.allclose(m(x), ref_topk, atol=1e-5), (
    'Weights must be softmax over the SELECTED logits so they sum to 1. '
    'Softmaxing all E and then truncating gives the (wrong) other answer.'
)
assert not jnp.allclose(ref_topk, ref_full, atol=1e-4), 'test is not discriminating'
""",
        },
        {
            "name": "Only the top-k experts contribute",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(4, 8, 4, top_k=1, rngs=nnx.Rngs(params=6))
m.router.kernel[...] = jnp.zeros((4, 4))
m.router.bias[...] = jnp.array([5.0, 0.0, 0.0, 0.0])   # expert 0 always wins

x = jax.random.normal(jax.random.key(7), (3, 4))
assert jnp.allclose(m(x), m.experts[0](x), atol=1e-5), (
    'With top_k=1 and expert 0 always selected, the output must be exactly '
    'expert 0 (its softmax weight over one logit is 1.0)'
)

# Perturbing an unused expert must change nothing.
before = m(x)
m.experts[2].layers[0].kernel[...] += 100.0
assert jnp.allclose(m(x), before, atol=1e-5), (
    'Changing an unselected expert altered the output — non-top-k experts must '
    'receive weight exactly 0'
)
""",
        },
        {
            "name": "Routing is per token, not per sequence",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, 4, top_k=2, rngs=nnx.Rngs(params=8))
x = jax.random.normal(jax.random.key(9), (1, 6, 8))

# A (B, S, D) call must equal flattening to (B*S, D) and back.
flat = m(x.reshape(-1, 8)).reshape(1, 6, 8)
assert jnp.allclose(m(x), flat, atol=1e-5), (
    'Reshape to (N, D) before routing — the shape must not change the result'
)

# Different tokens can select different experts.
logits = m.router(x.reshape(-1, 8))
_, idx = jax.lax.top_k(logits, 2)
assert idx.shape == (6, 2), f'routing indices {idx.shape} — one row per token'
""",
        },
        {
            "name": "top_k and num_experts are honoured",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

x = jax.random.normal(jax.random.key(10), (4, 8))
for E, k in ((2, 1), (4, 2), (8, 4), (4, 4)):
    m = {fn}(8, 16, E, top_k=k, rngs=nnx.Rngs(params=11))
    assert len(m.experts) == E, f'E={E}: got {len(m.experts)} experts'
    out = m(x)
    assert out.shape == (4, 8), f'E={E}, k={k}: {out.shape}'
    assert jnp.isfinite(out).all(), f'E={E}, k={k}: non-finite output'

# k == E is a dense mixture over every expert.
m = {fn}(8, 16, 4, top_k=4, rngs=nnx.Rngs(params=12))
w = jax.nn.softmax(m.router(x), axis=-1)
ref = sum(w[:, e:e+1] * m.experts[e](x) for e in range(4))
assert jnp.allclose(m(x), ref, atol=1e-5), 'With k == E this is a dense weighted mixture'
""",
        },
        {
            "name": "Gradients reach the router and the selected experts",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

m = {fn}(8, 16, 4, top_k=2, rngs=nnx.Rngs(params=13))
x = jax.random.normal(jax.random.key(14), (8, 8))

grads = nnx.grad(lambda mod: jnp.sum(mod(x) ** 2))(m)
state = nnx.state(grads)

rk = state["router"]["kernel"]
rv = rk[...] if isinstance(rk, nnx.Variable) else rk
assert jnp.isfinite(rv).all(), 'Non-finite router gradient'
assert float(jnp.abs(rv).sum()) > 0, (
    'No gradient reached the router — the softmax weights must stay in the '
    'differentiable path'
)

leaves = [v for v in jax.tree.leaves(state) if jnp.size(v)]
assert leaves and all(jnp.isfinite(v).all() for v in leaves), 'Non-finite expert gradient'
""",
        },
    ],
}
