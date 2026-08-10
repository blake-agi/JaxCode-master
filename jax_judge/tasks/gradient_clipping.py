"""Global-norm gradient clipping over a pytree."""

TASK = {
    "title": "Gradient Norm Clipping",
    "category": "Training",
    "number": "21",
    "difficulty": "Easy",
    "function_name": "clip_grad_norm",
    "hint": (
        "One norm for the WHOLE tree, not one per leaf: square every leaf, sum "
        "them all, take the square root. Then scale every leaf by the same "
        "coefficient max_norm / (total + 1e-6), and only when that coefficient "
        "is below 1 — clipping should never make a small gradient bigger. "
        "jax.tree.leaves gets you the leaves; jax.tree.map applies the scale."
    ),
    "description": r"""
Implement **global-norm gradient clipping**.

$$g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|_2 + \epsilon}\right)
\qquad \|g\|_2 = \sqrt{\sum_{\text{all leaves}} \sum_i g_i^2}$$

### Signature
```python
def clip_grad_norm(grads, max_norm):
    ...  # -> (clipped_grads, total_norm)
```

### Rules
- The norm is **global** — one number across the entire gradient pytree, not
  per-tensor
- Use `1e-6` in the denominator, matching the reference
- Only scale when the coefficient is `< 1`
- Do not use `optax.clip_by_global_norm`

### Why global and not per-tensor
Scaling every leaf by the *same* coefficient preserves the **direction** of the
update — you shorten the step without rotating it. Per-tensor clipping rescales
each tensor independently, which changes the relative sizes of the layer
updates and therefore points you somewhere else entirely. Global-norm is what
every large-model training script uses, usually at `max_norm=1.0`.

### What it is actually for
Clipping is a guard against loss spikes. A single bad batch — a long sequence,
a degenerate example — can produce a gradient orders of magnitude larger than
usual, and one such step is enough to knock a large model into a region it
never recovers from. Clipping bounds the damage of that step.

How often it actually fires depends on the run: with `max_norm=1.0` it is
common for a large fraction of early pretraining steps to be clipped, tailing
off as training settles. So treat it as an always-on safety rail whose binding
rate you should watch — a clip rate near 100% late in training usually means
the threshold is too low, not that the model is unstable.

### ⚠️ JAX-forced signature change
PyTorch takes an iterable of parameters, reads `p.grad`, and mutates it via
`p.grad.mul_(coef)`. In JAX gradients are a plain pytree returned by
`jax.grad`, and arrays are immutable — so this takes the **gradient pytree**
and **returns** the clipped one, alongside the norm.
""",
    "stub": '''import jax
import jax.numpy as jnp


def clip_grad_norm(grads, max_norm):
    """Clip a gradient pytree by its global L2 norm.

    Args:
        grads:    pytree of gradient arrays
        max_norm: maximum allowed global norm

    Returns:
        (clipped_grads, total_norm) — total_norm is measured BEFORE clipping.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def clip_grad_norm(grads, max_norm):
    # One norm across every leaf — this is what makes it "global".
    total_norm = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
    )

    clip_coef = max_norm / (total_norm + 1e-6)
    # Never scale UP: a gradient already inside the ball is left alone.
    clip_coef = jnp.minimum(clip_coef, 1.0)

    clipped = jax.tree.map(lambda g: g * clip_coef, grads)
    return clipped, total_norm
''',
    "demo": '''import jax
import jax.numpy as jnp

grads = {"w": jnp.array([3.0, 4.0]), "b": jnp.array(0.0)}   # norm = 5
clipped, norm = clip_grad_norm(grads, max_norm=1.0)

print("norm before:", float(norm))
print("clipped    :", clipped["w"], "-> norm", float(jnp.linalg.norm(clipped["w"])))
print("direction preserved:", clipped["w"] / jnp.linalg.norm(clipped["w"]))

small = {"w": jnp.array([0.1, 0.0])}
out, n = clip_grad_norm(small, max_norm=1.0)
print("\\nsmall gradient untouched:", out["w"], "(norm", float(n), ")")
''',
    "tests": [
        {
            "name": "Clips to exactly max_norm",
            "code": """
import jax
import jax.numpy as jnp

grads = {"w": jnp.array([3.0, 4.0])}     # norm = 5
clipped, total = {fn}(grads, 1.0)

assert jnp.allclose(total, 5.0, atol=1e-5), (
    f'total_norm should be the norm BEFORE clipping (5.0), got {float(total)}'
)
out_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(clipped)))
assert jnp.allclose(out_norm, 1.0, atol=1e-4), (
    f'After clipping the global norm should be ~max_norm=1.0, got {float(out_norm)}'
)
""",
        },
        {
            "name": "The norm is global, not per-leaf",
            "code": """
import jax
import jax.numpy as jnp

# Two leaves of norm 3 and 4 -> global norm 5, NOT 3 and 4 clipped separately.
grads = {"a": jnp.array([3.0, 0.0]), "b": jnp.array([0.0, 4.0])}
clipped, total = {fn}(grads, 1.0)

assert jnp.allclose(total, 5.0, atol=1e-5), (
    f'Global norm should be sqrt(3^2 + 4^2) = 5, got {float(total)}. '
    'Computing a norm per leaf and combining wrongly gives 3 or 4.'
)

# Same coefficient (1/5) on both leaves -> direction preserved.
assert jnp.allclose(clipped["a"], jnp.array([0.6, 0.0]), atol=1e-5), f'{clipped["a"]}'
assert jnp.allclose(clipped["b"], jnp.array([0.0, 0.8]), atol=1e-5), f'{clipped["b"]}'
""",
        },
        {
            "name": "Small gradients pass through untouched",
            "code": """
import jax
import jax.numpy as jnp

grads = {"w": jnp.array([0.1, 0.2]), "b": jnp.array(0.05)}
clipped, total = {fn}(grads, 10.0)

for k in grads:
    assert jnp.allclose(clipped[k], grads[k], atol=1e-6), (
        f'Leaf {k} changed even though the norm ({float(total):.3f}) is well '
        'under max_norm=10. Clipping must never scale a gradient UP.'
    )
""",
        },
        {
            "name": "Direction is preserved",
            "code": """
import jax
import jax.numpy as jnp

g = {"w": jax.random.normal(jax.random.key(0), (16,)) * 50.0}
clipped, _ = {fn}(g, 1.0)

a = g["w"] / jnp.linalg.norm(g["w"])
b = clipped["w"] / jnp.linalg.norm(clipped["w"])
assert jnp.allclose(a, b, atol=1e-5), (
    'The unit vector changed — every leaf must be scaled by the SAME coefficient'
)
""",
        },
        {
            "name": "Nested pytree structure preserved",
            "code": """
import jax
import jax.numpy as jnp

grads = {
    "l1": {"w": jnp.ones((3, 2)) * 2.0, "b": jnp.zeros(2)},
    "l2": [jnp.full((4,), 3.0), jnp.array(1.0)],
}
clipped, total = {fn}(grads, 1.0)

assert jax.tree.structure(clipped) == jax.tree.structure(grads), 'Structure changed'
for a, b in zip(jax.tree.leaves(clipped), jax.tree.leaves(grads)):
    assert a.shape == b.shape, f'Leaf shape changed: {a.shape} vs {b.shape}'

expected = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
assert jnp.allclose(total, expected, atol=1e-4), f'{float(total)} vs {float(expected)}'
""",
        },
        {
            "name": "Does not mutate the input",
            "code": """
import jax
import jax.numpy as jnp

g = jnp.array([3.0, 4.0])
grads = {"w": g}
before = g.copy()
clipped, _ = {fn}(grads, 1.0)

assert jnp.allclose(grads["w"], before), (
    'The input gradients must be untouched — JAX arrays are immutable, which is '
    'why this returns the clipped tree instead of mutating p.grad'
)
""",
        },
        {
            "name": "jit and zero gradients",
            "code": """
import functools
import jax
import jax.numpy as jnp

grads = {"w": jnp.array([3.0, 4.0]), "b": jnp.array(1.0)}
c1, n1 = jax.jit(functools.partial({fn}, max_norm=1.0))(grads)
c2, n2 = {fn}(grads, 1.0)
assert jnp.allclose(c1["w"], c2["w"], atol=1e-6), 'jit changes the result'
assert jnp.allclose(n1, n2, atol=1e-6), 'jit changes the norm'

zero = {"w": jnp.zeros(4)}
cz, nz = {fn}(zero, 1.0)
assert jnp.isfinite(nz) and jnp.allclose(nz, 0.0), f'Zero gradient norm should be 0, got {nz}'
assert jnp.isfinite(cz["w"]).all(), 'Zero gradients produced non-finite output — the 1e-6 guards this'
""",
        },
    ],
}
