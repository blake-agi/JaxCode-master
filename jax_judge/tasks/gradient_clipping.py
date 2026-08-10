"""Global-norm gradient clipping over a whole pytree — one scalar for every leaf."""

TASK = {
    "title": "Global-Norm Gradient Clipping",
    "category": "Training",
    "order": 5,
    "difficulty": "Medium",
    "function_name": "clip_by_global_norm",
    "hint": (
        "jax.tree.leaves flattens any pytree to a list of arrays. Accumulate the "
        "sum of squares across ALL of them and take a single square root at the "
        "end — one norm for the whole tree, not one per leaf — then one scalar "
        "multiplier, broadcast back over the tree with jax.tree.map. "
        "Do not write `if norm > max_norm:`: that is a Python branch on a traced "
        "value and dies under jit. The branch-free version is a jnp.minimum "
        "against 1.0, which conveniently also handles the all-zero tree — but "
        "look hard at what the quotient inside that minimum evaluates to when "
        "the norm is exactly 0 before you trust it."
    ),
    "description": r"""
Clip a gradient pytree by its **global** norm and return the rescaled pytree.

$$\|g\|_2 = \sqrt{\sum_{p \in \text{leaves}} \sum_i g_{p,i}^2}
\qquad
\hat g = g \cdot \min\!\left(1, \frac{\tau}{\|g\|_2}\right)$$

One scalar $\|g\|_2$ is computed across **all** parameters concatenated into a
single vector, and one scalar factor is applied to **every** leaf.

### Rules
- Signature: `clip_by_global_norm(grads, max_norm)`
- `grads` is an arbitrary pytree (nested dicts/lists/tuples, or an `nnx.State`)
- Return the tuple `(clipped_grads, global_norm)` where `global_norm` is the
  norm **before** clipping, as a JAX scalar
- The returned tree must have the **same structure** as the input
- Banned: `optax.clip_by_global_norm`, `optax.global_norm`
- No Python `if` on the norm — the function must survive `jax.jit`
- All-zero gradients must return all-zero gradients, **not** `NaN`

### Why global, and not per-tensor
Per-tensor clipping (`each leaf independently scaled to at most tau`) uses a
*different* multiplier per leaf. That does not merely shrink the update — it
**rotates** it. In the flattened parameter space the direction of the update is
a unit vector; multiplying block $A$ by 0.1 and block $B$ by 1.0 produces a
descent direction that is no longer parallel to $-\nabla L$, so you are no
longer doing gradient descent on $L$ at all. You are doing gradient descent on
some silently reweighted objective whose per-layer weights change every step.

Global-norm clipping is the exact **Euclidean projection** of $g$ onto the ball
$\{v : \|v\|_2 \le \tau\}$: it is the closest point in the ball, it is a
*positive* multiple of $g$, and therefore
$\cos(\hat g, g) = 1$ exactly. Only the step *length* changes. That is the whole
point — a loss spike should shorten your step, not steer it somewhere else.

Two more consequences worth being able to say out loud:

- The threshold $\tau$ is a property of the **model**, not of a tensor. Under
  per-tensor clipping a good $\tau$ depends on each tensor's fan-in and element
  count, so it silently changes meaning when you widen a layer.
- Clipping is **not** a no-op under Adam. Adam normalises by a *running*
  second moment, so a single clipped step still lowers that step's contribution
  to $\hat v$ and damps the spike for many steps afterwards.

### The two traps
`if global_norm > max_norm: ...` raises `TracerBoolConversionError` the moment
you `jit` it — the norm is a traced value with no concrete truth value. Use
`jnp.minimum` (or `jnp.where`), which is branch-free and compiles to a select.

And every gradient can be exactly zero — a fully masked batch, a frozen
submodule, a dead ReLU block. Then the direct
`g * max_norm / global_norm` evaluates $0 \cdot \infty = \text{NaN}$ and poisons
every parameter on the next update. Writing the factor as
`jnp.minimum(1.0, max_norm / (global_norm + 1e-6))` fixes it twice over: the
`minimum` selects the safe branch, and the epsilon keeps the quotient finite in
the first place.

That fixes the **forward** pass. It does not fix the backward pass, and that is
the follow-up worth being ready for. $\sqrt{\cdot}$ has an *infinite* derivative
at zero, so `jax.grad` of `jnp.sqrt(jnp.sum(g ** 2))` at `g = 0` is
$0/0 = \text{NaN}$. Note where the epsilon above actually sits: in the
*division*, not under the root — so the implementation asked for here still
differentiates to `NaN` at `g = 0`.

That is fine for ordinary training, where the clip is applied to gradients and
never differentiated through. It is not fine for meta-learning, learned
optimizers, or unrolled inner loops, where clipping ends up inside the graph. To
make it safe there the epsilon has to move *inside* the root —
`jnp.sqrt(sq_sum + 1e-12)` — and you pay for it by no longer reporting a global
norm of exactly `0.0` on a zero tree (you get $10^{-6}$). Two different
epsilons, two different failures; knowing which one you need is the question.
""",
    "stub": '''import jax
import jax.numpy as jnp


def clip_by_global_norm(grads, max_norm):
    """Rescale a gradient pytree so its global L2 norm is at most `max_norm`.

    Args:
        grads:    arbitrary pytree of arrays
        max_norm: scalar clipping threshold

    Returns:
        (clipped_grads, global_norm) — the rescaled pytree (same structure as
        `grads`) and the global norm measured BEFORE clipping.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def clip_by_global_norm(grads, max_norm):
    leaves = jax.tree.leaves(grads)

    # ONE norm for the whole tree: treat every parameter as one long vector.
    sq_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    global_norm = jnp.sqrt(sq_sum)

    # Branch-free so it survives jit. On a zero tree the quotient would be
    # max_norm/0 = inf and `g * inf` would be 0 * inf = NaN; the eps keeps the
    # quotient finite, and the minimum then pins the scale at exactly 1.0.
    # Exactly 1.0 also means an under-threshold tree comes back bit-for-bit.
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))

    # The SAME scalar hits every leaf -> the direction is preserved exactly.
    clipped = jax.tree.map(lambda g: g * scale, grads)
    return clipped, global_norm
''',
    "demo": '''import jax
import jax.numpy as jnp

grads = {"enc": jnp.array([3.0, 4.0]), "dec": jnp.array([0.0, 0.0, 1.0])}

clipped, norm = clip_by_global_norm(grads, max_norm=1.0)
print("global norm:", norm)                       # sqrt(25 + 1) = 5.099
print("global-clipped:", clipped)

# What per-tensor clipping would have done instead:
per_tensor = jax.tree.map(
    lambda g: g * jnp.minimum(1.0, 1.0 / (jnp.linalg.norm(g) + 1e-6)), grads
)
print("per-tensor    :", per_tensor)

flat = lambda t: jnp.concatenate([l.ravel() for l in jax.tree.leaves(t)])
cos = lambda a, b: jnp.dot(flat(a), flat(b)) / (
    jnp.linalg.norm(flat(a)) * jnp.linalg.norm(flat(b))
)
print("cos(g, global)     =", cos(grads, clipped))      # exactly 1.0
print("cos(g, per-tensor) =", cos(grads, per_tensor))   # < 1.0 -> direction moved
''',
    "tests": [
        {
            "name": "Hand-computed 3-4-5 case across two leaves",
            "code": """
import jax
import jax.numpy as jnp

grads = {"a": jnp.array([3.0]), "b": jnp.array([4.0])}
clipped, norm = {fn}(grads, 1.0)

assert abs(float(norm) - 5.0) < 1e-5, (
    f'global_norm should be sqrt(3^2 + 4^2) = 5.0, got {float(norm)} — '
    'the norm must run over ALL leaves, not one leaf at a time'
)
assert jnp.allclose(clipped["a"], 0.6, atol=1e-4), f'a: {clipped["a"]} vs 0.6'
assert jnp.allclose(clipped["b"], 0.8, atol=1e-4), f'b: {clipped["b"]} vs 0.8'

new_norm = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree.leaves(clipped)))
assert float(new_norm) <= 1.0 + 1e-5, f'Clipped norm {float(new_norm)} exceeds max_norm 1.0'
assert float(new_norm) >= 1.0 - 1e-3, (
    f'Clipped norm {float(new_norm)} is well under 1.0 — clipping should land ON '
    'the boundary of the ball, not shrink further'
)
""",
        },
        {
            "name": "Returns the pre-clipping norm and no-ops below threshold",
            "code": """
import jax
import jax.numpy as jnp

grads = {"w": jnp.array([[0.003, -0.004], [0.0, 0.0]]), "b": jnp.array([0.0])}
expected = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree.leaves(grads)))

clipped, norm = {fn}(grads, 100.0)
assert abs(float(norm) - float(expected)) < 1e-7, (
    f'Returned {float(norm)}, expected the ORIGINAL norm {float(expected)} — '
    'do not return the norm measured after rescaling'
)
assert jnp.array_equal(clipped["w"], grads["w"]), (
    'Gradients under max_norm must come back bit-for-bit unchanged; '
    'the scale should be exactly 1.0, not tau/||g||'
)
assert jnp.array_equal(clipped["b"], grads["b"]), 'Zero leaf was modified'
""",
        },
        {
            "name": "Global, not per-tensor",
            "code": """
import jax
import jax.numpy as jnp

# Leaf "a" has norm 5, leaf "b" has norm 1, global norm is sqrt(26) = 5.099.
grads = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0, 0.0, 1.0])}
clipped, norm = {fn}(grads, 1.0)

assert abs(float(norm) - 26.0 ** 0.5) < 1e-4, (
    f'global_norm {float(norm)} should be sqrt(26) = 5.0990'
)

nb = float(jnp.linalg.norm(clipped["b"]))
assert nb < 0.5, (
    f'Leaf b still has norm {nb:.4f}. Per-tensor clipping would leave it at 1.0 '
    'because it is already under max_norm — but global clipping shrinks EVERY '
    'leaf by the same factor tau/||g||_global = 0.196'
)

# The ratio between leaves must be untouched: one scalar for the whole tree.
ratio_a = float(clipped["a"][0] / grads["a"][0])
ratio_b = float(clipped["b"][2] / grads["b"][2])
assert abs(ratio_a - ratio_b) < 1e-5, (
    f'Leaf a was scaled by {ratio_a:.5f} but leaf b by {ratio_b:.5f} — '
    'a single shared scalar must multiply every leaf'
)
""",
        },
        {
            "name": "Direction is preserved exactly",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.key(0)
k1, k2, k3 = jax.random.split(key, 3)
grads = {
    "enc": {"w": jax.random.normal(k1, (16, 8)) * 30.0},
    "dec": [jax.random.normal(k2, (8,)) * 0.01, jax.random.normal(k3, (4, 4)) * 5.0],
}

clipped, norm = {fn}(grads, 1.0)

flat_b = jnp.concatenate([l.ravel() for l in jax.tree.leaves(grads)])
flat_a = jnp.concatenate([l.ravel() for l in jax.tree.leaves(clipped)])
cos = float(
    jnp.dot(flat_b, flat_a) / (jnp.linalg.norm(flat_b) * jnp.linalg.norm(flat_a))
)
assert abs(cos - 1.0) < 1e-5, (
    f'cos(g, clipped) = {cos:.6f}, must be 1.0. A per-leaf multiplier rotates the '
    'update; global-norm clipping may only shorten it'
)
assert float(jnp.linalg.norm(flat_a)) <= 1.0 + 1e-5, (
    f'Clipped norm {float(jnp.linalg.norm(flat_a))} exceeds 1.0'
)
""",
        },
        {
            "name": "All-zero gradients do not produce NaN",
            "code": """
import jax
import jax.numpy as jnp

grads = {"w": jnp.zeros((4, 3)), "b": jnp.zeros((3,))}
clipped, norm = {fn}(grads, 1.0)

assert float(norm) == 0.0, f'Norm of a zero tree should be 0.0, got {float(norm)}'
for name, leaf in clipped.items():
    assert jnp.isfinite(leaf).all(), (
        f'Leaf {name} is non-finite: {leaf}. max_norm / 0.0 is inf and 0 * inf is '
        'NaN — guard the division, e.g. max_norm / (norm + 1e-6)'
    )
    assert jnp.allclose(leaf, 0.0), f'Leaf {name} should stay all zeros, got {leaf}'
""",
        },
        {
            "name": "Jittable and structure-preserving on a nested tree",
            "code": """
import jax
import jax.numpy as jnp

grads = {
    "blocks": [
        {"w": jnp.full((3, 3), 2.0), "b": jnp.full((3,), -1.0)},
        {"w": jnp.full((3, 3), 0.5), "b": jnp.zeros((3,))},
    ],
    "head": (jnp.full((3, 2), 4.0), jnp.array(7.0)),
}

eager, n_eager = {fn}(grads, 2.0)
assert jax.tree.structure(eager) == jax.tree.structure(grads), (
    'Output pytree structure differs from the input'
)
assert isinstance(eager["head"], tuple), 'Tuple nodes must stay tuples'
assert isinstance(eager["blocks"], list), 'List nodes must stay lists'

# A Python `if` on the norm dies here with a tracer error.
jitted, n_jit = jax.jit({fn})(grads, 2.0)
assert abs(float(n_jit) - float(n_eager)) < 1e-4, f'{float(n_jit)} vs {float(n_eager)}'
for a, b in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager)):
    assert jnp.allclose(a, b, atol=1e-6), 'jit and eager results disagree'

new_norm = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree.leaves(jitted)))
assert float(new_norm) <= 2.0 + 1e-4, f'Clipped norm {float(new_norm)} exceeds 2.0'
""",
        },
        {
            "name": "Works on real nnx gradients",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

model = nnx.Linear(6, 4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(1), (8, 6)) * 50.0
y = jax.random.normal(jax.random.key(2), (8, 4)) * 50.0

grads = nnx.grad(lambda m: jnp.mean((m(x) - y) ** 2))(model)
raw = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree.leaves(grads)))
assert float(raw) > 1.0, 'Test setup problem: gradients were already tiny'

clipped, norm = {fn}(grads, 1.0)
assert abs(float(norm) - float(raw)) < 1e-2 * max(1.0, float(raw)), (
    f'Returned norm {float(norm)} vs true {float(raw)}'
)
assert jax.tree.structure(clipped) == jax.tree.structure(grads), (
    'The nnx.State structure must be preserved so the optimizer can consume it'
)

new = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree.leaves(clipped)))
assert float(new) <= 1.0 + 1e-4, f'Clipped norm {float(new)} exceeds 1.0'
assert all(jnp.isfinite(l).all() for l in jax.tree.leaves(clipped)), 'Non-finite output'
""",
        },
    ],
}
