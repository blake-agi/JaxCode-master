"""Stack a list of identically-structured pytrees — tree_map over many trees."""

TASK = {
    "title": "Stack a List of Pytrees",
    "category": "JAX Fundamentals",
    "order": 4,
    "number": "b_04",
    "difficulty": "Medium",
    "function_name": "tree_stack",
    "hint": (
        "jax.tree.map(f, tree1, tree2, ...) walks several trees in lockstep and "
        "calls f once per corresponding group of leaves. So "
        "jax.tree.map(lambda *ls: jnp.stack(ls), *trees) does the whole job — "
        "the *ls catches one leaf from each tree. Validate structures first with "
        "jax.tree.structure so mismatches fail loudly."
    ),
    "description": r"""
Given a **list of pytrees** that all share the same structure, produce a single
pytree of the same structure where each leaf is the `jnp.stack` of the
corresponding leaves.

```
[{"w": (3, 2), "b": (2,)},        ->   {"w": (4, 3, 2), "b": (4, 2)}
 {"w": (3, 2), "b": (2,)},
 {"w": (3, 2), "b": (2,)},
 {"w": (3, 2), "b": (2,)}]
```

### Rules
- Handle arbitrary nesting: dicts, lists, tuples, and mixtures of them
- Raise `ValueError` if the input list is empty
- Raise `ValueError` if the trees do not all share the same structure
- No hand-written recursion over dicts — use the `jax.tree` utilities

### Why it matters
This is how you build an **ensemble**: stack N independently-initialised
parameter sets into one pytree, then `vmap` your model over the leading axis to
run all N models in a single batched call. The same trick collects per-step
metrics from a training loop into arrays, and assembles the `xs` argument for
`lax.scan`.

Interviewers like it because the naive answer is a pile of nested loops, and the
JAX answer is one line of `tree.map` with a variadic lambda.
""",
    "stub": '''import jax
import jax.numpy as jnp


def tree_stack(trees):
    """Stack a list of same-structure pytrees into one pytree.

    Args:
        trees: non-empty list of pytrees, all with identical structure

    Returns:
        A pytree with the same structure, where each leaf has a new leading
        axis of size len(trees).

    Raises:
        ValueError: if `trees` is empty or the structures do not match.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def tree_stack(trees):
    trees = list(trees)
    if not trees:
        raise ValueError("tree_stack requires at least one tree")

    ref = jax.tree.structure(trees[0])
    for i, t in enumerate(trees[1:], start=1):
        s = jax.tree.structure(t)
        if s != ref:
            raise ValueError(
                f"tree {i} has structure {s}, expected {ref}"
            )

    # tree.map walks all trees in lockstep; *leaves collects one leaf per tree.
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)
''',
    "demo": '''import jax
import jax.numpy as jnp

# Three "models", each a small parameter pytree.
models = [
    {"w": jnp.full((3, 2), float(i)), "b": jnp.full((2,), float(i))}
    for i in range(3)
]

stacked = tree_stack(models)
print("leaf shapes:", jax.tree.map(lambda a: a.shape, stacked))
print("stacked['b']:\\n", stacked["b"])
''',
    "tests": [
        {
            "name": "Flat dict of arrays",
            "code": """
import jax
import jax.numpy as jnp

trees = [
    {"w": jnp.full((3, 2), float(i)), "b": jnp.full((2,), float(i))}
    for i in range(4)
]
out = {fn}(trees)

assert jax.tree.structure(out) == jax.tree.structure(trees[0]), 'Structure changed'
assert out["w"].shape == (4, 3, 2), f'w shape {out["w"].shape} vs (4, 3, 2)'
assert out["b"].shape == (4, 2), f'b shape {out["b"].shape} vs (4, 2)'
assert jnp.allclose(out["w"][2], 2.0), 'Leading axis must index the input list'
assert jnp.allclose(out["b"][0], 0.0), 'Wrong ordering along the new axis'
""",
        },
        {
            "name": "Nested mixed containers",
            "code": """
import jax
import jax.numpy as jnp

def make(i):
    return {
        "enc": [{"w": jnp.full((2, 2), float(i))}, {"w": jnp.full((2,), float(i))}],
        "dec": ({"scale": jnp.array(float(i))}, jnp.full((3,), float(i))),
    }

trees = [make(i) for i in range(5)]
out = {fn}(trees)

assert jax.tree.structure(out) == jax.tree.structure(trees[0]), 'Structure changed'
assert out["enc"][0]["w"].shape == (5, 2, 2), f'{out["enc"][0]["w"].shape}'
assert out["enc"][1]["w"].shape == (5, 2), f'{out["enc"][1]["w"].shape}'
assert out["dec"][0]["scale"].shape == (5,), f'{out["dec"][0]["scale"].shape}'
assert out["dec"][1].shape == (5, 3), f'{out["dec"][1].shape}'
assert isinstance(out["dec"], tuple), 'Tuple nodes must stay tuples'
assert isinstance(out["enc"], list), 'List nodes must stay lists'
assert jnp.allclose(out["dec"][0]["scale"], jnp.arange(5.0)), 'Scalar leaves mis-stacked'
""",
        },
        {
            "name": "Single-element list",
            "code": """
import jax.numpy as jnp

out = {fn}([{"a": jnp.ones((2, 3))}])
assert out["a"].shape == (1, 2, 3), f'{out["a"].shape} vs (1, 2, 3)'
assert jnp.allclose(out["a"][0], 1.0)
""",
        },
        {
            "name": "Empty list raises ValueError",
            "code": """
try:
    {fn}([])
except ValueError:
    pass
else:
    raise AssertionError('Empty input must raise ValueError')
""",
        },
        {
            "name": "Mismatched structures raise ValueError",
            "code": """
import jax.numpy as jnp

a = {"w": jnp.ones((2,)), "b": jnp.ones((2,))}
b = {"w": jnp.ones((2,))}                      # missing key
try:
    {fn}([a, b])
except ValueError:
    pass
else:
    raise AssertionError('Mismatched structures must raise ValueError')

c = {"w": jnp.ones((2,)), "b": [jnp.ones((2,))]}   # different nesting
try:
    {fn}([a, c])
except ValueError:
    pass
else:
    raise AssertionError('Different nesting must raise ValueError')
""",
        },
        {
            "name": "Enables vmap over an ensemble",
            "code": """
import jax
import jax.numpy as jnp

# Build 8 independent linear models, stack them, run all 8 at once with vmap.
keys = jax.random.split(jax.random.key(0), 8)
models = [
    {"w": jax.random.normal(k, (4, 3)), "b": jnp.zeros(3)} for k in keys
]
stacked = {fn}(models)

x = jnp.ones((4,))
apply = lambda p, v: v @ p["w"] + p["b"]
batched = jax.vmap(apply, in_axes=(0, None))(stacked, x)

assert batched.shape == (8, 3), f'{batched.shape} vs (8, 3)'
for i in range(8):
    assert jnp.allclose(batched[i], apply(models[i], x), atol=1e-5), (
        f'Ensemble member {i} disagrees with running the model on its own'
    )
""",
        },
    ],
}
