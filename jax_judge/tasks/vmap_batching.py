"""Pairwise squared distances via nested vmap — the canonical vmap exercise."""

TASK = {
    "title": "Pairwise Distances with vmap",
    "category": "JAX Fundamentals",
    "number": "b_02",
    "difficulty": "Easy",
    "function_name": "pairwise_sq_dist",
    "hint": (
        "Write the problem for ONE pair of vectors first: two 1-D arrays in, a "
        "scalar out. Then add the batch axes with vmap, one at a time. Each vmap "
        "maps over one argument while holding the other fixed, and in_axes=None is "
        "how you say 'this argument is not batched'. You need two nested vmaps — "
        "and the order you nest them in decides whether you get (N, M) or its "
        "transpose, so reason about which axis the OUTER one contributes."
    ),
    "description": r"""
Given `X` of shape `(N, D)` and `Y` of shape `(M, D)`, compute the matrix of
**squared Euclidean distances** of shape `(N, M)`:

$$D_{ij} = \|x_i - y_j\|_2^2 = \sum_{d} (X_{id} - Y_{jd})^2$$

The point of this problem is *not* the math — it is learning to write the
function for a **single example** and let `jax.vmap` add the batch dimensions.
This "write one, vmap the rest" habit is what interviewers are looking for.

### Rules
- Write a single-pair helper, then compose **two** `vmap`s around it
- No Python `for` loops over N or M
- Do **not** use the expand-dims broadcasting trick (`X[:, None] - Y[None]`);
  the exercise is specifically about `vmap` and `in_axes`
- Do not use `jnp.linalg.norm` or `scipy` distance helpers

### Example
```
X shape (3, 2), Y shape (5, 2)  ->  output shape (3, 5)
```

### Why it matters
The expand-dims version materializes an `(N, M, D)` intermediate. `vmap` expresses
the same computation without you hand-managing axes, and it composes with `grad`
and `jit`. Being fluent with `in_axes=(None, 0)` vs `(0, None)` is a very common
JAX screening question.
""",
    "stub": '''def pairwise_sq_dist(X, Y):
    """Squared Euclidean distances between every row of X and every row of Y.

    Args:
        X: (N, D) array
        Y: (M, D) array

    Returns:
        (N, M) array where out[i, j] = ||X[i] - Y[j]||^2
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def pairwise_sq_dist(X, Y):
    # The whole problem, for one pair of vectors.
    def sq_dist(x, y):
        diff = x - y
        return jnp.sum(diff * diff)

    # One row: fix x, sweep over every y.
    row = jax.vmap(sq_dist, in_axes=(None, 0))
    # Full matrix: sweep that row-builder over every x.
    return jax.vmap(row, in_axes=(0, None))(X, Y)
''',
    "demo": '''import jax.numpy as jnp

X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
Y = jnp.array([[0.0, 0.0], [1.0, 1.0]])

out = pairwise_sq_dist(X, Y)
print("X:", X.shape, " Y:", Y.shape)
print("out shape:", out.shape, "(expected (3, 2))")
print(out)
# Row 0 is distance from origin to each Y -> [0., 2.]
''',
    "tests": [
        {
            "name": "Known small case",
            "code": """
import jax.numpy as jnp

X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
Y = jnp.array([[0.0, 0.0], [1.0, 1.0]])
out = {fn}(X, Y)

expected = jnp.array([[0.0, 2.0], [1.0, 1.0], [1.0, 1.0]])
assert out.shape == (3, 2), f'Shape mismatch: {out.shape} vs (3, 2)'
assert jnp.allclose(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Rectangular shapes",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(0), (7, 4))
Y = jax.random.normal(jax.random.key(1), (11, 4))
out = {fn}(X, Y)

assert out.shape == (7, 11), f'Shape mismatch: {out.shape} vs (7, 11)'
expected = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
assert jnp.allclose(out, expected, atol=1e-4), 'Values differ from reference'
""",
        },
        {
            "name": "Self-distance diagonal is zero",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(2), (6, 3))
out = {fn}(X, X)

assert out.shape == (6, 6), f'{out.shape}'
assert jnp.allclose(jnp.diag(out), 0.0, atol=1e-5), 'Diagonal must be zero'
assert jnp.allclose(out, out.T, atol=1e-5), 'Self-distance matrix must be symmetric'
assert (out >= -1e-6).all(), 'Squared distances must be non-negative'
""",
        },
        {
            "name": "Single row edge cases",
            "code": """
import jax.numpy as jnp

X = jnp.array([[1.0, 2.0, 3.0]])
Y = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
out = {fn}(X, Y)

assert out.shape == (1, 2), f'Shape mismatch on N=1: {out.shape}'
assert jnp.allclose(out, jnp.array([[14.0, 0.0]])), f'{out}'

# D = 1 should still work
out1 = {fn}(jnp.array([[1.0], [3.0]]), jnp.array([[0.0]]))
assert out1.shape == (2, 1), f'Shape mismatch on D=1: {out1.shape}'
assert jnp.allclose(out1, jnp.array([[1.0], [9.0]])), f'{out1}'
""",
        },
        {
            "name": "Differentiable and jittable",
            "code": """
import jax
import jax.numpy as jnp

X = jax.random.normal(jax.random.key(3), (5, 2))
Y = jax.random.normal(jax.random.key(4), (4, 2))

jitted = jax.jit({fn})
assert jnp.allclose(jitted(X, Y), {fn}(X, Y), atol=1e-5), 'jit changes the result'

g = jax.grad(lambda a, b: jnp.sum({fn}(a, b)))(X, Y)
assert g.shape == X.shape, f'Gradient shape {g.shape} vs {X.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'

# d/dX sum_ij ||xi - yj||^2 = sum_j 2 (xi - yj)
expected_g = 2 * (Y.shape[0] * X - jnp.sum(Y, axis=0))
assert jnp.allclose(g, expected_g, atol=1e-4), 'Gradient value mismatch'
""",
        },
    ],
}
