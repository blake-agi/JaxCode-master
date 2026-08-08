"""Hessian via jacfwd-of-jacrev — composing JAX's autodiff transforms."""

TASK = {
    "title": "Hessian with jacfwd(jacrev(f))",
    "category": "JAX Fundamentals",
    "order": 9,
    "difficulty": "Medium",
    "function_name": "hessian_matrix",
    "hint": (
        "jax.jacrev(f) gives the (D,) gradient; differentiating THAT gives the "
        "(D, D) Hessian. Use jax.jacfwd(jax.jacrev(f)) — forward-over-reverse is "
        "the efficient order, because the inner reverse pass gives you D outputs "
        "and forward mode costs one pass per input. Then just evaluate it at x."
    ),
    "description": r"""
Return the **Hessian matrix** of a scalar function at a point.

$$H_{ij} = \frac{\partial^2 f}{\partial x_i\, \partial x_j}$$

Given `f: (D,) -> scalar` and `x: (D,)`, return the `(D, D)` matrix of second
derivatives.

### Rules
- Compose `jax.jacfwd` and `jax.jacrev` — do not use `jax.hessian`
- Use the **forward-over-reverse** order: `jacfwd(jacrev(f))`
- Must work for any `D` and stay `jit`-able

### Signature
```python
def hessian_matrix(f, x):  # -> (D, D)
    ...
```

### Why the order matters
This is the real question behind the problem. For `f: R^D -> R`:

- `jacrev` costs **one** pass regardless of `D` (one output), so it is the right
  choice for the inner gradient
- `jacfwd` costs **one pass per input**, and it is applied to the `D`-dimensional
  gradient, giving `D` passes total
- Doing it the other way, `jacrev(jacfwd(f))`, computes the inner Jacobian in
  `D` forward passes and then reverse-differentiates that whole thing — same
  asymptotics but a much larger tape and more memory

Reverse mode is cheap in the number of **outputs**; forward mode is cheap in the
number of **inputs**. Being able to say that out loud is the point.
""",
    "stub": '''import jax
import jax.numpy as jnp


def hessian_matrix(f, x):
    """Hessian of a scalar function f at point x.

    Args:
        f: callable (D,) -> scalar
        x: (D,) array

    Returns:
        (D, D) array of second partial derivatives.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def hessian_matrix(f, x):
    # Inner jacrev: one reverse pass for the scalar output -> (D,) gradient.
    # Outer jacfwd: D forward passes over that gradient -> (D, D) Hessian.
    return jax.jacfwd(jax.jacrev(f))(x)
''',
    "demo": '''import jax.numpy as jnp

# f(x) = x0^2 + 3*x0*x1 + 2*x1^2
# H = [[2, 3], [3, 4]]
f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2

print(hessian_matrix(f, jnp.array([1.0, 1.0])))

# A quadratic form x^T A x has Hessian A + A^T, independent of x:
A = jnp.array([[1.0, 2.0], [0.0, 3.0]])
q = lambda x: x @ A @ x
print(hessian_matrix(q, jnp.array([5.0, -7.0])))   # == A + A.T
''',
    "tests": [
        {
            "name": "Hand-computed 2-D quadratic",
            "code": """
import jax.numpy as jnp

f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2
H = {fn}(f, jnp.array([1.0, 1.0]))

expected = jnp.array([[2.0, 3.0], [3.0, 4.0]])
assert H.shape == (2, 2), f'Shape mismatch: {H.shape} vs (2, 2)'
assert jnp.allclose(H, expected, atol=1e-5), f'{H} vs {expected}'
""",
        },
        {
            "name": "Quadratic form: H = A + A^T, constant in x",
            "code": """
import jax
import jax.numpy as jnp

A = jax.random.normal(jax.random.key(0), (5, 5))
f = lambda x: x @ A @ x

for seed in [1, 2, 3]:
    x = jax.random.normal(jax.random.key(seed), (5,))
    H = {fn}(f, x)
    assert H.shape == (5, 5), f'{H.shape} vs (5, 5)'
    assert jnp.allclose(H, A + A.T, atol=1e-4), 'Hessian of x^T A x must be A + A^T'
""",
        },
        {
            "name": "Symmetry and a non-quadratic function",
            "code": """
import jax
import jax.numpy as jnp

f = lambda x: jnp.sum(jnp.sin(x) * jnp.exp(x[0]))
x = jnp.array([0.3, -0.7, 1.1])
H = {fn}(f, x)

assert H.shape == (3, 3), f'{H.shape}'
assert jnp.allclose(H, H.T, atol=1e-4), f'Hessian must be symmetric:\\n{H}'
assert jnp.isfinite(H).all(), 'Non-finite entries'
assert jnp.allclose(H, jax.hessian(f)(x), atol=1e-4), 'Disagrees with jax.hessian'
""",
        },
        {
            "name": "Diagonal case",
            "code": """
import jax.numpy as jnp

# f(x) = sum(x^3)  ->  H = diag(6x)
f = lambda x: jnp.sum(x ** 3)
x = jnp.array([1.0, 2.0, 3.0, 4.0])
H = {fn}(f, x)

assert H.shape == (4, 4), f'{H.shape}'
assert jnp.allclose(jnp.diag(H), 6 * x, atol=1e-4), f'diag {jnp.diag(H)} vs {6 * x}'
off = H - jnp.diag(jnp.diag(H))
assert jnp.allclose(off, 0.0, atol=1e-5), 'Off-diagonal entries should be zero'
""",
        },
        {
            "name": "D=1 and jit",
            "code": """
import jax
import jax.numpy as jnp

f = lambda x: x[0] ** 4
H = {fn}(f, jnp.array([2.0]))
assert H.shape == (1, 1), f'{H.shape} vs (1, 1)'
assert jnp.allclose(H, 48.0, atol=1e-3), f'12*x^2 at x=2 is 48, got {H}'

g = lambda x: jnp.sum(x ** 2) + x[0] * x[1]
jitted = jax.jit(lambda x: {fn}(g, x))
Hj = jitted(jnp.array([1.0, 2.0]))
assert jnp.allclose(Hj, jnp.array([[2.0, 1.0], [1.0, 2.0]]), atol=1e-5), f'{Hj}'
""",
        },
    ],
}
