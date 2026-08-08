"""Newton's method under lax.while_loop — data-dependent control flow in jit."""

TASK = {
    "title": "Newton's Method with lax.while_loop",
    "category": "JAX Fundamentals",
    "order": 7,
    "difficulty": "Medium",
    "function_name": "newton_sqrt",
    "hint": (
        "jax.lax.while_loop(cond_fun, body_fun, init_val) needs both functions to "
        "take and return the SAME carry structure, and cond_fun must return a "
        "scalar boolean array. Carry a tuple (guess, iteration) so you can stop on "
        "either convergence or the iteration cap. Handle x == 0 up front — "
        "the update divides by the guess."
    ),
    "description": r"""
Implement $\sqrt{x}$ with **Newton's method**, using `jax.lax.while_loop` so it
runs inside `jit`.

$$g_{n+1} = \frac{1}{2}\left(g_n + \frac{x}{g_n}\right)$$

Iterate until $|g_{n+1}^2 - x| < \text{tol}$ or `max_iters` is reached.

### Rules
- Use `jax.lax.while_loop` — a Python `while` on a traced value raises
  `ConcretizationTypeError` under `jit`
- `x` is a **scalar**; the function must be `jit`-able and `vmap`-able
- Return `0.0` for `x == 0` (the update would divide by zero)
- Do not call `jnp.sqrt`, `x ** 0.5`, or `jnp.power`

### Signature
```python
def newton_sqrt(x, tol=1e-6, max_iters=50):
    ...
```

### Why it matters
`while_loop` is the escape hatch for **data-dependent** iteration counts — the
loop runs until a condition on traced values is met, which a Python loop cannot
express under tracing. The catch, and the thing interviewers probe: `while_loop`
is **not reverse-mode differentiable**, because the trip count is not known at
trace time. If you need gradients through an iterative solver, you either use
`lax.scan` with a fixed count, or `implicit differentiation` on the fixed point.
""",
    "stub": '''import jax
import jax.numpy as jnp


def newton_sqrt(x, tol=1e-6, max_iters=50):
    """Square root of a non-negative scalar via Newton's method.

    Args:
        x:         non-negative scalar
        tol:       stop when |guess**2 - x| < tol
        max_iters: hard iteration cap

    Returns:
        Scalar approximation of sqrt(x).
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def newton_sqrt(x, tol=1e-6, max_iters=50):
    x = jnp.asarray(x, dtype=jnp.float32)

    def cond(carry):
        guess, i = carry
        return (jnp.abs(guess * guess - x) >= tol) & (i < max_iters)

    def body(carry):
        guess, i = carry
        return 0.5 * (guess + x / guess), i + 1

    # Start from x itself (any positive seed converges for x > 0).
    init = (jnp.maximum(x, 1.0), jnp.array(0))
    guess, _ = jax.lax.while_loop(cond, body, init)

    # x == 0 would divide by zero in the body, so special-case it.
    return jnp.where(x == 0, 0.0, guess)
''',
    "demo": '''import jax
import jax.numpy as jnp

for v in [0.0, 1.0, 2.0, 16.0, 1e6]:
    print(f"newton_sqrt({v:>9}) = {newton_sqrt(v):.6f}   (jnp.sqrt = {jnp.sqrt(v):.6f})")

# Works under vmap, so you can do a whole array at once:
print(jax.vmap(newton_sqrt)(jnp.array([4.0, 9.0, 25.0])))
''',
    "tests": [
        {
            "name": "Known values",
            "code": """
import jax.numpy as jnp

for v, want in [(4.0, 2.0), (9.0, 3.0), (16.0, 4.0), (2.0, 1.4142135), (1.0, 1.0)]:
    got = float({fn}(v))
    assert abs(got - want) < 1e-3, f'sqrt({v}) = {got}, expected {want}'
""",
        },
        {
            "name": "Zero does not blow up",
            "code": """
import jax.numpy as jnp

out = {fn}(0.0)
assert jnp.isfinite(out), f'sqrt(0) gave {out} — guard the division by the guess'
assert abs(float(out)) < 1e-5, f'sqrt(0) should be 0.0, got {out}'
""",
        },
        {
            "name": "Wide dynamic range",
            "code": """
import jax.numpy as jnp

for v in [1e-4, 0.25, 100.0, 12345.0, 1e6]:
    got = float({fn}(v))
    want = float(jnp.sqrt(jnp.asarray(v, jnp.float32)))
    assert abs(got - want) / max(want, 1e-6) < 1e-3, (
        f'sqrt({v}) = {got}, expected ~{want}'
    )
""",
        },
        {
            "name": "Runs under jit (a Python while would fail here)",
            "code": """
import functools
import jax
import jax.numpy as jnp

f = jax.jit({fn})
out = f(2.0)
assert abs(float(out) - 1.4142135) < 1e-3, f'{out}'

# Tracing must not depend on the VALUE of x.
for v in [4.0, 100.0, 0.0]:
    got, want = float(f(v)), float(jnp.sqrt(jnp.asarray(v, jnp.float32)))
    assert abs(got - want) < 1e-2, f'jit sqrt({v}) = {got}, expected {want}'
""",
        },
        {
            "name": "vmap over a batch",
            "code": """
import jax
import jax.numpy as jnp

xs = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0, 100.0])
out = jax.vmap({fn})(xs)

assert out.shape == (6,), f'{out.shape} vs (6,)'
assert jnp.allclose(out, jnp.sqrt(xs), atol=1e-3), f'{out} vs {jnp.sqrt(xs)}'
""",
        },
        {
            "name": "Respects the iteration cap",
            "code": """
import jax.numpy as jnp

# With a single iteration from the seed the answer must be far from converged,
# which proves max_iters is actually consulted by the loop condition.
coarse = float({fn}(1e6, tol=1e-12, max_iters=1))
exact = float(jnp.sqrt(jnp.asarray(1e6, jnp.float32)))
assert jnp.isfinite(coarse), f'Non-finite result with max_iters=1: {coarse}'
assert abs(coarse - exact) > 1.0, (
    f'max_iters=1 returned {coarse}, which is already converged to {exact} — '
    'the iteration cap does not appear to be part of the loop condition'
)

# And with plenty of iterations it does converge.
fine = float({fn}(1e6, tol=1e-6, max_iters=100))
assert abs(fine - exact) / exact < 1e-3, f'{fine} vs {exact}'
""",
        },
    ],
}
