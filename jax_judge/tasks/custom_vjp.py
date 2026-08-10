"""Numerically stable log1pexp via custom_vjp — overriding autodiff."""

TASK = {
    "title": "Stable log(1+exp(x)) with custom_vjp",
    "category": "JAX Fundamentals",
    "order": 8,
    "difficulty": "Hard",
    "function_name": "log1pexp",
    "hint": (
        "The three pieces have fixed shapes and none is optional. fwd(x) returns "
        "(output, residuals) — residuals are whatever the backward rule needs, so "
        "ask yourself whether that is x or the output here. bwd(residuals, g) "
        "receives the INCOMING cotangent g and must return a tuple, one entry per "
        "primal argument, so a single input still needs the trailing comma — and g "
        "has to be multiplied into your derivative, not dropped (with jax.grad it "
        "is 1.0, which hides the bug). Note that fwd, not the decorated function, "
        "is what runs when you differentiate, so fwd needs to be stable too. "
        "Nothing is registered until you call .defvjp(fwd, bwd)."
    ),
    "description": r"""
Implement the softplus function

$$f(x) = \log(1 + e^{x})$$

so that **both** the value and its gradient are numerically stable for large `x`.

### The problem
The naive implementation is a classic trap:

```python
def log1pexp(x):
    return jnp.log(1.0 + jnp.exp(x))

log1pexp(100.0)            # inf   — exp(100) overflows
jax.grad(log1pexp)(100.0)  # nan   — autodiff differentiates through the inf
```

The true value at `x = 100` is `100.0` (to float precision) and the true
gradient is `1.0`. Autodiff is doing exactly what you told it — the fix is to
tell it something better.

### Rules
- Decorate with `@jax.custom_vjp` and register `defvjp(fwd, bwd)`
- The **forward** pass must be stable — `logaddexp`, not `log(1 + exp(x))`
- The **backward** pass must return the analytic derivative
  $f'(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$
- `bwd` must return a **tuple**, one entry per primal input
- Must remain `jit`-able and `vmap`-able

### Why it matters
`custom_vjp` is how you rescue a gradient that is mathematically fine but
numerically catastrophic — the same technique behind stable `logsumexp`, the
straight-through estimator in quantization, gradient reversal layers, and
implicit differentiation of solvers. Knowing *when* autodiff needs help, and how
to hand it the analytic answer, is a strong senior-level signal.
""",
    "stub": '''import jax
import jax.numpy as jnp


@jax.custom_vjp
def log1pexp(x):
    """Numerically stable log(1 + exp(x))."""
    pass  # Replace this


def log1pexp_fwd(x):
    """Return (output, residuals_needed_by_bwd)."""
    pass  # Replace this


def log1pexp_bwd(res, g):
    """Return a TUPLE of gradients, one per primal input."""
    pass  # Replace this


log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)
''',
    "solution": '''import jax
import jax.numpy as jnp


@jax.custom_vjp
def log1pexp(x):
    # logaddexp(0, x) == log(exp(0) + exp(x)) == log(1 + exp(x)),
    # computed with the max factored out so nothing overflows.
    return jnp.logaddexp(0.0, x)


def log1pexp_fwd(x):
    return jnp.logaddexp(0.0, x), x      # stash x for the backward pass


def log1pexp_bwd(res, g):
    x = res
    return (g * jax.nn.sigmoid(x),)      # d/dx log(1+e^x) = sigmoid(x)


log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)
''',
    "demo": '''import jax
import jax.numpy as jnp

naive = lambda x: jnp.log(1.0 + jnp.exp(x))

for x in [0.0, 10.0, 100.0]:
    print(f"x={x:>6}  yours={log1pexp(x):>10.4f}  naive={naive(x):>10.4f}")

print()
for x in [0.0, 100.0]:
    print(f"x={x:>6}  grad yours={jax.grad(log1pexp)(x):.6f}  "
          f"grad naive={jax.grad(naive)(x):.6f}")
''',
    "tests": [
        {
            "name": "Forward values on a safe range",
            "code": """
import jax.numpy as jnp

for x in [-5.0, -1.0, 0.0, 1.0, 5.0, 20.0]:
    got = float({fn}(x))
    want = float(jnp.logaddexp(0.0, jnp.asarray(x, jnp.float32)))
    assert abs(got - want) < 1e-4, f'log1pexp({x}) = {got}, expected {want}'

# log(1 + e^0) = log 2
assert abs(float({fn}(0.0)) - 0.6931472) < 1e-5, f'{float({fn}(0.0))}'
""",
        },
        {
            "name": "Forward is stable for large x",
            "code": """
import jax.numpy as jnp

for x in [50.0, 100.0, 500.0]:
    out = float({fn}(x))
    assert jnp.isfinite(out), (
        f'log1pexp({x}) = {out} — overflowed. Use jnp.logaddexp(0.0, x) '
        'instead of jnp.log(1 + jnp.exp(x)).'
    )
    # For large x, log(1+e^x) -> x
    assert abs(out - x) < 1e-3, f'log1pexp({x}) should approach {x}, got {out}'
""",
        },
        {
            "name": "Gradient is stable for large x",
            "code": """
import jax
import jax.numpy as jnp

g = float(jax.grad({fn})(100.0))
assert not jnp.isnan(g), (
    'grad at x=100 is NaN — the custom_vjp backward rule is not being used. '
    'Register it with log1pexp.defvjp(fwd, bwd).'
)
assert abs(g - 1.0) < 1e-5, f'grad at x=100 should be 1.0, got {g}'

g500 = float(jax.grad({fn})(500.0))
assert jnp.isfinite(g500) and abs(g500 - 1.0) < 1e-5, f'grad at x=500: {g500}'
""",
        },
        {
            "name": "Gradient equals sigmoid everywhere",
            "code": """
import jax
import jax.numpy as jnp

xs = jnp.array([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
got = jax.vmap(jax.grad({fn}))(xs)
want = jax.nn.sigmoid(xs)

assert jnp.allclose(got, want, atol=1e-5), f'{got} vs sigmoid {want}'
assert abs(float(jax.grad({fn})(0.0)) - 0.5) < 1e-6, 'grad at 0 must be 0.5'

# The incoming cotangent must be USED. jax.grad hands bwd g = 1.0, which hides
# a rule that returns sigmoid(x) instead of g * sigmoid(x); scaling exposes it.
scaled = float(jax.grad(lambda t: 3.0 * {fn}(t))(0.0))
assert abs(scaled - 1.5) < 1e-5, (
    f'd/dx [3 * log1pexp(x)] at 0 should be 1.5, got {scaled} — bwd must return '
    'g * sigmoid(x), not sigmoid(x). The cotangent g is its second argument.'
)

# Composition, so the cotangent reaching bwd is not a constant either.
chained = float(jax.grad(lambda t: {fn}(2.0 * t))(0.0))
assert abs(chained - 1.0) < 1e-5, f'chain rule through log1pexp(2x) at 0: {chained}'
""",
        },
        {
            "name": "Works under jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

xs = jnp.array([-2.0, 0.0, 2.0, 100.0])

vals = jax.jit(jax.vmap({fn}))(xs)
assert jnp.isfinite(vals).all(), f'Non-finite under jit+vmap: {vals}'
assert jnp.allclose(vals[:3], jnp.logaddexp(0.0, xs[:3]), atol=1e-5), f'{vals}'
assert abs(float(vals[3]) - 100.0) < 1e-3, f'{vals[3]}'

grads = jax.jit(jax.vmap(jax.grad({fn})))(xs)
assert jnp.isfinite(grads).all(), f'Non-finite grads under jit+vmap: {grads}'
assert jnp.allclose(grads, jax.nn.sigmoid(xs), atol=1e-5), f'{grads}'
""",
        },
        {
            "name": "custom_vjp is actually registered",
            "code": """
import jax

assert isinstance({fn}, jax.custom_derivatives.custom_vjp), (
    f'{fn} is not a jax.custom_vjp object (got {type({fn})}). '
    'Decorate the function with @jax.custom_vjp.'
)
assert getattr({fn}, "fwd", None) is not None and getattr({fn}, "bwd", None) is not None, (
    'defvjp(fwd, bwd) was never called — the custom rule is not registered.'
)
""",
        },
    ],
}
