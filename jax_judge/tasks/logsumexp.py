"""logsumexp — the max trick again, but the shape discipline is inverted."""

TASK = {
    "title": "LogSumExp",
    "category": "Core Ops & Layers",
    "number": "b_12",
    "difficulty": "Medium",
    "function_name": "logsumexp",
    "extra_names": ["logsumexp_merge"],
    "hint": (
        "Part 1: subtract the per-slice max before exp(), exactly as in softmax "
        "— but watch what happens when you add it back. You need keepdims=True "
        "on the max so it broadcasts against x for the subtraction, and then the "
        "sum reduces the axis away, so the two operands no longer line up. "
        "Adding a (..., 1) max to a (...,) sum does not error: it broadcasts "
        "into an outer sum of the wrong rank. Squeeze the axis back out, or keep "
        "it on both and let keepdims decide at the end. "
        "Part 2: to merge two (max, sum-of-exp) states, take the new max and "
        "rescale BOTH sums onto it before adding — l * exp(m_old - m_new). The "
        "empty state is (-inf, 0), and exp(-inf - -inf) is nan, so pin the "
        "exponent when the new max is not finite."
    ),
    "description": r"""
Implement **logsumexp** two ways: as a normal reduction, and as a *streaming*
merge that never sees all the data at once.

$$\text{logsumexp}(x) = \log \sum_i e^{x_i}
= m + \log \sum_i e^{x_i - m}, \qquad m = \max_i x_i$$

### Signatures
```python
def logsumexp(x, axis=-1, keepdims=False): ...
def logsumexp_merge(m1, l1, m2, l2): ...      # -> (m, l)
```

### Rules
- No `jax.scipy.special.logsumexp` and no `jax.nn.logsumexp`
- Stable for large positive **and** large negative inputs
- `axis` may be any valid axis; `keepdims` must behave like every other
  reduction

---

## Part 1 — the reduction

### The trap that makes this its own problem
Softmax uses the same max trick, so it is tempting to assume the same shape
handling carries over. It does not, and the difference is the point.

Softmax **divides** by its sum. With `keepdims=True` on both reductions the
shapes cancel, so uniform `keepdims` is simply correct:

```python
z = x - x.max(axis, keepdims=True)             # (2, 3)
e = jnp.exp(z) / jnp.sum(..., keepdims=True)   # (2, 3) / (2, 1) -> (2, 3)  ✓
```

logsumexp **adds** the max back. The sum has already reduced the axis away, so
the operands no longer match:

```python
x_max = jnp.max(x, axis=axis, keepdims=True)              # (2, 1)
s = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis))       # (2,)
s + x_max     # (2,) + (2, 1) -> (2, 2)   WRONG, and it does not raise
```

That is an outer sum. On a `(2, 3)` input it silently returns `(2, 2)` with each
row's value duplicated. Nothing errors, and a test that only checks values at
`[0]` still passes. You need the max at the *reduced* rank —
`x_max.squeeze(axis)` — or `keepdims=True` on both and one squeeze at the end.

A square input hides this completely: `(3,) + (3, 1)` broadcasts to `(3, 3)`
without complaint. Test with something like `(2, 3)`.

### Why the max must be per-slice
Shifting by any constant is exact, so on a single vector a global `max(x)` also
works. Across slices on different scales it does not: subtract a global `1000`
from a row sitting near `-1000` and every term underflows to `0`, leaving
`log(0) = -inf`. Reduce along the axis you are reducing over.

---

## Part 2 — the streaming merge

Represent a partial result as the pair $(m, \ell)$ with
$m = \max x_i$ and $\ell = \sum_i e^{x_i - m}$, so the answer is
$m + \log \ell$. `logsumexp_merge` combines two such pairs:

$$m = \max(m_1, m_2), \qquad
\ell = \ell_1 e^{m_1 - m} + \ell_2 e^{m_2 - m}$$

Both sums are rescaled onto the *new* max before adding — that rescale is the
whole trick, and it is why the running total never overflows no matter what
order the chunks arrive in.

This is exactly FlashAttention's inner loop (problem 25): the running `m` and
`l` carried across key tiles, with everything accumulated so far retroactively
rescaled whenever a later tile raises the maximum. It is also why attention
never needs to materialise the full `(seq_q, seq_k)` score matrix.

### The empty state
The identity element is $(-\infty,\, 0)$ — merging it changes nothing, which is
what lets you start a fold from it. Watch the arithmetic: when *both* inputs are
$-\infty$ the new max is $-\infty$ too, and `exp(-inf - -inf)` is `nan`, not `0`.
Pin the exponent when the max is not finite.

### A useful identity
$$\frac{\partial}{\partial x_i}\,\text{logsumexp}(x) = \text{softmax}(x)_i$$

so the gradient is a probability distribution and sums to 1 — the fastest way to
check your implementation differentiates correctly.
""",
    "stub": '''import jax
import jax.numpy as jnp


def logsumexp(x, axis=-1, keepdims=False):
    """Numerically stable log(sum(exp(x))) along `axis`.

    Args:
        x:        array of any shape
        axis:     axis to reduce over
        keepdims: if True, keep the reduced axis as a length-1 dimension

    Returns:
        Array with `axis` reduced (or kept as length 1 when keepdims=True).
    """
    pass  # Replace this


def logsumexp_merge(m1, l1, m2, l2):
    """Combine two (max, sum-of-exp) partial states into one.

    Each state means "some chunk of data whose max is m and whose shifted
    exponential sum is l", so its logsumexp is m + log(l).

    Args:
        m1, l1: the first chunk's running max and shifted sum
        m2, l2: the second chunk's

    Returns:
        (m, l) for the two chunks combined.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def logsumexp(x, axis=-1, keepdims=False):
    # keepdims=True here so the max broadcasts against x for the subtraction.
    x_max = jnp.max(x, axis=axis, keepdims=True)
    # An all -inf slice has max -inf, and -inf - -inf is nan. Shift by 0 there.
    x_max = jnp.where(jnp.isfinite(x_max), x_max, 0.0)

    # Keep the axis on the sum too, so both operands still line up when the max
    # is added back. Adding a (..., 1) max to an already-reduced (...,) sum
    # broadcasts into an outer sum instead — silently, and with the wrong rank.
    out = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True)) + x_max

    return out if keepdims else jnp.squeeze(out, axis=axis)


def logsumexp_merge(m1, l1, m2, l2):
    m = jnp.maximum(m1, m2)
    # Same -inf guard: the empty state is (-inf, 0), and merging two empties
    # would otherwise compute exp(-inf - -inf) = nan instead of 0.
    safe = jnp.where(jnp.isfinite(m), m, 0.0)
    # Rescale BOTH sums onto the new max before adding — this is what keeps the
    # running total finite regardless of the order chunks arrive in.
    return m, l1 * jnp.exp(m1 - safe) + l2 * jnp.exp(m2 - safe)
''',
    "demo": '''import jax
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # (2, 3), deliberately not square

print("mine     :", logsumexp(x, axis=-1))
print("reference:", jax.scipy.special.logsumexp(x, axis=-1))

# Failure 1 — the max added back at the wrong rank. No error, wrong shape.
x_max = jnp.max(x, axis=-1, keepdims=True)          # (2, 1)
s = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=-1))   # (2,)
print("\\n(2,) + (2,1) ->", (s + x_max).shape, "  <- outer sum, silently wrong")
print(s + x_max)
print("squeezed     ->", (s + x_max.squeeze(-1)).shape, s + x_max.squeeze(-1))

# Failure 2 — a global max instead of a per-slice one.
rows = jnp.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
m = jnp.max(rows)                                   # one number for both rows
print("\\nglobal max :", jnp.log(jnp.sum(jnp.exp(rows - m), axis=-1)) + m)
print("per-slice  :", logsumexp(rows, axis=-1))

# Failure 3 — no shift at all.
big = jnp.array([1000.0, 1001.0, 1002.0])
print("\\nnaive      :", jnp.log(jnp.sum(jnp.exp(big))))
print("stable     :", logsumexp(big))

# Streaming: fold over chunks, never holding them all at once.
chunks = [jnp.array([1.0, 2.0, 3.0]), jnp.array([100.0, 101.0]), jnp.array([-50.0])]
state = (-jnp.inf, 0.0)                             # the empty state
for c in chunks:
    cm = jnp.max(c)
    state = logsumexp_merge(*state, cm, jnp.sum(jnp.exp(c - cm)))
m_all, l_all = state
print("\\nstreamed   :", m_all + jnp.log(l_all))
print("one-shot   :", logsumexp(jnp.concatenate(chunks)))

# The gradient is softmax — a correctness check that costs one line.
g = jax.grad(lambda v: logsumexp(v))(jnp.array([1.0, 2.0, 3.0]))
print("\\ngrad       :", g)
print("softmax    :", jax.nn.softmax(jnp.array([1.0, 2.0, 3.0])))
''',
    "tests": [
        {
            "name": "Basic 1-D, and the n=2 case is logaddexp",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
out = {fn}(x)
expected = jax.scipy.special.logsumexp(x)

assert jnp.ndim(out) == 0, (
    f'A full reduction of a 1-D input must be a scalar, got shape {jnp.shape(out)}. '
    'If this is (1,), the max was added back with its keepdims axis still on.'
)
assert jnp.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'

# Two elements is exactly logaddexp — the special case behind log1pexp.
pair = jnp.array([0.0, 100.0])
assert jnp.allclose({fn}(pair), jnp.logaddexp(0.0, 100.0), atol=1e-5), (
    f'{fn}([0, 100]) should equal logaddexp(0, 100) = {jnp.logaddexp(0.0, 100.0)}, '
    f'got {{fn}}(pair)'
)
""",
        },
        {
            "name": "Shape is the reduced shape, not a broadcast of it",
            "code": """
import jax
import jax.numpy as jnp

# Deliberately NOT square: a square input hides the bug below, because
# (n,) + (n, 1) broadcasts to (n, n) and still looks plausible.
x = jax.random.normal(jax.random.key(0), (2, 3))

for axis in (-1, 0, 1):
    out = {fn}(x, axis=axis)
    expected = jax.scipy.special.logsumexp(x, axis=axis)
    assert out.shape == expected.shape, (
        f'axis={axis}: got shape {out.shape}, expected {expected.shape}. '
        'Adding a keepdims max to an already-reduced sum broadcasts into an '
        'outer sum — squeeze the axis back out, or keep it on both reductions.'
    )
    assert jnp.allclose(out, expected, atol=1e-5), (
        f'axis={axis}: {out} vs {expected}'
    )

# 3-D, middle axis — the reduced axis must be removed, not just resized.
y = jax.random.normal(jax.random.key(1), (2, 3, 4))
assert {fn}(y, axis=1).shape == (2, 4), f'3-D axis=1 gave a wrong shape'
""",
        },
        {
            "name": "keepdims",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(2), (2, 3))

for axis in (-1, 0):
    kept = {fn}(x, axis=axis, keepdims=True)
    ref = jax.scipy.special.logsumexp(x, axis=axis, keepdims=True)
    assert kept.shape == ref.shape, (
        f'axis={axis}, keepdims=True: got {kept.shape}, expected {ref.shape}'
    )
    assert jnp.allclose(kept, ref, atol=1e-5), 'keepdims values differ'
    # And it must actually differ from the keepdims=False shape.
    assert kept.shape != {fn}(x, axis=axis).shape, (
        f'axis={axis}: keepdims=True and keepdims=False gave the same shape — '
        'the keepdims argument is being ignored'
    )
""",
        },
        {
            "name": "Stable on large positive and large negative inputs",
            "code": """
import jax
import jax.numpy as jnp

big = jnp.array([1000.0, 1001.0, 1002.0])
out = {fn}(big)
assert jnp.isfinite(out).all(), (
    f'Got {out} on a large input — exp() overflowed. Subtract the max first.'
)
assert jnp.allclose(out, jax.scipy.special.logsumexp(big), atol=1e-3), (
    f'{out} vs {jax.scipy.special.logsumexp(big)}'
)

small = jnp.array([-1000.0, -1001.0, -1002.0])
out_s = {fn}(small)
assert jnp.isfinite(out_s).all(), (
    f'Got {out_s} on a large negative input — every term underflowed to 0 and '
    'log(0) = -inf. Shifting by the max keeps the largest term at exp(0) = 1.'
)
assert jnp.allclose(out_s, jax.scipy.special.logsumexp(small), atol=1e-3), (
    f'{out_s} vs {jax.scipy.special.logsumexp(small)}'
)
""",
        },
        {
            "name": "The max is per-slice, not global",
            "code": """
import jax
import jax.numpy as jnp

# Two rows on wildly different scales. A single global max is exact for the
# big row and annihilates the small one.
rows = jnp.array([[1000.0, 1001.0, 1002.0],
                  [-1000.0, -1001.0, -1002.0]])
out = {fn}(rows, axis=-1)
expected = jax.scipy.special.logsumexp(rows, axis=-1)

assert jnp.isfinite(out).all(), (
    f'Got {out}. A global jnp.max(x) shifts the second row by +1000, so every '
    'term underflows to 0 and log(0) = -inf. Reduce along `axis`.'
)
assert jnp.allclose(out, expected, atol=1e-3), f'{out} vs {expected}'
""",
        },
        {
            "name": "Gradient is softmax",
            "code": """
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
g = jax.grad(lambda v: {fn}(v))(x)

assert jnp.isfinite(g).all(), f'Non-finite gradient: {g}'
assert jnp.allclose(g, jax.nn.softmax(x), atol=1e-5), (
    f'd/dx logsumexp(x) must equal softmax(x); got {g} vs {jax.nn.softmax(x)}'
)
assert jnp.allclose(jnp.sum(g), 1.0, atol=1e-5), 'The gradient must sum to 1'

# Still finite where the naive formulation would have produced inf/nan.
gb = jax.grad(lambda v: {fn}(v))(jnp.array([1000.0, 1001.0, 1002.0]))
assert jnp.isfinite(gb).all(), f'Non-finite gradient on a large input: {gb}'
""",
        },
        {
            "name": "jit and vmap",
            "code": """
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(3), (2, 3))

jitted = jax.jit(lambda v: {fn}(v, axis=-1))
assert jnp.allclose(jitted(x), {fn}(x, axis=-1), atol=1e-5), 'jit changes the result'

# vmap over rows must agree with reducing the last axis directly.
mapped = jax.vmap(lambda row: {fn}(row))(x)
assert mapped.shape == (2,), f'vmap gave {mapped.shape}, expected (2,)'
assert jnp.allclose(mapped, {fn}(x, axis=-1), atol=1e-5), (
    'vmap over rows disagrees with axis=-1'
)
""",
        },
        {
            "name": "Merging two chunks equals reducing the whole",
            "code": """
import jax
import jax.numpy as jnp

def state(c):
    m = jnp.max(c)
    return m, jnp.sum(jnp.exp(c - m))

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([0.5, -4.0])
m, l = logsumexp_merge(*state(a), *state(b))
got = m + jnp.log(l)
want = jax.scipy.special.logsumexp(jnp.concatenate([a, b]))

assert jnp.allclose(got, want, atol=1e-5), (
    f'Merged {got} but the whole reduces to {want}. Both partial sums must be '
    'rescaled onto the NEW max before adding: l * exp(m_old - m_new).'
)

# Order must not matter — the merge is commutative.
m2, l2 = logsumexp_merge(*state(b), *state(a))
assert jnp.allclose(m2 + jnp.log(l2), want, atol=1e-5), (
    'Merging b then a gave a different answer from a then b'
)
""",
        },
        {
            "name": "Merging survives chunks on wildly different scales",
            "code": """
import jax
import jax.numpy as jnp

def state(c):
    m = jnp.max(c)
    return m, jnp.sum(jnp.exp(c - m))

# The point of carrying (m, l) instead of a raw running sum: the second chunk
# is ~e^1000 times larger, so anything unshifted overflows here.
small = jnp.array([1.0, 2.0])
huge = jnp.array([1000.0, 1001.0])

m, l = logsumexp_merge(*state(small), *state(huge))
got = m + jnp.log(l)
want = jax.scipy.special.logsumexp(jnp.concatenate([small, huge]))
assert jnp.isfinite(got), f'Got {got} — the earlier sum was not rescaled onto the new max'
assert jnp.allclose(got, want, atol=1e-3), f'{got} vs {want}'

# ...and in the other order, where the max ARRIVES first and the later chunk
# must be scaled down instead.
m2, l2 = logsumexp_merge(*state(huge), *state(small))
assert jnp.isfinite(m2 + jnp.log(l2)), 'Non-finite when the larger chunk comes first'
assert jnp.allclose(m2 + jnp.log(l2), want, atol=1e-3), 'Order changed the answer'
""",
        },
        {
            "name": "The empty state (-inf, 0) is an identity, and folds",
            "code": """
import jax
import jax.numpy as jnp

def state(c):
    m = jnp.max(c)
    return m, jnp.sum(jnp.exp(c - m))

EMPTY = (-jnp.inf, 0.0)

# Merging the empty state must change nothing — that is what lets a fold start.
a = jnp.array([1.0, 2.0, 3.0])
m, l = logsumexp_merge(*EMPTY, *state(a))
assert jnp.allclose(m + jnp.log(l), jax.scipy.special.logsumexp(a), atol=1e-5), (
    'Merging the empty state (-inf, 0) changed the result — it must be an identity'
)

# Two empties must stay empty, not become nan: exp(-inf - -inf) is nan, so the
# exponent has to be pinned when the new max is not finite.
me, le = logsumexp_merge(*EMPTY, *EMPTY)
assert not jnp.isnan(me) and not jnp.isnan(le), (
    f'Merging two empty states gave ({me}, {le}) — exp(-inf - -inf) is nan. '
    'Guard the shift when the merged max is not finite.'
)
assert le == 0.0, f'Two empty states must stay empty, got l={le}'

# A fold over many chunks must equal the one-shot reduction.
chunks = [jnp.array([1.0, 2.0, 3.0]), jnp.array([100.0, 101.0]),
          jnp.array([-50.0]), jnp.array([7.0, 7.5])]
acc = EMPTY
for c in chunks:
    acc = logsumexp_merge(*acc, *state(c))
folded = acc[0] + jnp.log(acc[1])
want = jax.scipy.special.logsumexp(jnp.concatenate(chunks))
assert jnp.allclose(folded, want, atol=1e-4), (
    f'Folding chunk-by-chunk gave {folded}, one-shot gives {want}'
)
""",
        },
    ],
}
