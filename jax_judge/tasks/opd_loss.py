"""On-policy distillation — reverse KL from student to teacher(s)."""

TASK = {
    "title": "OPD (On-Policy Distillation) Loss",
    "category": "RLHF & Preference Losses",
    "order": 4,
    "difficulty": "Hard",
    "function_name": "opd_loss",
    "hint": (
        "Temperature-scale BOTH sets of logits before the softmax. The KL is "
        "reverse — sum_v p_student(v) * (log p_student(v) - log p_teacher(v)) — "
        "so the expectation is under the STUDENT. Get both log-probs with "
        "jax.nn.log_softmax and recover p_student as exp(log p_student). Combine "
        "multiple teachers with teacher_weights, apply the token mask, and "
        "multiply the whole thing by temperature ** 2."
    ),
    "description": r"""
Implement the **on-policy distillation** loss: match a student's next-token
distribution to one or more teachers, on sequences the *student* generated.

$$\mathcal{L} = T^2 \cdot \frac{\sum_t m_t \sum_k w_k\,
\mathbb{D}_{KL}\!\left[\pi_S \,\|\, \pi_{T_k}\right]_t}{\sum_t m_t}$$

with the **reverse** KL

$$\mathbb{D}_{KL}[\pi_S\|\pi_T] = \sum_v \pi_S(v)\big(\log \pi_S(v) - \log \pi_T(v)\big)$$

and both distributions taken at temperature $T$, i.e.
$\pi = \text{softmax}(\text{logits}/T)$.

### Signature
```python
def opd_loss(student_logits, teacher_logits, temperature=1.0,
             teacher_weights=None, mask=None):
    # student_logits: (batch, seq, vocab)
    # teacher_logits: (batch, seq, vocab) or (n_teachers, batch, seq, vocab)
    # teacher_weights: (n_teachers,) or None -> uniform
    # mask: (batch, seq) of 1.0 real / 0.0 padding, or None
    ...  # -> scalar
```

### Rules
- **Reverse** KL: the expectation is under the student, not the teacher
- Apply the temperature to **both** logit sets before softmax
- Multiply the result by $T^2$
- Support a single teacher `(B, S, V)` *and* a stack `(K, B, S, V)`
- `teacher_weights` defaults to uniform and should be normalised to sum to 1
- Mask-average over real tokens
- Use `jax.nn.log_softmax`; do not write `log(softmax(x))`

### Why the $T^2$ factor
Raising the temperature flattens both distributions, which shrinks the gradients
roughly as $1/T^2$. Multiplying the loss by $T^2$ cancels that, so the same
learning rate works across temperatures. Hinton's distillation paper introduces
this exact correction; forgetting it means every temperature change silently
becomes a learning-rate change.

### Forward vs reverse KL — the part that matters
| | Direction | Behaviour |
|---|---|---|
| Forward $\mathbb{D}[\pi_T\|\pi_S]$ | expectation under teacher | **mode-covering** — the student must put mass everywhere the teacher does, so it smears over all modes |
| Reverse $\mathbb{D}[\pi_S\|\pi_T]$ | expectation under student | **mode-seeking** — the student is only penalised where *it* puts mass, so it can safely ignore modes and commit to one |

Reverse KL is the right choice here: a smeared-out student that hedges across
every plausible continuation generates worse text than one that commits. The
cost is that the student can quietly drop teacher behaviours entirely.

### Why *on-policy* is the other half of the idea
Off-policy distillation trains on the teacher's own outputs, so the student only
ever sees states a competent model reaches. At inference the student drifts into
states its teacher never visited and has no idea what to do — the classic
exposure-bias / compounding-error failure. On-policy distillation samples from
the **student**, so the teacher supervises exactly the states the student
actually lands in. That is why the sequences here are the student's, and why the
loss is evaluated token-wise over them.
""",
    "stub": '''import jax
import jax.numpy as jnp


def opd_loss(student_logits, teacher_logits, temperature=1.0,
             teacher_weights=None, mask=None):
    """On-policy distillation loss (reverse KL, temperature-scaled).

    Args:
        student_logits:  (batch, seq, vocab)
        teacher_logits:  (batch, seq, vocab) or (n_teachers, batch, seq, vocab)
        temperature:     softmax temperature applied to both
        teacher_weights: (n_teachers,) mixing weights, or None for uniform
        mask:            (batch, seq) 1.0 real / 0.0 padding, or None

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def opd_loss(student_logits, teacher_logits, temperature=1.0,
             teacher_weights=None, mask=None):
    # Normalise a single teacher up to a stack of one.
    teacher_logits = jnp.asarray(teacher_logits)
    if teacher_logits.ndim == student_logits.ndim:
        teacher_logits = teacher_logits[None, ...]
    n_teachers = teacher_logits.shape[0]

    # Temperature applies to BOTH sides, before the softmax.
    s_logp = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    t_logp = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)

    s_p = jnp.exp(s_logp)

    # Reverse KL: expectation under the STUDENT. -> (n_teachers, batch, seq)
    kl = jnp.sum(s_p[None, ...] * (s_logp[None, ...] - t_logp), axis=-1)

    if teacher_weights is None:
        weights = jnp.full((n_teachers,), 1.0 / n_teachers)
    else:
        weights = jnp.asarray(teacher_weights, dtype=kl.dtype)
        weights = weights / jnp.sum(weights)

    per_token = jnp.sum(weights[:, None, None] * kl, axis=0)   # (batch, seq)

    if mask is None:
        loss = jnp.mean(per_token)
    else:
        mask = mask.astype(per_token.dtype)
        loss = jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # Cancel the 1/T^2 gradient shrinkage introduced by the temperature.
    return loss * temperature ** 2
''',
    "demo": '''import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(0), (2, 3, 6))

print("student == teacher:", float(opd_loss(logits, logits)))          # ~0
print("different teacher: ", float(opd_loss(logits, logits * 2.0)))    # > 0

# T^2 scaling keeps the loss on a comparable scale as temperature changes.
for T in (1.0, 2.0, 4.0):
    print(f"  T={T}: {float(opd_loss(logits, logits * 2.0, temperature=T)):.4f}")
''',
    "tests": [
        {
            "name": "Scalar output and zero at a perfect match",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(0), (2, 3, 5))
loss = {fn}(logits, logits)

assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'
assert jnp.allclose(loss, 0.0, atol=1e-5), (
    f'KL of a distribution with itself is 0, got {float(loss)}'
)
assert loss >= -1e-6, 'KL must be non-negative'

# A logit shift is a softmax no-op, so it must not change the loss.
shifted = {fn}(logits, logits + 7.0)
assert jnp.allclose(shifted, 0.0, atol=1e-5), (
    f'Adding a constant to all teacher logits leaves the distribution unchanged, '
    f'so the loss should still be 0. Got {float(shifted)}'
)
""",
        },
        {
            "name": "Reverse KL, not forward",
            "code": """
import jax
import jax.numpy as jnp

# Deliberately asymmetric: a permutation-symmetric pair like
# [2,0,0] vs [0,0,2] gives the SAME value in both directions and would not
# distinguish them. Here reverse ~ 1.816 and forward ~ 3.186.
s = jnp.array([[[4.0, 0.0, -1.0]]])
t = jnp.array([[[0.0, 0.5, 1.5]]])

got = float({fn}(s, t))

s_logp = jax.nn.log_softmax(s, -1)
t_logp = jax.nn.log_softmax(t, -1)
reverse = float(jnp.sum(jnp.exp(s_logp) * (s_logp - t_logp)))
forward = float(jnp.sum(jnp.exp(t_logp) * (t_logp - s_logp)))

assert not jnp.allclose(reverse, forward, atol=1e-3), 'test setup is not asymmetric'
assert jnp.allclose(got, reverse, atol=1e-5), (
    f'Got {got}; reverse KL (expectation under the STUDENT) is {reverse}, '
    f'forward KL is {forward}. The student distribution is the weighting term.'
)
""",
        },
        {
            "name": "Temperature and the T^2 factor",
            "code": """
import jax
import jax.numpy as jnp

s = jax.random.normal(jax.random.key(1), (2, 3, 7))
t = jax.random.normal(jax.random.key(2), (2, 3, 7))

for T in (0.5, 2.0, 4.0):
    s_logp = jax.nn.log_softmax(s / T, -1)
    t_logp = jax.nn.log_softmax(t / T, -1)
    expected = float(jnp.mean(jnp.sum(jnp.exp(s_logp) * (s_logp - t_logp), -1)) * T ** 2)
    got = float({fn}(s, t, temperature=T))
    assert jnp.allclose(got, expected, atol=1e-4), (
        f'T={T}: {got} vs {expected}. Scale BOTH logit sets by 1/T and multiply '
        'the result by T**2.'
    )

# Without the T^2 factor the loss would collapse toward 0 as T grows.
hot = float({fn}(s, t, temperature=4.0))
assert hot > 1e-4, f'Loss vanished at T=4 ({hot}) — the T**2 correction is missing'
""",
        },
        {
            "name": "Multiple teachers and weights",
            "code": """
import jax
import jax.numpy as jnp

s = jax.random.normal(jax.random.key(3), (2, 3, 5))
t1 = jax.random.normal(jax.random.key(4), (2, 3, 5))
t2 = jax.random.normal(jax.random.key(5), (2, 3, 5))
stack = jnp.stack([t1, t2])

l1 = float({fn}(s, t1))
l2 = float({fn}(s, t2))

# Uniform weights == the mean of the individual losses (KL is linear in the sum).
uniform = float({fn}(s, stack))
assert jnp.allclose(uniform, 0.5 * (l1 + l2), atol=1e-5), (
    f'Uniform two-teacher loss should be the mean of {l1} and {l2}, got {uniform}'
)

# All weight on teacher 1 reproduces the single-teacher loss.
only1 = float({fn}(s, stack, teacher_weights=jnp.array([1.0, 0.0])))
assert jnp.allclose(only1, l1, atol=1e-5), f'weights=[1,0] gave {only1}, expected {l1}'

# Unnormalised weights must be normalised internally.
raw = float({fn}(s, stack, teacher_weights=jnp.array([3.0, 1.0])))
norm = float({fn}(s, stack, teacher_weights=jnp.array([0.75, 0.25])))
assert jnp.allclose(raw, norm, atol=1e-5), (
    f'Weights [3,1] and [0.75,0.25] must agree: {raw} vs {norm}'
)

# A single teacher passed as (B,S,V) must still work.
assert jnp.allclose({fn}(s, t1), l1, atol=1e-6), 'Single-teacher path broke'
""",
        },
        {
            "name": "Mask averages over real tokens only",
            "code": """
import jax
import jax.numpy as jnp

s = jax.random.normal(jax.random.key(6), (2, 4, 5))
t = jax.random.normal(jax.random.key(7), (2, 4, 5))
mask = jnp.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0, 0.0]])

# Corrupting masked positions must not move the loss at all.
noise = jax.random.normal(jax.random.key(8), (2, 4, 5)) * 100
t_corrupt = t + noise * (1 - mask)[..., None]

a = float({fn}(s, t, mask=mask))
b = float({fn}(s, t_corrupt, mask=mask))
assert jnp.allclose(a, b, atol=1e-4), (
    f'Masked positions leaked into the loss: {a} vs {b}'
)

# And it must equal the loss computed on the kept slice alone.
c = float({fn}(s[:, :2], t[:, :2]))
assert jnp.allclose(a, c, atol=1e-4), (
    f'Masked mean {a} should equal the loss over the kept tokens {c} — the '
    'denominator must count only unmasked tokens.'
)

full = float({fn}(s, t, mask=jnp.ones((2, 4))))
none = float({fn}(s, t))
assert jnp.allclose(full, none, atol=1e-6), 'An all-ones mask changed the result'
""",
        },
        {
            "name": "Non-negative and numerically stable",
            "code": """
import jax
import jax.numpy as jnp

key = jax.random.key(9)
for i in range(5):
    k1, k2, key = jax.random.split(key, 3)
    s = jax.random.normal(k1, (2, 3, 8)) * 5
    t = jax.random.normal(k2, (2, 3, 8)) * 5
    v = float({fn}(s, t))
    assert v >= -1e-5, f'KL must be non-negative, got {v}'
    assert jnp.isfinite(v), f'Non-finite loss {v}'

# Extreme logits: log(softmax(x)) would overflow/underflow here.
big = jnp.array([[[1000.0, -1000.0, 0.0]]])
small = jnp.array([[[-1000.0, 1000.0, 0.0]]])
v = {fn}(big, small)
assert jnp.isfinite(v), f'Got {v} on extreme logits — use jax.nn.log_softmax'
""",
        },
        {
            "name": "Gradient flows to the student only",
            "code": """
import functools
import jax
import jax.numpy as jnp

s = jax.random.normal(jax.random.key(10), (2, 3, 5))
t = jax.random.normal(jax.random.key(11), (2, 3, 5))

g = jax.grad({fn})(s, t)
assert g.shape == s.shape, f'Gradient shape {g.shape} vs {s.shape}'
assert jnp.isfinite(g).all(), 'Non-finite gradient'
assert jnp.abs(g).max() > 1e-6, 'Zero gradient — the student logits are not connected'

# A gradient step in the -grad direction must reduce the loss.
before = float({fn}(s, t))
after = float({fn}(s - 0.01 * g, t))
assert after < before, f'Descent step increased the loss: {before} -> {after}'

jitted = jax.jit(functools.partial({fn}, temperature=2.0))
assert jnp.allclose(jitted(s, t), {fn}(s, t, temperature=2.0), atol=1e-5), (
    'jit changes the result'
)
""",
        },
    ],
}
