"""One reverse step of a masked diffusion LM — the sampler that inverts b_16."""

TASK = {
    "title": "Masked Diffusion Sampling Step",
    "category": "Inference & Decoding",
    "number": "b_17",
    "difficulty": "Medium",
    "function_name": "denoise_step",
    "hint": (
        "Each still-masked position unmasks independently with probability "
        "(alpha_s - alpha_t) / (1 - alpha_t), which for the linear schedule is "
        "(t - s) / t — one Bernoulli draw PER POSITION, not one for the whole "
        "sequence. Where it unmasks, fill in a token sampled from the model: "
        "jax.random.categorical over the logits with the MASK column set to "
        "-inf. Positions that already hold a real token are copied through "
        "untouched. Split the key so the unmask decision and the token draw do "
        "not share randomness."
    ),
    "description": r"""
Masked diffusion generates by starting from an all-`[MASK]` sequence and
repeatedly stepping backwards in time, revealing a few tokens each step.
Implement **one** such step: given $x_t$ and the model's logits, produce
$x_s$ for some $s < t$.

Because the forward kernel is absorbing and per-token, the exact posterior
factorises over positions and has no matrix in it:

$$q(x_s^i \mid x_t^i, x_0^i) =
\begin{cases}
\delta_{x_t^i} & x_t^i \neq \texttt{[MASK]}\\[4pt]
\dfrac{\alpha_s - \alpha_t}{1 - \alpha_t}\,\delta_{x_0^i}
\;+\; \dfrac{1 - \alpha_s}{1 - \alpha_t}\,\delta_{\texttt{[MASK]}}
& x_t^i = \texttt{[MASK]}
\end{cases}$$

Sampling replaces the unknown $x_0^i$ with a draw from the model. With the
linear schedule $\alpha_t = 1 - t$ the unmasking probability is just
$(t - s)/t$.

### Rules
- Signature: `denoise_step(key, logits, xt, t, s, *, mask_id)` -> `x_s`, shape of `xt`
- `logits` is `(B, L, V)`, `xt` is `(B, L)` integer ids, `t` and `s` are scalars
  with $0 \le s < t \le 1$
- Every position decides **independently** whether to unmask
- A position already holding a real token is **never** touched — not re-masked,
  not resampled
- A sampled token is never `[MASK]`
- Deterministic given `key`; use `jax.random.split` rather than passing one key
  to two draws
- `jax.nn.softmax` / `log_softmax` are allowed here; `optax` is not
- Must work under `jit`

### Carry-over unmasking, and why it is not optional
Once a token is revealed it stays. That is not a design choice bolted on for
convenience — it is what the absorbing posterior above *says*: the top branch is
a point mass on $x_t^i$. An implementation that resamples every position each
step is not sampling from this model at all; it will still produce fluent-looking
text, which is exactly what makes the bug survive review.

The practical consequence is that the number of masked tokens falls
monotonically, and at $s = 0$ the probability becomes $(t - 0)/t = 1$, so the
last step is guaranteed to leave nothing masked. You get termination for free
from the schedule rather than from a special case in the loop.

### The ratio, not the difference
$\alpha_s - \alpha_t$ is the *unconditional* mass that leaves the masked state
between the two times. But you are already conditioning on this position being
masked at $t$, an event of probability $1 - \alpha_t$, so the conditional rate
is the ratio $\frac{\alpha_s - \alpha_t}{1 - \alpha_t}$. Use the bare difference
and the sampler under-unmasks — badly at small $t$, where $1 - \alpha_t$ is
small — leaving `[MASK]` tokens in the output of a loop that thought it was done.

### One draw per position
`jax.random.bernoulli(key, p)` with no shape returns a **scalar**: every position
in the batch then makes the same decision, and the sampler reveals either
everything or nothing. Pass the shape explicitly. Same for the token draw —
`jax.random.categorical` reduces the last axis, so `(B, L, V)` logits give
`(B, L)` samples, one per position.

### What this leaves out
Real samplers add a *remasking* policy on top: rather than unmasking a random
subset, LLaDA-style decoding unmasks the positions the model is most confident
about, and some schedules deliberately re-mask low-confidence tokens later. That
last part steps outside the posterior above — it is where the discrete-flow-
matching view (which allows corrector steps) buys you something the plain
diffusion sampler cannot express. This problem is the exact posterior; the
policies are a layer above it.
""",
    "stub": '''import jax
import jax.numpy as jnp


def denoise_step(key, logits, xt, t, s, *, mask_id):
    """One reverse step of a masked diffusion LM: x_t -> x_s, with s < t.

    Args:
        key:     PRNG key
        logits:  (B, L, V) model scores for x0 given xt
        xt:      (B, L) current token ids, [MASK] where undecided
        t:       current time in (0, 1]
        s:       target time, 0 <= s < t
        mask_id: the [MASK] token id

    Returns:
        (B, L) token ids at time s.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def denoise_step(key, logits, xt, t, s, *, mask_id):
    logits = jnp.asarray(logits)
    xt = jnp.asarray(xt)

    # The model never emits [MASK].
    logits = logits.at[..., mask_id].set(-jnp.inf)

    # Two independent sources of randomness: which positions reveal, and what
    # they reveal.
    k_unmask, k_token = jax.random.split(key)

    # (alpha_s - alpha_t) / (1 - alpha_t) with alpha = 1 - t. Conditional on
    # this position being masked at t, hence the ratio and not the difference.
    p_unmask = (t - s) / t

    # One decision PER POSITION.
    reveal = jax.random.bernoulli(k_unmask, p_unmask, xt.shape) & (xt == mask_id)
    x0_hat = jax.random.categorical(k_token, logits, axis=-1)

    # Carry-over: anything already decoded is copied through untouched.
    return jnp.where(reveal, x0_hat, xt)
''',
    "demo": '''import jax
import jax.numpy as jnp

V, MASK, L = 8, 7, 10

# A model that always wants token (position % 7), sharply.
logits = jax.nn.one_hot(jnp.arange(L) % 7, V)[None] * 30.0

# Start from all [MASK] and run the schedule down to 0.
x = jnp.full((1, L), MASK)
steps = 5
print("start:", x[0])
key = jax.random.key(0)
for i in range(steps):
    t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
    key, sub = jax.random.split(key)
    x = denoise_step(sub, logits, x, t, s, mask_id=MASK)
    print(f"t={t:.1f} -> s={s:.1f}:", x[0], f" masked: {int(jnp.sum(x == MASK))}")

print("\\nno [MASK] left:", bool(jnp.all(x != MASK)))

# The unmasking rate is the RATIO, not the difference. Late in the schedule the
# two are wildly apart.
for t, s in [(1.0, 0.8), (0.4, 0.2), (0.1, 0.05)]:
    print(f"t={t}, s={s}:  ratio {(t - s) / t:.3f}   difference {t - s:.3f}")

# Carry-over: an already-decoded token survives any number of steps.
partial = jnp.array([[3, MASK, MASK, 3, MASK, 3, MASK, MASK, MASK, MASK]])
out = denoise_step(jax.random.key(1), logits, partial, 0.5, 0.25, mask_id=MASK)
print("\\nbefore:", partial[0], "\\nafter :", out[0])
print("decoded positions unchanged:",
      bool(jnp.all(out[partial != MASK] == partial[partial != MASK])))
''',
    "tests": [
        {
            "name": "Carry-over: decoded tokens are never touched",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L, B = 8, 7, 12, 4
logits = jax.random.normal(jax.random.key(0), (B, L, V))
xt = jax.random.randint(jax.random.key(1), (B, L), 0, V)   # a mix of tokens and MASK

decoded = xt != MASK
for k in range(12):
    out = {fn}(jax.random.key(k), logits, xt, 0.7, 0.3, mask_id=MASK)
    assert out.shape == xt.shape, f'shape changed: {out.shape} vs {xt.shape}'
    assert jnp.all(jnp.where(decoded, out == xt, True)), (
        'a position that already held a real token was changed — the posterior '
        'is a point mass there, so it must be copied through'
    )
    # Never re-masked, so the mask count can only fall.
    assert int(jnp.sum(out == MASK)) <= int(jnp.sum(xt == MASK)), (
        'the number of [MASK] tokens increased — this sampler never re-masks'
    )
""",
        },
        {
            "name": "Unmasking rate is (t - s) / t, per position",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L, B = 6, 5, 64, 64
logits = jnp.zeros((B, L, V))
allmask = jnp.full((B, L), MASK)

for t, s in [(1.0, 0.5), (0.4, 0.2), (0.2, 0.15)]:
    out = {fn}(jax.random.key(int(t * 100) + int(s * 10)), logits, allmask, t, s,
               mask_id=MASK)
    rate = float(jnp.mean(out != MASK))
    want = (t - s) / t
    assert abs(rate - want) < 0.03, (
        f'unmasked {rate:.3f} of the positions going from t={t} to s={s}, expected '
        f'{want:.3f} = (t-s)/t. Conditional on being masked at t, the rate is the '
        f'RATIO (alpha_s - alpha_t)/(1 - alpha_t), not the difference {t - s:.3f}'
    )

# Per POSITION, not one draw for the whole batch: the count must vary.
out = {fn}(jax.random.key(3), logits, allmask, 0.8, 0.4, mask_id=MASK)
counts = jnp.sum(out != MASK, axis=-1)
assert jnp.std(counts) > 0.5, (
    f'per-sequence unmask counts are near-constant (std={float(jnp.std(counts)):.2f}) '
    '— jax.random.bernoulli(key, p) with no shape returns ONE scalar decision'
)

# s = 0 finishes the job: (t - 0) / t = 1, so nothing may stay masked.
done = {fn}(jax.random.key(4), logits, allmask, 0.35, 0.0, mask_id=MASK)
assert jnp.all(done != MASK), (
    'stepping to s=0 must leave no [MASK] tokens — the probability is exactly 1'
)
""",
        },
        {
            "name": "Revealed tokens come from the model, and are never [MASK]",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L, B = 8, 7, 16, 8
allmask = jnp.full((B, L), MASK)

# A sharp model: every revealed token must be its argmax.
want = jnp.arange(L) % 7
sharp = jnp.broadcast_to(jax.nn.one_hot(want, V)[None] * 30.0, (B, L, V))
out = {fn}(jax.random.key(0), sharp, allmask, 1.0, 0.5, mask_id=MASK)
revealed = out != MASK
assert jnp.any(revealed), 'nothing was unmasked at all'
assert jnp.all(jnp.where(revealed, out == want, True)), (
    'a revealed token did not match the model argmax — sample from the logits'
)

# Even when [MASK] is the most likely token by far, it must never be emitted.
maskish = jnp.zeros((B, L, V)).at[..., MASK].set(50.0)
out2 = {fn}(jax.random.key(1), maskish, allmask, 1.0, 0.0, mask_id=MASK)
assert jnp.all(out2 != MASK), (
    'the model was allowed to emit [MASK] — set that logit to -inf before sampling'
)

# A two-way tie is resolved by sampling, not by argmax: both must appear.
tie = jnp.zeros((B, L, V)).at[..., 0].set(10.0).at[..., 1].set(10.0)
out3 = {fn}(jax.random.key(2), tie, allmask, 1.0, 0.0, mask_id=MASK)
frac = float(jnp.mean(out3 == 0))
assert 0.35 < frac < 0.65, (
    f'token 0 took {frac:.2f} of the draws on a 50/50 tie — draw with '
    'jax.random.categorical, do not take the argmax'
)

# Determinism and key hygiene.
a = {fn}(jax.random.key(5), tie, allmask, 0.9, 0.4, mask_id=MASK)
b = {fn}(jax.random.key(5), tie, allmask, 0.9, 0.4, mask_id=MASK)
c = {fn}(jax.random.key(6), tie, allmask, 0.9, 0.4, mask_id=MASK)
assert jnp.array_equal(a, b), 'same key must give the same output'
assert not jnp.array_equal(a, c), 'a different key must give a different output'

# The unmask decision and the token draw must not share randomness: with a
# 50/50 tie they are independent, so which positions reveal cannot predict
# which token they reveal.
big = jnp.zeros((256, 32, V)).at[..., 0].set(10.0).at[..., 1].set(10.0)
o = {fn}(jax.random.key(7), big, jnp.full((256, 32), MASK), 1.0, 0.5, mask_id=MASK)
zeros = float(jnp.mean(o[o != MASK] == 0))
assert 0.4 < zeros < 0.6, (
    f'{zeros:.2f} of the revealed tokens were token 0 — split the key so the '
    'Bernoulli draw and the categorical draw are independent'
)
""",
        },
        {
            "name": "A full reverse loop terminates with a clean sequence",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L = 8, 7, 20
logits = jnp.broadcast_to(jax.nn.one_hot(jnp.arange(L) % 7, V)[None] * 30.0, (1, L, V))

x = jnp.full((1, L), MASK)
key = jax.random.key(0)
steps, prev = 8, L
for i in range(steps):
    t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
    key, sub = jax.random.split(key)
    x = {fn}(sub, logits, x, t, s, mask_id=MASK)
    n = int(jnp.sum(x == MASK))
    assert n <= prev, f'step {i}: masked count rose from {prev} to {n}'
    prev = n

assert jnp.all(x != MASK), f'the loop ended with {int(jnp.sum(x == MASK))} [MASK] left'
assert jnp.all(x[0] == jnp.arange(L) % 7), 'the decoded sequence is not the model argmax'
""",
        },
        {
            "name": "jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

V, MASK, L, B = 6, 5, 10, 3
logits = jax.random.normal(jax.random.key(0), (B, L, V))
xt = jnp.where(jax.random.uniform(jax.random.key(1), (B, L)) < 0.5,
               MASK, jax.random.randint(jax.random.key(2), (B, L), 0, MASK))

f = functools.partial({fn}, mask_id=MASK)
eager = f(jax.random.key(3), logits, xt, 0.6, 0.2)
jitted = jax.jit(f)(jax.random.key(3), logits, xt, 0.6, 0.2)
assert jnp.array_equal(jitted, eager), (
    'jit changed the answer — no Python branching on array values, and t/s must '
    'be usable as traced scalars'
)
""",
        },
    ],
}
