"""Speculative decoding — provably lossless, if you get the residual right."""

TASK = {
    "title": "Speculative Decoding",
    "category": "Inference & Decoding",
    "order": 3,
    "difficulty": "Hard",
    "function_name": "speculative_step",
    "hint": (
        "A plain Python loop over the k positions is fine here — this one is "
        "about the probability, not the vectorisation. Split the key per "
        "position and then split AGAIN: the accept/reject uniform and the "
        "resample draw have to be independent, or the token you fall back to is "
        "correlated with the coin that rejected it. Sample the residual with "
        "jax.random.categorical on its log rather than hand-rolling a CDF. Two "
        "invariants worth asserting as you go: the loop returns on the FIRST "
        "rejection and never looks at later positions, and n_accepted counts "
        "DRAFT tokens, so the number of ids you emit is always n_accepted + 1."
    ),
    "description": r"""
Implement one step of **speculative decoding**: a small draft model proposes $k$
tokens, the large target model verifies them all in a single forward pass, and
the accept/reject rule guarantees the output is distributed *exactly* as if you
had sampled from the target model directly.

### Signature
```python
def speculative_step(key, draft_tokens, draft_probs, target_probs):
    # draft_tokens: (k,) int   — tokens the draft model proposed
    # draft_probs:  (k, V)     — draft distribution at each of the k positions
    # target_probs: (k+1, V)   — target distribution at those k positions
    #                            PLUS one extra for the bonus token
    ...  # -> (accepted_tokens, n_accepted)
```

Return a `(k+1,)` int array padded with `-1` past the end, and the number of
draft tokens accepted.

### The rule, exactly
For each position $i$ in order, draw $u_i \sim U(0,1)$ and accept if

$$u_i < \min\left(1, \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)$$

- **Accept** → keep $x_i$, continue to $i+1$.
- **Reject** → stop, and sample a replacement from the **residual**
  $$p'(x) = \frac{\max\big(p_{\text{target}}(x) - p_{\text{draft}}(x),\ 0\big)}
  {\sum_x \max\big(p_{\text{target}}(x) - p_{\text{draft}}(x),\ 0\big)}$$
- **All $k$ accepted** → sample a **bonus** token from `target_probs[k]`.

So a step returns between 1 and $k+1$ tokens — never zero.

### Rules
- One uniform draw per position, and a *separate* key for the resample —
  reusing the accept draw correlates the fallback token with the rejection
- The residual must be renormalised, and clamped at zero *before* normalising
- Guard the degenerate case where the residual sums to 0
- Do not use a library implementation

### Why this is lossless — the property that matters
It is tempting to think this trades quality for speed. It does not, and the
proof is short: the probability of emitting token $x$ at a position is

$$\underbrace{p_d(x)\min\!\left(1, \tfrac{p_t(x)}{p_d(x)}\right)}_{\text{accepted}}
+ \underbrace{P(\text{reject})\cdot p'(x)}_{\text{resampled}} = p_t(x)$$

The first term is $\min(p_d(x), p_t(x))$. The residual contributes exactly the
missing $\max(p_t(x) - p_d(x), 0)$, and the rejection probability is precisely
its normalising constant — so they cancel and the total is $p_t(x)$.

**This is the whole point**, and it is why the residual must be
$\max(p_t - p_d, 0)$ renormalised and nothing else. Sampling the rejection from
$p_t$ directly is the classic bug: it over-weights tokens the draft already
liked, and quietly changes the output distribution. The tests below check the
resulting distribution statistically, not just the shapes.

### Where the speedup comes from
The target model runs once per *step*, not once per token, and a step emits up
to $k+1$ tokens. Decoding is memory-bandwidth bound — you pay to stream the
weights in regardless — so verifying $k$ tokens costs barely more than
generating one. The expected number of tokens per step rises with how well the
draft matches the target, which is why draft models are typically distilled from
their target.
""",
    "stub": '''import jax
import jax.numpy as jnp


def speculative_step(key, draft_tokens, draft_probs, target_probs):
    """One speculative decoding step.

    Args:
        key:          PRNG key
        draft_tokens: (k,) proposed token ids
        draft_probs:  (k, V) draft distribution at each position
        target_probs: (k+1, V) target distribution, including the bonus position

    Returns:
        (accepted, n_accepted)
          accepted   (k+1,) int32, emitted tokens then -1 padding
          n_accepted int, how many DRAFT tokens were accepted (0..k)
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def speculative_step(key, draft_tokens, draft_probs, target_probs):
    k, V = draft_probs.shape
    keys = jax.random.split(key, k + 1)

    out = [-1] * (k + 1)
    n_accepted = 0

    for i in range(k):
        # The accept/reject draw and the resample must be INDEPENDENT. Reusing
        # one key for both correlates them and silently skews the output.
        accept_key, resample_key = jax.random.split(keys[i])

        tok = int(draft_tokens[i])
        p_t = float(target_probs[i, tok])
        p_d = float(draft_probs[i, tok])

        ratio = 1.0 if p_d <= 0.0 else min(1.0, p_t / p_d)
        u = float(jax.random.uniform(accept_key))

        if u < ratio:
            out[i] = tok
            n_accepted += 1
            continue

        # Rejected: resample this position from the normalised residual. This
        # exact distribution is what makes the whole scheme lossless.
        residual = jnp.maximum(target_probs[i] - draft_probs[i], 0.0)
        total = jnp.sum(residual)
        residual = jnp.where(total > 0, residual / total, target_probs[i])
        out[i] = int(jax.random.categorical(resample_key, jnp.log(residual + 1e-38)))
        return jnp.array(out, dtype=jnp.int32), n_accepted

    # Every draft token survived, so we get a free extra token from the target.
    out[k] = int(jax.random.categorical(keys[k], jnp.log(target_probs[k] + 1e-38)))
    return jnp.array(out, dtype=jnp.int32), n_accepted
''',
    "demo": '''import jax
import jax.numpy as jnp

V, k = 4, 3
draft_probs = jnp.tile(jnp.array([0.7, 0.1, 0.1, 0.1]), (k, 1))
target_probs = jnp.tile(jnp.array([0.4, 0.3, 0.2, 0.1]), (k + 1, 1))
draft_tokens = jnp.array([0, 0, 0])

for seed in range(5):
    toks, n = speculative_step(jax.random.key(seed), draft_tokens,
                               draft_probs, target_probs)
    print(f"seed {seed}: accepted {n}/{k} draft tokens -> {toks.tolist()}")
''',
    "tests": [
        {
            "name": "Shapes, padding and the always-emit-something guarantee",
            "code": """
import jax
import jax.numpy as jnp

k, V = 4, 6
dp = jnp.tile(jnp.full((V,), 1.0 / V), (k, 1))
tp = jnp.tile(jnp.full((V,), 1.0 / V), (k + 1, 1))
dt = jnp.array([0, 1, 2, 3])

for seed in range(12):
    toks, n = {fn}(jax.random.key(seed), dt, dp, tp)
    toks = jnp.asarray(toks)

    assert toks.shape == (k + 1,), f'Expected shape ({k+1},), got {toks.shape}'
    assert 0 <= int(n) <= k, f'n_accepted must be in [0, {k}], got {int(n)}'

    emitted = [int(t) for t in toks if int(t) != -1]
    assert len(emitted) >= 1, 'A step must always emit at least one token'
    assert len(emitted) == int(n) + 1, (
        f'Accepting {int(n)} draft tokens must emit {int(n)+1} tokens '
        f'(the accepted ones plus a resampled or bonus token), got {len(emitted)}'
    )
    # Padding is a suffix, never interleaved.
    vals = [int(t) for t in toks]
    if -1 in vals:
        first_pad = vals.index(-1)
        assert all(v == -1 for v in vals[first_pad:]), f'Padding must be a suffix: {vals}'
    assert all(0 <= t < V for t in emitted), f'Token out of vocab range: {emitted}'
""",
        },
        {
            "name": "Identical distributions accept everything",
            "code": """
import jax
import jax.numpy as jnp

k, V = 5, 8
key = jax.random.key(0)
p = jax.random.uniform(key, (V,)) + 0.1
p = p / p.sum()

dp = jnp.tile(p, (k, 1))
tp = jnp.tile(p, (k + 1, 1))
dt = jnp.array([2, 5, 1, 0, 7])

for seed in range(15):
    toks, n = {fn}(jax.random.key(seed), dt, dp, tp)
    assert int(n) == k, (
        f'When draft == target the ratio is exactly 1 everywhere, so every draft '
        f'token must be accepted. Got n_accepted={int(n)} on seed {seed}.'
    )
    assert [int(t) for t in jnp.asarray(toks)[:k]] == [int(t) for t in dt], (
        'Accepted tokens must be the draft tokens, unchanged'
    )
    assert int(jnp.asarray(toks)[k]) != -1, 'Full acceptance must yield a bonus token'
""",
        },
        {
            "name": "Zero target probability always rejects",
            "code": """
import jax
import jax.numpy as jnp

k, V = 3, 4
# The draft loves token 0; the target assigns it probability 0.
dp = jnp.tile(jnp.array([0.9, 0.05, 0.03, 0.02]), (k, 1))
tp = jnp.tile(jnp.array([0.0, 0.5, 0.3, 0.2]), (k + 1, 1))
dt = jnp.array([0, 0, 0])

for seed in range(15):
    toks, n = {fn}(jax.random.key(seed), dt, dp, tp)
    assert int(n) == 0, (
        f'p_target = 0 makes the acceptance ratio 0, so the first token must '
        f'always be rejected. Got n_accepted={int(n)} on seed {seed}.'
    )
    first = int(jnp.asarray(toks)[0])
    assert first != 0, (
        f'The resampled token must come from the residual, which has zero mass '
        f'on token 0. Got {first}.'
    )
    assert first in (1, 2, 3), f'Resampled token out of support: {first}'
""",
        },
        {
            "name": "Losslessness: the output distribution IS the target",
            "code": """
import jax
import jax.numpy as jnp

# One draft token, so the first emitted token is exactly one speculative
# decision. Over many trials its distribution must match target_probs[0].
#
# The guarantee only holds when the draft TOKEN is itself sampled from the
# draft distribution — that is the premise of the proof — so each trial draws
# its own proposal rather than pinning it to a constant.
V = 5
dp = jnp.array([[0.60, 0.20, 0.10, 0.05, 0.05]])
tp = jnp.array([[0.20, 0.20, 0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20, 0.20, 0.20]])

N = 4000
counts = [0] * V
for seed in range(N):
    prop_key, step_key = jax.random.split(jax.random.key(seed))
    dt = jax.random.categorical(prop_key, jnp.log(dp[0]), shape=(1,))
    toks, n = {fn}(step_key, dt, dp, tp)
    counts[int(jnp.asarray(toks)[0])] += 1

empirical = jnp.array(counts, dtype=jnp.float32) / N
expected = tp[0]

assert jnp.allclose(empirical, expected, atol=0.035), (
    f'The emitted distribution must equal the TARGET distribution.\\n'
    f'  empirical: {[round(float(v), 3) for v in empirical]}\\n'
    f'  target:    {[round(float(v), 3) for v in expected]}\\n'
    f'  draft:     {[round(float(v), 3) for v in dp[0]]}\\n'
    'Drift toward the draft means the rejection branch is resampling from '
    'target_probs directly instead of from the normalised residual '
    'max(p_target - p_draft, 0).'
)
""",
        },
        {
            "name": "Residual is clamped and renormalised",
            "code": """
import jax
import jax.numpy as jnp

# Draft exceeds target on tokens 0 and 1, so the residual there is negative
# before clamping. All surviving mass belongs to tokens 2 and 3, split 1:3.
V = 4
dp = jnp.array([[0.50, 0.40, 0.05, 0.05]])
tp = jnp.array([[0.10, 0.10, 0.20, 0.60],
                [0.25, 0.25, 0.25, 0.25]])
dt = jnp.array([0])

counts = [0] * V
rejected = 0
for seed in range(3000):
    toks, n = {fn}(jax.random.key(seed), dt, dp, tp)
    if int(n) == 0:
        rejected += 1
        counts[int(jnp.asarray(toks)[0])] += 1

assert rejected > 500, f'Expected many rejections here, saw {rejected}'
emp = jnp.array(counts, dtype=jnp.float32) / rejected

# residual = max(tp - dp, 0) = [0, 0, 0.15, 0.55] -> normalised [0, 0, 0.214, 0.786]
assert emp[0] < 0.02 and emp[1] < 0.02, (
    f'Tokens where the draft already exceeds the target must get ZERO residual '
    f'mass (clamp at 0 before normalising). Got {[round(float(v),3) for v in emp]}'
)
assert jnp.allclose(emp[2], 0.15 / 0.70, atol=0.05), (
    f'Residual not renormalised correctly: token 2 got {float(emp[2]):.3f}, '
    f'expected {0.15/0.70:.3f}'
)
assert jnp.allclose(emp[3], 0.55 / 0.70, atol=0.05), (
    f'token 3 got {float(emp[3]):.3f}, expected {0.55/0.70:.3f}'
)
""",
        },
        {
            "name": "Better drafts accept more",
            "code": """
import jax
import jax.numpy as jnp

k, V = 4, 6
target = jnp.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
tp = jnp.tile(target, (k + 1, 1))
dt = jnp.zeros((k,), dtype=jnp.int32)

def mean_accepted(draft):
    dp = jnp.tile(draft, (k, 1))
    tot = 0
    for seed in range(300):
        _, n = {fn}(jax.random.key(seed), dt, dp, tp)
        tot += int(n)
    return tot / 300

good = mean_accepted(target)                                    # perfect draft
poor = mean_accepted(jnp.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]))

assert good > poor, (
    f'A draft matching the target should be accepted more often than a poor '
    f'one: {good:.2f} vs {poor:.2f}'
)
assert good > k - 0.01, f'A perfect draft should accept all {k}, got {good:.2f}'
""",
        },
        {
            "name": "Deterministic in the key",
            "code": """
import jax
import jax.numpy as jnp

k, V = 3, 5
dp = jnp.tile(jnp.array([0.5, 0.2, 0.15, 0.1, 0.05]), (k, 1))
tp = jnp.tile(jnp.array([0.3, 0.3, 0.2, 0.1, 0.1]), (k + 1, 1))
dt = jnp.array([0, 1, 2])

key = jax.random.key(123)
a_t, a_n = {fn}(key, dt, dp, tp)
b_t, b_n = {fn}(key, dt, dp, tp)

assert int(a_n) == int(b_n), f'Same key gave different n_accepted: {int(a_n)} vs {int(b_n)}'
assert (jnp.asarray(a_t) == jnp.asarray(b_t)).all(), (
    f'Same key gave different tokens: {a_t} vs {b_t}'
)

# Across many keys the outcome must actually vary — otherwise the key is
# being ignored somewhere.
seen = set()
for seed in range(40):
    t, n = {fn}(jax.random.key(seed), dt, dp, tp)
    seen.add((int(n), tuple(int(v) for v in jnp.asarray(t))))
assert len(seen) > 1, (
    f'All 40 keys produced the identical result {seen} — the PRNG key is not '
    'driving the accept/reject decision.'
)
""",
        },
    ],
}
