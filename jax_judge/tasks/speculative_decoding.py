"""Speculative decoding — draft, verify, and resample from the residual."""

TASK = {
    "title": "Speculative Decoding",
    "category": "Inference & Decoding",
    "number": "34",
    "difficulty": "Hard",
    "function_name": "speculative_decode",
    "hint": (
        "For each drafted token accept it with probability min(1, p_target / "
        "p_draft) — note that a token the target likes MORE than the draft has "
        "ratio > 1 and is always accepted. On the first rejection you must stop "
        "and emit exactly one token drawn from the normalised positive part of "
        "(p_target - p_draft), then return immediately: everything after a "
        "rejection was conditioned on a token that did not survive. Split the "
        "key per position so the accept coin and the resample draw are independent."
    ),
    "description": r"""
Implement the accept/reject loop at the heart of **speculative decoding**.

A small draft model proposes `K` tokens; the large target model scores them all
in one batched pass. You then decide how many to keep.

### Signature
```python
def speculative_decode(key, target_probs, draft_probs, draft_tokens):
    ...  # -> list[int]
```

- `target_probs`, `draft_probs`: `(K, V)` probability rows
- `draft_tokens`: `(K,)` proposed token ids
- returns the accepted tokens, plus one resampled token if a rejection happened

### The algorithm
For each position `i`:
1. Accept `draft_tokens[i]` with probability
   $\min\!\left(1, \frac{p_{\text{target}}(t)}{p_{\text{draft}}(t)}\right)$
2. On acceptance, continue to the next position
3. On **rejection**, sample one token from the normalised residual
   $\max(p_{\text{target}} - p_{\text{draft}}, 0)$, append it, and **stop**

If every draft token is accepted, return all `K`.

### Rules
- Guard the division with `1e-10`
- If the residual sums to zero, fall back to a uniform distribution
- Stop at the **first** rejection — do not keep going

### Why this is lossless, not an approximation
This is the property that matters and the one interviewers probe. The
accept/reject rule plus the residual resample is constructed so the emitted
token is distributed **exactly** as $p_{\text{target}}$ — not approximately.

The argument: a token $t$ survives either by being drafted and accepted, with
probability $p_d(t)\min(1, p_t(t)/p_d(t)) = \min(p_d(t), p_t(t))$, or by being
drawn from the residual after a rejection. The two paths sum to exactly
$p_t(t)$. So speculative decoding changes the *speed*, never the output
distribution — which is why you can deploy it without re-evaluating quality.

The speedup comes from the target model scoring `K` positions in **one**
forward pass. Acceptance rate depends on how well the draft mimics the target;
a good pairing keeps 60–80%, giving 2–3× fewer target calls.

### ⚠️ JAX-forced signature change
The original calls `torch.rand` and `torch.multinomial`, drawing from a hidden
global RNG. JAX has none, so the PRNG **key is the first argument**. Split it
per position so the accept coin and the resample draw never reuse randomness.
""",
    "stub": '''import jax
import jax.numpy as jnp


def speculative_decode(key, target_probs, draft_probs, draft_tokens):
    """Verify drafted tokens against the target distribution.

    Args:
        key:          jax.random key
        target_probs: (K, V) target-model probabilities
        draft_probs:  (K, V) draft-model probabilities
        draft_tokens: (K,) drafted token ids

    Returns:
        list[int] — accepted tokens, plus one resampled token on rejection.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def speculative_decode(key, target_probs, draft_probs, draft_tokens):
    K = len(draft_tokens)
    accepted = []

    for i in range(K):
        t = int(draft_tokens[i])

        # Ratio > 1 means the target likes this token more than the draft did,
        # so min(1, ratio) accepts it outright.
        ratio = target_probs[i, t] / jnp.maximum(draft_probs[i, t], 1e-10)

        key, k_accept, k_resample = jax.random.split(key, 3)
        if float(jax.random.uniform(k_accept)) < float(jnp.minimum(1.0, ratio)):
            accepted.append(t)
            continue

        # Rejected: emit one token from the normalised residual and stop.
        residual = jnp.maximum(target_probs[i] - draft_probs[i], 0.0)
        s = residual.sum()
        residual = jnp.where(
            s > 0, residual / jnp.maximum(s, 1e-10),
            jnp.ones_like(residual) / residual.shape[0],
        )
        accepted.append(int(jax.random.categorical(k_resample, jnp.log(residual + 1e-10))))
        return accepted

    return accepted
''',
    "demo": '''import jax
import jax.numpy as jnp

V = 5
# A draft that agrees with the target is accepted often.
target = jnp.tile(jnp.array([0.6, 0.1, 0.1, 0.1, 0.1]), (4, 1))
draft = jnp.tile(jnp.array([0.5, 0.2, 0.1, 0.1, 0.1]), (4, 1))
tokens = jnp.zeros(4, dtype=jnp.int32)          # all propose token 0

kept = [len(speculative_decode(jax.random.key(s), target, draft, tokens))
        for s in range(200)]
print("mean tokens emitted per round:", sum(kept) / len(kept), "of 4 drafted")
''',
    "tests": [
        {
            "name": "Accepts everything when the models agree",
            "code": """
import jax
import jax.numpy as jnp

# Identical distributions -> ratio is exactly 1 -> always accepted.
p = jnp.tile(jnp.array([0.7, 0.1, 0.1, 0.1]), (3, 1))
tokens = jnp.zeros(3, dtype=jnp.int32)

for seed in range(20):
    out = {fn}(jax.random.key(seed), p, p, tokens)
    assert isinstance(out, list), f'Must return a list, got {type(out)}'
    assert out == [0, 0, 0], (
        f'With target == draft the ratio is 1, so all 3 tokens must be accepted, got {out}'
    )
""",
        },
        {
            "name": "Ratio > 1 is always accepted",
            "code": """
import jax
import jax.numpy as jnp

# Target strongly prefers token 0; draft barely proposes it.
target = jnp.tile(jnp.array([0.9, 0.05, 0.05]), (2, 1))
draft = jnp.tile(jnp.array([0.1, 0.45, 0.45]), (2, 1))
tokens = jnp.zeros(2, dtype=jnp.int32)

for seed in range(20):
    out = {fn}(jax.random.key(seed), target, draft, tokens)
    assert out == [0, 0], (
        f'ratio = 0.9/0.1 = 9 > 1, so min(1, ratio) = 1 and both must be accepted, got {out}'
    )
""",
        },
        {
            "name": "Rejection stops immediately and emits one token",
            "code": """
import jax
import jax.numpy as jnp

# Target gives token 0 ~zero mass, so it is essentially always rejected.
target = jnp.tile(jnp.array([1e-9, 0.5, 0.5 - 1e-9]), (4, 1))
draft = jnp.tile(jnp.array([0.98, 0.01, 0.01]), (4, 1))
tokens = jnp.zeros(4, dtype=jnp.int32)

for seed in range(20):
    out = {fn}(jax.random.key(seed), target, draft, tokens)
    assert len(out) == 1, (
        f'The first token is rejected, so exactly one resampled token should come '
        f'back and the loop must stop. Got {len(out)} tokens: {out}'
    )
    assert out[0] in (1, 2), (
        f'The resampled token must come from the residual (tokens 1 or 2), got {out[0]}'
    )
""",
        },
        {
            "name": "Resamples from the residual, not the target",
            "code": """
import jax
import jax.numpy as jnp

# Residual = max(target - draft, 0) puts ALL its mass on token 2.
target = jnp.array([[0.0, 0.30, 0.70]])
draft = jnp.array([[1.0, 0.00, 0.00]])
tokens = jnp.zeros(1, dtype=jnp.int32)

got = set({fn}(jax.random.key(s), target, draft, tokens)[0] for s in range(40))
assert got <= {1, 2}, (
    f'Resampling must use max(target - draft, 0) normalised, which allows only '
    f'tokens 1 and 2 here. Got {sorted(got)}. Sampling from target directly '
    'would also be restricted to 1 and 2, but ignoring the subtraction entirely '
    'would let token 0 through.'
)
assert 0 not in got, 'Token 0 has zero residual mass and must never be resampled'
""",
        },
        {
            "name": "Partial acceptance",
            "code": """
import jax
import jax.numpy as jnp

# Positions 0 and 1 always accept; position 2 always rejects.
target = jnp.stack([
    jnp.array([0.9, 0.05, 0.05]),
    jnp.array([0.9, 0.05, 0.05]),
    jnp.array([1e-9, 0.5, 0.5 - 1e-9]),
])
draft = jnp.stack([
    jnp.array([0.9, 0.05, 0.05]),
    jnp.array([0.9, 0.05, 0.05]),
    jnp.array([0.98, 0.01, 0.01]),
])
tokens = jnp.zeros(3, dtype=jnp.int32)

for seed in range(15):
    out = {fn}(jax.random.key(seed), target, draft, tokens)
    assert len(out) == 3, f'Expected 2 accepted + 1 resampled = 3 tokens, got {out}'
    assert out[:2] == [0, 0], f'First two should be the accepted drafts, got {out[:2]}'
    assert out[2] in (1, 2), f'Third should be resampled from the residual, got {out[2]}'
""",
        },
        {
            "name": "Output is distributed as the target (lossless)",
            "code": """
import jax
import jax.numpy as jnp

# The key property. One drafted position, a deliberately poor draft, and the
# FIRST emitted token must still follow the target distribution.
target = jnp.array([[0.5, 0.3, 0.2]])
draft = jnp.array([[0.2, 0.2, 0.6]])

counts = [0, 0, 0]
N = 4000
for s in range(N):
    k1, k2 = jax.random.split(jax.random.key(s))
    tok = int(jax.random.categorical(k1, jnp.log(draft[0])))
    out = {fn}(k2, target, draft, jnp.array([tok], dtype=jnp.int32))
    counts[out[0]] += 1

emp = jnp.array(counts) / N
assert jnp.allclose(emp, target[0], atol=0.04), (
    f'Empirical distribution {emp} should match the target {target[0]}. '
    'Speculative decoding is LOSSLESS — the accept rule plus the residual '
    'resample reproduce the target exactly.'
)
""",
        },
        {
            "name": "Deterministic in the key, and guards degenerate input",
            "code": """
import jax
import jax.numpy as jnp

target = jnp.array([[0.4, 0.6]])
draft = jnp.array([[0.7, 0.3]])
tokens = jnp.zeros(1, dtype=jnp.int32)

a = {fn}(jax.random.key(3), target, draft, tokens)
b = {fn}(jax.random.key(3), target, draft, tokens)
assert a == b, 'The same key must give the same result'

# Zero residual everywhere -> uniform fallback, never NaN or a crash.
same = jnp.array([[0.5, 0.5]])
zero_draft = jnp.array([[1.0, 0.0]])
out = {fn}(jax.random.key(4), same, zero_draft, jnp.zeros(1, dtype=jnp.int32))
assert len(out) >= 1 and out[0] in (0, 1), f'Degenerate residual not handled: {out}'
""",
        },
    ],
}
