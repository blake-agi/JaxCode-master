"""Temperature + top-k + top-p filtering, then one categorical draw."""

TASK = {
    "title": "Top-k / Top-p Sampling",
    "category": "Inference & Decoding",
    "number": "32",
    "difficulty": "Medium",
    "function_name": "sample_top_k_top_p",
    "hint": (
        "Each stage is one JAX op, and the order in the spec is not negotiable. "
        "jax.lax.top_k hands you the k-th largest logit to threshold against; "
        "jnp.argsort plus jnp.cumsum give you the nucleus in sorted-rank space; "
        "a scatter (jnp.zeros(V, bool).at[...]) puts a rank mask back into "
        "vocabulary order; jax.random.categorical samples from logits directly, "
        "so mask with -jnp.inf and never renormalise by hand. The subtle part is "
        "WHICH running total: it has to exclude the token being tested, or the "
        "token that crosses the threshold is thrown away instead of kept. And "
        "check what your rule does when top_p is exactly 0."
    ),
    "description": r"""
Implement the standard **LLM sampling head**: temperature scaling, top-k
truncation, top-p (nucleus) truncation, and a single categorical draw.

```python
def sample_top_k_top_p(key, logits, *, temperature=1.0, top_k=None, top_p=1.0):
    ...  # -> scalar int32 token id
```

### The three knobs

| knob | what it does | effect |
|---|---|---|
| `temperature` | $z \leftarrow z / T$ | $T<1$ sharpens, $T>1$ flattens, $T\to0$ is argmax |
| `top_k` | keep the $k$ largest logits | fixed-size support |
| `top_p` | keep the smallest prefix of the sorted distribution with mass $\ge p$ | adaptive support |

Top-p keeps the **smallest** set $S$ of highest-probability tokens with
$\sum_{x \in S} p(x) \ge p$. Concretely, sort descending and keep rank $i$ iff

$$\sum_{j<i} p_{(j)} \;<\; p$$

The *exclusive* cumulative sum is what makes the token that crosses the
threshold get kept, and it keeps rank 0 alive for any `top_p` $> 0$.
`top_p = 0` is the one case it does not cover — force rank 0 in explicitly so
the filter can never mask the whole vocabulary.

### Rules
- Everything not kept is set to `-jnp.inf` **before** the draw — do not
  renormalise by hand, `jax.random.categorical` takes logits
- Apply in the order **temperature → top-k → top-p → sample**
- top-p sees the already-top-k-masked logits (the two filters compose)
- At least one token must always survive
- No Python loop over the vocabulary; mask with `jnp.where`
- Must work under `jax.jit` (with `top_k`/`top_p` as static Python values) and
  under `jax.vmap` over a batch of keys
- `logits` is 1-D of shape `(V,)`; return a **scalar** integer array

### Why the order matters (the interview question)
Temperature commutes with top-k — dividing by $T$ is monotone, so the identity
of the $k$ largest logits never changes. It does **not** commute with top-p:
the nucleus is defined on probabilities, and $T$ changes them. With
$z = [3,2,1,0]$ and $p = 0.9$, $T=1$ admits three tokens but $T=0.5$ admits
only two. Filtering before scaling gives you the nucleus of the *unscaled*
distribution — wider than you asked for whenever $T<1$, narrower whenever
$T>1$. Ship that bug and your "deterministic, low-temperature" endpoint keeps
emitting tokens the user thought they had truncated away.

The second trap is renormalising twice. Masking to `-inf` and letting the
softmax inside `categorical` normalise once is exact; explicitly dividing by
the surviving mass and then calling a sampler that softmaxes again squares your
probabilities.

The third is ties: `logits < kth_value` keeps *every* token equal to the
$k$-th largest, so a uniform distribution with `top_k=1` keeps the whole
vocabulary. That is the reference behaviour in every production stack, but you
should be able to say why.
""",
    "stub": '''import jax
import jax.numpy as jnp


def sample_top_k_top_p(key, logits, *, temperature=1.0, top_k=None, top_p=1.0):
    """Draw one token id from a filtered categorical distribution.

    Args:
        key:         a jax.random key
        logits:      (V,) unnormalised scores
        temperature: divide the logits by this before filtering
        top_k:       keep only the k largest logits (None = no top-k)
        top_p:       keep the smallest high-probability set with mass >= top_p

    Returns:
        Scalar int array — the sampled token id.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def sample_top_k_top_p(key, logits, *, temperature=1.0, top_k=None, top_p=1.0):
    logits = jnp.asarray(logits, dtype=jnp.float32)
    V = logits.shape[-1]

    # 1. Temperature FIRST. It reshapes the distribution the filters look at.
    logits = logits / jnp.maximum(jnp.asarray(temperature, logits.dtype), 1e-6)

    # 2. Top-k: everything below the k-th largest logit becomes -inf.
    if top_k is not None and top_k < V:
        kth = jax.lax.top_k(logits, top_k)[0][-1]
        logits = jnp.where(logits < kth, -jnp.inf, logits)

    # 3. Top-p on whatever survived top-k (softmax renormalises over it).
    if top_p is not None and top_p < 1.0:
        order = jnp.argsort(logits)[::-1]            # descending
        probs = jax.nn.softmax(logits[order])
        excl = jnp.cumsum(probs) - probs             # EXCLUSIVE cumsum
        keep_sorted = excl < top_p                   # rank 0 always survives
        keep_sorted = keep_sorted.at[0].set(True)    # ...even if top_p == 0
        keep = jnp.zeros(V, dtype=bool).at[order].set(keep_sorted)
        logits = jnp.where(keep, logits, -jnp.inf)

    # 4. One draw. categorical works on logits and treats -inf as probability 0.
    return jax.random.categorical(key, logits)
''',
    "demo": '''import jax
import jax.numpy as jnp

logits = jnp.log(jnp.array([0.5, 0.25, 0.125, 0.075, 0.05]))
keys = jax.random.split(jax.random.key(0), 4000)


def hist(**kw):
    toks = jax.vmap(lambda k: sample_top_k_top_p(k, logits, **kw))(keys)
    return jnp.bincount(toks, length=5) / 4000


print("raw            ", hist())
print("temperature 0.5", hist(temperature=0.5), "<- sharpened")
print("temperature 2.0", hist(temperature=2.0), "<- flattened")
print("top_k=2        ", hist(top_k=2), "<- support of size 2")
print("top_p=0.8      ", hist(top_p=0.8), "<- adaptive support")
print("T=0.5, p=0.8   ", hist(temperature=0.5, top_p=0.8),
      "<- nucleus shrinks because temperature ran first")
''',
    "tests": [
        {
            "name": "top_k=1 is argmax; top_k=2 restricts the support",
            "code": """
import jax
import jax.numpy as jnp

logits = jnp.array([1.0, 5.0, 2.0, 0.5])
keys = jax.random.split(jax.random.key(0), 64)

toks = jax.vmap(lambda k: {fn}(k, logits, top_k=1))(keys)
assert toks.shape == (64,), f'Expected a scalar per key, got shape {toks.shape[1:]} each'
assert bool(jnp.all(toks == 1)), (
    f'top_k=1 must always return the argmax (1), saw {jnp.unique(toks).tolist()}'
)

toks2 = jax.vmap(lambda k: {fn}(k, logits, top_k=2))(keys)
seen = set(int(t) for t in toks2)
assert seen <= {1, 2}, f'top_k=2 must only ever emit the two largest logits, saw {seen}'
assert seen == {1, 2}, f'top_k=2 should reach both survivors over 64 keys, saw {seen}'
""",
        },
        {
            "name": "Filtered distribution is renormalised correctly",
            "code": """
import jax
import jax.numpy as jnp

# p = [0.4, 0.3, 0.2, 0.1]; keeping the top 2 must give [4/7, 3/7].
logits = jnp.log(jnp.array([0.4, 0.3, 0.2, 0.1]))
keys = jax.random.split(jax.random.key(1), 20000)

toks = jax.vmap(lambda k: {fn}(k, logits, top_k=2))(keys)
freq = jnp.bincount(toks, length=4) / 20000.0
expected = jnp.array([4 / 7, 3 / 7, 0.0, 0.0])
assert jnp.allclose(freq, expected, atol=0.02), (
    f'Empirical frequencies {freq} vs expected {expected} — the surviving mass '
    'must be renormalised exactly once (mask to -inf and let softmax do it)'
)

# No filtering at all: every token must be reachable and match p.
toks_all = jax.vmap(lambda k: {fn}(k, logits))(keys)
freq_all = jnp.bincount(toks_all, length=4) / 20000.0
assert jnp.allclose(freq_all, jnp.array([0.4, 0.3, 0.2, 0.1]), atol=0.02), (
    f'With no filtering the output must follow softmax(logits), got {freq_all}'
)
""",
        },
        {
            "name": "Top-p keeps the minimal nucleus, including the boundary",
            "code": """
import jax
import jax.numpy as jnp

# p = [0.5, 0.25, 0.125, 0.125], exclusive cumsum = [0, 0.5, 0.75, 0.875]
logits = jnp.log(jnp.array([0.5, 0.25, 0.125, 0.125]))
keys = jax.random.split(jax.random.key(2), 20000)

# top_p=0.6 keeps ranks 0 and 1 -> renormalised [2/3, 1/3]
toks = jax.vmap(lambda k: {fn}(k, logits, top_p=0.6))(keys)
freq = jnp.bincount(toks, length=4) / 20000.0
assert jnp.allclose(freq, jnp.array([2 / 3, 1 / 3, 0.0, 0.0]), atol=0.02), (
    f'top_p=0.6 should keep exactly tokens 0 and 1 as [2/3, 1/3], got {freq}'
)

# Boundary: exclusive cumsum at rank 1 is exactly 0.5, and 0.5 < 0.5 is False,
# so top_p=0.5 keeps only rank 0.
toks_b = jax.vmap(lambda k: {fn}(k, logits, top_p=0.5))(keys[:200])
assert bool(jnp.all(toks_b == 0)), (
    'top_p=0.5 on p=[0.5, ...] must keep only token 0 — use a strictly-less-than '
    'test on the EXCLUSIVE cumulative sum'
)

# Degenerate top_p must still leave one token alive, not produce all -inf.
toks_z = jax.vmap(lambda k: {fn}(k, logits, top_p=0.0))(keys[:64])
assert bool(jnp.all(toks_z == 0)), f'top_p=0 must fall back to the argmax, got {jnp.unique(toks_z).tolist()}'
""",
        },
        {
            "name": "Temperature is applied BEFORE top-p",
            "code": """
import jax
import jax.numpy as jnp

# softmax([3,2,1,0])   = [.644, .237, .087, .032] -> nucleus(0.9) = 3 tokens
# softmax([6,4,2,0])   = [.865, .117, .016, .002] -> nucleus(0.9) = 2 tokens
logits = jnp.array([3.0, 2.0, 1.0, 0.0])
keys = jax.random.split(jax.random.key(3), 4000)

hot = jax.vmap(lambda k: {fn}(k, logits, temperature=1.0, top_p=0.9))(keys)
assert int(jnp.sum(hot == 2)) > 50, (
    'At temperature 1.0 the 0.9-nucleus contains token 2; it should show up. '
    f'Saw {int(jnp.sum(hot == 2))} occurrences in 4000 draws'
)

cold = jax.vmap(lambda k: {fn}(k, logits, temperature=0.5, top_p=0.9))(keys)
assert int(jnp.sum(cold == 2)) == 0, (
    f'At temperature 0.5 the 0.9-nucleus is only tokens 0 and 1, but token 2 was '
    f'emitted {int(jnp.sum(cold == 2))} times. Divide by the temperature BEFORE '
    'computing the nucleus, not after.'
)
assert set(int(t) for t in cold) == {0, 1}, (
    f'Expected support {{0, 1}} at temperature 0.5, got {sorted(set(int(t) for t in cold))}'
)

# Sanity: the same temperature with no top-p does reach token 2.
free = jax.vmap(lambda k: {fn}(k, logits, temperature=0.5))(keys)
assert int(jnp.sum(free == 2)) > 10, 'Without top-p, temperature 0.5 still reaches token 2'
""",
        },
        {
            "name": "Low temperature concentrates on the argmax",
            "code": """
import jax
import jax.numpy as jnp

logits = jnp.array([1.0, 3.0, 2.0])
keys = jax.random.split(jax.random.key(4), 500)

cold = jax.vmap(lambda k: {fn}(k, logits, temperature=0.01))(keys)
assert int(jnp.sum(cold == 1)) >= 495, (
    f'temperature=0.01 should be effectively greedy, got {int(jnp.sum(cold == 1))}/500 '
    'on the argmax — check you divide by T rather than multiply'
)

hot = jax.vmap(lambda k: {fn}(k, logits, temperature=100.0))(keys)
freq = jnp.bincount(hot, length=3) / 500.0
assert jnp.allclose(freq, 1 / 3, atol=0.08), (
    f'temperature=100 should be nearly uniform, got {freq}'
)
assert jnp.isfinite(jnp.asarray(freq)).all(), 'Non-finite frequencies'
""",
        },
        {
            "name": "Combined filters, valid ids, integer scalar output",
            "code": """
import jax
import jax.numpy as jnp

V = 100
logits = jax.random.normal(jax.random.key(5), (V,)) * 2.0
keys = jax.random.split(jax.random.key(6), 300)

one = {fn}(keys[0], logits, temperature=0.8, top_k=10, top_p=0.9)
one = jnp.asarray(one)
assert one.shape == (), f'Must return a SCALAR token id, got shape {one.shape}'
assert jnp.issubdtype(one.dtype, jnp.integer), f'Must return an integer, got dtype {one.dtype}'

toks = jax.vmap(lambda k: {fn}(k, logits, temperature=0.8, top_k=10, top_p=0.9))(keys)
assert bool(jnp.all((toks >= 0) & (toks < V))), f'Token id out of range: {toks.min()}..{toks.max()}'

# top_k=10 caps the support at 10 no matter what top_p does.
support = set(int(t) for t in toks)
assert len(support) <= 10, f'top_k=10 allows at most 10 distinct tokens, saw {len(support)}'

# Those survivors must be a subset of the 10 largest logits.
top10 = set(int(i) for i in jnp.argsort(logits)[-10:])
assert support <= top10, f'Sampled outside the top-10 logits: {sorted(support - top10)}'
""",
        },
        {
            "name": "Deterministic per key, and jit-compatible",
            "code": """
import jax
import jax.numpy as jnp

logits = jax.random.normal(jax.random.key(7), (32,))
key = jax.random.key(8)

a = {fn}(key, logits, temperature=0.7, top_k=5, top_p=0.95)
b = {fn}(key, logits, temperature=0.7, top_k=5, top_p=0.95)
assert int(a) == int(b), 'The same key must give the same token — do not fold in extra entropy'

c = {fn}(jax.random.key(9), logits, temperature=0.7, top_k=5, top_p=0.95)
toks = jax.vmap(lambda k: {fn}(k, logits, temperature=0.7, top_k=5, top_p=0.95))(
    jax.random.split(jax.random.key(10), 200)
)
assert len(set(int(t) for t in toks)) > 1, 'Different keys must be able to give different tokens'

f = jax.jit(lambda k, l: {fn}(k, l, temperature=0.7, top_k=5, top_p=0.95))
jitted = f(key, logits)
assert int(jitted) == int(a), (
    f'jit gave {int(jitted)} but eager gave {int(a)} — filtering must use jnp.where, '
    'not Python control flow on traced values'
)
""",
        },
    ],
}
