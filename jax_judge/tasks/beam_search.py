"""Beam search with length-normalised scoring over a black-box score function."""

TASK = {
    "title": "Beam Search Decoding",
    "category": "Inference & Decoding",
    "number": "33",
    "difficulty": "Medium",
    "function_name": "beam_search",
    "hint": (
        "Carry two Python lists — live hypotheses and finished ones — and let the "
        "array work be one (n_live, V) matrix per step: each beam's running score "
        "added to its next-token log-probs. Adding a column vector to that matrix "
        "IS the expansion, and top_k over the flattened matrix is the prune; "
        "divmod recovers which beam and which token a flat index came from. Two "
        "things trip people up. First, the eos column is not a candidate to keep "
        "alive — harvest it, then take it out of the running before the top_k. "
        "Second, decide deliberately where length normalisation belongs: inside "
        "the loop every live beam has the same length, so it cannot change any "
        "ranking there."
    ),
    "description": r"""
Implement **beam search** over a black-box scoring function, ranked by
length-normalised log-probability.

```python
def beam_search(score_fn, start_token, max_len, beam_width, eos_token,
                length_penalty=1.0):
    ...  # -> list[int], the best sequence, starting with start_token
```

`score_fn(tokens)` receives the sequence decoded so far (including
`start_token`) and returns a `(V,)` array of log-probabilities for the **next**
token. Read `V` off that first call — do not assume it.

### The algorithm, exactly
Start with one live hypothesis `(start_token,)` at score `0.0`. Repeat at most
`max_len - 1` times:

1. **Expand.** For every live beam $b$ and every token $v$, the candidate score
   is $s_b + \log p_b(v)$.
2. **Terminate.** Every candidate whose new token is `eos_token` is finished:
   move it to the finished set. It is never extended again.
3. **Prune.** The best `beam_width` *non-eos* candidates become the new live
   set. If none survive, stop.

Anything still live when the loop ends is also a (truncated) hypothesis. Return
the finished hypothesis maximising

$$\mathrm{score}_{\text{norm}} = \frac{\sum_t \log p(y_t \mid y_{<t})}{L^{\alpha}}, \qquad L = \text{tokens generated after } \texttt{start\_token}$$

with $\alpha =$ `length_penalty`. $\alpha = 0$ recovers the raw sum;
$\alpha = 1$ is the mean log-probability per token.

### Rules
- No Python sort over `n_live * V` candidate tuples — build the `(n_live, V)`
  score matrix and take `jax.lax.top_k` on its flattened view
- Prune on **raw** sums, rank the final answer on **normalised** scores
- `beam_width = 1` must reduce to greedy decoding
- Return a plain Python `list[int]` beginning with `start_token`
- `len(result) <= max_len`

### Why length normalisation is not optional
Every $\log p$ is negative, so appending a token can only ever *lower* a raw
sum. Raw-score beam search therefore has a structural bias toward stopping
early: it prefers a two-token answer at $-0.4$ over a five-token answer at
$-0.9$ even though the latter is a much more confident model of its own tokens
($-0.225$ vs $-0.4$ per token). Un-normalised beam search in machine
translation empirically truncates long sentences; GNMT's fix is
$((5+L)/6)^\alpha$, a smoothed version of the $L^\alpha$ used here.

### Why you should usually not use it anyway
Beam search answers "what is the most likely continuation?" — and for
open-ended generation that is the wrong question. The mode of a language model
is bland: high-likelihood text is generic, repetitive, and degenerate
("I don't know. I don't know. I don't know."), because real human text is not
the argmax of its own distribution — it sits in a band of moderate surprisal.
Widening the beam makes this *worse*, not better: you search harder for a mode
you did not want.

So beam search belongs where the output is genuinely constrained and near-unique
— translation, speech recognition, constrained code or JSON generation,
anything scored by BLEU/WER — and top-k/top-p sampling belongs in chat and
open-ended writing. The other cost is systems-level: a width-$B$ beam is $B$
concurrent KV caches and $B\times$ the attention memory, and every step needs a
global top-k across beams, which is a synchronisation point that pure sampling
never pays for.
""",
    "stub": '''import math

import jax
import jax.numpy as jnp


def beam_search(score_fn, start_token, max_len, beam_width, eos_token,
                length_penalty=1.0):
    """Decode the best sequence under score_fn with a beam of beam_width.

    Args:
        score_fn:       callable, tokens-so-far -> (V,) next-token log-probs
        start_token:    int, the first token of every hypothesis
        max_len:        int, maximum total sequence length (incl. start_token)
        beam_width:     int, number of live hypotheses kept per step
        eos_token:      int, token that terminates a hypothesis
        length_penalty: float alpha in score / L**alpha, L = generated length

    Returns:
        list[int] — the best sequence, starting with start_token.
    """
    pass  # Replace this
''',
    "solution": '''import math

import jax
import jax.numpy as jnp


def beam_search(score_fn, start_token, max_len, beam_width, eos_token,
                length_penalty=1.0):
    start_token, eos_token = int(start_token), int(eos_token)

    live_toks = [(start_token,)]
    live_scores = [0.0]
    finished = []                      # (tokens, raw cumulative log-prob)

    for _ in range(max_len - 1):
        if not live_toks:
            break

        # (n_live, V): every beam's running score broadcast over the vocabulary.
        logps = jnp.stack([jnp.asarray(score_fn(t)) for t in live_toks])
        totals = logps + jnp.asarray(live_scores, dtype=logps.dtype)[:, None]
        n_live, V = totals.shape

        # An eos extension terminates that hypothesis; it is never extended again.
        for b in range(n_live):
            finished.append((live_toks[b] + (eos_token,), float(totals[b, eos_token])))

        # Prune on the RAW sums — every live beam has the same length here, so
        # length normalisation could not reorder them anyway.
        masked = totals.at[:, eos_token].set(-jnp.inf).reshape(-1)
        vals, flat = jax.lax.top_k(masked, min(beam_width, masked.shape[0]))

        new_toks, new_scores = [], []
        for v, f in zip(vals.tolist(), flat.tolist()):
            if not math.isfinite(v):   # ran out of real candidates
                continue
            b, tok = divmod(f, V)
            new_toks.append(live_toks[b] + (tok,))
            new_scores.append(v)
        live_toks, live_scores = new_toks, new_scores

    # Beams still alive at max_len are truncated hypotheses, but still candidates.
    finished.extend(zip(live_toks, live_scores))

    def normalised(hyp):
        toks, score = hyp
        return score / max(len(toks) - 1, 1) ** length_penalty

    return list(max(finished, key=normalised)[0])
''',
    "demo": '''import jax.numpy as jnp

# A toy model: token 1 is a safe-but-mediocre step, token 2 is a gamble that
# pays off, token 3 is eos.
def score_fn(tokens):
    lp = jnp.full((4,), -100.0)
    if len(tokens) == 1:
        lp = lp.at[1].set(-0.5).at[2].set(-0.9)
    elif tokens[-1] == 1:
        lp = lp.at[3].set(-2.5)
    elif tokens[-1] == 2:
        lp = lp.at[3].set(-0.1)
    else:
        lp = lp.at[3].set(-50.0)
    return lp


print("greedy   (width 1):", beam_search(score_fn, 0, 4, 1, 3))
print("beam     (width 2):", beam_search(score_fn, 0, 4, 2, 3))
print("-> greedy commits to the locally better token 1 and pays for it later")


# Length normalisation flips the winner.
def steady(tokens):
    lp = jnp.full((4,), -20.0)
    if len(tokens) - 1 < 3:
        lp = lp.at[1].set(-0.3).at[3].set(-0.4)
    else:
        lp = lp.at[3].set(0.0)
    return lp


print("alpha=0 (raw sums):", beam_search(steady, 0, 5, 2, 3, length_penalty=0.0))
print("alpha=1 (per token):", beam_search(steady, 0, 5, 2, 3, length_penalty=1.0))
''',
    "tests": [
        {
            "name": "Contract: list of ints starting at start_token",
            "code": """
import jax.numpy as jnp

calls = []


def sf(tokens):
    calls.append(tuple(int(t) for t in tokens))
    return jnp.full((3,), -1.0).at[2].set(-0.5)


out = {fn}(sf, start_token=1, max_len=3, beam_width=2, eos_token=2)

assert isinstance(out, list), f'Must return a Python list, got {type(out).__name__}'
assert len(out) >= 1 and int(out[0]) == 1, f'Sequence must start with start_token=1, got {out}'
assert len(out) <= 3, f'Sequence longer than max_len=3: {out}'
assert all(0 <= int(t) < 3 for t in out), f'Token out of vocabulary range: {out}'

assert calls, 'score_fn was never called'
assert calls[0] == (1,), (
    f'The first score_fn call must receive just (start_token,), got {calls[0]}'
)
assert all(c[0] == 1 for c in calls), 'Every score_fn call must include start_token first'
assert all(2 not in c[1:] for c in calls), (
    'score_fn was called on a sequence that already contains eos — finished '
    'hypotheses must never be extended'
)
""",
        },
        {
            "name": "beam_width=1 reduces to greedy decoding",
            "code": """
import jax.numpy as jnp


def greedy_fn(tokens):
    lp = jnp.full((5,), -10.0)
    return lp.at[min(len(tokens), 4)].set(0.0)


out = [int(t) for t in {fn}(greedy_fn, start_token=0, max_len=5, beam_width=1, eos_token=4)]
assert out == [0, 1, 2, 3, 4], f'Greedy path should be [0, 1, 2, 3, 4], got {out}'
""",
        },
        {
            "name": "A wider beam escapes the greedy trap",
            "code": """
import jax.numpy as jnp


def tricky(tokens):
    # Token 1 looks better now (-0.5 vs -0.9) but leads to a terrible eos.
    lp = jnp.full((4,), -100.0)
    if len(tokens) == 1:
        lp = lp.at[1].set(-0.5).at[2].set(-0.9)
    elif tokens[-1] == 1:
        lp = lp.at[3].set(-2.5)
    elif tokens[-1] == 2:
        lp = lp.at[3].set(-0.1)
    else:
        lp = lp.at[3].set(-50.0)
    return lp


greedy = [int(t) for t in {fn}(tricky, start_token=0, max_len=4, beam_width=1, eos_token=3)]
assert greedy == [0, 1, 3], f'Width-1 search should take the myopic path [0, 1, 3], got {greedy}'

beam = [int(t) for t in {fn}(tricky, start_token=0, max_len=4, beam_width=2, eos_token=3)]
assert beam == [0, 2, 3], (
    f'Width-2 search should find [0, 2, 3] (total -1.0 vs -3.0), got {beam}. '
    'Keep the second-best hypothesis alive instead of pruning to the argmax.'
)
""",
        },
        {
            "name": "Length normalisation changes the winner",
            "code": """
import jax.numpy as jnp


def steady(tokens):
    # Continuing costs -0.3/token, stopping costs -0.4; after 3 tokens eos is free.
    lp = jnp.full((4,), -20.0)
    if len(tokens) - 1 < 3:
        lp = lp.at[1].set(-0.3).at[3].set(-0.4)
    else:
        lp = lp.at[3].set(0.0)
    return lp


raw = [int(t) for t in {fn}(steady, 0, 5, 2, 3, length_penalty=0.0)]
assert raw == [0, 3], (
    f'With length_penalty=0 the raw sums favour the shortest hypothesis [0, 3] '
    f'(-0.4 beats every longer path), got {raw}'
)

norm = [int(t) for t in {fn}(steady, 0, 5, 2, 3, length_penalty=1.0)]
assert norm == [0, 1, 1, 1, 3], (
    f'With length_penalty=1 the per-token score -0.9/4 = -0.225 beats -0.4/1, so '
    f'the answer should be [0, 1, 1, 1, 3], got {norm}. Divide by L**alpha where '
    'L counts the tokens generated AFTER start_token.'
)
""",
        },
        {
            "name": "eos terminates and is not extended",
            "code": """
import jax.numpy as jnp


def eos_fn(tokens):
    return jnp.zeros((4,)).at[3].set(10.0)


out = [int(t) for t in {fn}(eos_fn, start_token=0, max_len=12, beam_width=2, eos_token=3)]
assert out == [0, 3], (
    f'The model scores eos at +10 and everything else at 0, so [0, 3] wins on the '
    f'normalised score (10/1 vs 10/2 for [0, x, 3]); got {out}'
)

# max_len=1 leaves no room to generate anything.
short = [int(t) for t in {fn}(eos_fn, start_token=0, max_len=1, beam_width=2, eos_token=3)]
assert short == [0], f'max_len=1 must return just [start_token], got {short}'
""",
        },
        {
            "name": "A full-width beam finds the true optimum",
            "code": """
import jax.numpy as jnp

V, EOS, START, MAX_LEN = 4, 3, 0, 4


def sf(tokens):
    h = 17
    for t in tokens:
        h = (h * 31 + int(t) + 7) % 1009
    vals = jnp.array([float((h * (7 * i + 5)) % 29 + 1) for i in range(V)])
    return jnp.log(vals / vals.sum())


def seq_score(seq):
    s = 0.0
    for i in range(1, len(seq)):
        s += float(sf(tuple(seq[:i]))[seq[i]])
    return s


def normalised(seq):
    return seq_score(seq) / max(len(seq) - 1, 1)


# Enumerate every hypothesis the spec can produce.
hyps = []


def rec(seq, depth):
    if depth == MAX_LEN - 1:
        hyps.append(tuple(seq))            # truncated at max_len
        return
    for v in range(V):
        if v == EOS:
            hyps.append(tuple(seq + [v]))  # terminated
        else:
            rec(seq + [v], depth + 1)


rec([START], 0)
best = max(hyps, key=normalised)

out = [int(t) for t in {fn}(sf, start_token=START, max_len=MAX_LEN, beam_width=64, eos_token=EOS)]
assert tuple(out) in hyps, f'{out} is not a hypothesis this search can produce'
assert abs(normalised(out) - normalised(best)) < 1e-4, (
    f'Beam width 64 is exhaustive here, so it must find the optimum '
    f'{list(best)} (score {normalised(best):.5f}); got {out} '
    f'(score {normalised(out):.5f})'
)
""",
        },
        {
            "name": "Wider beams never score worse",
            "code": """
import jax.numpy as jnp

V, EOS = 5, 4

# A hand-built garden-path model. Token 1 is the best move RIGHT NOW, but it
# leads to a dead end; token 2 looks worse and then pays off. Greedy takes the
# bait and scores -1.35; any beam of width >= 2 keeps token 2 alive and finds
# -0.825. This is the whole reason beam search exists, so it is what the test
# is built around — a scoring function on which greedy is already optimal
# cannot tell a real beam search apart from an argmax loop.
TRAP = {
    (0,):   [-9.0, -0.2, -1.6, -9.0, -9.0],
    (0, 1): [-9.0, -9.0, -9.0, -9.0, -2.50],
    (0, 2): [-9.0, -9.0, -9.0, -9.0, -0.05],
}
FALLBACK = [-9.0, -9.0, -9.0, -9.0, -1.0]


def sf(tokens):
    return jnp.array(TRAP.get(tuple(int(t) for t in tokens), FALLBACK))


def normalised(seq):
    s = 0.0
    for i in range(1, len(seq)):
        s += float(sf(tuple(seq[:i]))[seq[i]])
    return s / max(len(seq) - 1, 1)


scores = []
for width in (1, 2, 4, 8):
    out = [int(t) for t in {fn}(sf, start_token=0, max_len=5, beam_width=width, eos_token=EOS)]
    assert out[0] == 0 and len(out) <= 5, f'width={width} produced an invalid sequence {out}'
    scores.append(normalised(out))

for i in range(1, len(scores)):
    assert scores[i] >= scores[i - 1] - 1e-4, (
        f'Widening the beam made the answer worse: {scores} for widths [1, 2, 4, 8]. '
        'A wider beam is a superset of the search a narrower one performs.'
    )
assert scores[-1] > scores[0] + 1e-6, (
    f'Every beam width scored the same ({scores}) on a model built so that greedy '
    'is provably suboptimal. A width-8 beam must find the token-2 branch that '
    'greedy misses — this usually means the search only ever keeps the single '
    'best candidate, i.e. it is an argmax loop rather than a beam.'
)
""",
        },
    ],
}
