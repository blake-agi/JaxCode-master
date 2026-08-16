"""Masked (absorbing-state) diffusion language model — the training objective.

The loss every deployed diffusion LM actually optimises (MDLM, MD4, LLaDA,
Dream): a weighted cross-entropy over the masked positions only. b_17 is the
sampler that inverts it.
"""

TASK = {
    "title": "Masked Diffusion LM Loss",
    "category": "Training",
    "number": "b_16",
    "difficulty": "Medium",
    "function_name": "masked_diffusion_loss",
    "extra_names": ["q_sample"],
    "hint": (
        "Forward: mask each position independently with probability t — one "
        "uniform draw per position, compared against t broadcast over the "
        "sequence. Loss: force the MASK column of the logits to -inf BEFORE "
        "the log-softmax (the model must never predict [MASK]), gather "
        "-log p(x0) per position, zero out the positions that were not "
        "masked, sum over the sequence and divide by the sequence LENGTH — "
        "not by how many tokens happened to be masked — then scale by 1/t and "
        "average over the batch."
    ),
    "description": r"""
A masked diffusion LM is trained by corrupting a sequence to `[MASK]` and asking
the model to fill it back in. The forward process is per-token and independent:
with the linear schedule $\alpha_t = 1 - t$, each position survives with
probability $\alpha_t$ and becomes `[MASK]` with probability $t$.

The continuous-time NELBO collapses into something you already know — a
cross-entropy, weighted by $1/t$ and restricted to the masked positions:

$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1]}\;
\frac{1}{t}\,\frac{1}{L}\sum_{i=1}^{L}
\mathbb{1}[x_t^i = \texttt{[MASK]}]\;\bigl(-\log p_\theta(x_0^i \mid x_t)\bigr)$$

Implement both halves: the forward corruption, and the loss for one $(t, x_t)$
sample.

> Want the mechanism before the objective? Skim **b_17**'s problem statement
> first — it is the reverse process this loss trains, and watching an
> all-`[MASK]` sequence fill in makes the $1/t$ weighting below easier to see.
> Come back here to implement, though: `q_sample` is the forward kernel b_17's
> posterior is derived from.

### Rules
- `q_sample(key, x0, t, *, mask_id)` -> `x_t`, same shape as `x0`
- `masked_diffusion_loss(logits, x0, xt, t, *, mask_id)` -> **scalar**
- `logits` is `(B, L, V)`, `x0` and `xt` are `(B, L)` integer ids, `t` is `(B,)`
  with $t \in (0, 1]$ — one time per sequence, not one per token
- Each position is masked **independently**; `q_sample` must not mask a fixed
  count per sequence
- The model may never predict `[MASK]`: drive that column to $-\infty$ **before**
  normalising, so it holds no probability mass at all
- Positions that were not masked contribute **exactly zero**
- `jax.nn.log_softmax` and `logsumexp` are allowed here — you wrote both in 16
  and b_14, and the point of this problem is elsewhere. `optax` is still banned
- Must work under `jit` and `vmap`

### The denominator is the whole problem
Divide by the number of **masked** tokens and it will look right, train, and be
wrong. The $1/t$ factor already accounts for how many tokens the forward process
masks: $\mathbb{E}[\#\text{masked}] = tL$, so $\frac{1}{t}\sum_{\text{masked}}$
is an unbiased estimator of the full NELBO. Dividing by the *realised* count
corrects for the same thing a second time, and the estimator stops being the
bound you meant to optimise — the gradient is now systematically reweighted
towards the high-noise end.

Divide by $L$ instead, and what you get is the bound in per-token units, which
is exactly what a bits-per-token number needs.

There is a cheap way to remember which is which: take a batch where every
sequence sees the same $t$ but happens to get a different number of masked
tokens. The loss **should** differ between them — more masked tokens is more
evidence, and averaging that away throws it out.

### Why the MASK column must be $-\infty$
`[MASK]` is in the vocabulary, so an unconstrained softmax will happily put mass
on it — mass that can only be wrong, since $x_0$ never contains `[MASK]`. Worse,
it is the easy answer: predicting the token that is literally sitting in the
input is the shortcut every underfit model finds first.

Zeroing it *after* the softmax is not the same thing — the remaining
probabilities no longer sum to one, so the loss is no longer a log-likelihood.
Setting the logit to $-\infty$ before normalising redistributes that mass over
the real vocabulary, which is what the SUBS ("zero masking probability")
parameterisation means in the MDLM paper.

### Absorbing vs uniform, and where flow matching fits
The other classic discrete kernel is **uniform** (D3PM-uniform, SEDD-uniform):
instead of an absorbing `[MASK]`, a corrupted token jumps to a *random* token.
That kernel has no absorbing state, so its posterior needs the full transition
matrix $Q = \alpha I + \frac{1-\alpha}{V}\mathbf{1}\mathbf{1}^\top$ and the loss
becomes a KL between categoricals. Absorbing wins in practice for a structural
reason worth being able to say out loud: `[MASK]` is *self-identifying*. The
model can see exactly which positions are corrupted, already-decoded tokens are
never destroyed again, and the posterior collapses to "stay masked, or jump
straight to $x_0$" — no matrix products anywhere.

**Discrete flow matching** (Campbell et al., Gat et al. 2024) with a masked
source distribution and a linear $\kappa$ schedule produces *this same
objective*, term for term. The framings differ, and the sampler you get out of
the flow view is more general (it can un-decode a token, which the diffusion
posterior above cannot), but if you are asked "how is discrete flow matching
different from masked diffusion", the honest answer starts with "the training
loss is the same". MeanFlow is a different animal again: average-velocity fields
in *continuous* space for one-step generation, not a discrete-token method.
""",
    "stub": '''import jax
import jax.numpy as jnp


def q_sample(key, x0, t, *, mask_id):
    """Corrupt x0: mask each position independently with probability t.

    Args:
        key:     PRNG key
        x0:      (B, L) integer token ids
        t:       (B,) times in (0, 1]
        mask_id: the [MASK] token id

    Returns:
        (B, L) partially masked token ids.
    """
    pass  # Replace this


def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    """Weighted cross-entropy over the masked positions.

    Args:
        logits:  (B, L, V) model scores for x0 given xt
        x0:      (B, L) clean token ids
        xt:      (B, L) masked token ids, as returned by q_sample
        t:       (B,) times in (0, 1]
        mask_id: the [MASK] token id

    Returns:
        Scalar loss.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def q_sample(key, x0, t, *, mask_id):
    x0 = jnp.asarray(x0)
    t = jnp.asarray(t)
    # One draw PER POSITION, compared against this sequence's t: independent
    # masking, so the count varies from sequence to sequence.
    u = jax.random.uniform(key, x0.shape)
    return jnp.where(u < t[..., None], mask_id, x0)


def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    logits = jnp.asarray(logits)
    x0 = jnp.asarray(x0)
    t = jnp.asarray(t)

    # The model may never predict [MASK]. Killing the logit BEFORE the
    # normalisation is what redistributes that mass over the real vocabulary.
    logits = logits.at[..., mask_id].set(-jnp.inf)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    nll = -jnp.take_along_axis(log_probs, x0[..., None], axis=-1)[..., 0]

    # Only the masked positions carry loss; the rest contribute exactly zero.
    masked = xt == mask_id
    per_token = jnp.where(masked, nll, 0.0)

    # Divide by the sequence LENGTH, not by the number of masked tokens: the
    # 1/t factor already accounts for E[#masked] = t * L.
    per_seq = jnp.sum(per_token, axis=-1) / x0.shape[-1]
    return jnp.mean(per_seq / t)
''',
    "demo": '''import jax
import jax.numpy as jnp

V, MASK = 8, 7                      # ids 0..6 are real tokens, 7 is [MASK]
x0 = jnp.array([[0, 1, 2, 3, 4, 5]])
t = jnp.array([0.5])

xt = q_sample(jax.random.key(0), x0, t, mask_id=MASK)
print("x0:", x0[0], "\\nxt:", xt[0], " masked:", int(jnp.sum(xt == MASK)), "of 6")

# Independent masking: the count varies run to run, it is not exactly t * L.
counts = [int(jnp.sum(q_sample(jax.random.key(k), x0, t, mask_id=MASK) == MASK))
          for k in range(8)]
print("counts over 8 keys:", counts)

# A perfect model scores zero loss.
perfect = jax.nn.one_hot(x0, V) * 100.0
print("\\nperfect model:", float(masked_diffusion_loss(perfect, x0, xt, t, mask_id=MASK)))

# A uniform model over the 7 REAL tokens: each masked position costs log(7).
flat = jnp.zeros((1, 6, V))
n_masked = int(jnp.sum(xt == MASK))
print("uniform model:", float(masked_diffusion_loss(flat, x0, xt, t, mask_id=MASK)))
print("by hand      :", float((1 / t[0]) * n_masked / 6 * jnp.log(V - 1)))

# The MASK column is inert: poisoning it changes nothing.
poisoned = flat.at[..., MASK].set(50.0)
print("\\npoisoned MASK logit:", float(masked_diffusion_loss(poisoned, x0, xt, t, mask_id=MASK)))

# Same t, more masked tokens -> MORE loss. Dividing by #masked would flatten
# this to a constant, which is the bug this problem is about.
for k in (2, 5):
    xk = q_sample(jax.random.key(k), x0, t, mask_id=MASK)
    print(f"  {int(jnp.sum(xk == MASK))} masked -> "
          f"{float(masked_diffusion_loss(flat, x0, xk, t, mask_id=MASK)):.4f}")
''',
    "tests": [
        {
            "name": "q_sample masks independently, with probability t",
            "code": """
import jax
import jax.numpy as jnp

V, MASK = 10, 9
x0 = jax.random.randint(jax.random.key(0), (4, 16), 0, 9)
t = jnp.array([0.25, 0.5, 0.75, 1.0])

xt = q_sample(jax.random.key(1), x0, t, mask_id=MASK)
assert xt.shape == x0.shape, f'q_sample changed the shape: {xt.shape} vs {x0.shape}'

# Every position is either untouched or [MASK] — nothing else may appear.
kept = xt != MASK
assert jnp.all(jnp.where(kept, xt == x0, True)), (
    'a position that is not [MASK] must still hold its original token'
)

# t = 1.0 masks everything.
assert jnp.all(xt[3] == MASK), f'row with t=1.0 must be fully masked, got {xt[3]}'

# Rates: masked fraction ~ t, averaged over many keys.
big = jnp.zeros((256, 64), dtype=jnp.int32)
for tv in (0.2, 0.6):
    tt = jnp.full((256,), tv)
    m = q_sample(jax.random.key(int(tv * 10)), big, tt, mask_id=MASK) == MASK
    rate = float(jnp.mean(m))
    assert abs(rate - tv) < 0.02, f'masked fraction {rate:.3f} for t={tv}'

# INDEPENDENT per position: the count must vary between sequences, not be
# pinned to round(t * L) the way a top-k / fixed-count implementation would.
counts = jnp.sum(q_sample(jax.random.key(7), jnp.zeros((64, 32), dtype=jnp.int32),
                          jnp.full((64,), 0.5), mask_id=MASK) == MASK, axis=-1)
assert jnp.std(counts) > 1.0, (
    f'mask counts per sequence are near-constant (std={float(jnp.std(counts)):.2f}); '
    'each position must be masked independently, not a fixed count per sequence'
)

# Per-sequence t, not one t for the whole batch.
tt = jnp.array([0.0, 1.0, 0.0, 1.0])
xt2 = q_sample(jax.random.key(3), x0, tt, mask_id=MASK)
assert jnp.all(xt2[0] == x0[0]) and jnp.all(xt2[2] == x0[2]), 't=0 rows must be untouched'
assert jnp.all(xt2[1] == MASK) and jnp.all(xt2[3] == MASK), 't=1 rows must be fully masked'
""",
        },
        {
            "name": "Hand-computed loss values",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L = 8, 7, 6
x0 = jnp.array([[0, 1, 2, 3, 4, 5]])
xt = jnp.array([[7, 1, 7, 3, 7, 5]])          # 3 of 6 masked
t = jnp.array([0.5])

# Uniform logits over the whole vocabulary. With the MASK column removed the
# model is uniform over V-1 = 7 real tokens, so each masked position costs
# log(7), and the loss is (1/t) * (n_masked / L) * log(7).
flat = jnp.zeros((1, L, V))
out = {fn}(flat, x0, xt, t, mask_id=MASK)
expected = (1 / 0.5) * (3 / L) * jnp.log(V - 1.0)
assert jnp.ndim(out) == 0, f'loss must be a scalar, got shape {out.shape}'
assert jnp.allclose(out, expected, atol=1e-5), (
    f'{float(out):.6f} vs {float(expected):.6f} — uniform over the V-1 real tokens '
    f'costs log({V - 1}) per masked position, weighted by 1/t and averaged over L'
)

# A perfect model costs nothing.
perfect = jax.nn.one_hot(x0, V) * 200.0
assert float({fn}(perfect, x0, xt, t, mask_id=MASK)) < 1e-4, 'perfect model must score ~0'

# Nothing masked -> nothing to predict -> exactly 0, and no nan from 1/t.
tiny_t = jnp.array([1e-6])
assert jnp.allclose({fn}(flat, x0, x0, tiny_t, mask_id=MASK), 0.0, atol=1e-9), (
    'with no masked positions the loss is 0 — and 1/t must not turn that into nan'
)
""",
        },
        {
            "name": "The 1/t weight and the L denominator",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L = 8, 7, 6
x0 = jnp.array([[0, 1, 2, 3, 4, 5]])
xt = jnp.array([[7, 1, 7, 3, 7, 5]])
flat = jnp.zeros((1, L, V))

# Same corruption, twice the t -> exactly half the loss.
a = {fn}(flat, x0, xt, jnp.array([0.25]), mask_id=MASK)
b = {fn}(flat, x0, xt, jnp.array([0.5]), mask_id=MASK)
assert jnp.allclose(a, 2 * b, atol=1e-5), (
    f'loss at t=0.25 is {float(a):.4f} and at t=0.5 is {float(b):.4f}; the 1/t '
    'weight makes the first exactly twice the second'
)

# THE denominator test. Same t, different numbers of masked tokens: the loss
# must scale with the count. Dividing by #masked instead of L makes these equal.
one = jnp.array([[7, 1, 2, 3, 4, 5]])
four = jnp.array([[7, 7, 7, 7, 4, 5]])
l1 = {fn}(flat, x0, one, jnp.array([0.5]), mask_id=MASK)
l4 = {fn}(flat, x0, four, jnp.array([0.5]), mask_id=MASK)
assert jnp.allclose(l4, 4 * l1, atol=1e-5), (
    f'1 masked token gives {float(l1):.4f} and 4 give {float(l4):.4f} — they must be '
    'in a 1:4 ratio. Divide by the sequence LENGTH, not by the number of masked '
    'tokens: the 1/t factor already accounts for E[#masked] = t*L'
)

# Per-sequence t: row 0 and row 1 are identical apart from t.
x2 = jnp.concatenate([x0, x0])
xt2 = jnp.concatenate([xt, xt])
both = {fn}(jnp.zeros((2, L, V)), x2, xt2, jnp.array([0.25, 0.5]), mask_id=MASK)
assert jnp.allclose(both, 0.5 * (a + b), atol=1e-5), (
    f'{float(both):.6f} vs {float(0.5 * (a + b)):.6f} — t is per SEQUENCE, and the '
    'batch is averaged after each row is weighted by its own 1/t'
)
""",
        },
        {
            "name": "[MASK] holds no probability mass",
            "code": """
import jax
import jax.numpy as jnp

V, MASK, L = 8, 7, 6
x0 = jnp.array([[0, 1, 2, 3, 4, 5]])
xt = jnp.array([[7, 1, 7, 3, 7, 5]])
t = jnp.array([0.5])
logits = jax.random.normal(jax.random.key(0), (1, L, V)) * 2.0

base = {fn}(logits, x0, xt, t, mask_id=MASK)
assert jnp.isfinite(base), f'non-finite loss: {base}'

# The MASK column is inert: whatever it holds, the answer is the same.
for fill in (-30.0, 0.0, 30.0):
    poisoned = logits.at[..., MASK].set(fill)
    got = {fn}(poisoned, x0, xt, t, mask_id=MASK)
    assert jnp.allclose(got, base, atol=1e-5), (
        f'setting the MASK logit to {fill} changed the loss ({float(got):.6f} vs '
        f'{float(base):.6f}) — drive that column to -inf BEFORE normalising'
    )

# ...and the mass really is redistributed, not just dropped: the loss must equal
# a log-softmax over the V-1 real columns.
real = jnp.delete(logits, MASK, axis=-1)
lp = jax.nn.log_softmax(real, axis=-1)
nll = -jnp.take_along_axis(lp, x0[..., None], axis=-1)[..., 0]   # ids < MASK, so
ref = (1 / t[0]) * jnp.sum(jnp.where(xt == MASK, nll, 0.0)) / L   # deleting MASK
assert jnp.allclose(base, ref, atol=1e-5), (                      # keeps the index
    f'{float(base):.6f} vs {float(ref):.6f} — zeroing the MASK probability AFTER '
    'the softmax leaves the rest summing to less than 1, which is no longer a '
    'log-likelihood'
)

# Unmasked positions contribute nothing, so their logits cannot matter either.
poison = logits.at[0, 1].set(jnp.array([-50.0] * V)).at[0, 3].set(jnp.array([50.0] * V))
assert jnp.allclose({fn}(poison, x0, xt, t, mask_id=MASK), base, atol=1e-5), (
    'changing the logits at an UNMASKED position changed the loss'
)
""",
        },
        {
            "name": "Gradients, jit and vmap",
            "code": """
import functools
import jax
import jax.numpy as jnp

V, MASK, L, B = 8, 7, 6, 3
x0 = jax.random.randint(jax.random.key(0), (B, L), 0, MASK)
xt = q_sample(jax.random.key(1), x0, jnp.full((B,), 0.6), mask_id=MASK)
t = jnp.array([0.3, 0.6, 0.9])
logits = jax.random.normal(jax.random.key(2), (B, L, V))

f = functools.partial({fn}, mask_id=MASK)
eager = f(logits, x0, xt, t)
assert jnp.allclose(jax.jit(f)(logits, x0, xt, t), eager, atol=1e-6), 'jit changed the answer'

per = jax.vmap(f)(logits, x0, xt, t)
assert per.shape == (B,), f'vmap gave {per.shape}, expected ({B},)'
assert jnp.allclose(jnp.mean(per), eager, atol=1e-5), (
    'the mean of the per-sequence losses should equal the batched loss'
)

g = jax.grad(lambda z: f(z, x0, xt, t))(logits)
assert g.shape == logits.shape, f'gradient shape {g.shape} vs {logits.shape}'
assert jnp.isfinite(g).all(), (
    'non-finite gradient — a -inf logit must be introduced with .at[].set(), not '
    'by adding -inf to a value autodiff still threads through'
)
assert jnp.allclose(g[..., MASK], 0.0, atol=1e-9), (
    'the MASK column must receive no gradient'
)
# Unmasked positions get no gradient either.
unmasked = xt != MASK
assert jnp.allclose(jnp.where(unmasked[..., None], g, 0.0), 0.0, atol=1e-9), (
    'unmasked positions must receive no gradient'
)
""",
        },
        {
            "name": "The estimator is unbiased",
            "code": """
import jax
import jax.numpy as jnp

# With logits held fixed, averaging the loss over t and over the masking noise
# must recover the plain per-token cross-entropy of x0. This is the property
# the 1/t weight exists for: E[#masked] = t*L cancels the 1/t exactly. Getting
# the weight or the denominator wrong breaks it.
V, MASK, L, B = 6, 5, 8, 4096
x0 = jax.random.randint(jax.random.key(0), (1, L), 0, MASK)
logits = jax.random.normal(jax.random.key(1), (1, L, V)) * 1.5

lp = jax.nn.log_softmax(logits.at[..., MASK].set(-jnp.inf), axis=-1)
target = float(jnp.mean(-jnp.take_along_axis(lp, x0[..., None], axis=-1)))

x0b = jnp.repeat(x0, B, axis=0)
lb = jnp.repeat(logits, B, axis=0)
# That cancellation holds for ANY distribution over t, so this sweeps a
# stratified grid over [0.05, 1] rather than (0, 1]: the tiny-t end contributes
# nothing but variance (rare masks, huge 1/t) and would make the check flaky.
t = 0.05 + 0.95 * (jnp.arange(B) + 0.5) / B
xt = q_sample(jax.random.key(2), x0b, t, mask_id=MASK)
est = float({fn}(lb, x0b, xt, t, mask_id=MASK))

assert abs(est - target) < 0.06 * target, (
    f'averaged estimate {est:.4f} vs the true cross-entropy {target:.4f}. The '
    'weighted-masked-CE estimator must be unbiased: check the 1/t weight and '
    'that the denominator is L, not the number of masked tokens'
)
""",
        },
    ],
}
