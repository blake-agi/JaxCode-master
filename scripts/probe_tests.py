#!/usr/bin/env python3
"""Probe JAXCode test suites with deliberately-wrong implementations.

`verify_tasks.py` proves each solution passes its own tests — self-consistency.
This proves the tests actually REJECT wrong answers, which self-consistency
cannot. Several attacks come straight from the upstream TorchCode issue tracker,
where the PyTorch original's tests were reported as too weak:

  - issue #17: causal attention accepted BOTH sqrt(d_k) and d_k scaling
  - issue #21: vit_patch did not check that patchification was correct
  - issue  #9: relu accepted implementations that broke on N-D / autodiff

A test suite that does NOT reject an attack has a hole; add a test that closes it.

    python scripts/probe_tests.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jax_judge._contract import build_namespace, render_test
from jax_judge._term import BOLD, DIM, GREEN, RED, RESET
from jax_judge.tasks import get_task


def probe(task_id, label, src, symbol=None):
    t = get_task(task_id)
    fn_name = t["function_name"]
    ns = {}
    try:
        exec(src, ns)
    except Exception as e:
        print(f"  {DIM}(attack failed to compile: {e}){RESET}")
        return
    if symbol:                      # attack defines the impl under another name
        ns = {**ns, fn_name: ns[symbol]}
    base = build_namespace(t, ns)
    caught = []
    for test in t["tests"]:
        code = render_test(t, test)
        try:
            exec(compile(code, "<t>", "exec"), dict(base))
        except AssertionError as e:
            caught.append((test["name"], str(e)[:100]))
        except Exception as e:
            caught.append((test["name"], f"{type(e).__name__}: {e}"[:100]))
    n_tests = len(t["tests"])
    if caught:
        print(f"  {GREEN}✅ CAUGHT{RESET} {label} {DIM}({len(caught)}/{n_tests} tests fail){RESET}")
        print(f"     {DIM}first: {caught[0][0]} — {caught[0][1]}{RESET}")
    else:
        print(f"  {RED}❌ HOLE — SLIPS THROUGH{RESET}  {BOLD}{label}{RESET} "
              f"{RED}(0/{n_tests} tests catch it){RESET}")
    return not caught


holes = []

print(f"\n{BOLD}=== attention: scaling (upstream issue #17) ==={RESET}")
holes.append(("attention", probe("attention", "d_k instead of sqrt(d_k)", '''
import jax, jax.numpy as jnp
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = (q @ jnp.swapaxes(k, -1, -2)) / d_k
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))
    return jax.nn.softmax(scores, axis=-1) @ v
''')))
holes.append(("attention", probe("attention", "no scaling at all", '''
import jax, jax.numpy as jnp
def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ jnp.swapaxes(k, -1, -2)
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, scores.dtype))
    return jax.nn.softmax(scores, axis=-1) @ v
''')))
holes.append(("attention", probe("attention", "mask multiplied after softmax", '''
import jax, jax.numpy as jnp
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, q.dtype))
    w = jax.nn.softmax(scores, axis=-1)
    if mask is not None:
        w = w * mask
    return w @ v
''')))

print(f"\n{BOLD}=== relu (upstream issue #9) ==={RESET}")
holes.append(("relu", probe("relu", "abs() instead of clamp", '''
import jax.numpy as jnp
def relu(x):
    return jnp.abs(x)
''')))
holes.append(("relu", probe("relu", "leaky (0.01 slope) passed off as relu", '''
import jax.numpy as jnp
def relu(x):
    return jnp.where(x > 0, x, 0.01 * x)
''')))

print(f"\n{BOLD}=== softmax ==={RESET}")
holes.append(("softmax", probe("softmax", "no max subtraction (unstable)", '''
import jax.numpy as jnp
def my_softmax(x, axis=-1):
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=axis, keepdims=True)
''')))

print(f"\n{BOLD}=== layernorm ==={RESET}")
holes.append(("layernorm", probe("layernorm", "unbiased variance (ddof=1)", '''
import jax.numpy as jnp
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    n = x.shape[-1]
    var = jnp.sum((x - mean) ** 2, axis=-1, keepdims=True) / (n - 1)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta
''')))
holes.append(("layernorm", probe("layernorm", "normalises over the batch axis", '''
import jax.numpy as jnp
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta
''')))

print(f"\n{BOLD}=== vit_patch (upstream issue #21) ==={RESET}")
holes.append(("vit_patch", probe("vit_patch", "reshape without the transpose (scrambles patches)", '''
import jax, jax.numpy as jnp
from flax import nnx
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, *, rngs):
        self.img_size=img_size; self.patch_size=patch_size
        self.in_channels=in_channels; self.embed_dim=embed_dim
        grid=img_size//patch_size; self.num_patches=grid*grid
        pd=patch_size*patch_size*in_channels
        self.w=nnx.Param(jax.random.normal(rngs.params(), (pd, embed_dim))/jnp.sqrt(pd))
        self.b=nnx.Param(jnp.zeros((embed_dim,)))
    def __call__(self, x):
        B,H,W,C=x.shape; P=self.patch_size
        # BUG: straight reshape, no transpose -> rows of the image, not patches
        patches=x.reshape(B,(H//P)*(W//P),P*P*C)
        return patches @ self.w + self.b
''')))

print(f"\n{BOLD}=== adam ==={RESET}")
holes.append(("adam", probe("adam", "no bias correction", '''
import jax, jax.numpy as jnp
class MyAdam:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr; self.beta1, self.beta2 = betas; self.eps = eps
    def init(self, params):
        return {"m": jax.tree.map(jnp.zeros_like, params),
                "v": jax.tree.map(jnp.zeros_like, params), "t": 0}
    def update(self, params, grads, state):
        t = state["t"] + 1
        m = jax.tree.map(lambda m_, g: self.beta1*m_ + (1-self.beta1)*g, state["m"], grads)
        v = jax.tree.map(lambda v_, g: self.beta2*v_ + (1-self.beta2)*g*g, state["v"], grads)
        p = jax.tree.map(lambda p_, m_, v_: p_ - self.lr*m_/(jnp.sqrt(v_)+self.eps), params, m, v)
        return p, {"m": m, "v": v, "t": t}
''')))

print(f"\n{BOLD}=== ppo_loss ==={RESET}")
holes.append(("ppo_loss", probe("ppo_loss", "clip without min (the classic)", '''
import jax, jax.numpy as jnp
def ppo_loss(new_logps, old_logps, advantages, clip_eps=0.2, mask=None):
    ratio = jnp.exp(new_logps - old_logps)
    surr = jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * advantages
    if mask is None:
        return -jnp.mean(surr)
    mask = mask.astype(surr.dtype)
    return -jnp.sum(surr*mask)/jnp.maximum(jnp.sum(mask), 1.0)
''')))

print(f"\n{BOLD}=== dpo_loss ==={RESET}")
holes.append(("dpo_loss", probe("dpo_loss", "reference model ignored", '''
import jax, jax.numpy as jnp
def dpo_loss(pc, pr, rc, rr, beta=0.1):
    return -jnp.mean(jax.nn.log_sigmoid(beta*(pc - pr)))
''')))

print(f"\n{BOLD}=== logsumexp ==={RESET}")
# This task needs two symbols, so each probe supplies a correct partner for the
# one it is breaking — otherwise the namespace is incomplete and the probe fails
# for the wrong reason.
_LSE_OK = """
import jax.numpy as jnp
def logsumexp(x, axis=-1, keepdims=False):
    m = jnp.max(x, axis=axis, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    out = jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True)) + m
    return out if keepdims else jnp.squeeze(out, axis=axis)
"""
_MERGE_OK = """
def logsumexp_merge(m1, l1, m2, l2):
    m = jnp.maximum(m1, m2)
    safe = jnp.where(jnp.isfinite(m), m, 0.0)
    return m, l1 * jnp.exp(m1 - safe) + l2 * jnp.exp(m2 - safe)
"""

# The bug this problem exists for: `m` is added back still carrying keepdims,
# so it broadcasts instead of adding. Values stay correct on 1-D — only the
# rank is wrong — which is exactly why the suite must assert shapes.
holes.append(("logsumexp", probe("logsumexp", "keepdims max added back to a reduced sum", '''
import jax.numpy as jnp
def logsumexp(x, axis=-1, keepdims=False):
    m = jnp.max(x, axis=axis, keepdims=True)
    out = jnp.log(jnp.sum(jnp.exp(x - m), axis=axis)) + m
    return out if keepdims else jnp.squeeze(out, axis)
''' + _MERGE_OK)))
holes.append(("logsumexp", probe("logsumexp", "global max instead of per-slice", '''
import jax.numpy as jnp
def logsumexp(x, axis=-1, keepdims=False):
    m = jnp.max(x)
    out = jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True)) + m
    return out if keepdims else jnp.squeeze(out, axis)
''' + _MERGE_OK)))

# The streaming merge must rescale BOTH partial sums onto the new max. Rescaling
# only the incoming one is the natural mistake, and stays invisible until the
# chunks sit on different scales.
holes.append(("logsumexp", probe("logsumexp", "merge rescales only one side", _LSE_OK + '''
def logsumexp_merge(m1, l1, m2, l2):
    m = jnp.maximum(m1, m2)
    return m, l1 + l2 * jnp.exp(m2 - m)
''')))
# Correct for every non-empty input; only the (-inf, 0) identity exposes it,
# because exp(-inf - -inf) is nan rather than 0.
holes.append(("logsumexp", probe("logsumexp", "merge has no -inf guard (empty state -> nan)", _LSE_OK + '''
def logsumexp_merge(m1, l1, m2, l2):
    m = jnp.maximum(m1, m2)
    return m, l1 * jnp.exp(m1 - m) + l2 * jnp.exp(m2 - m)
''')))

print(f"\n{BOLD}=== mini_gpt ==={RESET}")
# The assembly bugs: each one runs, trains, and is a different model.
from jax_judge.tasks import TASKS as _T
_SOL = _T["mini_gpt"]["solution"]
holes.append(("mini_gpt", probe("mini_gpt", "RoPE added to the embedding, not q/k",
    _SOL.replace("q, k = apply_rope(q, positions), apply_rope(k, positions)", "pass")
        .replace("x = self.tok_emb(ids)                      # no position added here",
                 "x = apply_rope(self.tok_emb(ids), jnp.arange(ids.shape[1]))"))))
holes.append(("mini_gpt", probe("mini_gpt", "v rotated as well as q/k",
    _SOL.replace("q, k = apply_rope(q, positions), apply_rope(k, positions)",
                 "q, k, v = apply_rope(q, positions), apply_rope(k, positions), apply_rope(v, positions)"))))
holes.append(("mini_gpt", probe("mini_gpt", "no final norm before the head",
    _SOL.replace("        x = self.norm_f(x)\n", ""))))
holes.append(("mini_gpt", probe("mini_gpt", "untied head (separate Linear)",
    _SOL.replace("self.norm_f = nnx.RMSNorm(d_model, rngs=rngs)",
                 "self.norm_f = nnx.RMSNorm(d_model, rngs=rngs)\n        self.head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)")
        .replace("return self.tok_emb.attend(x)", "return self.head(x)"))))

print(f"\n{BOLD}=== cross_entropy ==={RESET}")
holes.append(("cross_entropy", probe("cross_entropy", "log(softmax()) instead of log_softmax", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    p = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
    lp = jnp.log(p)
    oh = jax.nn.one_hot(labels, logits.shape[-1])
    return jnp.mean(-jnp.sum(oh * lp, axis=-1))
''')))
holes.append(("cross_entropy", probe("cross_entropy", "sum instead of mean over the batch", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    lp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.sum(jnp.take_along_axis(lp, labels[:, None], axis=-1))
''')))
holes.append(("cross_entropy", probe("cross_entropy", "logits[labels] indexes rows, not one per row", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    return jnp.mean(jax.scipy.special.logsumexp(logits, axis=-1) - logits[labels])
''')))

print(f"\n{BOLD}=== cross_entropy_fused (b_14: logsumexp banned) ==={RESET}")
# The ban itself is not probeable here: it is enforced by reading the function's
# source, and attacks are exec'd from strings, where inspect.getsource has
# nothing to read. That test deliberately no-ops rather than failing everyone in
# CI. Its live behaviour is covered by scripts/smoke_notebooks.py --banned.
holes.append(("cross_entropy_fused", probe(
    "cross_entropy_fused", "no max shift: exp() overflows", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    lse = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return jnp.mean(lse - jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0])
''')))
holes.append(("cross_entropy_fused", probe(
    "cross_entropy_fused", "shift subtracted but never added back", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))
    return jnp.mean(lse - jnp.take_along_axis(logits, labels[:, None], axis=-1))
''')))
holes.append(("cross_entropy_fused", probe(
    "cross_entropy_fused", "max/sum without keepdims: broadcasts over the batch axis", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels):
    m = jnp.max(logits, axis=-1)
    lse = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1))
    return jnp.mean(lse - jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0])
''')))

print(f"\n{BOLD}=== cross_entropy_full (b_15) ==={RESET}")
holes.append(("cross_entropy_full", probe(
    "cross_entropy_full", "log(softmax()) instead of log_softmax", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels, *, label_smoothing=0.0, ignore_index=-1):
    p = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
    lp = jnp.log(p)
    n = logits.shape[-1]
    oh = jax.nn.one_hot(labels, n)
    if label_smoothing:
        oh = oh*(1-label_smoothing) + label_smoothing/n
    loss = -jnp.sum(oh*lp, axis=-1)
    m = (labels != ignore_index)
    return jnp.sum(loss*m)/jnp.maximum(jnp.sum(m), 1)
''')))
holes.append(("cross_entropy_full", probe(
    "cross_entropy_full", "ignored positions masked but still in the denominator", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels, *, label_smoothing=0.0, ignore_index=-1):
    lp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    valid = labels != ignore_index
    nll = -jnp.take_along_axis(lp, jnp.where(valid, labels, 0)[..., None], axis=-1)[..., 0]
    per = (1 - label_smoothing) * nll + label_smoothing * (-jnp.mean(lp, axis=-1))
    return jnp.mean(jnp.where(valid, per, 0.0))
''')))
holes.append(("cross_entropy_full", probe(
    "cross_entropy_full", "smoothing spread over the OTHER C-1 classes", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels, *, label_smoothing=0.0, ignore_index=-1):
    lp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    n = logits.shape[-1]
    valid = labels != ignore_index
    safe = jnp.where(valid, labels, 0)
    oh = jax.nn.one_hot(safe, n)
    q = oh * (1 - label_smoothing) + (1 - oh) * label_smoothing / (n - 1)
    per = -jnp.sum(q * lp, axis=-1)
    per = jnp.where(valid, per, 0.0)
    return jnp.sum(per) / jnp.maximum(jnp.sum(valid), 1)
''')))
holes.append(("cross_entropy_full", probe(
    "cross_entropy_full", "gathers the raw ignore_index, masks with * (0 * NaN = NaN)", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels, *, label_smoothing=0.0, ignore_index=-1):
    lp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    nll = -jnp.take_along_axis(lp, labels[..., None], axis=-1)[..., 0]
    per = (1 - label_smoothing) * nll + label_smoothing * (-jnp.mean(lp, axis=-1))
    valid = (labels != ignore_index).astype(logits.dtype)
    return jnp.sum(per * valid) / jnp.maximum(jnp.sum(valid), 1)
''')))
holes.append(("cross_entropy_full", probe(
    "cross_entropy_full", "unguarded denominator: all-padding batch -> nan", '''
import jax, jax.numpy as jnp
def cross_entropy_loss(logits, labels, *, label_smoothing=0.0, ignore_index=-1):
    lp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    valid = labels != ignore_index
    safe = jnp.where(valid, labels, 0)
    nll = -jnp.take_along_axis(lp, safe[..., None], axis=-1)[..., 0]
    per = (1 - label_smoothing) * nll + label_smoothing * (-jnp.mean(lp, axis=-1))
    return jnp.sum(jnp.where(valid, per, 0.0)) / jnp.sum(valid)
''')))

print(f"\n{BOLD}=== masked_diffusion (b_16) ==={RESET}")
holes.append(("masked_diffusion", probe(
    "masked_diffusion", "averages over the MASKED tokens instead of the length", '''
import jax, jax.numpy as jnp
def q_sample(key, x0, t, *, mask_id):
    return jnp.where(jax.random.uniform(key, x0.shape) < t[..., None], mask_id, x0)
def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    lp = jax.nn.log_softmax(logits.at[..., mask_id].set(-jnp.inf), axis=-1)
    nll = -jnp.take_along_axis(lp, x0[..., None], axis=-1)[..., 0]
    m = xt == mask_id
    n = jnp.maximum(jnp.sum(m, axis=-1), 1)
    return jnp.mean(jnp.sum(jnp.where(m, nll, 0.0), axis=-1) / n / t)
''', symbol=None)))
holes.append(("masked_diffusion", probe(
    "masked_diffusion", "drops the 1/t weight", '''
import jax, jax.numpy as jnp
def q_sample(key, x0, t, *, mask_id):
    return jnp.where(jax.random.uniform(key, x0.shape) < t[..., None], mask_id, x0)
def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    lp = jax.nn.log_softmax(logits.at[..., mask_id].set(-jnp.inf), axis=-1)
    nll = -jnp.take_along_axis(lp, x0[..., None], axis=-1)[..., 0]
    m = xt == mask_id
    return jnp.mean(jnp.sum(jnp.where(m, nll, 0.0), axis=-1) / x0.shape[-1])
''')))
holes.append(("masked_diffusion", probe(
    "masked_diffusion", "zeroes the MASK probability AFTER the softmax", '''
import jax, jax.numpy as jnp
def q_sample(key, x0, t, *, mask_id):
    return jnp.where(jax.random.uniform(key, x0.shape) < t[..., None], mask_id, x0)
def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    p = jax.nn.softmax(logits, axis=-1).at[..., mask_id].set(0.0)
    nll = -jnp.log(jnp.take_along_axis(p, x0[..., None], axis=-1)[..., 0])
    m = xt == mask_id
    return jnp.mean(jnp.sum(jnp.where(m, nll, 0.0), axis=-1) / x0.shape[-1] / t)
''')))
holes.append(("masked_diffusion", probe(
    "masked_diffusion", "q_sample masks a fixed count per sequence", '''
import jax, jax.numpy as jnp
def q_sample(key, x0, t, *, mask_id):
    L = x0.shape[-1]
    order = jax.random.permutation(key, L)[None, :]
    k = jnp.round(t * L)[..., None]
    return jnp.where(order < k, mask_id, x0)
def masked_diffusion_loss(logits, x0, xt, t, *, mask_id):
    lp = jax.nn.log_softmax(logits.at[..., mask_id].set(-jnp.inf), axis=-1)
    nll = -jnp.take_along_axis(lp, x0[..., None], axis=-1)[..., 0]
    m = xt == mask_id
    return jnp.mean(jnp.sum(jnp.where(m, nll, 0.0), axis=-1) / x0.shape[-1] / t)
''')))

print(f"\n{BOLD}=== diffusion_sampling (b_17) ==={RESET}")
holes.append(("diffusion_sampling", probe(
    "diffusion_sampling", "unmask probability is the difference, not the ratio", '''
import jax, jax.numpy as jnp
def denoise_step(key, logits, xt, t, s, *, mask_id):
    logits = logits.at[..., mask_id].set(-jnp.inf)
    ku, kt = jax.random.split(key)
    reveal = jax.random.bernoulli(ku, t - s, xt.shape) & (xt == mask_id)
    return jnp.where(reveal, jax.random.categorical(kt, logits, axis=-1), xt)
''')))
holes.append(("diffusion_sampling", probe(
    "diffusion_sampling", "one Bernoulli for the whole batch", '''
import jax, jax.numpy as jnp
def denoise_step(key, logits, xt, t, s, *, mask_id):
    logits = logits.at[..., mask_id].set(-jnp.inf)
    ku, kt = jax.random.split(key)
    reveal = jax.random.bernoulli(ku, (t - s) / t) & (xt == mask_id)
    return jnp.where(reveal, jax.random.categorical(kt, logits, axis=-1), xt)
''')))
holes.append(("diffusion_sampling", probe(
    "diffusion_sampling", "resamples every position, ignoring carry-over", '''
import jax, jax.numpy as jnp
def denoise_step(key, logits, xt, t, s, *, mask_id):
    logits = logits.at[..., mask_id].set(-jnp.inf)
    ku, kt = jax.random.split(key)
    reveal = jax.random.bernoulli(ku, (t - s) / t, xt.shape)
    return jnp.where(reveal, jax.random.categorical(kt, logits, axis=-1), xt)
''')))
holes.append(("diffusion_sampling", probe(
    "diffusion_sampling", "lets the model emit [MASK]", '''
import jax, jax.numpy as jnp
def denoise_step(key, logits, xt, t, s, *, mask_id):
    ku, kt = jax.random.split(key)
    reveal = jax.random.bernoulli(ku, (t - s) / t, xt.shape) & (xt == mask_id)
    return jnp.where(reveal, jax.random.categorical(kt, logits, axis=-1), xt)
''')))
holes.append(("diffusion_sampling", probe(
    "diffusion_sampling", "takes the argmax instead of sampling", '''
import jax, jax.numpy as jnp
def denoise_step(key, logits, xt, t, s, *, mask_id):
    logits = logits.at[..., mask_id].set(-jnp.inf)
    reveal = jax.random.bernoulli(key, (t - s) / t, xt.shape) & (xt == mask_id)
    return jnp.where(reveal, jnp.argmax(logits, axis=-1), xt)
''')))

print(f"\n{BOLD}{'='*60}{RESET}")
real_holes = [t for t, ok in holes if ok is True]
if real_holes:
    print(f"{RED}{BOLD}HOLES FOUND in: {sorted(set(real_holes))}{RESET}")
else:
    print(f"{GREEN}{BOLD}No holes — every attack was rejected.{RESET}")

raise SystemExit(1 if real_holes else 0)
