"""Gradient accumulation over micro-batches — exact equivalence to one big batch."""

TASK = {
    "title": "Gradient Accumulation",
    "category": "Training",
    "order": 6,
    "difficulty": "Medium",
    "function_name": "accumulate_grads",
    "hint": (
        "Accumulate into a zero-filled copy of the params tree. The whole problem "
        "is the weighting: grad_fn returns the MEAN over its micro-batch, and an "
        "unweighted mean of means equals the full-batch mean only when every "
        "micro-batch is the same size. Weight each micro-batch's gradient by its "
        "row count and divide by the total at the end. Batch sizes are static "
        "shapes, so that arithmetic is plain Python and stays jit-friendly."
    ),
    "description": r"""
Accumulate gradients across a list of micro-batches into **one** gradient
pytree, purely functionally, so that a single optimizer step is mathematically
identical to one step on the concatenated batch.

Let $\ell_j$ be the per-example loss and let micro-batch $i$ hold $n_i$
examples. `grad_fn` returns the gradient of the **mean** loss over the batch it
is given:

$$g_i = \nabla_\theta \frac{1}{n_i}\sum_{j \in B_i} \ell_j
\qquad\text{but the full-batch gradient is}\qquad
g^\star = \nabla_\theta \frac{1}{N}\sum_{j} \ell_j,\; N = \sum_i n_i$$

Recovering $g^\star$ therefore means undoing each micro-batch's own denominator:

$$g^\star = \frac{1}{N}\sum_i n_i\, g_i$$

### Rules
- Signature: `accumulate_grads(grad_fn, params, micro_batches)`
- `grad_fn(params, batch) -> (loss, grads)`; `loss` is the **mean** over `batch`
  and `grads` is a pytree matching `params`
- A batch is a pytree whose leaves all share a leading example axis; its size is
  `jax.tree.leaves(batch)[0].shape[0]`. **Sizes may differ between micro-batches.**
- Return `(loss, grads)` — the size-weighted mean loss and the size-weighted
  mean gradient, i.e. exactly what `grad_fn(params, concat(micro_batches))`
  would have returned
- Call `grad_fn` **once per micro-batch**. Concatenating the micro-batches and
  calling it once defeats the entire purpose
- Combine trees with `jax.tree.map`, not hand-written recursion over dict keys
- Raise `ValueError` on an empty `micro_batches`

### The averaging subtlety
Almost every tutorial writes `loss / n_micro_batches` and stops. That is correct
**only when all micro-batches are the same size**, and it fails silently the
moment they are not — which is precisely the common case: the ragged last chunk
of an epoch, a bucketed-by-length dataloader, or a per-host shard that does not
divide evenly.

With sizes $(3, 5, 2)$ the naive $\frac{1}{3}(g_1+g_2+g_3)$ weights each of the
2 examples in the last micro-batch by $\frac{1}{3}\cdot\frac{1}{2} = 0.167$ while
each of the 5 examples in the middle one gets $\frac{1}{3}\cdot\frac{1}{5} =
0.067$. Short micro-batches quietly dominate the update. Nothing crashes, no
shape is wrong, the loss curve still goes down — you just are not optimising the
objective you think you are.

The same bug in language-model training is worse, because there the natural unit
is the **token**, not the sequence. If your loss is a mean over unmasked tokens,
the accumulation weight must be each micro-batch's unmasked-token count, not its
sequence count. Padding-heavy micro-batches otherwise get over-weighted, and the
effective objective drifts with your batch-shuffling seed.

### Why it matters
Accumulation buys memory with **time**, not with compute: the FLOP count is
identical, but you only ever hold the activations of one micro-batch, and you
pay for that in arithmetic intensity — the parameters are re-read from HBM once
per micro-batch instead of once per step — and in lost parallelism. That is the
trade that lets a large *effective* batch size survive on limited HBM, and it is
the sequential twin of data parallelism: the weighted sum here is exactly the
weighted all-reduce a multi-host job performs across devices.

One thing JAX gives you for free: there is no mutable `.grad` buffer, so the
classic "forgot to zero the gradients" bug cannot be written. The accumulator is
an explicit value you create with `jnp.zeros_like` and thread through the loop.
The failure mode moves entirely to the *weighting*, which is the part the
interviewer is actually probing.
""",
    "stub": '''import jax
import jax.numpy as jnp


def accumulate_grads(grad_fn, params, micro_batches):
    """Accumulate micro-batch gradients into one full-batch-equivalent gradient.

    Args:
        grad_fn:       callable (params, batch) -> (loss, grads); `loss` is the
                       MEAN loss over `batch` and `grads` matches `params`
        params:        parameter pytree
        micro_batches: list of batches. The number of examples in a batch is
                       jax.tree.leaves(batch)[0].shape[0] and may differ
                       between micro-batches.

    Returns:
        (loss, grads) identical to calling `grad_fn` on the concatenation of
        every micro-batch.

    Raises:
        ValueError: if `micro_batches` is empty.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp


def accumulate_grads(grad_fn, params, micro_batches):
    micro_batches = list(micro_batches)
    if not micro_batches:
        raise ValueError("accumulate_grads requires at least one micro-batch")

    # Explicit accumulator instead of a mutable .grad buffer.
    acc = jax.tree.map(jnp.zeros_like, params)
    total_loss = jnp.zeros(())
    total_n = 0

    for batch in micro_batches:
        # Leading axis is the example axis; shapes are static, so n is a
        # plain Python int and all of this stays jit-friendly.
        n = jax.tree.leaves(batch)[0].shape[0]
        loss, grads = grad_fn(params, batch)

        # grad_fn already divided by n, so multiply it back out before summing.
        # Skipping this factor is the silent bug when the n_i differ.
        acc = jax.tree.map(lambda a, g: a + n * g, acc, grads)
        total_loss = total_loss + n * loss
        total_n += n

    return total_loss / total_n, jax.tree.map(lambda a: a / total_n, acc)
''',
    "demo": '''import jax
import jax.numpy as jnp

params = {"w": jnp.array([[1.0], [-2.0], [0.5]]), "b": jnp.array([0.25])}


def loss_fn(p, batch):
    x, y = batch
    return jnp.mean((x @ p["w"] + p["b"] - y) ** 2)


grad_fn = jax.value_and_grad(loss_fn)

key = jax.random.key(0)
X = jax.random.normal(key, (10, 3))
Y = jax.random.normal(jax.random.key(1), (10, 1))

full_loss, full_grads = grad_fn(params, (X, Y))

# Deliberately ragged: 3 + 5 + 2 = 10
chunks = [(X[:3], Y[:3]), (X[3:8], Y[3:8]), (X[8:], Y[8:])]
acc_loss, acc_grads = accumulate_grads(grad_fn, params, chunks)

naive = jax.tree.map(
    lambda *gs: sum(gs) / len(gs), *[grad_fn(params, c)[1] for c in chunks]
)

print("full batch  w-grad:", full_grads["w"].ravel())
print("accumulated w-grad:", acc_grads["w"].ravel(), "  <- matches")
print("naive mean  w-grad:", naive["w"].ravel(), "  <- silently different")
print("loss:", float(full_loss), float(acc_loss))
''',
    "tests": [
        {
            "name": "Unequal micro-batches match the full batch exactly",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.array([[0.5, -0.25], [1.0, 0.75], [-0.5, 0.1], [0.2, 0.3]]),
          "b": jnp.array([0.1, -0.2])}

def loss_fn(p, batch):
    x, y = batch
    return jnp.mean((x @ p["w"] + p["b"] - y) ** 2)

grad_fn = jax.value_and_grad(loss_fn)

X = jax.random.normal(jax.random.key(0), (10, 4))
Y = jax.random.normal(jax.random.key(1), (10, 2))
full_loss, full_grads = grad_fn(params, (X, Y))

chunks = [(X[:3], Y[:3]), (X[3:8], Y[3:8]), (X[8:], Y[8:])]   # 3 + 5 + 2
acc_loss, acc_grads = {fn}(grad_fn, params, chunks)

assert jax.tree.structure(acc_grads) == jax.tree.structure(params), (
    'Returned gradient pytree does not match the params structure'
)
for k in ("w", "b"):
    assert jnp.allclose(acc_grads[k], full_grads[k], atol=1e-5, rtol=1e-5), (
        f'grad[{k}] mismatch: {acc_grads[k]} vs full-batch {full_grads[k]} — '
        'weight each micro-batch gradient by its example count n_i before summing'
    )

# Sanity: the naive unweighted mean really is a different answer here, so the
# assertion above has teeth.
naive = jax.tree.map(lambda *gs: sum(gs) / len(gs),
                     *[grad_fn(params, c)[1] for c in chunks])
assert not jnp.allclose(naive["w"], full_grads["w"], atol=1e-4), (
    'Test setup problem: the naive mean happened to coincide with the full batch'
)
""",
        },
        {
            "name": "Returned loss is the size-weighted mean",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.array([[1.0], [-2.0], [0.5]]), "b": jnp.array([0.25])}

def loss_fn(p, batch):
    x, y = batch
    return jnp.mean((x @ p["w"] + p["b"] - y) ** 2)

grad_fn = jax.value_and_grad(loss_fn)

X = jax.random.normal(jax.random.key(2), (9, 3))
Y = jax.random.normal(jax.random.key(3), (9, 1)) * 3.0
full_loss, _ = grad_fn(params, (X, Y))

chunks = [(X[:1], Y[:1]), (X[1:7], Y[1:7]), (X[7:], Y[7:])]   # 1 + 6 + 2
acc_loss, _ = {fn}(grad_fn, params, chunks)

assert jnp.ndim(acc_loss) == 0, f'Loss should be a scalar, got shape {jnp.shape(acc_loss)}'
assert jnp.allclose(acc_loss, full_loss, atol=1e-5, rtol=1e-5), (
    f'Loss {float(acc_loss)} vs full-batch {float(full_loss)} — the reported loss '
    'must also be weighted by n_i, not averaged over micro-batches'
)

per_chunk = [float(grad_fn(params, c)[0]) for c in chunks]
naive_loss = sum(per_chunk) / len(per_chunk)
assert abs(naive_loss - float(full_loss)) > 1e-4, (
    'Test setup problem: unweighted loss mean coincided with the full batch'
)
""",
        },
        {
            "name": "Equal sizes and the single-micro-batch identity",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.array([[0.3, 0.7], [-1.1, 0.2]]), "b": jnp.array([0.0, 1.0])}

def loss_fn(p, batch):
    x, y = batch
    return jnp.mean((jnp.tanh(x @ p["w"] + p["b"]) - y) ** 2)

grad_fn = jax.value_and_grad(loss_fn)

X = jax.random.normal(jax.random.key(4), (8, 2))
Y = jax.random.normal(jax.random.key(5), (8, 2))

# One micro-batch: must reproduce grad_fn's own output.
one_loss, one_grads = {fn}(grad_fn, params, [(X, Y)])
ref_loss, ref_grads = grad_fn(params, (X, Y))
assert jnp.allclose(one_loss, ref_loss, atol=1e-6), f'{float(one_loss)} vs {float(ref_loss)}'
assert jnp.allclose(one_grads["w"], ref_grads["w"], atol=1e-6), (
    'A single micro-batch must pass grad_fn through untouched — check you are not '
    'dividing by len(micro_batches) somewhere as well'
)

# Equal sizes: weighting by n_i must reduce to the plain average.
chunks = [(X[i:i + 2], Y[i:i + 2]) for i in range(0, 8, 2)]
acc_loss, acc_grads = {fn}(grad_fn, params, chunks)
full_loss, full_grads = grad_fn(params, (X, Y))
assert jnp.allclose(acc_grads["w"], full_grads["w"], atol=1e-5, rtol=1e-5), (
    f'Equal-size case failed: {acc_grads["w"]} vs {full_grads["w"]}'
)
assert jnp.allclose(acc_grads["b"], full_grads["b"], atol=1e-5, rtol=1e-5), 'b mismatch'
assert jnp.allclose(acc_loss, full_loss, atol=1e-5, rtol=1e-5), 'loss mismatch'
""",
        },
        {
            "name": "grad_fn runs once per micro-batch (no concatenation)",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((3, 1))}

def base(p, batch):
    x, y = batch
    return jnp.mean((x @ p["w"] - y) ** 2)

seen = []

def counting_grad_fn(p, batch):
    seen.append(int(batch[0].shape[0]))
    return jax.value_and_grad(base)(p, batch)

X = jax.random.normal(jax.random.key(6), (11, 3))
Y = jax.random.normal(jax.random.key(7), (11, 1))
chunks = [(X[:4], Y[:4]), (X[4:9], Y[4:9]), (X[9:], Y[9:])]   # 4 + 5 + 2

{fn}(counting_grad_fn, params, chunks)

assert len(seen) == 3, (
    f'grad_fn was called {len(seen)} times for 3 micro-batches. Calling it once on '
    'the concatenated data materialises every activation at the same time, which '
    'is exactly the memory cost accumulation exists to avoid'
)
assert seen == [4, 5, 2], (
    f'grad_fn saw batch sizes {seen}, expected [4, 5, 2] — pass each micro-batch '
    'through unchanged, in order'
)
""",
        },
        {
            "name": "One SGD step equals the full-batch SGD step",
            "code": """
import jax
import jax.numpy as jnp

params = {"enc": {"w": jnp.array([[0.4, -0.9, 0.1], [1.2, 0.3, -0.7]])},
          "dec": {"w": jnp.array([[0.5], [-0.2], [0.8]]), "b": jnp.array([0.05])}}

def loss_fn(p, batch):
    x, y = batch
    h = jnp.tanh(x @ p["enc"]["w"])
    return jnp.mean((h @ p["dec"]["w"] + p["dec"]["b"] - y) ** 2)

grad_fn = jax.value_and_grad(loss_fn)

X = jax.random.normal(jax.random.key(8), (7, 2))
Y = jax.random.normal(jax.random.key(9), (7, 1))
chunks = [(X[:5], Y[:5]), (X[5:], Y[5:])]   # 5 + 2, deliberately ragged

_, acc_grads = {fn}(grad_fn, params, chunks)
_, full_grads = grad_fn(params, (X, Y))

lr = 0.1
p_acc = jax.tree.map(lambda p, g: p - lr * g, params, acc_grads)
p_ref = jax.tree.map(lambda p, g: p - lr * g, params, full_grads)

assert jax.tree.structure(p_acc) == jax.tree.structure(params), 'Structure changed'
for a, b in zip(jax.tree.leaves(p_acc), jax.tree.leaves(p_ref)):
    assert jnp.allclose(a, b, atol=1e-6, rtol=1e-5), (
        f'Post-update params diverge: {a} vs {b}. One step on the accumulated '
        'gradient must land exactly where one step on the full batch lands'
    )

moved = jax.tree.map(lambda p, q: jnp.abs(p - q).sum(), params, p_acc)
assert float(sum(jax.tree.leaves(moved))) > 1e-6, 'Parameters did not move at all'
""",
        },
        {
            "name": "Empty micro_batches raises ValueError",
            "code": """
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((2, 2))}
grad_fn = jax.value_and_grad(lambda p, b: jnp.mean((b[0] @ p["w"] - b[1]) ** 2))

try:
    {fn}(grad_fn, params, [])
except ValueError:
    pass
except ZeroDivisionError as e:
    raise AssertionError(
        'Got ZeroDivisionError instead of ValueError — check for an empty list '
        'before dividing by the total example count'
    )
else:
    raise AssertionError('An empty micro_batches list must raise ValueError')
""",
        },
        {
            "name": "Jittable, and works on an nnx model's state",
            "code": """
import jax
import jax.numpy as jnp
from flax import nnx

model = nnx.Linear(5, 3, rngs=nnx.Rngs(0))
graphdef, state = nnx.split(model)

def grad_fn(st, batch):
    x, y = batch
    def loss(s):
        m = nnx.merge(graphdef, s)
        return jnp.mean((m(x) - y) ** 2)
    return jax.value_and_grad(loss)(st)

X = jax.random.normal(jax.random.key(10), (12, 5))
Y = jax.random.normal(jax.random.key(11), (12, 3))
chunks = [(X[:7], Y[:7]), (X[7:10], Y[7:10]), (X[10:], Y[10:])]   # 7 + 3 + 2

eager_loss, eager_grads = {fn}(grad_fn, state, chunks)
assert jax.tree.structure(eager_grads) == jax.tree.structure(state), (
    'The returned tree must mirror the nnx.State so an optimizer can apply it'
)

full_loss, full_grads = grad_fn(state, (X, Y))
for a, b in zip(jax.tree.leaves(eager_grads), jax.tree.leaves(full_grads)):
    assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5), (
        f'nnx gradients differ from the full batch: {a} vs {b}'
    )

# grad_fn is a Python callable, so it must be static.
jf = jax.jit({fn}, static_argnums=0)
jit_loss, jit_grads = jf(grad_fn, state, chunks)
assert jnp.allclose(jit_loss, eager_loss, atol=1e-5, rtol=1e-5), (
    f'jit loss {float(jit_loss)} vs eager {float(eager_loss)}'
)
for a, b in zip(jax.tree.leaves(jit_grads), jax.tree.leaves(eager_grads)):
    assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5), 'jit and eager gradients disagree'
""",
        },
    ],
}
