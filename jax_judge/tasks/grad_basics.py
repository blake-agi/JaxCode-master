"""One SGD step over a pytree of parameters — the core JAX training idiom."""

TASK = {
    "title": "SGD Step with value_and_grad",
    "category": "JAX Fundamentals",
    "number": "b_01",
    "difficulty": "Easy",
    "function_name": "sgd_step",
    "hint": (
        "Two things to look up. First: one transform in the jax.grad family hands "
        "back the value alongside the gradient, so you never pay for a second "
        "forward pass. Second: the gradient it returns has exactly the same tree "
        "structure as params, which means the update is a single traversal that "
        "visits params and grads in lockstep — reach for the jax.tree helper that "
        "accepts more than one tree, not a Python loop over dict keys."
    ),
    "description": r"""
Implement a single **stochastic gradient descent step**.

This is the innermost loop of essentially every JAX training script — if you
only learn one JAX idiom, learn this one.

Given a loss function, a pytree of parameters, a batch, and a learning rate,
return the updated parameters and the loss *at the old parameters*.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t, \text{batch})$$

### Rules
- Use `jax.value_and_grad` — one forward and one backward pass give you both
  numbers (`loss_fn(...)` followed by `jax.grad(loss_fn)(...)` runs the forward
  computation twice)
- `params` is an arbitrary **pytree** (nested dicts/lists/tuples of arrays),
  not a flat array — do not assume it is a single array
- Differentiate with respect to `params` only, not the batch
- The returned loss must be the loss **before** the update

### The traps
- **Reporting the loss after the update.** Re-evaluating `loss_fn(new_params, …)`
  costs an extra forward pass *and* reports a number that no training curve
  in the literature plots. `value_and_grad` gives you the value at the point
  where the gradient was taken, which is what you want.
- **Assuming a flat array.** `params - lr * grads` is fine for the toy case and
  dies with `TypeError: unsupported operand type(s) for -: 'dict' and 'float'`
  the moment `params` is a real parameter tree.
- **Reaching for in-place updates.** JAX arrays are immutable; there is no
  `p -= lr * g`. The step *returns* new parameters, which is why JAX training
  loops thread state through explicitly instead of hiding it inside a mutable
  optimizer object.

### Signature
```python
def sgd_step(loss_fn, params, batch, lr):
    # loss_fn(params, batch) -> scalar
    # returns (new_params, loss)
    ...
```

### Why it matters
Production code uses Optax, and `optax.apply_updates` is exactly the `jax.tree`
traversal you are writing here (it adds already-negated updates leaf by leaf).
Interviewers ask for the hand-rolled version to check you understand the
contract underneath it: gradients mirror the parameter tree leaf for leaf, and
the step is a pure function of `(params, batch)` — nothing is mutated — so the
whole thing can be wrapped in `jax.jit` once and reused unchanged.
""",
    "stub": '''def sgd_step(loss_fn, params, batch, lr):
    """One SGD step.

    Args:
        loss_fn: callable (params, batch) -> scalar loss
        params:  pytree of parameters
        batch:   whatever loss_fn expects as its second argument
        lr:      float learning rate

    Returns:
        (new_params, loss) where loss is measured at the OLD params
    """
    pass  # Replace this
''',
    "solution": '''import jax


def sgd_step(loss_fn, params, batch, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss
''',
    "demo": '''import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
batch = (jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([1.0, 1.0]))


def loss_fn(p, b):
    x, y = b
    pred = x @ p["w"] + p["b"]
    return jnp.mean((pred - y) ** 2)


new_params, loss = sgd_step(loss_fn, params, batch, lr=0.1)
print("loss before step:", loss)
print("old params:", params)
print("new params:", new_params)
''',
    "tests": [
        {
            "name": "Analytic gradient on a quadratic",
            "code": """
import jax.numpy as jnp

# loss = sum(p**2)  ->  grad = 2p  ->  p_new = p - lr*2p
params = jnp.array([1.0, -2.0, 3.0])
loss_fn = lambda p, b: jnp.sum(p ** 2)
new_params, loss = {fn}(loss_fn, params, None, 0.1)

assert jnp.allclose(loss, 14.0), f'loss should be 14.0 (at OLD params), got {loss}'
expected = params - 0.1 * 2 * params
assert jnp.allclose(new_params, expected), f'{new_params} vs {expected}'
""",
        },
        {
            "name": "Loss is measured before the update",
            "code": """
import jax.numpy as jnp

params = jnp.array([5.0])
loss_fn = lambda p, b: jnp.sum(p ** 2)
new_params, loss = {fn}(loss_fn, params, None, 0.5)

assert jnp.allclose(loss, 25.0), (
    f'Expected the loss at the OLD params (25.0), got {loss}. '
    'Compute the loss and grads together, before applying the update.'
)
assert not jnp.allclose(new_params, params), 'Params did not change'
""",
        },
        {
            "name": "Nested pytree params",
            "code": """
import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.ones((3, 2)), "b": jnp.zeros(2)},
    "layer2": [jnp.full((2,), 0.5), jnp.array(1.0)],
}
x = jnp.ones((4, 3))


def loss_fn(p, b):
    h = b @ p["layer1"]["w"] + p["layer1"]["b"]
    return jnp.sum(h * p["layer2"][0]) + p["layer2"][1]


new_params, loss = {fn}(loss_fn, params, x, 0.01)

assert jax.tree.structure(new_params) == jax.tree.structure(params), (
    'Returned params must keep the same pytree structure as the input'
)
for a, b in zip(jax.tree.leaves(new_params), jax.tree.leaves(params)):
    assert a.shape == b.shape, f'Leaf shape changed: {a.shape} vs {b.shape}'
assert jnp.ndim(loss) == 0, f'Loss must be a scalar, got shape {jnp.shape(loss)}'
""",
        },
        {
            "name": "Loss actually decreases over many steps",
            "code": """
import jax.numpy as jnp

key_x = jnp.linspace(-1.0, 1.0, 32).reshape(32, 1)
y = 3.0 * key_x[:, 0] - 1.0
params = {"w": jnp.zeros((1,)), "b": jnp.zeros(())}


def loss_fn(p, b):
    x_, y_ = b
    return jnp.mean((x_ @ p["w"] + p["b"] - y_) ** 2)


losses = []
for _ in range(200):
    params, l = {fn}(loss_fn, params, (key_x, y), 0.1)
    losses.append(float(l))

assert losses[-1] < losses[0], 'Loss did not decrease'
assert losses[-1] < 1e-3, f'Should converge on a linear problem, final loss {losses[-1]}'
assert jnp.allclose(params["w"], 3.0, atol=1e-1), f'w should approach 3.0, got {params["w"]}'
""",
        },
        {
            "name": "Works under jit",
            "code": """
import functools
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0])}
loss_fn = lambda p, b: jnp.sum(p["w"] ** 2 * b)

step = jax.jit(functools.partial({fn}, loss_fn), static_argnums=())
new_params, loss = step(params, jnp.array(2.0), 0.1)

assert jnp.allclose(loss, 10.0), f'{loss}'
assert jnp.allclose(new_params["w"], jnp.array([0.6, 1.2])), f'{new_params["w"]}'
""",
        },
    ],
}
