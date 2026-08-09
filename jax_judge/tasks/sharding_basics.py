"""Data-parallel mean with shard_map and lax.pmean — SPMD in miniature."""

TASK = {
    "title": "Data-Parallel Mean with shard_map",
    "category": "JAX Fundamentals",
    "order": 11,
    "difficulty": "Hard",
    "function_name": "data_parallel_mean",
    "hint": (
        "Build the mesh with jax.make_mesh((n_devices,), ('data',), "
        "axis_types=(jax.sharding.AxisType.Auto,)) — the Auto axis type lets "
        "shard_map take an ordinary unsharded array and place it for you. Inside "
        "shard_map, each device sees only its own (B/n, D) shard, so compute the "
        "LOCAL mean and then combine across devices with "
        "jax.lax.pmean(local, axis_name='data'). Use in_specs=P('data', None) to "
        "split rows across devices, and out_specs=P() so the scalar comes back "
        "replicated. Because every shard has the same number of rows, the mean of "
        "the local means is the true global mean."
    ),
    "description": r"""
Compute the mean of a `(B, D)` array **across devices**, using JAX's SPMD tools.

Split the batch dimension across all available devices, have each device reduce
its own shard, then combine with a collective. Return the scalar global mean.

### Rules
- Build a 1-D mesh over `jax.devices()` with axis name `"data"`
- Use `jax.shard_map` with `in_specs=P("data", None)` and `out_specs=P()`
- Combine the per-device results with `jax.lax.pmean`
- `B` is divisible by the device count
- The result must equal `x.mean()` exactly (to float tolerance)

### Explicit vs Auto axis types
Recent JAX versions give mesh axes a *type*. `jax.make_mesh(...)` defaults to
**Explicit**, which means `shard_map` insists the input already carries a
matching sharding and errors otherwise:

```
ValueError: in_specs passed to shard_map: P('data', None) does not match
the specs of the input: P(None, None)
```

Two ways out — either place the array yourself first with
`jax.device_put(x, NamedSharding(mesh, P("data", None)))`, or declare the axis
**Auto** and let `shard_map` handle placement:

```python
mesh = jax.make_mesh((n,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
```

This task uses the `Auto` route.

### Signature
```python
def data_parallel_mean(x):  # (B, D) -> scalar
    ...
```

### Running this without a TPU pod
On a single CPU you can fake 8 devices — this must be set **before** the first
`jax` import:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
print(jax.devices())   # 8 CpuDevices
```

### Why it matters
Inside `shard_map` you write code from the perspective of a **single device**:
shapes are per-shard, and cross-device communication is explicit via collectives
(`pmean`, `psum`, `all_gather`, `ppermute`). That explicitness is the whole
point — it is why data parallelism, tensor parallelism, and FSDP are all just
different `PartitionSpec` choices over the same code.

The classic follow-up: *why is the mean-of-means correct here, and when does it
break?* (It breaks the moment shards have different row counts — then you need
`psum` of sums and `psum` of counts.)
""",
    # Runs before the jax import in the generated notebook — XLA reads this flag
    # once, at initialisation.
    "notebook_setup": '''import os

# Fake 8 devices on a single CPU so the mesh has something to shard across.
# MUST run before jax is imported for the first time.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"''',
    "stub": '''import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def data_parallel_mean(x):
    """Global mean of x, computed with the batch sharded across devices.

    Args:
        x: (B, D) array, B divisible by len(jax.devices())

    Returns:
        Scalar mean, replicated across devices.
    """
    pass  # Replace this
''',
    "solution": '''import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def data_parallel_mean(x):
    n = len(jax.devices())
    # Auto axis types let shard_map accept a plain, unsharded array. With the
    # default (Explicit) you would have to jax.device_put the input onto the
    # mesh yourself before calling in.
    mesh = jax.make_mesh((n,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))

    def local_mean(shard):
        # `shard` is this device's slice only: (B // n, D).
        # Every shard has the same row count, so mean-of-means is exact.
        return jax.lax.pmean(jnp.mean(shard), axis_name="data")

    f = jax.shard_map(
        local_mean,
        mesh=mesh,
        in_specs=P("data", None),
        out_specs=P(),
    )
    return f(x)
''',
    "demo": '''import jax
import jax.numpy as jnp

print("devices:", len(jax.devices()))

x = jnp.arange(64.0).reshape(16, 4)
got = data_parallel_mean(x)
print("sharded mean:", got, " numpy mean:", x.mean())
''',
    "tests": [
        {
            "name": "Matches the global mean",
            "code": """
import jax
import jax.numpy as jnp

n = len(jax.devices())
x = jnp.arange(float(n * 4 * 3)).reshape(n * 4, 3)

out = {fn}(x)
assert jnp.ndim(out) == 0, f'Expected a scalar, got shape {jnp.shape(out)}'
assert jnp.allclose(out, x.mean(), atol=1e-4), f'{out} vs {x.mean()}'
""",
        },
        {
            "name": "Random data, several batch sizes",
            "code": """
import jax
import jax.numpy as jnp

n = len(jax.devices())
for mult in [1, 2, 5]:
    x = jax.random.normal(jax.random.key(mult), (n * mult, 8))
    out = {fn}(x)
    assert jnp.allclose(out, x.mean(), atol=1e-4), (
        f'B={n * mult}: got {out}, expected {x.mean()}'
    )
""",
        },
        {
            "name": "Result is replicated, not per-device",
            "code": """
import jax
import jax.numpy as jnp

n = len(jax.devices())
x = jax.random.normal(jax.random.key(0), (n * 4, 6))
out = {fn}(x)

assert jnp.ndim(out) == 0, (
    f'Output shape {jnp.shape(out)} — out_specs=P() should give one replicated '
    'scalar, not one value per device. Did you forget the pmean?'
)
assert jnp.isfinite(out), 'Non-finite result'

# Without the collective you would get only the FIRST shard's mean.
first_shard_mean = x[: x.shape[0] // n].mean()
if n > 1 and not jnp.allclose(first_shard_mean, x.mean(), atol=1e-4):
    assert not jnp.allclose(out, first_shard_mean, atol=1e-6), (
        'Result equals the first shard mean — the per-device values were never '
        'combined. Use jax.lax.pmean(..., axis_name="data").'
    )
""",
        },
        {
            "name": "Uses a mesh and a collective",
            "code": """
import jax
import jax.numpy as jnp

n = len(jax.devices())
x = jnp.ones((n * 2, 4))

jaxpr = str(jax.make_jaxpr({fn})(x))
assert ("shard_map" in jaxpr) or ("psum" in jaxpr) or ("pmean" in jaxpr), (
    'The jaxpr shows no shard_map or cross-device collective — this looks like '
    'a plain jnp.mean. Use jax.shard_map with jax.lax.pmean.'
)
assert jnp.allclose({fn}(x), 1.0, atol=1e-6), 'Mean of all-ones must be 1.0'
""",
        },
        {
            "name": "Composes with jit and grad",
            "code": """
import jax
import jax.numpy as jnp

n = len(jax.devices())
x = jax.random.normal(jax.random.key(7), (n * 3, 5))

jitted = jax.jit({fn})
assert jnp.allclose(jitted(x), x.mean(), atol=1e-4), 'jit changes the result'

g = jax.grad({fn})(x)
assert g.shape == x.shape, f'Gradient shape {g.shape} vs {x.shape}'
expected = jnp.full_like(x, 1.0 / x.size)
assert jnp.allclose(g, expected, atol=1e-6), 'd(mean)/dx should be 1/N everywhere'
""",
        },
    ],
}
