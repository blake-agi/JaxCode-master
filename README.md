<div align="center">

# ⚡ JAXCode

**Crack the JAX interview.**

Implement operators, layers, and training machinery from scratch — in JAX and Flax NNX.

*An interactive coding platform, but for tensors. Self-hosted. Jupyter-based. Instant feedback.*

[![JAX](https://img.shields.io/badge/JAX-0.10-blue?style=for-the-badge)](https://docs.jax.dev)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-4c8bf5?style=for-the-badge)](https://flax.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

![GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen?style=flat-square)

</div>

---

## 🎯 Why JAXCode

JAX interviews are not PyTorch interviews with different syntax. They test a
different set of reflexes:

- Can you write the **single-example** function and let `vmap` add the batch axis?
- Do you know why `axis` must be **static** under `jit` but the array must not be?
- Can you express a recurrence as `lax.scan` instead of a Python loop that
  unrolls into a 10,000-node graph?
- Do you understand that a `while_loop` **cannot** be reverse-mode differentiated?
- When autodiff produces `NaN` on a mathematically fine function, can you reach
  for `custom_vjp` and hand it the analytic gradient?

JAXCode gives you a graded environment for exactly those reflexes, plus the
standard ML-implementation canon (attention, normalization, optimizers, RLHF
losses) written the way JAX actually wants them.

| | Feature | |
|---|---|---|
| 🧩 | **Curated problems** | JAX fundamentals first, then the ML canon |
| ⚖️ | **Real judge** | Every problem ships a test suite: correctness, edge cases, gradients, `jit`/`vmap` |
| 🎨 | **Instant feedback** | Colored pass/fail per test, like competitive programming |
| 💡 | **Hints, then solutions** | `hint()` nudges; `solution()` shows the reference |
| 📊 | **Progress tracking** | Dashboard of solved / attempted / todo by category |
| 🧪 | **Self-verifying** | Every reference solution is CI-checked against its own tests |
| 🔥 | **Flax NNX** | Layers are real `nnx.Module`s — mutable state, `nnx.Param`, `nnx.BatchStat` |

No cloud. No signup. No GPU needed.

---

## 🚀 Quick start

### Option 1 — Google Colab (zero install)

Every notebook carries an *Open in Colab* badge. In Colab, install the judge:

```bash
!pip install jax-judge flax
```

Then in a cell:

```python
from jax_judge import check, hint, solution, status

status()             # dashboard of every problem
check("relu")        # grade your implementation
hint("relu")         # a nudge, not the answer
solution("relu")     # spoiler: the reference implementation
```

> The Colab badges are built from `JAXCODE_REPO`. After you push this to your own
> GitHub, regenerate them so the links resolve:
> ```bash
> JAXCODE_REPO="you/JAXCode" make notebooks
> ```

### Option 2 — Docker (full JupyterLab)

```bash
make run
```

Opens JupyterLab at **http://localhost:8888** with every notebook preloaded and
progress persisted to `./data/progress.json`.

```bash
make stop     # stop it
make clean    # stop, drop volumes, wipe progress
```

### Option 3 — Local virtualenv

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" jupyterlab
make setup-local
jupyter lab notebooks/
```

---

## 🧠 How it works

Each problem is a notebook with a blank implementation cell and a submit cell:

```python
# ✏️ YOUR IMPLEMENTATION HERE
def my_softmax(x, axis=-1):
    pass

# ✅ SUBMIT
from jax_judge import check
check("softmax")
```

The judge pulls your function out of the notebook namespace and runs the real
test suite against it:

```
🧪 Testing: Implement Softmax (Easy)
────────────────────────────────────────────────────────
  ✅ [1/6] Basic 1-D (12.3ms)
  ✅ [2/6] 2-D along the last axis (8.1ms)
  ❌ [3/6] Numerical stability on large inputs
     NaN in output: [nan nan nan] — subtract the max along the axis before exp()
  ...
────────────────────────────────────────────────────────
  📊 5/6 tests passed.
```

Tests are written to catch the *specific* bug and say so, rather than just
reporting a diff.

---

## 📚 Problems

<!-- PROBLEMS:START -->
**27 problems** — 🟢 12 Easy · 🟡 13 Medium · 🔴 2 Hard


### JAX Fundamentals (11)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 01 | [SGD Step with value_and_grad](templates/01_grad_basics.ipynb) | 🟢 Easy | `grad_basics` |
| 02 | [Pairwise Distances with vmap](templates/02_vmap_batching.ipynb) | 🟢 Easy | `vmap_batching` |
| 03 | [jit with static_argnames](templates/03_jit_static.ipynb) | 🟢 Easy | `jit_static` |
| 04 | [Stack a List of Pytrees](templates/04_pytree_ops.ipynb) | 🟡 Medium | `pytree_ops` |
| 05 | [PRNG Keys and Splitting](templates/05_prng_keys.ipynb) | 🟢 Easy | `prng_keys` |
| 06 | [Discounted Returns with lax.scan](templates/06_lax_scan.ipynb) | 🟡 Medium | `lax_scan` |
| 07 | [Newton's Method with lax.while_loop](templates/07_lax_control_flow.ipynb) | 🟡 Medium | `lax_control_flow` |
| 08 | [Stable log(1+exp(x)) with custom_vjp](templates/08_custom_vjp.ipynb) | 🔴 Hard | `custom_vjp` |
| 09 | [Hessian with jacfwd(jacrev(f))](templates/09_higher_order_grad.ipynb) | 🟡 Medium | `higher_order_grad` |
| 10 | [Gradient Checkpointing with jax.checkpoint](templates/10_remat_checkpoint.ipynb) | 🟡 Medium | `remat_checkpoint` |
| 11 | [Data-Parallel Mean with shard_map](templates/11_sharding_basics.ipynb) | 🔴 Hard | `sharding_basics` |

### Core Ops & Layers (11)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 12 | [Implement ReLU](templates/12_relu.ipynb) | 🟢 Easy | `relu` |
| 13 | [Implement GELU (exact and tanh)](templates/13_gelu.ipynb) | 🟢 Easy | `gelu` |
| 14 | [Implement Softmax](templates/14_softmax.ipynb) | 🟢 Easy | `softmax` |
| 15 | [Linear Layer (nnx.Module)](templates/15_linear.ipynb) | 🟢 Easy | `linear` |
| 16 | [LayerNorm (nnx.Module)](templates/16_layernorm.ipynb) | 🟢 Easy | `layernorm` |
| 17 | [RMSNorm (nnx.Module)](templates/17_rmsnorm.ipynb) | 🟢 Easy | `rmsnorm` |
| 18 | [BatchNorm with Running Stats (nnx.Module)](templates/18_batchnorm.ipynb) | 🟡 Medium | `batchnorm` |
| 19 | [Inverted Dropout (nnx.Module)](templates/19_dropout.ipynb) | 🟡 Medium | `dropout` |
| 20 | [Embedding Lookup (nnx.Module)](templates/20_embedding.ipynb) | 🟢 Easy | `embedding` |
| 21 | [Two-Layer MLP (nnx.Module)](templates/21_mlp.ipynb) | 🟢 Easy | `mlp` |
| 22 | [Xavier and He Initialisation](templates/22_weight_init.ipynb) | 🟡 Medium | `weight_init` |

### Attention & Transformers (3)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 23 | [Scaled Dot-Product Attention](templates/23_attention.ipynb) | 🟡 Medium | `attention` |
| 24 | [Multi-Head Attention (nnx.Module)](templates/24_mha.ipynb) | 🟡 Medium | `mha` |
| 25 | [Rotary Position Embeddings (RoPE)](templates/25_rope.ipynb) | 🟡 Medium | `rope` |

### Training (2)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 26 | [Global-Norm Gradient Clipping](templates/26_gradient_clipping.ipynb) | 🟡 Medium | `gradient_clipping` |
| 27 | [Gradient Accumulation](templates/27_gradient_accumulation.ipynb) | 🟡 Medium | `gradient_accumulation` |
<!-- PROBLEMS:END -->

---

## 🗺️ Suggested route

**Do the JAX Fundamentals first**, even if you know the ML content cold. Every
later problem assumes fluency with `vmap`, `scan`, pytrees, and explicit PRNG
keys — and those are what a JAX-specific interview actually probes. The ML
problems are where you demonstrate that you can express known algorithms
*functionally*.

After that, follow the categories in order, or jump to whatever your interview
targets.

---

## 🏗️ Repo layout

```
jax_judge/
  engine.py           check() / hint() / solution() — framework-agnostic runner
  progress.py         solved/attempted tracking + status() dashboard
  tasks/
    _registry.py      auto-discovers task modules, defines curriculum order
    <task_id>.py      ONE TASK dict: description, stub, solution, tests
templates/            generated — blank practice notebooks
solutions/            generated — reference solution notebooks
scripts/
  generate_notebooks.py   renders templates/ + solutions/ from tasks
  verify_tasks.py         runs each reference solution against its own tests
  smoke_notebooks.py      executes notebooks in a real kernel
```

**The task files are the single source of truth.** Notebooks are generated
artifacts — never edit them by hand, your changes will be overwritten.

---

## ➕ Adding a problem

Create `jax_judge/tasks/my_task.py`:

```python
TASK = {
    "title": "My Problem",
    "category": "Core Ops & Layers",   # must match _registry.CATEGORIES
    "order": 13,                        # position within the category
    "difficulty": "Medium",             # Easy | Medium | Hard
    "function_name": "my_fn",           # symbol the judge looks for
    "hint": "Name the API and the shape trick, without giving the answer.",
    "description": "Markdown problem statement — this becomes the notebook.",
    "stub": "def my_fn(x):\n    pass",
    "solution": "def my_fn(x):\n    return x * 2",
    "demo": "print(my_fn(jnp.arange(3)))",   # optional scratch cell
    "tests": [
        {"name": "Basic", "code": "import jax.numpy as jnp\nassert {fn}(2) == 4"},
    ],
}
```

`{fn}` is replaced with `function_name` before the test runs. The registry
auto-discovers the module — no imports to register.

Then:

```bash
make verify      # your reference solution must pass your own tests
make notebooks   # regenerate the notebooks
```

Gotcha: each test snippet is standalone and must do its own imports.
`import jax.numpy as jnp` does **not** bind the name `jax` — if you use
`jax.grad`, import `jax` too.

---

## 🧰 Development

```bash
make verify      # every reference solution vs its own tests
make notebooks   # regenerate all notebooks from task definitions
make smoke       # execute solution notebooks in a real Jupyter kernel
make check       # verify + assert notebooks are not stale (CI)
```

`make verify` is the important one. A task whose own published solution fails
its own tests is worse than no task at all, so it runs on every change.

---

## 📄 License

MIT.

Adapted from [TorchCode](https://github.com/duoan/TorchCode) by duoan — same
judge architecture and notebook workflow, rebuilt for JAX and Flax NNX with an
added JAX-fundamentals track.
