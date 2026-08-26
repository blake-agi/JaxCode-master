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

## 🚀 How to run

### Option 1 — VS Code + uv (recommended) ✅ tested

The whole loop runs locally with no Docker and no GitHub account. Every
command below was run verbatim on macOS with Python 3.11 and uv 0.12, including
a full wrong-answer → hint → correct-answer cycle through the judge.

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
make setup-local
code .
```

Open `notebooks/01_relu.ipynb`, choose the `.venv` interpreter as the kernel when
VS Code asks, and work down the notebook. `.vscode/settings.json` already pins
the interpreter and sets `jupyter.notebookFileRoot` to the repo root, so relative
paths behave the same from every notebook.

`make setup-local` gives you a practice folder:

| path | what it is |
|---|---|
| `notebooks/*.ipynb` | **your copies — edit these** |
| `notebooks/_solutions/` | reference implementations, out of sight |
| `notebooks/_pristine/` | untouched blanks; `cp` one back to reset a problem |

`notebooks/` is gitignored, so nothing you write gets committed. Regenerating the
notebooks (`make notebooks`) rewrites `templates/` and `solutions/` but never
touches your copies.

After pulling changes, `make refresh` copies in new and updated problems while
**skipping any notebook you have edited** — it diffs each one against
`notebooks/_pristine/` and leaves your work alone, listing what it protected.
(`make setup-local` is the same command; it is safe to re-run at any time.)

### The loop

Each problem is one notebook. Fill in the ✏️ cell, run the ✅ cell:

```python
from jax_judge import check, hint, solution, status

check("relu")        # grade your implementation
hint("relu")         # a nudge, not the answer
solution("relu")     # spoiler: the reference implementation
status()             # dashboard of all 52 problems
```

Progress is saved to `data/progress.json`, anchored to the repo — so it is the
same file no matter which directory you launch a notebook from.

### Option 2 — Docker ⚠️ not tested

```bash
make run     # JupyterLab at http://localhost:8888
make stop
make clean   # also wipes progress
```

Heavier than the local route: the image builds the JupyterLab extension, so it
pulls Node as well as the Python stack. Use it if you would rather not install
anything on the host.

> **Not tested.** Docker is not installed on the machine this was set up on, so
> `make run` has not been exercised. The Dockerfile and Makefile targets are
> carried over from the upstream PyTorch project with the package renamed and
> the JAX dependencies swapped in; nothing about them is verified here.

### Option 3 — Google Colab ⚠️ not tested

**Requires pushing this repo to GitHub first.** `jax-judge` is not published on
PyPI, so the notebooks install the judge straight from your fork, and the
*Open in Colab* badges point at it too. Both are templated from `JAXCODE_REPO`:

```bash
git remote add origin https://github.com/YOU/JAXCode.git
git push -u origin master
JAXCODE_REPO="YOU/JAXCode" make notebooks     # rewrites badges AND install cells
```

Until you do that, both point at a placeholder and will 404.

One caveat: a Colab VM is discarded when the session ends, so `data/progress.json`
does not survive unless you mount Drive. The local route keeps it for free.

> **Not tested.** The install cell is valid Python and the badge URLs are
> well-formed, but both point at a placeholder repo until you push, so the path
> has not been run end to end. Expect to debug the first notebook you open.

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
**69 problems** — 🟢 16 Easy · 🟡 28 Medium · 🔴 25 Hard


### JAX Fundamentals (11)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| b_01 | [SGD Step with value_and_grad](templates/b_01_grad_basics.ipynb) | 🟢 Easy | `grad_basics` |
| b_02 | [Pairwise Distances with vmap](templates/b_02_vmap_batching.ipynb) | 🟢 Easy | `vmap_batching` |
| b_03 | [jit with static_argnames](templates/b_03_jit_static.ipynb) | 🟢 Easy | `jit_static` |
| b_04 | [Stack a List of Pytrees](templates/b_04_pytree_ops.ipynb) | 🟡 Medium | `pytree_ops` |
| b_05 | [PRNG Keys and Splitting](templates/b_05_prng_keys.ipynb) | 🟢 Easy | `prng_keys` |
| b_06 | [Discounted Returns with lax.scan](templates/b_06_lax_scan.ipynb) | 🟡 Medium | `lax_scan` |
| b_07 | [Newton's Method with lax.while_loop](templates/b_07_lax_control_flow.ipynb) | 🟡 Medium | `lax_control_flow` |
| b_08 | [Stable log(1+exp(x)) with custom_vjp](templates/b_08_custom_vjp.ipynb) | 🔴 Hard | `custom_vjp` |
| b_09 | [Hessian with jacfwd(jacrev(f))](templates/b_09_higher_order_grad.ipynb) | 🟡 Medium | `higher_order_grad` |
| b_10 | [Gradient Checkpointing with jax.checkpoint](templates/b_10_remat_checkpoint.ipynb) | 🟡 Medium | `remat_checkpoint` |
| b_11 | [Data-Parallel Mean with shard_map](templates/b_11_sharding_basics.ipynb) | 🔴 Hard | `sharding_basics` |

### Core Ops & Layers (17)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 01 | [Implement ReLU](templates/01_relu.ipynb) | 🟢 Easy | `relu` |
| 02 | [Implement Softmax](templates/02_softmax.ipynb) | 🟢 Easy | `softmax` |
| 03 | [Simple Linear Layer](templates/03_linear.ipynb) | 🟡 Medium | `linear` |
| 04 | [Implement LayerNorm](templates/04_layernorm.ipynb) | 🟡 Medium | `layernorm` |
| 07 | [Implement BatchNorm](templates/07_batchnorm.ipynb) | 🟡 Medium | `batchnorm` |
| 08 | [Implement RMSNorm](templates/08_rmsnorm.ipynb) | 🟡 Medium | `rmsnorm` |
| 15 | [SwiGLU MLP](templates/15_mlp.ipynb) | 🟡 Medium | `mlp` |
| 17 | [Implement Dropout](templates/17_dropout.ipynb) | 🟢 Easy | `dropout` |
| 18 | [Embedding Layer](templates/18_embedding.ipynb) | 🟢 Easy | `embedding` |
| 19 | [GELU Activation](templates/19_gelu.ipynb) | 🟢 Easy | `gelu` |
| 20 | [Kaiming Initialization](templates/20_weight_init.ipynb) | 🟢 Easy | `weight_init` |
| 22 | [2D Convolution](templates/22_conv2d.ipynb) | 🟡 Medium | `conv2d` |
| b_12 | [LogSumExp](templates/b_12_logsumexp.ipynb) | 🟡 Medium | `logsumexp` |
| b_22 | [Numerically Stable Sigmoid](templates/b_22_stable_sigmoid.ipynb) | 🟡 Medium | `stable_sigmoid` |
| b_23 | [Linear Layer without Flax](templates/b_23_linear_pure.ipynb) | 🟢 Easy | `linear_pure` |
| b_24 | [Embedding without Flax](templates/b_24_embedding_pure.ipynb) | 🟢 Easy | `embedding_pure` |
| b_25 | [Dropout without Flax](templates/b_25_dropout_pure.ipynb) | 🟢 Easy | `dropout_pure` |

### Attention & Transformers (19)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 05 | [Softmax Attention](templates/05_attention.ipynb) | 🔴 Hard | `attention` |
| 06 | [Multi-Head Attention](templates/06_mha.ipynb) | 🔴 Hard | `mha` |
| 09 | [Causal Self-Attention](templates/09_causal_attention.ipynb) | 🔴 Hard | `causal_attention` |
| 10 | [Grouped Query Attention](templates/10_gqa.ipynb) | 🔴 Hard | `gqa` |
| 11 | [Sliding Window Attention](templates/11_sliding_window.ipynb) | 🔴 Hard | `sliding_window` |
| 12 | [Linear Self-Attention](templates/12_linear_attention.ipynb) | 🔴 Hard | `linear_attention` |
| 13 | [GPT-2 Transformer Block](templates/13_gpt2_block.ipynb) | 🔴 Hard | `gpt2_block` |
| 14 | [KV Cache Attention](templates/14_kv_cache.ipynb) | 🔴 Hard | `kv_cache` |
| 23 | [Cross-Attention](templates/23_cross_attention.ipynb) | 🟡 Medium | `cross_attention` |
| 24 | [Rotary Position Embedding (RoPE)](templates/24_rope.ipynb) | 🔴 Hard | `rope` |
| 25 | [Flash Attention (Tiled)](templates/25_flash_attention.ipynb) | 🔴 Hard | `flash_attention` |
| 26 | [LoRA (Low-Rank Adaptation)](templates/26_lora.ipynb) | 🟡 Medium | `lora` |
| 27 | [Vision Transformer Patch Embedding](templates/27_vit_patch.ipynb) | 🟡 Medium | `vit_patch` |
| 28 | [Mixture of Experts (MoE)](templates/28_moe.ipynb) | 🔴 Hard | `moe` |
| b_13 | [Mini GPT — assemble the whole model](templates/b_13_mini_gpt.ipynb) | 🔴 Hard | `mini_gpt` |
| b_20 | [Causal Attention with Padding](templates/b_20_causal_attention_padded.ipynb) | 🔴 Hard | `causal_attention_padded` |
| b_26 | [Multi-Head Attention without Flax](templates/b_26_mha_pure.ipynb) | 🟡 Medium | `mha_pure` |
| b_27 | [Cross-Attention without Flax](templates/b_27_cross_attention_pure.ipynb) | 🟡 Medium | `cross_attention_pure` |
| b_28 | [KV Cache Attention without Flax](templates/b_28_kv_cache_pure.ipynb) | 🔴 Hard | `kv_cache_pure` |

### Training (11)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 16 | [Cross-Entropy Loss](templates/16_cross_entropy.ipynb) | 🟢 Easy | `cross_entropy` |
| 21 | [Gradient Norm Clipping](templates/21_gradient_clipping.ipynb) | 🟢 Easy | `gradient_clipping` |
| 29 | [Implement Adam Optimizer](templates/29_adam.ipynb) | 🟡 Medium | `adam` |
| 30 | [Cosine LR Scheduler with Warmup](templates/30_cosine_lr.ipynb) | 🟡 Medium | `cosine_lr` |
| 31 | [Gradient Accumulation](templates/31_gradient_accumulation.ipynb) | 🟢 Easy | `gradient_accumulation` |
| 40 | [Linear Regression](templates/40_linear_regression.ipynb) | 🟡 Medium | `linear_regression` |
| b_14 | [Cross-Entropy Without logsumexp](templates/b_14_cross_entropy_fused.ipynb) | 🟡 Medium | `cross_entropy_fused` |
| b_15 | [Cross-Entropy: Smoothing & Padding Mask](templates/b_15_cross_entropy_full.ipynb) | 🟡 Medium | `cross_entropy_full` |
| b_16 | [Masked Diffusion LM Loss](templates/b_16_masked_diffusion.ipynb) | 🟡 Medium | `masked_diffusion` |
| b_18 | [Linear Regression with lax.scan](templates/b_18_linear_regression_scan.ipynb) | 🟡 Medium | `linear_regression_scan` |
| b_19 | [Mini-batch SGD with nested lax.scan](templates/b_19_minibatch_sgd_scan.ipynb) | 🔴 Hard | `minibatch_sgd_scan` |

### Inference & Decoding (7)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 32 | [Top-k / Top-p Sampling](templates/32_topk_sampling.ipynb) | 🟡 Medium | `topk_sampling` |
| 33 | [Beam Search Decoding](templates/33_beam_search.ipynb) | 🟡 Medium | `beam_search` |
| 34 | [Speculative Decoding](templates/34_speculative_decoding.ipynb) | 🔴 Hard | `speculative_decoding` |
| 35 | [Byte-Pair Encoding (BPE)](templates/35_bpe.ipynb) | 🔴 Hard | `bpe` |
| 36 | [INT8 Quantized Linear](templates/36_int8_quantization.ipynb) | 🔴 Hard | `int8_quantization` |
| b_17 | [Masked Diffusion Sampling Step](templates/b_17_diffusion_sampling.ipynb) | 🟡 Medium | `diffusion_sampling` |
| b_21 | [RoPE with a KV Cache](templates/b_21_rope_cached.ipynb) | 🔴 Hard | `rope_cached` |

### RLHF & Preference Losses (4)

| # | Problem | Difficulty | `task_id` |
|---|---|---|---|
| 37 | [DPO (Direct Preference Optimization) Loss](templates/37_dpo_loss.ipynb) | 🔴 Hard | `dpo_loss` |
| 38 | [GRPO (Group Relative Policy Optimization) Loss](templates/38_grpo_loss.ipynb) | 🔴 Hard | `grpo_loss` |
| 39 | [PPO (Proximal Policy Optimization) Clipped Loss](templates/39_ppo_loss.ipynb) | 🔴 Hard | `ppo_loss` |
| 41 | [OPD (On-Policy Distillation) Loss](templates/41_opd_loss.ipynb) | 🔴 Hard | `opd_loss` |
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
    "number": "b_12",                   # notebook number; b_* = JAX-only problem
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

`{fn}` is replaced with `function_name` before the test runs, and each test sees
**only** `function_name` plus anything listed in `extra_names` — see
`jax_judge/_contract.py`. The registry auto-discovers the module, so there is
nothing to import or register.

The `number` is the curriculum order and the notebook filename. The 41 problems
ported from PyTorch keep the original's numbers so the two repos line up; new
JAX-only problems take the next free `b_*`.

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
make probe       # attack every test suite with wrong implementations
make align       # assert the 41 ported problems still match the PyTorch original
make notebooks   # regenerate all notebooks from task definitions
make smoke       # execute notebooks in a real Jupyter kernel
make check       # verify + probe + align + notebooks --check (what CI runs)
```

### Four layers of validation

A task whose published solution fails its own tests is worse than no task, but
passing your own tests only proves *self-consistency*. Each layer catches what
the one above it cannot:

| Layer | Proves | Blind spot |
|---|---|---|
| `make verify` | all 52 solutions pass their own tests | a formula wrong in **both** solution and tests |
| `make probe` | the tests **reject** wrong answers | a misconception shared by every check |
| `make align` | the ported problems still match the PyTorch original | nothing about correctness |
| `crosscheck_vs_torch.py` | agreement with an independent implementation | — |

`make probe` attacks each suite with deliberately-broken implementations —
attention without the `1/sqrt(d_k)` scale, a mask multiplied after the softmax
instead of added before it, Adam without bias correction, DPO ignoring the
reference model. Two of the attacks come from **open bugs in the upstream
PyTorch project** ([#17](https://github.com/duoan/TorchCode/issues/17),
[#21](https://github.com/duoan/TorchCode/issues/21)), where those wrong
implementations pass. Here they are rejected, and the attacks are permanent
regression tests.

The outermost layer lives in [`jax_pytorch_comparison/`](jax_pytorch_comparison/),
which also holds a side-by-side PyTorch→JAX notebook. It is deliberately outside
the package so `jax_judge` never depends on `torch`.

### Stack

JAX + **Flax NNX** + Optax — the current Google DeepMind stack. Flax Linen is in
maintenance mode and [dm-haiku has been maintenance-only since
2023](https://github.com/google-deepmind/dm-haiku), so neither appears here.
Optimizers and losses are implemented by hand rather than imported from Optax:
in an interview you are asked to *write* Adam, not call it.

---

## 📄 License

MIT.

Adapted from [TorchCode](https://github.com/duoan/TorchCode) by duoan — same
judge architecture and notebook workflow, rebuilt for JAX and Flax NNX with an
added JAX-fundamentals track.
