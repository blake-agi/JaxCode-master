# 🔥 → ⚡ PyTorch to JAX

A side-by-side comparison for someone fluent in PyTorch who needs to be
productive in JAX + Flax NNX — and who is likely to be asked about the
differences in an interview.

This folder is **separate from the JAXCode package on purpose**: `jax_judge`
must never depend on `torch`. Everything here is the only place the two
frameworks meet.

---

## Contents

| File | What it is |
|---|---|
| `pytorch_to_jax.ipynb` | The comparison notebook (generated — see below) |
| `build_notebook.py` | Source of truth for the notebook |
| `crosscheck_vs_torch.py` | Validates JAXCode's reference solutions against PyTorch |

---

## The notebook

13 sections, each running **both frameworks on the same inputs and asserting
they agree numerically**. Nothing is a claim you take on faith — if a cell
prints ✅, the two implementations really did produce the same numbers.

1. Arrays are immutable — `x[0] = 5` vs `x.at[0].set(5)`
2. Autograd — `.backward()` vs `jax.grad`
3. Modules — `nn.Module` vs `nnx.Module` (**and the weight-transpose trap**)
4. Optimizers — stateful object vs pure function
5. A full training loop, side by side
6. Randomness — global seed vs explicit keys
7. Batching — manual broadcasting vs `vmap`
8. Compilation — `jax.jit` and the tracing model
9. Loops — Python `for` vs `lax.scan`
10. Devices
11. Train/eval mode
12. **⚠️ BatchNorm: where the frameworks genuinely disagree**
13. Translation cheat sheet

Open it in Colab, or run locally in an environment with both frameworks:

```bash
pip install jax flax torch --index-url https://download.pytorch.org/whl/cpu
```

To edit it, change `build_notebook.py` and regenerate — do not hand-edit the
`.ipynb`:

```bash
python build_notebook.py
```

---

## The cross-check

JAXCode has three layers of validation, and this is the outermost one:

| Check | What it proves | What it cannot catch |
|---|---|---|
| `scripts/verify_tasks.py` | each solution passes its own tests | a formula wrong in *both* |
| `scripts/probe_tests.py` | the tests reject wrong answers | a shared misconception |
| **`crosscheck_vs_torch.py`** | agreement with an independent implementation | — |

PyTorch was written by other people and is battle-tested, so numeric agreement
with it is genuine outside evidence:

```bash
python crosscheck_vs_torch.py
```

21 comparisons covering ReLU, both GELU variants, softmax, LayerNorm, BatchNorm,
Linear, Embedding, Conv2D (three stride/padding combinations), scaled dot-product
and causal attention, cross-entropy (plus label smoothing and `ignore_index`),
and 25 steps of Adam against `torch.optim.Adam`.

### The one real divergence it found

`running_var` in BatchNorm:

- `flax.nnx.BatchNorm` updates it with the **biased** variance (`ddof=0`)
- `torch.nn.BatchNorm1d` updates it with the **unbiased** variance (`ddof=1`)

They differ by exactly the Bessel factor `n/(n-1)`. Training outputs are
identical; only **inference** diverges. JAXCode follows the Flax convention, and
the cross-check asserts the gap to torch is precisely `n/(n-1)` — so the
divergence is pinned down rather than papered over.

Conventions that must be lined up for the other comparisons to be meaningful:

- **Linear** — Flax kernel is `(in, out)`; torch is `(out, in)` → transpose
- **Conv2D** — JAXCode uses NHWC/HWIO; torch uses NCHW/OIHW → transpose
- **LayerNorm** — both use the biased variance, so no adjustment
- **`ignore_index`** — JAXCode defaults to `-1`, torch to `-100`
