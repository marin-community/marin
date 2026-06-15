# Re-entrant (self-looping) models: research findings & first-model recommendation

**Status:** research complete, implementation not started.
**Author:** weaver session `weaver/re-entrant-model-testing` (issue #139).
**Date:** 2026-06-15.

## TL;DR

A "re-entrant" model that re-applies its own layers is not a fringe idea — it is a
well-studied family usually called **recurrent-depth**, **looped**, or **recursive**
transformers. The thesis you are after — *latent recurrence can substitute for explicit
"thinking" tokens* — is supported both theoretically (a stack looped `T` times can
simulate `T` steps of chain-of-thought) and empirically (a `k`-layer block looped `L`
times nearly matches a `kL`-layer model on reasoning, at a fraction of the parameters).

Your three design questions all have established answers:

1. **How to train it?** Weight-tie the looped block; train with a *randomized* number of
   loop iterations per step (so the model learns to be correct at many depths); use
   *truncated* backprop through only the last few iterations to bound memory. This is
   the Huginn recipe. Alternatively, solve for the fixed point directly and use implicit
   gradients (the Deep Equilibrium Model recipe).
2. **How does it signal "more time"?** Three options of increasing cost: (a) a *training-free*
   inference heuristic — stop looping when consecutive latent states stop changing (KL
   convergence); (b) a learned *halting head* (ACT / PonderNet) with a "ponder cost"
   regularizer; (c) a *per-token recursion router* (Mixture-of-Recursions) that routes
   each token to its own loop depth.
3. **The "input/output consistency" soft constraint** is exactly the **fixed-point /
   equilibrium** framing (DEQ): the looped block `f` is trained so `z* ≈ f(z*; x)`.
   Note: the strongest modern result (Huginn) deliberately does *not* enforce strict
   convergence — convergence *emerges* from random-depth training. Treat an explicit
   consistency penalty as an ablation, not the default.

**Recommendation:** build a `experiments/grug/reentrant/` variant (copy-first, per the
`change-grug` skill) using a **prelude → shared recurrent block → coda** structure on the
small 130M dense Grug config. Phase 1 is fixed-depth (validate plumbing + match a dense
baseline at equal effective depth). Phase 2 adds randomized-depth training with truncated
BPTT for test-time depth scaling. Phase 3 adds a KL-convergence early-exit (training-free).
Phase 4, only if 1–3 win, adds a learned router/halting head. Start dense, not MoE — keep
the number of moving variables minimal. Details in [§5](#5-recommendation-a-first-re-entrant-grug)
and [§6](#6-phased-plan--issues).

---

## 1. The concept and why it is interesting

Standard test-time scaling spends extra compute by emitting more *tokens* (chain-of-thought).
A re-entrant model instead spends extra compute by re-applying *layers* to the same
positions — extra computation in latent space, with no extra tokens emitted. The appeal:

- **Decouple compute from parameter count.** A small weight-tied block looped many times
  has the FLOPs of a deep model with the parameter footprint of a shallow one.
- **Reason in latent space.** Some computations ("types of reasoning not easily represented
  in words", per Huginn) may be awkward to serialize into discrete tokens.
- **Adaptive per-input/per-token compute.** Hard inputs can loop more; easy ones exit early.

The flip side of your "skip layers" intuition (early-exit / layer-drop) is precisely this:
instead of *removing* depth for easy inputs, you *add* depth (by re-entry) for hard ones.

---

## 2. Literature survey

All claims below were adversarially fact-checked against primary sources (arXiv abstracts/PDFs)
in a 3-vote verification pass; citations are in [§7](#7-references). Empirical numbers are
self-reported by the originating papers in controlled/synthetic settings unless noted.

### 2.1 Foundations (adaptive compute + weight sharing + equilibrium)

- **Adaptive Computation Time (ACT)** — Graves 2016 (arXiv:1603.08983). Lets an RNN *learn
  how many iterations* of the same recurrent step to run per input, via a halting unit that
  accumulates a halting probability until it crosses a threshold; a "ponder cost" penalizes
  extra steps. Established the core idea: **the model decides its own compute budget.**

- **Universal Transformer (UT)** — Dehghani et al. 2018 (arXiv:1807.03819, ICLR 2019).
  Applies the *same* transformer transition function recurrently across depth (**weight
  sharing across recurrent depth**) and adds a **dynamic per-position ACT halting** so
  different token positions stop after different numbers of steps. The direct ancestor of
  every model here: depth-via-recurrence + per-token adaptive compute.

- **PonderNet** — Banino, Balaguer, Blundell 2021 (arXiv:2107.05407). Reframes halting
  probabilistically: a per-step scalar halting probability `λ_n` induces a (generalized)
  geometric distribution `p_n = λ_n ∏_{j<n}(1−λ_j)` over which step to stop at; the
  prediction is taken at the sampled halting step (vs ACT's weighted average). Trained with
  a two-term loss — reconstruction `Σ_n p_n L(y, y_n)` + `β·KL(p_n ‖ geometric prior)`. The
  KL term biases toward an expected step count `1/λ_p` and *incentivizes exploration* rather
  than bluntly penalizing steps. Empirically solved all 20 bAbI tasks (0.15 vs UT's 0.29 avg
  error) with ~6× fewer steps, and **extrapolated** parity to 96-element inputs (trained on
  ≤48) where ACT stayed at chance — by spending more thinking steps. Strong evidence that
  *learned latent pondering substitutes for fixed compute*.

- **ALBERT** — Lan et al. 2019 (arXiv:1909.11942, ICLR 2020). Introduced **cross-layer
  parameter sharing** (layer tying) as a parameter-reduction technique (plus factorized
  embeddings). Not framed as "reasoning", but it is the canonical proof that tying weights
  across all transformer layers is trainable and competitive — the structural prerequisite
  for looping.

- **Deep Equilibrium Models (DEQ)** — Bai, Kolter, Koltun 2019 (arXiv:1909.01377, NeurIPS).
  The **fixed-point / equilibrium framing** you asked about. Instead of stacking discrete
  layers, represent an infinite-depth weight-tied network by *directly solving* for its
  fixed point `z* = f(z*; x)` via root-finding. Crucially, training uses **implicit
  differentiation** to backprop analytically through the equilibrium point, so memory is
  **constant regardless of effective depth**. This is the "input/output consistency" soft
  constraint made into the whole architecture.

### 2.2 The modern reasoning-focused wave (2023–2025)

- **CoTFormer** — Mohtashami, Pagliardini, Jaggi 2023 (arXiv:2310.10845). Explicitly frames
  CoT as "employing a deeper transformer by re-applying the model multiple times" and builds
  an architecture that mimics CoT at the token level by reusing/looping layers, with
  **budget-adaptive compute at inference**. Reaches accuracies close to much larger models.

- **Recursive & Relaxed Recursive Transformers** — Bae et al. 2024 (arXiv:2410.20672,
  ICLR 2025). A **training recipe to convert existing pretrained LLMs into looped form**:
  reuse a single block of unique layers repeated `L` times in a loop, *initialized from a
  standard pretrained transformer* (not from scratch) with minimal performance loss.
  **Relaxed** Recursive Transformers loosen strict weight-tying with depth-wise **LoRA**
  modules so each loop iteration can differ slightly while staying compact. Directly relevant
  if we ever want to "loop-ify" an existing Grug checkpoint instead of training fresh.

- **Reasoning with Latent Thoughts: On the Power of Looped Transformers** — Saunshi, Dikkala,
  Li, Kumar, Reddi 2025 (arXiv:2502.17416, ICLR 2025). The key **theoretical + empirical**
  case: looped models "implicitly generate latent thoughts and can simulate `T` steps of CoT
  with `T` loops" (their Theorem 5.4). Empirically, a `k`-layer transformer looped `L` times
  **nearly matches a `kL`-layer non-looped model** on reasoning (addition, p-hop induction,
  math) despite far fewer parameters — i.e. *effective depth, not parameter count, drives
  reasoning*. Looped and non-looped models scale with effective depth analogously to
  inference-time CoT scaling. (Follow-up arXiv:2509.25239 corroborates "looped transformers
  subsume deterministic CoT" but flags an open gap under stochastic decoding.)

- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach ("Huginn")**
  — Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein
  2025 (arXiv:2502.05171). **The closest existing model to what you want** and the template
  for my recommendation. A 3.5B-param model with a **prelude → recurrent block → coda**
  ("sandwich") structure: a few non-looped layers embed the input, a core block is **looped a
  variable number of times**, and a few non-looped layers decode. Key properties:
  - **No specialized (CoT) training data required** — it learns latent reasoning from
    ordinary pretraining (trained to 800B tokens).
  - Trained with a **randomly sampled number of recurrent iterations per step** (a
    log-normal-Poisson schedule), so the model is simultaneously trained to produce good
    outputs at many depths. Backprop is **truncated** to the last few iterations to bound
    memory (truncated BPTT), and the recurrent state is randomly initialized with the
    embedded input **injected at every iteration**.
  - At **test time it can scale compute by looping more** — up to a compute load
    "equivalent to 50B parameters" — improving reasoning benchmarks, "sometimes dramatically".
  - Exhibits **emergent per-token convergence**: the latent state's change between
    consecutive iterations shrinks, enabling a training-free per-token early-exit (stop when
    the KL/Δ between steps is small). Claims latent recurrence captures reasoning "not easily
    represented in words".

- **Mixture-of-Recursions (MoR)** — Bae, Kim, et al. 2025 (arXiv:2507.10524, NeurIPS 2025).
  Unifies parameter sharing + **per-token adaptive compute** in one recursive transformer:
  a shared stack is reused across recursion steps, while **lightweight routers dynamically
  assign different recursion depths to individual tokens** — the cleanest answer to "how does
  a token get *more time*". At equal training FLOPs and smaller sizes, MoR lowers validation
  perplexity, raises few-shot accuracy, and improves throughput vs vanilla and recursive
  baselines (e.g. 43.1% vs 42.3% avg few-shot at 16.5e18 FLOPs with ~50% fewer params).
  Public code exists (github.com/raymin0223/mixture_of_recursions).

### 2.3 How the pieces map to your three questions

| Your question | Established mechanisms |
|---|---|
| **How to train a self-looping model?** | Weight-tie the looped block (ALBERT/UT). Train with **randomized loop count** + **truncated BPTT** (Huginn) so the model is good at many depths and memory is bounded. *Or* solve for the fixed point and use **implicit/equilibrium gradients** (DEQ). *Or* convert a pretrained model with layer-tying + LoRA (Relaxed Recursive). |
| **How does it ask for "more time"?** | (a) **Training-free**: early-exit when consecutive latent states converge (Huginn KL/Δ heuristic). (b) **Learned halting head**: ACT threshold or PonderNet geometric halting + ponder-cost KL. (c) **Per-token router**: MoR assigns each token a recursion depth. |
| **Soft input/output consistency constraint?** | This is the **fixed-point / equilibrium** idea (DEQ): train so `z ≈ f(z; x)`. Can be added as an auxiliary penalty `‖f(z)−z‖`. **Caveat:** Huginn shows you may not *want* to force this — convergence emerges from random-depth training, and over-constraining can cap expressivity. Recommend it as an ablation. |

---

## 3. Where Grug stands today (codebase map)

Grug is **template-first** (per `.agents/skills/change-grug`): the canonical edit surface is
`experiments/grug/base/` (`model.py`, `train.py`, `launch.py`); variants are copied from `base`
and edited locally (this is how `experiments/grug/moe/` exists). Plain JAX arrays + Equinox
modules with `init`/`__call__`, explicit sharding, minimal config knobs, legibility first.

The layer stack is a **plain Python `for`-loop over a tuple of blocks** — there is no
weight-tying, scan-over-shared-params, or adaptive compute anywhere yet. The dense base
forward pass (`experiments/grug/base/model.py:197-200`):

```python
for i, block in enumerate(self.blocks):
    with jax.named_scope(f"block_{i}"):
        block_fn = block if is_backward_flow_active() else eqx.filter_checkpoint(block)
        hidden = block_fn(hidden, mask)
```

`Block` is a standard pre-norm residual unit (`model.py:136-158`): `x = x + attn(norm(x));
x = x + mlp(norm(x))`. Blocks are built as `tuple(Block.init(cfg, key=k) for k in block_keys)`
(`model.py:175`). The MoE variant (`experiments/grug/moe/model.py:544-547`) is the same loop
but collects per-layer router stats and alternates sliding-window masks every 4th layer.

**Implications for a re-entrant variant — all favorable:**

- Converting the for-loop into "prelude loop → looped core → coda loop" is a *local* edit to
  `Transformer.__call__` plus a few config knobs. No framework surface needed.
- Weight-tying = build *one* core block and apply it `R` times instead of building `N`
  distinct blocks. Trivial in Equinox.
- `eqx.filter_checkpoint` is already used per-block; truncated BPTT can reuse the same remat
  machinery (`jax.lax.stop_gradient` on the carry for all but the last `k` iterations).
- `haliax.nn.Stacked` (`lib/haliax/src/haliax/nn/scan.py`) provides `scan`/`fold` over layers
  with checkpointing, *but it's for distinct per-layer params*. For tied params the simplest
  primitive is `jax.lax.scan(step, init=hidden, xs=None, length=R)` where `step` closes over
  the single shared block — keep it inline and grug-style rather than reaching for Stacked.
- A small tiny config already exists — `GRUG_130M_MODEL` (`experiments/grug/base/launch.py:62`,
  6 layers / hidden 512) — ideal for the first experiments.
- MoE's existing per-layer router-stats plumbing is a ready template if we later add a
  MoR-style per-token recursion router.

---

## 4. Design space & recommended choices

| Axis | Options | Recommendation for v1 |
|---|---|---|
| **What is looped** | single block (pure UT) · a small stack · whole model | **Prelude/coda + a small shared core** (Huginn). Pure single-block looping is the simplest sanity check but empirically weaker; prelude/coda is the proven sweet spot. |
| **Weight tying** | strict tie · tie + per-iteration LoRA (Relaxed Recursive) | **Strict tie** for v1. LoRA-per-iteration is a later expressivity lever. |
| **Loop count at train time** | fixed `R` · randomized per step | **Fixed `R`** in phase 1 (clean baseline), **randomized** in phase 2 (enables test-time scaling). |
| **Gradient** | full BPTT · truncated BPTT · implicit/DEQ | **Truncated BPTT** (last `k≈2` iters) once depth is randomized. DEQ implicit-grad is a heavier, higher-risk alternative — defer. |
| **State init / input injection** | zero/random init · inject embedded input each iter | **Random init + input injection every iteration** (Huginn) — improves robustness to variable depth. |
| **Adaptive compute** | none · KL-convergence exit · ACT/PonderNet head · MoR router | **None → KL-exit (training-free) → learned router**, in that order across phases. |
| **Consistency penalty** | none · `‖f(z)−z‖` aux loss | **None by default**; add as an ablation only. |
| **Dense vs MoE** | dense · MoE | **Dense first.** MoE adds routing variance that confounds the looping signal; combine only after looping is validated. |

---

## 5. Recommendation: a first re-entrant Grug

Build **`experiments/grug/reentrant/`** by copying `experiments/grug/base/` and making the
forward pass re-entrant. Concrete shape, starting from `GRUG_130M_MODEL`:

```
tokens → embed
       → PRELUDE  : P unique blocks            (e.g. P = 1)
       → CORE     : C shared blocks, looped R× (e.g. C = 1, R ∈ {2,4,8})   ← the re-entry
       → CODA     : K unique blocks            (e.g. K = 1)
       → final norm → logits
```

So "effective depth" = `P + C·R + K`. With `P=K=1, C=1`, looping `R=4` gives effective depth
6 — matching the 6-layer dense baseline at **~1/3 the transformer-block parameters** (3 unique
blocks vs 6). That equal-effective-depth, unequal-params comparison is the headline experiment.

Config additions (new frozen dataclass `ReentrantGrugModelConfig`, or fields on a copied
`GrugModelConfig`):

```python
num_prelude_layers: int = 1
num_core_layers: int = 1          # unique layers in the shared, looped block
num_coda_layers: int = 1
recurrence_steps: int = 4         # R: loop count (fixed in phase 1)
randomize_recurrence: bool = False        # phase 2: sample R per microbatch
recurrence_dist: str = "lognormal_poisson"  # Huginn-style schedule (phase 2)
backprop_steps: int = 2           # truncated BPTT window (phase 2)
inject_input_each_step: bool = True       # add prelude output into core input every iter
# adaptive compute lives in a later phase; keep it out of v1
```

Forward-pass sketch (dense, inline, grug-style — replaces the `for` loop):

```python
hidden = embed(token_ids)
for block in self.prelude:                      # P unique blocks
    hidden = block(hidden, mask)
prelude_out = hidden

def core_step(state, _):
    x = state + prelude_out if cfg.inject_input_each_step else state
    for block in self.core:                     # C shared blocks, reused every iteration
        x = block(x, mask)
    return x, None

# phase 1: fixed R. phase 2: R sampled; stop_gradient on carry except last `backprop_steps`.
hidden, _ = jax.lax.scan(core_step, init=hidden, xs=None, length=R)

for block in self.coda:                         # K unique blocks
    hidden = block(hidden, mask)
hidden = self.final_norm(hidden)
```

(For phase-2 truncated BPTT, run the first `R−k` iterations under `jax.lax.stop_gradient` and
the last `k` with gradients — a small wrapper around the scan, or two scans.)

### Headline experiments (small, cheap, on 130M-scale configs)

1. **Equal effective depth, fewer params.** Looped (`P=K=1, C=1, R=4`, effective depth 6,
   3 unique blocks) vs dense 6-layer baseline. Question: how much val-loss / reasoning gap
   does parameter-sharing cost? (Saunshi predicts: small, on reasoning.)
2. **Effective-depth scaling at fixed params.** Same looped model, evaluate at `R ∈ {2,4,8,16}`.
   Question: does looping more at *test time* (phase 2, randomized-depth training) keep
   improving reasoning, à la Huginn?
3. **Reasoning vs memorization split.** Track loss separately on a reasoning probe (arithmetic
   / multi-hop) vs a memorization-heavy slice. Saunshi's claim is the benefit concentrates on
   reasoning — worth confirming on Marin data/evals.
4. **Adaptive-compute (phase 3):** add the training-free KL-convergence early-exit; measure
   accuracy vs average loop count. Does it cut compute on easy tokens without hurting hard ones?

### Validation / checks

- `./infra/pre-commit.py --all-files` and `uv run pytest tests/test_grug_variant_contracts.py`
  (the contract test the `change-grug` skill mandates for variants).
- Record the variant in `docs/reports/grug-archive.md` (path, origin=`base`, purpose, status).
- Add a focused test that the re-entrant forward at `R=1, P+C+K=N` is numerically equivalent
  to the dense baseline (a clean plumbing regression).

---

## 6. Phased plan & issues

Issues filed under this branch (see `weaver issue ls --mine`):

- **#158 Phase 1 — fixed-depth plumbing.** Copy `base` → `reentrant`, add prelude/core/coda
  config, weight-tied fixed-`R` scan, equivalence test, train 130M, compare to dense baseline
  at equal effective depth.
- **#159 Phase 2 — randomized depth + truncated BPTT.** Sample `R` per step
  (log-normal-Poisson), truncate backprop to last `k`, input injection + random state init.
  Demonstrate test-time depth scaling (experiment 2).
- **#160 Phase 3 — training-free adaptive compute.** KL/Δ-convergence per-token early-exit at
  inference; compute-vs-accuracy curve (experiment 4).
- **#161 Phase 4 (stretch, only if 1–3 win) — learned adaptive compute.** Either a
  PonderNet-style halting head + ponder-cost loss, or a MoR-style per-token recursion router
  reusing the MoE router-stats plumbing. Optionally combine with MoE.

This document is the `plan` artifact; the dashboard projects live issue state — keep the issue
list and this section reconciled as work moves.

---

## 7. References

| # | Title | Authors / year | arXiv |
|---|---|---|---|
| 1 | Adaptive Computation Time for Recurrent Neural Networks | Graves 2016 | [1603.08983](https://arxiv.org/abs/1603.08983) |
| 2 | Universal Transformers | Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser 2018 (ICLR'19) | [1807.03819](https://arxiv.org/abs/1807.03819) |
| 3 | ALBERT: A Lite BERT for Self-supervised Learning of Language Representations | Lan, Chen, Goodman, Gimpel, Sharma, Soricut 2019 (ICLR'20) | [1909.11942](https://arxiv.org/abs/1909.11942) |
| 4 | Deep Equilibrium Models | Bai, Kolter, Koltun 2019 (NeurIPS) | [1909.01377](https://arxiv.org/abs/1909.01377) |
| 5 | PonderNet: Learning to Ponder | Banino, Balaguer, Blundell 2021 | [2107.05407](https://arxiv.org/abs/2107.05407) |
| 6 | CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference | Mohtashami, Pagliardini, Jaggi 2023 | [2310.10845](https://arxiv.org/abs/2310.10845) |
| 7 | Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA | Bae, Fisch, Harutyunyan, Ji, Kim, Schuster 2024 (ICLR'25) | [2410.20672](https://arxiv.org/abs/2410.20672) |
| 8 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn) | Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein 2025 | [2502.05171](https://arxiv.org/abs/2502.05171) |
| 9 | Reasoning with Latent Thoughts: On the Power of Looped Transformers | Saunshi, Dikkala, Li, Kumar, Reddi 2025 (ICLR'25) | [2502.17416](https://arxiv.org/abs/2502.17416) |
| 10 | Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation | Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun 2025 (NeurIPS'25) | [2507.10524](https://arxiv.org/abs/2507.10524) |
| 11 | (follow-up) Looped transformers subsume deterministic CoT — stochastic-decoding gap | 2025 | [2509.25239](https://arxiv.org/abs/2509.25239) |

## 8. Caveats & open questions

- Most empirical numbers are **self-reported** by the originating papers in controlled/synthetic
  settings; independent replication at Marin scale is exactly what these experiments would
  provide.
- Saunshi's "looped simulates CoT" is proven for **deterministic decoding**; the stochastic
  case is open (arXiv:2509.25239).
- The looped-substitutes-for-CoT benefit appears to concentrate on **reasoning**, not
  memorization — our eval split should separate the two.
- DEQ-style implicit gradients are elegant but add solver complexity and can be unstable;
  truncated-BPTT-over-random-depth (Huginn) is the lower-risk first bet.
- Strict input/output consistency may **cap expressivity**; do not enforce it by default.
- MoE + looping together multiplies routing variance — validate looping on a dense model first.
