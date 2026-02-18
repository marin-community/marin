# Adaptive-compute Mixture-of-Experts landscape
_Date: 2026-02-18_

This is a decision-oriented landscape review of **adaptive-compute MoE** mechanisms where **tokens can consume variable compute** by activating **variable numbers of experts**, including **“route-to-none” / null experts / early-stop** style routing.

The emphasis is on:
- mechanisms that plausibly work in **JAX/TPU** MoE stacks,
- **quality vs compute** tradeoffs with measurable throughput/latency,
- and routing/dispatch constraints (load balance, capacity, all-to-all overhead).

We also ground recommendations in recent Marin internal perf work on null routing and MoE microbench harnesses (issues #2704 and #2844).

---

## 1 Executive summary

### Strong takeaways (10–15)

1. **Null experts are the most implementation-friendly path to variable experts-per-token** in token-choice MoE because they keep the router and top-k machinery mostly unchanged while introducing an explicit “no compute” option. (AdaMoE; Perceptron’s 2026 data+weight sparsity.)
2. **Load balancing with null experts is subtle**: balancing *among* null experts is actively harmful because null experts are identical and the router gets over-constrained. AdaMoE fixes this by treating null experts as a *group* (average null load) in the load-balance objective.
3. **Compute adaptivity that changes only FLOPs but not communication can disappoint**. If “routing to null” still performs the same all-to-all/packing/dispatch work, most speedups vanish at small batch or decode-time. Real wins require a path that **bypasses dispatch** for null-routed tokens (or otherwise reduces routed token-expert pairs).
4. In Marin’s TPU microbench sweep, **null routing produced strong throughput gains with near-linear scaling in null fraction** under the tested shapes/top-k. This suggests the current implementation likely reduces meaningful work (not just MLP FLOPs), and is worth pursuing as a primary axis.
5. Marin’s sweep also found **`take_until_null` is ~perf-neutral vs `independent`** at fixed shape/top-k/null settings, implying that *the number of true experts* is what matters for throughput; ordering/early-stop semantics matter more for **training signal and stability** than for raw throughput.
6. **Dynamic top-k / threshold-based routers** (e.g., Top-P / cumulative-probability thresholds) can deliver variable experts-per-token without introducing explicit null experts, but they typically still require a **K_max** for static shapes and may create **capacity spikes** on “hard” tokens.
7. **XMoE-style threshold routing** and **Huang et al. (ACL 2024) dynamic routing** both report that variable experts-per-token can improve quality at similar or lower compute, but systems wins depend on how well you handle **load balance + capacity**.
8. **RL-based “expert count allocators” (Ada-K, ICLR 2025)** are a strong quality/compute story on paper (>25% FLOPs reduction and >20% inference speedup while improving benchmark accuracy), but they are **higher-risk for JAX/TPU training** because you introduce non-differentiable discrete decisions + PPO training dynamics.
9. **Expert-choice routing** (experts select tokens) naturally yields variable experts per token and very strong load balance properties, but it is generally **not causal** and introduces train/inference mismatch for autoregressive decoding. Many newer causal token-choice adaptive methods can be read as ways to capture some benefits of expert-choice without breaking causality.
10. **Compute-adaptive sparse transformers beyond MoE** (e.g., Mixture-of-Depths token skipping) show that “route-to-none” can be applied at **layer depth** too: some tokens bypass FFN/MoE layers entirely. These methods can provide larger wall-time benefits because they reduce whole-layer work, but they also alter optimization dynamics and often require careful budget control.
11. **Capacity/drop behavior is the hidden failure mode** for many adaptive routers: decreasing average compute is easy; keeping **overflow/drop rate low** and **training stable** is hard. Any adaptive method needs observability around overflow, per-expert load CV, and per-device stragglers.
12. A practical compute-adaptive objective usually looks like **(task loss) + λ·(expected compute)**, but for MoE the most actionable “compute proxy” is **expected number of true expert dispatches per token**, not raw router entropy.
13. The best “next” experiments are those that:
    - preserve static shapes and SPMD friendliness,
    - keep top-k low in the worst case,
    - and give you a clean knob to target an average compute budget.
14. The strongest near-term opportunity: **null experts + compute-target controller + robust load-balance** (AdaMoE-style null grouping), because it aligns with what we already have and targets quality-per-compute (the next axis Marin identified).

### What to try next in our JAX/TPU stack

These are ordered by expected impact per engineering hour.

1. **AdaMoE-style null balancing + true-expert-only weight normalization**, dropped into our null-routing block.
   - Goal: stabilize training and prevent “true experts undertrained” pathologies.
2. **Compute-targeted null fraction**: treat “avg true experts per token” as the controlled variable, and tune a controller to hit a target (even a simple proportional controller to start).
   - Goal: make iso-compute comparisons clean and automate sweeping.
3. **Threshold-based dynamic top-k with K_max fixed** (XMoE / Top-P style): keep top-8 static, but mask out experts beyond the threshold and optionally map masked slots to null.
   - Goal: compare “probability-threshold” adaptivity vs explicit null-fraction adaptivity.
4. **Layer-level route-to-none (MoD-style bypass)**: allow tokens to skip the MoE FFN entirely in some layers, gated by a compute budget.
   - Goal: test whether bypassing dispatch/work at the layer granularity beats per-expert adaptivity.
5. **Only if the above works**: evaluate Ada-K-style learned K allocator (but consider doing this as an offline fine-tune of the allocator only, frozen base model, as in the paper).

---

## 2 Taxonomy

This taxonomy is structured around how the model reduces compute:
- by **adding a “do nothing” option** (null routing),
- by **choosing fewer experts** (dynamic top-k),
- by **stopping early** (halting),
- by **skipping tokens or layers** (token/layer sparsity),
- or by **hybrids** that combine dense and sparse pathways.

### 2.1 Fixed-topk + null routing
**Definition:** Keep a fixed `K_max` in the router, but add **null experts** (or a special “none” action) so a token may select `0..K_max` *true* experts. Null slots consume (ideally) no expert FLOPs and preferably also bypass dispatch.

**Representative:** AdaMoE (Findings EMNLP 2024), Perceptron 2026 weight+data sparsity (arXiv 2601.15370).

**Pros:**
- Minimal router changes.
- Easy to keep static shapes (always route `K_max`, but with null ids).
- Natural knob for compute: expected number of true experts.

**Cons / failure modes:**
- If null still participates in dispatch/packing, speedups collapse.
- Load balancing must treat null experts carefully (group/null-average).
- Risk of “degenerate routing” where model overuses null and undertrains experts unless constrained.

### 2.2 Dynamic top-k (probability threshold / top-p routing)
**Definition:** Router produces expert probabilities; token chooses the *smallest* number of experts such that a threshold is met (e.g., cumulative probability ≥ τ). This yields variable `k(x)`.

**Representative:** Huang et al. (ACL 2024) dynamic routing (“Harder Task Needs More Experts”), XMoE (Findings ACL 2024), DTop-p MoE (arXiv 2025).

**Pros:**
- No need for explicit null experts.
- Intuitive: “confident tokens use fewer experts.”

**Cons / failure modes:**
- Still often implemented with `K_max` and masking → compute savings depend on kernels.
- Creates heterogeneous expert demand → capacity spikes and load imbalance.
- Threshold tuning is brittle across scales/datasets.

### 2.3 Learned K allocators / resource schedulers
**Definition:** Add a lightweight module that predicts `k(x)` (or a distribution over k), trained to trade off quality and compute.

**Representative:** Ada-K (ICLR 2025; PPO-trained allocator), DA-MoE (arXiv 2024; attention-derived importance → K).

**Pros:**
- Potentially better than fixed heuristics (learns token-type/layer/task structure).
- Provides interpretability: which tokens/layers demand more compute.

**Cons / failure modes:**
- Non-differentiable decisions typically require RL/REINFORCE or relaxations.
- Additional training instability; reproducibility challenges.

### 2.4 Halting / pondering style routing
**Definition:** Token iteratively applies experts or layers until a learned halting condition triggers. Equivalent to “take-until-stop” in time/depth.

**Representative:** ACT (Graves 2016), PonderNet (Banino et al. 2021), MoEUT (ICLR 2025; MoE + universal transformer recurrence).

**Pros:**
- Gives fine-grained compute adaptivity.
- Conceptually matches “reason longer on harder tokens.”

**Cons / failure modes:**
- JAX/XLA dynamic control flow can be tricky; static unrolling kills wall-time adaptivity.
- Often increases sequential dependency (hurts parallelism) and can worsen tail latency.

### 2.5 Confidence-threshold routing / early exit
**Definition:** Exit early from depth (stop applying layers) when confidence is high. Not strictly MoE, but often complementary.

**Representative:** Early-exit transformers (e.g., DeeBERT), dynamic depth variants.

**Pros:**
- Targets latency directly (skip whole layers).

**Cons / failure modes:**
- Adds train/inference mismatch unless trained carefully.

### 2.6 Token dropping / pruning variants
**Definition:** Drop tokens from expensive computation (attention/FFN) or reduce their representation updates based on importance.

**Representative:** Token pruning literature; MoD can be read as a structured token-skipping method.

**Pros:**
- Can give real wall-time wins if it reduces matrix sizes.

**Cons / failure modes:**
- Can break quality on long-range dependencies; often brittle.

### 2.7 Hybrid dense+sparse adaptive schemes
**Definition:** Combine a dense shared pathway (shared expert / dense FFN) with sparse experts and adapt which pathway is used.

**Representative:** Shared expert MoE practices (including Marin’s harness results: shared expert often helps), MoD variants, multi-branch conditional compute.

**Pros:**
- Dense path stabilizes quality; sparse path adds capacity.

**Cons / failure modes:**
- Additional routing complexity; can reduce the pure compute benefit if dense path is always active.

---

## 3 Paper/method table (compact, information-dense)

**Legend:**
- **Routing rule** describes per-token expert selection.
- **Reg/Objective** includes compute penalties, load-balance, RL reward, etc.
- **Gains** are as reported by authors; treat as *directional* unless explicitly iso-FLOP / iso-walltime.

| Work | Year / venue | Core idea | Routing rule | Training signal / regularization | Reported gains (quality + compute) | HW setting | Caveats / failure modes |
|---|---:|---|---|---|---|---|---|
| **AdaMoE** — “Token-Adaptive Routing with Null Experts for MoE LMs” ([ACL Anthology PDF](https://aclanthology.org/2024.findings-emnlp.361.pdf)) | 2024, Findings EMNLP | Add **m null experts** (zero/identity mapping) to allow variable true experts per token with minimal changes | Increase `k` and route top-k over (true + null). #true experts varies depending on how many nulls appear in top-k | Modified load-balance: include nulls but **do not balance among null experts**; anneal load-balance weight from tight → loose; normalize weights over **true experts only** | Example: Mixtral-8x7B fine-tuning on ARC-C: **−14.5% FLOPs** with **+1.69% accuracy** (also other datasets); can reach loads ~1.5 true experts avg (from their tables) | Not clearly specified (fine-tuning, likely GPU) | Speedups depend on dispatch path; if null participates in comm, gains shrink. Requires careful null treatment in load-balance; sensitive to (m, k) choices.
| **Improving MoE compute efficiency by composing weight + data sparsity** ([arXiv abs](https://arxiv.org/abs/2601.15370)) | 2026, arXiv (Perceptron) | Treat “data sparsity” (subset tokens per expert) as complementary to weight sparsity; recover it causally via **null experts** | Token-choice MoE with null experts to emulate data sparsity in expectation (vs non-causal expert-choice) | Standard load-balance over (real + null) gives uniform expert use; nulls create data sparsity without causality violation | Claims a better compute-efficient frontier at matched expected FLOPs; learns modality-aware allocation (routes vision tokens to null more than text) | Reported on vision-language training (details in paper) | Preprint; validate on our workloads. Systems wins depend on bypassing dispatch for null-routed tokens.
| **TC-MoE** — “Fast yet Expressive… Ternary Compute MoE” ([OpenReview PDF](https://openreview.net/pdf?id=dsP91M4hDL)) | 2025, ICLR | Expand expert “actions” to ternary {−1, 0, +1} (incl. **0 = skip**) to reduce compute with richer functional space | Router selects ternary expert contributions; includes “zero-expert” option | Adds load-balance + new losses (paper proposes additional objectives to stabilize ternary choices) | Reported improved performance with reduced compute vs baselines | Not specified in snippet (likely GPUs) | More complex semantics than null-only; risk of instability; harder to implement and debug.
| **XMoE** ([ACL Anthology PDF](https://aclanthology.org/2024.findings-acl.170.pdf)) | 2024, Findings ACL | Variable experts-per-token using probability thresholds + scheduling to reduce compute and improve wall time | Threshold-based selection: choose minimal #experts meeting a probability criterion (and/or per-expert threshold), with mechanisms to handle capacity/priority | Includes strategies to reduce imbalance; focuses on wall-time gains rather than only FLOPs | Reports both FLOPs and wall-time comparisons; argues wall-time is the right metric | Implemented on Megatron-LM (GPU) | Threshold tuning; can increase peak load; scheduling complexity.
| **Harder Task Needs More Experts: Dynamic Routing in MoE Models** ([ACL Anthology PDF](https://aclanthology.org/2024.acl-long.380.pdf)) | 2024, ACL | “Hard tokens use more experts”; dynamic routing reduces compute while improving some tasks | Dynamic routing based on router probability threshold / cumulative probability (Top-P-like); average experts can drop below 2 in training | Uses routing objective + task loss; explores dynamic routing in pretrain + finetune; reports routing statistics | Reports improved performance (esp. harder tasks) at lower average activated experts | Not specified in snippet | Needs careful capacity mgmt; may be more beneficial in finetune than pretrain depending on setup.
| **Ada-K routing** ([ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/df22a19686a558e74f038e6277a51f68-Paper-Conference.pdf)) | 2025, ICLR | Learn a per-token (and per-layer) distribution over k using a lightweight “allocator” trained with PPO | Allocator samples `k* ~ Softmax(W_alloc x)` then router activates top-`k*` experts | PPO reward uses log-likelihood at last layer; regularization on expected activated expert count; warm-start strategy | Reports **30–40% fewer activated experts**, **>25% FLOPs reduction**, **>20% inference speedup** while improving benchmark accuracy; only ~2M trainable params; training <8h even for Mixtral-8x22B (per paper) | Not specified (likely GPU cluster) | Higher-risk engineering: RL training loop, variance, reproducibility. Inference speedups depend on real kernels/dispatch savings.
| **DA-MoE** ([arXiv abs](https://arxiv.org/abs/2409.06669)) | 2024, arXiv | Use attention-derived token importance to decide how many experts to allocate | Importance score from attention weights; choose `K = ceil(score * N_experts)` then route to top-K | Uses attention-based importance; capacity constraints handled in algorithm | Reports improved GLUE performance vs baseline MoE | Not specified | K can become large; likely needs cap K_max for LLM scale; attention-derived importance might be noisy/late.
| **MoD — Mixture of Depths** ([arXiv abs](https://arxiv.org/abs/2404.02258)) | 2024, arXiv | **Route tokens through fewer FFN/MoE layers** (depth sparsity) to save compute; can be combined with MoE | Router chooses whether token takes “depth” computation; structured skipping yields fewer token-layer updates | Budgeted routing (select subset of tokens) + training objectives to keep usage controlled | Reports step-time gains (e.g., **~66% faster** than dense transformer; **~31% faster** than standard MoE in their setting) and improved compute-quality frontier | Not specified in snippet | Requires token selection infrastructure; may change optimization dynamics; interacts with attention (token stays in sequence but may skip MLP updates).
| **Expert Choice Routing** ([arXiv abs](https://arxiv.org/abs/2202.09368)) | 2022, arXiv / (published) | Invert routing: experts choose tokens → fixed tokens per expert, variable experts per token | Expert-level top-k token selection; each expert processes a fixed bucket | Eliminates need for aux load-balance; improves training efficiency | Reports >2× faster convergence vs Switch/GShard at similar compute in their study; better downstream in iso-compute settings | Google TPU/GPU (paper) | Not causal for autoregressive decoding; train/inference mismatch for LM generation.
| **BASE Layers** ([arXiv abs](https://arxiv.org/abs/2103.16716)) | 2021, ICML | Formulate token→expert assignment as linear assignment to guarantee balance | Balanced assignment per batch (each expert gets equal tokens) | Removes aux load-balance hyperparams | Improves stability/efficiency of sparse layers | (paper) | Solving assignment at scale can be expensive; may not fit streaming/AR decoding.
| **Switch Transformer** ([arXiv abs](https://arxiv.org/abs/2101.03961)) | 2021, arXiv/ (NeurIPS) | Simplify MoE with top-1 routing and capacity/dropping | Top-1 routing with capacity factor; dropped tokens use residual | Load-balance + router stabilizers (z-loss in later work) | Efficient scaling; establishes baseline systems practices | TPU | Not adaptive compute per token, but crucial for capacity/drop + load-balance discussions.
| **ACT — Adaptive Computation Time** ([paper](https://arxiv.org/abs/1603.08983)) | 2016, arXiv | Learn halting; tokens take variable number of recurrent steps | Halting probability accumulates until stop | Ponder cost regularizer | Enables variable compute | n/a | Efficient wall-time adaptivity is hard in compiled SPMD; risk of sequential slowdown.
| **PonderNet** ([paper](https://arxiv.org/abs/2107.05407)) | 2021, arXiv | Probabilistic halting distribution; more stable than ACT in some settings | Stochastic halting distribution over steps | KL-based regularization to target compute | Compute-adaptive inference/training | n/a | Same systems issues as ACT.
| **MoEUT** — Mixture-of-Experts Universal Transformers ([OpenReview](https://openreview.net/forum?id=7fh6uFFwW0)) | 2025, ICLR (per OpenReview) | Combine universal transformer recurrence + MoE, enabling dynamic depth/compute | Recurrent application with gating; can incorporate halting | Halting/pondering objectives | Reported improvements in reasoning/efficiency (paper) | Not specified | Complexity; may not yield wall-time wins on TPU without careful kernel work.

> Note: we include classic MoE references (Switch, BASE, Expert Choice) because they define the load-balance/capacity constraints that dominate system behavior. The adaptive-compute papers are the main focus.

---

## 4 Implementation notes for our JAX/TPU MoE stack

### 4.1 Baseline assumptions and constraints

From Marin’s MoE hillclimb harness (#2704) and null-routing perf sweep (#2844), we can assume:
- We have an MoE microbench harness with configurable `topk`, expert count, routing distributions, and backends.
- **`gmm` backend is consistently faster than `ragged_dot`** in the harness.
- **Throughput drops as `topk` rises** (so any adaptive method should keep worst-case `K_max` modest).
- **Shared expert** tends to be beneficial; `--shared-fused` is a modest positive knob.
- Some routing locality tricks (“runs”) did not materially help throughput in the harness.
- In #2844, **null routing fraction produced near-linear throughput gains**, and **`take_until_null` was perf-neutral vs `independent`**.

Key TPU/JAX constraints to respect:
- XLA likes **static shapes**; varying per-token expert counts must usually be represented as **masking within a fixed `K_max`**.
- Real speedups require that masked/“null” paths reduce actual executed work (avoid dispatch or reduce routed token-expert pairs).
- All-to-all + packing/permutation overhead can dominate at low batch (inference), even if it’s amortized at large training batch.

### 4.2 What is straightforward to implement

#### A) Null experts (AdaMoE-style)
**Mechanics:**
- Add `m` null experts to router logits.
- Select top-`K_max` experts across true+null.
- Convert selected expert ids into two subsets:
  - **true expert dispatch** (goes through all-to-all / expert MLP)
  - **null path** (identity or zero mapping) computed locally with no dispatch

**Key detail:** implement the “null path” so it **does not participate** in packing/all-to-all.

**Training objective changes (low diff):**
- Use AdaMoE’s modified load-balance term:
  - include null experts **as a group**; do not balance among them.
- Normalize expert weights over **true experts only** to keep output scale stable.

#### B) Masked dynamic top-k with fixed K_max
If you already have `topk` selection, dynamic-k can be done as:
- compute `topk_idx, topk_prob` for `K_max`
- compute a per-token cut `k(x)` based on threshold/cumulative probability
- define `mask = arange(K_max) < k(x)` and zero-out weights for masked-out experts
- optionally replace masked-out experts with a canonical “null id” to avoid dispatch

This keeps shapes static while enabling variable expected expert load.

#### C) A compute regularizer in pretraining / finetuning
A minimal compute proxy:

- `E_true = mean_over_tokens(sum_j 1[idx_j is true])`.

Add either:
- a penalty: `L = L_task + λ * E_true`, or
- a constraint-style controller: adjust λ or null-bias to hit `E_true_target`.

This is easy to instrument and makes iso-compute sweeps clean.

### 4.3 What is hard / risky (and why)

#### A) RL-based allocators (Ada-K)
- Requires a PPO/REINFORCE-style training loop.
- Adds another optimizer state, reward baselines, warm-start, etc.
- Debuggability is poor when routing collapses.

**Risk mitigation:** do allocator-only training with the base model frozen (as Ada-K does). This is tractable as a “policy fine-tune” stage but still adds moving parts.

#### B) Expert-choice routing for causal LMs
- Expert-choice depends on seeing a batch of tokens and letting experts pick.
- For autoregressive generation, this breaks causality or requires nontrivial buffering.

**Best use:** pretraining (non-streaming) or encoder-only tasks; otherwise treat it as a conceptual reference.

#### C) True halting/pondering with wall-time savings
- JAX control flow (`lax.while_loop`) might not translate into proportional wall-time savings on TPU depending on compilation and hardware scheduling.
- Even if it does, sequential steps reduce parallelism.

### 4.4 Expected bottlenecks and what to instrument

A practical instrumentation checklist:

| Concern | What to log (per layer, per step) | Why it matters |
|---|---|---|
| True expert usage | avg #true experts/token; histogram; per-token percentiles | primary compute proxy; needed for iso-compute eval |
| Null usage | null fraction per layer; positions of null in top-k (for take-until-null) | tells whether adaptivity is actually happening |
| Load balance | per-expert token counts; CV/variance; per-device max load | predicts stragglers + overflow |
| Capacity/drop | overflow rate; dropped tokens fraction; “second choice” usage | silent quality regressions |
| Dispatch overhead | bytes sent in all-to-all; time spent packing/unpacking; sort/argsort cost | determines whether skipping compute saves wall time |
| Kernel time breakdown | XLA profile: matmul time vs comm vs permute | tells where to optimize |
| Router stability | router logits mean/var; entropy; z-loss (if present) | routing collapse diagnostics |
| Inference tail latency | p50/p95/p99 token latency under small batch | adaptive routing can hurt tail |

### 4.5 Minimal viable experiments per promising method

| Method | MVP implementation sketch (JAX/TPU) | Key ablations |
|---|---|---|
| Null experts (AdaMoE) | `K_max=4/8`, add `m=K_max` null experts; bypass dispatch for null tokens; AdaMoE null-group load-balance | (m,k) grid; null-group LB vs naive LB; normalize true-only vs all |
| Take-until-null | Sort top-k by probability; stop at first null; dispatch only prefix true experts | compare `independent` vs `take_until_null`; check routing stability |
| Threshold dynamic top-k | compute cumulative prob over sorted experts; choose minimal k meeting τ; mask rest to null | τ sweep; K_max sweep; with/without controller |
| Controller for compute | maintain τ or null-bias to hit target `E_true` | controller speed/stability; iso-compute fairness |
| MoD-style layer bypass | per layer, select subset tokens for MoE FFN; others take residual | budget sweep; which layers allow bypass |
| Ada-K allocator-only fine-tune | freeze base, train allocator modules for k | compare RL vs differentiable relaxation; variance |

---

## 5 Comparison matrix

Ratings: **L/M/H** for risk/complexity; “Perf impact” assumes real dispatch savings.

| Candidate method | Impl complexity | Expected perf impact | Quality risk | Load-balance / stability risk | Observability / debuggability | Fit: training | Fit: inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Null experts + dispatch bypass** (AdaMoE-style) | M | H | M | M | H | H | M–H (depends on batch) |
| **Take-until-null** (prefix stop) | M | M (often) | M | M | M | H | M |
| **Threshold dynamic top-k (Top-P/XMoE)** w/ mask-to-null | M | M–H | M | H | M | H | M |
| **Compute-target controller** (on top of above) | M | M | L | M | H | H | H |
| **MoD-style layer bypass** | H | H | H | M | M | M | H |
| **Expert-choice routing** | H | M | M | L | M | M | L (AR mismatch) |
| **Ada-K allocator (PPO)** | H | M–H | M | H | L–M | M | M |
| **Halting/pondering (ACT/PonderNet)** | H | ? | H | H | L | M | M–H (tail latency risk) |
| **BASE balanced assignment** | H | M | L–M | L | M | H | L |

Interpretation:
- If we want to move fast, **null experts + controller + robust load-balance** is the best “low regret” path.
- If we want bigger wins at inference latency, **layer bypass (MoD-style)** is attractive but higher quality risk.

---

## 6 Proposed experiment plan

This plan is explicitly staged to de-risk quality regressions and to keep the “systems cost” visible.

### Phase 1 — Quick wins (1–2 weeks of focused work)

**Goal:** Establish a clean, reproducible **quality vs compute frontier** using our existing null routing machinery.

1. **Integrate AdaMoE null-group load-balance + true-only normalization** into our null MoE layer.
2. Add **first-class compute metrics** to training logs:
   - `E_true` (avg true experts/token),
   - `null_frac`,
   - per-expert load CV,
   - overflow/drop.
3. Run a **null-fraction sweep** (or `E_true_target` sweep) on a small pretrain proxy (e.g., 300M–1B) and on a representative fine-tune suite.
4. Add a “compute penalty” baseline: `L_task + λ·E_true`, compare vs directly forcing null fraction.

**Metrics to track (exact):**
- Training: loss vs step, tokens/sec, step time breakdown (XLA), MFU/TFLOPs (if available), all-to-all bytes.
- Quality: validation perplexity, downstream accuracy.
- Routing: `E_true` mean/p50/p95, null position distribution, load CV, overflow.

**Ablations:**
- null expert definition: zero vs identity.
- load-balance: naive vs AdaMoE null-group.
- normalization: normalize over true experts only vs all selected.

**Stopping criteria / thresholds:**
- If overflow/drop > 0.1% sustained → adjust capacity factor or routing constraints.
- If perplexity worsens by >0.2–0.3 at matched compute vs baseline top-k → treat as a red flag.
- If tokens/sec gains are <5% at high null fraction in training → likely dispatch is dominating; shift focus to dispatch bypass and layer bypass.

### Phase 2 — Medium-term (2–6 weeks)

**Goal:** Compare **heuristic dynamic-k** vs **null-based adaptivity** under controlled compute budgets.

1. Implement **threshold-based dynamic-k** (Top-P / XMoE-like) with `K_max` fixed and masked-to-null.
2. Add a **compute-target controller** to keep `E_true` near a target automatically:
   - Start with proportional control on a router bias or threshold τ.
3. Add **capacity-aware scheduling** (at minimum: priority by router prob, drop lowest prob when over capacity) if overflow appears.
4. Evaluate iso-compute on:
   - pretraining proxy,
   - “hard reasoning” evaluations where adaptive compute should help (e.g., GSM8K-like, ARC-C).

**Metrics:**
- All Phase 1 metrics.
- Plus: routing confidence stats (max prob, entropy), threshold hits per layer.

**Ablations:**
- threshold τ sweep vs null-fraction sweep at same `E_true`.
- per-layer vs global controller (allow different compute budgets per layer).

**Decision thresholds:**
- Promote method if it achieves **≥10% wall-time improvement at matched perplexity**, or **better perplexity at matched wall time**, with stable load/overflow.

### Phase 3 — High-risk / high-reward (6+ weeks)

**Goal:** Explore methods that may yield larger gains but need more engineering and risk tolerance.

1. **MoD-style layer bypass** in MoE FFNs: allow some tokens to skip the entire MoE layer.
2. **Ada-K allocator-only RL fine-tune** (base model frozen): evaluate whether learned k patterns generalize to our data and whether it transfers across tasks.
3. Explore **ternary compute actions** (TC-MoE-like) if we want richer “negative/zero” expert semantics.

**Key additional metrics:**
- Tail latency (decode p95/p99), variance of per-step time.
- Stability metrics: routing collapse indicators.

**Stop conditions:**
- If training becomes significantly non-deterministic or regressions are hard to diagnose, pause and return to Phase 2 methods.

---

## 7 Bibliography (primary sources only)

### Strong evidence (peer-reviewed or widely-cited, clear experiments)
- Mike Lewis et al. **BASE Layers: Simplifying Training of Large, Sparse Models** (ICML 2021). https://arxiv.org/abs/2103.16716
- William Fedus et al. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (2021). https://arxiv.org/abs/2101.03961
- Yanqi Zhou et al. **Mixture-of-Experts with Expert Choice Routing** (2022). https://arxiv.org/abs/2202.09368
- Hongcheng Gao et al. **AdaMoE: Token-Adaptive Routing with Null Experts for MoE LMs** (Findings EMNLP 2024). https://aclanthology.org/2024.findings-emnlp.361.pdf
- Weijia Huang et al. **Harder Task Needs More Experts: Dynamic Routing in MoE Models** (ACL 2024). https://aclanthology.org/2024.acl-long.380.pdf
- Nianyu Yang et al. **XMoE** (Findings ACL 2024). https://aclanthology.org/2024.findings-acl.170.pdf
- Tianyu Yue et al. **Ada-K Routing Strategy** (ICLR 2025). https://proceedings.iclr.cc/paper_files/paper/2025/file/df22a19686a558e74f038e6277a51f68-Paper-Conference.pdf

### Strong but “non-causal / less directly applicable to AR decoding”
- Yanqi Zhou et al. **Expert Choice Routing** (variable experts per token, but causality concerns). https://arxiv.org/abs/2202.09368

### Promising preprints / newer work (evidence still developing)
- Maciej Kilian et al. **Improving MoE Compute Efficiency by Composing Weight and Data Sparsity** (arXiv 2026). https://arxiv.org/abs/2601.15370
- Roberto Raposo et al. **Mixture of Depths** (arXiv 2024). https://arxiv.org/abs/2404.02258
- Maryam Akhavan Aghdam et al. **DA-MoE: Towards Dynamic Expert Allocation** (arXiv 2024). https://arxiv.org/abs/2409.06669
- (OpenReview) **TC-MoE** (ICLR 2025). https://openreview.net/pdf?id=dsP91M4hDL
- Can Jin et al. **DTop-p MoE** (arXiv 2025). https://arxiv.org/abs/2512.13996
  _Note: I could not access the full text in this environment; treat details beyond the abstract as uncertain._

### Foundational halting / compute-adaptive references
- Alex Graves. **Adaptive Computation Time** (2016). https://arxiv.org/abs/1603.08983
- Andrea Banino et al. **PonderNet** (2021). https://arxiv.org/abs/2107.05407

---

## Recommended next 5 experiments (tailored to our current null-routing / take-until-null work)

1. **Quality-vs-compute curve for null routing**: run an `E_true_target` sweep (or null-fraction sweep) and report **perplexity vs tokens/sec** and **perplexity vs all-to-all bytes**.
2. **AdaMoE null-group load-balance**: implement the null-group modification and compare to naive balancing on the same compute targets.
3. **True-expert-only weight normalization**: evaluate stability and quality impact (especially under high null usage).
4. **Threshold dynamic-k (Top-P / XMoE-like) masked-to-null**: compare to null-fraction routing at matched `E_true` and matched wall time.
5. **Layer-level bypass prototype (MoD-style)**: allow tokens to skip MoE FFN in a subset of layers under a fixed budget; compare iso-walltime.

