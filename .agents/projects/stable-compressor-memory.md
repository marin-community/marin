# Stable-compressor memory: a fixed-encoder, sum-friendly memory the LM reads

Status: research deep dive for Marin Ladder Climb v1 (#9451), 2026-09-27. Branch `lc1-compressor` (based on
`5eb9a5dd69`). The code is a default-off flag set on `experiments/grug/fast_track`, plus a CPU toy in
`experiments/grug/fast_track/compressor_proto/`. No GPU run has happened yet; section 6 has the launch specs.

## TL;DR

- **The idea, made precise.** Take a softmax readout `W` on features `φ(x)`. The gradient of its cross-entropy
  over a corpus is `Σ φ(x)(p_W(x) − e_y)ᵀ`. That sum has two halves:
  - The **data half**, `M = Σ φ(x) e_yᵀ`, depends only on the data once `φ` is fixed. It can be summed forever
    and never goes stale.
  - The **model half**, `Σ φ(x) p_W(x)ᵀ`, is the stale part.

  Momentum stores both halves under a `φ` that keeps moving, so it goes stale. A *stable compressor* fixes `φ`
  and keeps only the data half. In the linear-feature case, that sum is the kernel's sufficient statistic: for
  squared loss, `M` together with `C = Σ φφᵀ` gives the exact ridge solution. It is also a kernel conditional
  mean embedding. With one-hot hashed n-gram features, `M` is simply a table of next-token counts.
- **The concrete system (recommended).** Keep one hashed table per n-gram order, 2–4. Each row accumulates
  `[Σ code(next token), count]`, where `code` is a fixed random `[vocab, 64]` matrix.
  - The table is **never trained**. After each step, the trainer scatter-adds the batch into it (write after
    read).
  - Optionally, before step 0, it is **pre-filled** from tokens the run never trains on. This is untimed and
    costs a counting pass, not a training run.
  - The model reads `[mean code, log count]` of every order through a small learned MLP reader. The result is
    an AttnRes source next to the existing trained bigram table.
  - Gradients flow into the reader only. The memory has no optimizer state and cannot go stale.
- **Toy results** (CPU, 4-layer d128, PG-19 books, held-out books):
  - **Null: the fixed tables do not replace an SGD-trained bigram table.** At equal data, stat-only is 0.11 to
    0.13 worse than the learned table.
  - **They complement it, early.** At 400 steps (3.3M tokens), learned bigram + stat 2/3/4 beats learned 2/3/4
    tables by −0.047 / −0.034 (2 seeds) and learned bigram alone by −0.068 / −0.044. At d256 × 6 layers the
    gain over learned bigram is still −0.051.
  - Pre-filling the stat tables from tokens the run never trains on adds a further −0.005 / −0.012 / −0.023 at
    0.6× / 1.2× / 2.4× the run's tokens.
  - **Null: the same-data gain fades with training length.** Against learned 2/3/4, it is −0.047 at 400 steps,
    −0.017 at 700 and −0.005 at 1000 (8.2M tokens, seed 0). The learned tables catch up once each row has seen
    enough data.
  - The pre-fill gain also shrinks, but slower: at 700 steps with 1× extra tokens it is still −0.026 in total.
  - So in the toy the memory is mainly (a) an accelerator of early learning and (b) a way to use data the model
    never trains on.
- **Falsifiable d512 prediction.** d512 sees about 100× more tokens per table row than the toy's 1000-step run.
  - Online (same data): the toy trend predicts |Δ| < 0.003 at equal steps, i.e. no win.
  - Pre-filled from 2× the run's tokens: Δ ≤ −0.005.
  - From 8×: a larger gain than 2× (roughly log-linear in pre-fill tokens).
  - If pre-fill 8× gives < 0.005, the stable memory does not pay at this scale, and the idea's remaining value is
    capacity per byte at hero scale (section 3A).

## 1. The idea, formalized

### 1.1 Gradients as document encodings, momentum as an average

The step is `θ_{t+1} = θ_t − η·m_t` with `m_t = Σ_s β^{t−s} g(x_s; θ_s)`. Each `g(x; θ)` is an encoding of
document `x` in weight space. Averaging is meaningful because the loss is an expectation over documents:
`∇L(θ) = E_x g(x; θ)`, a linear functional of the data distribution. The encoding is only valid at the `θ` it
was computed at, though. `g(x_s; θ_s) ≠ g(x_s; θ_t)`, and the gap grows with `‖θ_t − θ_s‖` and with how
nonlinear the network is.

Staleness in the current recipe is modest. Candidate 6's Adam β1 is 0.8, a horizon of about 5 steps. AdEMAMix
β3 = 0.98 gives about 50 steps, and adopting it was worth −0.004 at d512. The AdEMAMix paper keeps gradients
useful for about 10k steps when the slow EMA is ramped in carefully. So old encodings are *not* worthless.
They are just not exact.

### 1.2 What "retains meaning under averaging" requires

A sum of encodings `Σ_x E(x)` is a sufficient statistic for the training signal exactly when that signal is
linear in `E(x)` with `E` fixed. Three cases:

1. **Linear readout on fixed features** (the kernel / NTK / lazy regime). For `f(x) = Wᵀφ(x)` and squared
   loss, `C = Σ φφᵀ` and `M = Σ φ yᵀ` give the exact optimum `W* = (C + λI)⁻¹ M`. That is kernel ridge
   regression, and `M` evaluated at a query is the conditional mean embedding of `p(y | x)` (Song et al. 2009;
   Muandet et al. 2016 review, 1605.09522).
   - For cross-entropy, `M` is still the exact *data half* of the gradient, `∇_W = Σ φ(p_W − e_y)ᵀ`. The
     model half needs the current predictions.
   - With `φ` = one-hot of a hashed n-gram, `M` is the count table. The count table is the full sufficient
     statistic of an n-gram model's likelihood.
2. **Linear sketches** of either of the above (count-sketch, random projections; FetchSGD 2007.07682 keeps
   momentum in sketch space). Sums of sketches are sketches of sums. Here the "code" of the next token is a
   random `r`-dim projection of its one-hot, so a row holds a JL sketch of the next-token distribution.
3. **Gradients of a frozen model** (neural tangent features, TRAK 2303.14186). Sums are exact at the anchor
   `θ₀`, but they only describe the model's behaviour near `θ₀`. Pretraining moves far from any anchor, which
   is why TRAK has to ensemble checkpoints. SVRG/SAG-style stored gradients fail on deep nets for the same
   reason (Defazio & Bottou 1812.04529).

So the stable part of the per-document encoding is the part that depends only on the data (`φ(x) ⊗ e_y`). The
stale part is the model's own prediction (`p_θ`) and its moving feature map (`J_θ`).

**Design principle.** Put the stable part in a non-parametric memory that is only ever summed. Let a small
learned reader, which is ordinary weights trained by ordinary SGD, turn sufficient statistics into whatever the
network needs. The reader is shared across all rows. It amortizes the "many GD steps from these statistics"
that a trained table performs row by row. It is effectively a hypernetwork from statistics to embeddings.

### 1.3 Capacity

- **Dense superposed memory.** `M = Σ k vᵀ` with `k ∈ R^D` holds about `D` associations before crosstalk
  dominates. Examples are linear attention and fast weights (Schlag et al. 2102.11174), and classical Hopfield
  at 0.14·D. A corpus-level dense memory therefore needs either a ridge/delta-rule read, which is the
  normal-equation solve `C⁻¹M`, or very high-dimensional sparse keys.
- **Sparse addressing** (hashing, Kanerva SDM, product keys). Capacity scales with the number of rows, and
  collisions cost ≈ (distinct contexts)/(rows). At d512, 1.5B tokens and a 16k vocab, there are O(10⁷)
  distinct bigrams and O(10⁸) 3- and 4-grams. The 524k-row learned bigram table therefore averages about 50
  bigrams per row. A statistic row is a *frequency-weighted* mixture, so heavy hitters dominate. That is fine
  for frequent contexts and noise for rare ones. Hence 2M rows per order in the GPU spec.
- **Stat vs trained per row.** A stat row costs `r+1` floats and no optimizer state. A trained row costs
  `hidden` floats plus 2–3× for Adam and AdEMAMix. At equal HBM a stat table can have about 20× the rows.

### 1.4 What the model can gain over just training on the data

1. **One-shot exact writes.** A context seen once gets its exact statistic immediately. A trained row seen
   once gets one Adam step of size `lr`.
   - This matters most for rare, high-order contexts, which trained tables handle badly.
   - In the toy, stat tables gain most at orders 3–4. Learned 3–4 tables add only −0.02 over a learned
     bigram.
2. **No staleness.** Row `h` means the same thing at step 10 and step 10⁵. A trained row's meaning drifts
   with the head and residual stream that read it.
3. **More tokens than the forward pass sees.** The memory can be filled from data the run never trains on
   (untimed pre-fill), at the cost of a counting pass (~1 ms/batch on GPU, or on CPU hosts). This is the
   literal "holds much more data" property.
4. **Leave-one-out for free.** Memory built over the *training* corpus leaks each position's own target into
   its row. Because rows are sums, training can subtract the position's own contribution exactly: read
   `(S − code(y_t), n − 1)`. That lets a hero run use a full-corpus memory without leakage. Not needed for the
   d512 test, whose pre-fill is disjoint from the trained tokens.

What it cannot do: learn features. The keys, which are hashed n-grams, are fixed by construction. Everything
beyond the memory's fixed feature space still comes from ordinary training. The toy says the two are
complements: learned bigram rows carry representational content the statistics don't.

## 2. Literature map

| work | what it shows | implication here |
|---|---|---|
| Engram, 2601.07372 (DeepSeek 2026) | Hashed 2/3-gram tables trained by Adam at 5× LR, with a content gate. Beat iso-FLOP MoE at 27B; the best split puts 20–25% of sparse parameters in the tables; host-DRAM offload costs ≤ 2.8% | Trained-table baseline to beat; it is already in candidate 6 (bigram ×10 LR, rank-16 gate). Deterministic addressing means stat tables can offload the same way |
| Memory Grafting, 2605.20948 (2026) | Frozen hidden states of a separate pretrained model as table values for frequent n-grams, read by light projections. Beat Engram at 0.9B and 2.8B | Closest prior work: frozen/precomputed values beat SGD values. Ours is the cheapest version (count statistics, no teacher) |
| TF-Engram 2607.07388; NGM 2605.16893 | Train-free n-gram memories help post hoc on frozen models | Small evidence that fixed memories carry signal |
| SCONE 2502.01637; Over-tokenized Transformer 2501.16975; N-grammer 2207.06366 | n-gram / input-vocabulary scaling improves log-linearly | Learned n-gram rows add *representation*, not just statistics; consistent with the toy's learned ≫ stat-only |
| "Can Transformers Learn n-gram LMs?" 2410.03001 | Transformers do worse than count estimators on arbitrary n-gram LMs | Argues for giving counts as input |
| Infini-gram 2401.17377; kNN-LM 1911.00172; RETRO 2112.04426; residual n-gram LM 2210.14431 | Corpus statistics and retrieval still hold information large LMs lack; kNN-LM stores frozen hidden states | Output-side interpolation is an alternative reader; we read on the input side so the network can compose |
| Linear transformers = fast weights, 2102.11174; Modern Hopfield 2008.02217; Kanerva 1804.01756 | Additive outer-product memory saturates at ~key dim; delta rule / sharp read / sparse addresses fix it | Dense corpus memory (design B) needs a ridge/delta read or sparse keys |
| TTT 2407.04620; Titans 2501.00663 | Memory = a model trained by gradient steps as tokens arrive, per sequence, with *learned* key maps | Per-sequence and not stable across the corpus; ours is the corpus-level, fixed-key limit |
| Kernel mean embeddings 1605.09522; conditional mean embeddings (Song 2009) | `μ_{Y|x} = Φ_Y (K + nλI)⁻¹ k(X, x)` = ridge regression on fixed features | Formal statement of "averages that retain meaning" |
| TRAK 2303.14186; linearized nets 2103.01439; eNTK for fine-tuning 2210.05643 | Frozen-gradient features work near a fixed θ and need checkpoint ensembles otherwise | Gradients of a frozen snapshot are a poor stable compressor for pretraining |
| SVRG ineffective 1812.04529; SAG/SAGA | Stored per-example gradients go stale on deep nets | Negative evidence for design C |
| AdEMAMix 2409.03137; SNOO 2510.15830; DiLoCo 2311.08105; Lookahead 1907.08610 | Very old or coarse gradient information still helps (10k-step EMAs; K-step outer Nesterov with 1.5–2.5× compute gains to 1e23) | Staleness is tolerable when damped; the optimizer route is already partly harvested (AdEMAMix adopted) |
| FetchSGD 2007.07682 | Count-sketched gradients sum across workers, with momentum in sketch space | Linear sketches keep sums meaningful |
| Model soups 2203.05482; task arithmetic 2212.04089 | Weight-space averages are meaningful within one basin | The same condition that bounds momentum staleness |
| Cartridges 2506.06266; Text-to-LoRA 2506.06105; Doc-to-LoRA 2602.15902 | Documents → KV prefixes or LoRA deltas via self-study or hypernetworks | Per-document weight-space encoders tied to a *frozen* base; stale if the base trains |
| Dataset distillation 1811.10959 | Learned corpus summaries in gradient space need bilevel unrolling and have not scaled to LM pretraining | Negative signal for "learned average document" designs |
| BYOL 2006.07733; Mean Teacher 1703.01780 | A slowly moving EMA target is stable enough to regress onto | Middle ground between a frozen and a live encoder (design B′) |

The literature scout found no paper that feeds raw count statistics (sums of fixed next-token codes plus counts)
into a pretraining transformer as an input source. Memory Grafting's frozen values come from a pretrained
teacher model, not from counts.

## 3. Candidate systems

The speedrun rules for all three: the 8-minute clock counts train steps only. Compile, eval, startup and
anything before step 0 are untimed, and compressor training is untimed by the owner's rule.

### A. Statistic n-gram memory (recommended; implemented)

- **Compressor.** A fixed hash of the last `k` tokens, for `k ∈ {2, 3, 4}`, plus a fixed next-token code
  `code ∈ R^{V×64}` (seeded Gaussian, `/√64`). Nothing is trained.
  - The toy also tried a PMI-SVD code fitted once on held-aside text, which is an "untimed compressor". It was
    better than random by 0.08 online and 0.08 with pre-fill. On GPU v1 uses the random code; a fitted code is
    follow-up (section 7).
- **Memory.** `[len(orders) × rows, 65]` fp32 per order block, holding sums of `code(next)` and counts.
  - Update rule: after each optimizer step, each position adds `[code(y_{t+1}), 1]` times its loss weight.
    Every GPU all-gathers the batch's row ids, next tokens and weights (about 10 MB) and scatter-adds the whole
    batch into its replicated table. No table-sized collective runs.
  - Optional untimed pre-fill from `N` batches of the stream *after* the run's last step.
  - Sums are exact and order-independent, so the memory never goes stale.
- **Reader.** For each position, gather its row for every order and form `[S/max(n,1), log1p(n)/4]` per order.
  Then a GELU MLP (195 → 512 → 512), RMSNorm, and an Engram scalar content gate. The result is one AttnRes
  source.
  - Gradient flows into the MLP, norm and gate. The table is `stop_gradient` and sits in the optimizer's
    `frozen` group (`optax.set_to_zero`, no state).
- **Timed cost (estimate).** Gather 3×65 floats per token; scatter 1.6M rows per step; a (195→512→512) MLP per
  token; a 10 MB all-gather. That is about 1–2 ms of a ~175 ms step, and **no** Adam or AdEMAMix state (a
  trained table has 3×).
  - The table stays fp32 in the compute copy (`_cast_to_compute`), so there is no per-step bf16 cast. The weight
    EMA keeps the live table instead of blending it.
- **Falsifiable prediction.** At d512 equal steps on candidate 6:
  1. Online stat 2/3/4 gives |Δ| < 0.003, a null, extrapolating the toy's decay with tokens per row.
  2. Pre-fill 2× gives Δ ≤ −0.005.
  3. Pre-fill 8× beats pre-fill 2×.
  4. Replacing the trained bigram table by the pre-filled stat tables is worse than keeping both. The toy says
     the stat-only memory is much worse than a learned table.
- **Scaling to 1e24.**
  - n-gram memory already transfers well on this ladder: the learned bigram kept 109–138% of its log-k at
    d768.
  - Engram reports gains at 27B.
  - The stable version's extra lever at scale is pre-filling from the whole corpus with leave-one-out
    subtraction, plus host-memory offload, since addressing is known from token ids before the forward pass.
    Tables of 10⁹ rows are plausible in host DRAM.
  - The risk, which the toy already shows: the same-data gain fades once the learned parameters have seen each
    context enough times. At 1e24, the durable value is then (i) the long tail of contexts too rare to learn,
    (ii) statistics from more data than the run trains on, and (iii) memory capacity per byte. A stat row is
    65 floats with no optimizer state; a trained row is `hidden` floats plus 2–3× for optimizer state.
  - The d768 and d1024 retention check is the gate.

### B. Frozen-key corpus fast-weight memory (dense generalisation of A)

- **Compressor.** A frozen small encoder `f₀`, trained once and untimed; for example the layer-2 residual of a
  d512 run or a tiny LM. It maps each context to a key `k = ψ(f₀(context))`, where `ψ` is a sparse top-k random
  feature map. That is locality-sensitive, so *similar* contexts share rows, not only exact n-grams.
- **Memory.** `M = Σ k ⊗ code(y)` and `C = Σ k kᵀ` (or diagonal counts for sparse `ψ`), accumulated over any
  number of tokens. Read `W* = (C + λ)⁻¹ M`, the conditional mean embedding.
- **Reader.** Query with the same frozen `f₀` on the current context. This is an extra forward of `f₀` per
  token, which is timed unless keys for training data are precomputed offline. A learned MLP turns
  `W*ᵀψ(f₀(x))` into an AttnRes source.
- **Pros.** It generalises across paraphrases, and `f₀` can be tiny.
- **Cons.**
  - Timed `f₀` FLOPs.
  - Precomputing keys means storing per-token keys for the run's data.
  - Gains risk being plain distillation from `f₀`, so a control reading `f₀`'s hidden state directly is needed.
  - Dense keys saturate at key dimension (section 1.3).
- Not prototyped. Design A with hash addressing is the `f₀ = identity on the last k tokens` special case.
- **B′ variant.** `f₀` is an EMA of the main model's own early layers, refreshed rarely. The memory is rebuilt
  on refresh, which costs a pass and brings staleness back in a controlled way.

### C. Anchored gradient sketch (the optimizer route; not recommended)

- **Compressor.** Gradients at a frozen anchor `θ_a`, count-sketched to `D` dims (FetchSGD-style), summed over
  many more documents than a step sees. This is computed untimed at the anchor, for example over the next 10×
  of data.
- **Use.** A long-horizon control variate `g̃ = g(x; θ) − g(x; θ_a) + μ_a`, which is SVRG. Or a slow momentum
  term re-anchored every K steps, which is SNOO/Lookahead-like.
- **Why not.**
  - SVRG gives no benefit on deep nets (1812.04529). The correction term needs a second forward and backward
    at the anchor, which is timed.
  - AdEMAMix and SNOO already capture most of the value of long-horizon gradient memory within the optimizer.
  - A frozen-anchor gradient sum is only valid near `θ_a`: it is the same staleness, moved elsewhere.

## 4. Prototype (CPU)

`experiments/grug/fast_track/compressor_proto/`:

- `prep_pg19.py` tokenizes the locally cached PG-19 test split. It trains a 4096 BPE on the 60 train books and
  writes book-disjoint streams: 8.7M train, 2.6M extra and 1.2M eval tokens.
- `proto.py` is a 4-layer d128 transformer (QK-norm, GELU MLP, Adam 3e-3 with cosine decay), trained for 400
  steps × 32 × 256 = 3.3M tokens.
  - Eval is on 256 windows from 10 held-out books.
  - `--learned-orders` adds Adam-trained hashed tables of 262k rows × d at LR ×10 (×30 is the same, ×1 is much
    worse). Each is added to the input and to the pre-head stream through zero-init gates.
  - `--stat-orders` adds the fixed statistic tables of design A (262k rows per order, code dim 128). They are
    read by a linear or MLP reader into the same two places.
  - `--prefill extra` fills the stat tables before step 0 from the 8.0M tokens the run never trains on: the
    unused train windows plus the extra books.

Held-out loss after 400 steps (lower is better; seed 0 / seed 1 where run):

| # | memory | data for memory | eval loss | Δ vs learned bigram |
|---|---|---|---|---|
| 1 | none | — | 5.029 / 4.999 | +0.437 / +0.434 |
| 2 | learned bigram, LR ×10 (≈ candidate 6's table) | run | **4.592 / 4.565** | 0 |
| 3 | learned bigram, LR ×1 | run | 4.777 | +0.185 |
| 4 | learned bigram, LR ×30 | run | 4.598 | +0.006 |
| 5 | learned 2, 3, 4 | run | 4.571 / 4.556 | −0.021 / −0.009 |
| 6 | stat bigram, random code, linear reader | run (online) | 4.831 | +0.239 |
| 7 | stat bigram, PMI code, linear | run (online) | 4.756 | +0.164 |
| 8 | stat bigram, PMI, MLP reader | run (online) | 4.718 | +0.126 |
| 9 | stat bigram, random, linear | + 8.0M pre-fill | 4.784 | +0.192 |
| 10 | stat bigram, PMI, linear | + pre-fill | 4.705 | +0.113 |
| 11 | stat bigram, PMI, MLP | + pre-fill | 4.698 | +0.106 |
| 12 | stat 2, 3, 4, PMI, MLP | + pre-fill | 4.643 | +0.051 |
| 13 | stat 2, 3, 4, 6, PMI, MLP | + pre-fill | 4.637 | +0.045 |
| 14 | stat 2, 3, 4, horizon-4 code (next 4 tokens) | run (online) | 4.725 | (null: no better than next-token) |
| 15 | **learned bigram + stat 2, 3, 4** (PMI, MLP) | run (online) | **4.524 / 4.521** | **−0.068 / −0.044** |
| 16 | learned bigram + stat 2, 3, 4, horizon-4 | run (online) | 4.533 | −0.059 |
| 17 | learned bigram + stat bigram | + pre-fill | 4.539 | −0.053 |
| 18 | **learned bigram + stat 2, 3, 4** | + pre-fill | **4.501 / 4.495** | **−0.091 / −0.070** |
| 19 | learned bigram + stat bigram | run (online) | 4.553 | −0.039 |
| 20 | learned bigram + stat 3, 4 | run (online) | 4.549 | −0.043 |
| 21 | learned bigram + stat 2, 3, 4 | + 25% of pre-fill (2.0M) | 4.519 | −0.073 |
| 22 | learned bigram + stat 2, 3, 4 | + 50% of pre-fill (4.0M) | 4.512 | −0.080 |

Model size (400 steps, 6 layers, d256, seed 0): none 4.791, learned bigram 4.398, learned bigram + stat 2/3/4
online **4.347 (−0.051)**. The d128 gain was −0.068, so most of it survives a ~4× larger model.

Training length (seed 0; the cosine schedule spans each run; Δ vs learned 2/3/4 at the same step):

| run length | tokens | learned 2, 3, 4 | + stat 2, 3, 4 online | Δ online | + stat 2, 3, 4 pre-filled | Δ pre-filled |
|---|---|---|---|---|---|---|
| 400 steps | 3.3M | 4.571 | 4.524 | −0.047 | 4.501 (2.4× extra) | −0.070 |
| 700 steps | 5.7M | 4.355 | 4.338 | −0.017 | 4.329 (1.0× extra) | −0.026 |
| 1000 steps | 8.2M | 4.238 | 4.233 | −0.005 | — (only 0.4× extra left) | — |

For reference, no memory at 1000 steps is 4.642.

Within each run the gap is largest early. In the 1000-step run it is −0.124 at step 100, −0.014 at 400 and
−0.005 at 1000. In the 400-step runs it locks in at about −0.046 from step 250 on, as the LR decays.

Reading the table:

- **Seed noise.** Unpaired, the two seeds differ by up to 0.03 (row 1). Paired differences (same seed and
  data order) are much tighter: row 15 − row 5 is −0.047 / −0.034, and row 18 − row 15 is −0.023 / −0.026.
- **Stat-only does not replace a trained table** (rows 8 and 11 vs row 2, +0.11 to +0.13). A trained row is a
  free `d`-dim vector for its n-gram: vocabulary expansion (Over-tokenized, SCONE). A statistic row only
  describes what follows the n-gram, and a reader shared across rows can't recover per-row representations
  from that. This is the main null result for the "stable encoder replaces training" reading of the idea.
- **Stat complements a trained table, at the same data, early in training** (row 15 vs rows 2 and 5). Adding
  fixed tables of orders 2–4 beats adding *trained* tables of orders 3–4: learned 3–4 add −0.02 / −0.01, while
  stat 2–4 add −0.07 / −0.04. That is the one-shot-write advantage on rare contexts. Even a stat table of the
  *same* order as the trained one helps (row 19, −0.039), so it is not only extra orders.
- **The same-data advantage fades as the learned tables saturate** (training-length table). This matches the
  mechanism: a stat row is exact after one visit, but a trained row catches up after tens of visits.
  - The toy's 1000-step run has about 31 tokens per learned row.
  - d512 has about 2800 tokens per bigram row (1.48B tokens, 524k rows).
  - So at d512 the online gain from frequent contexts is expected to be gone. Only the tail of rare 3/4-gram
    contexts remains.
- **More data in the memory helps, monotonically** (rows 21, 22, 18: −0.005, −0.012, −0.023 at 0.6×, 1.2× and
  2.4× extra tokens). The model reads statistics from more tokens than it trained on, at no timed cost. This is
  the one lever SGD cannot copy within the budget.
- The fixed-code choice matters (random vs PMI, −0.08). The reader shape matters somewhat (MLP vs linear,
  −0.04 online, −0.01 with pre-fill). A multi-token horizon code does not help (rows 14 and 16).

## 5. GPU implementation (`lc1-compressor`)

Everything is default-off.

- `GrugModelConfig` fields:
  - `ngram_stat_rows`: rows per order, 0 = off.
  - `ngram_stat_orders`, default `(2,)`.
  - `ngram_stat_dim`, default 64.
  - `ngram_stat_mlp_dim`: 0 means a linear reader.
  - `ngram_stat_gate`, default true.
- Model leaves: `ngram_stat_table` and `ngram_stat_code` (both optimizer group `frozen`), `ngram_stat_hidden`
  and `ngram_stat_up` (MuonH, like `embed2_up`), `ngram_stat_norm`, and `ngram_stat_gate_{w,b}` (Adam). The
  source goes last in the AttnRes extra-source list.
- `write_ngram_stats(model, tokens, loss_weight, segment_ids)` runs in `train_step` after the optimizer update.
  Document starts use the same sentinel as the trained bigram hash.
- `GrugTrainerConfig.ngram_stat_prefill_batches` / `--ngram-stat-prefill-batches N` writes `N` batches from
  `train_loader.iter_from_step(num_train_steps)` before step 0. These are tokens the run never trains on.
  Skipped on resume, since the table is in the checkpoint.
- `_cast_to_compute` keeps the table fp32 in the compute copy, in both train and eval. The weight EMA carries the
  live table.
- Tests: `experiments/grug/fast_track/test_ngram_stat.py`.
  - Writes are exact per-order sums of `code(next)`.
  - The model stays causal.
  - The table gets zero gradient and routes to `frozen`, while the reader trains.
  - A 4-device CPU mesh (data 2 × expert 2) writes the same table as one device.
  - A local `_make_train_step` / `_prefill_ngram_stats` smoke confirmed one write per step and an EMA that carries
    the live table.
- Not tested on GPU: step-time cost, the loader's pre-fill throughput, and memory. The d512 settings add
  2M × 3 × 65 × 4 B = 1.6 GB per GPU of table.
- Note: W&B `parameter_count` includes the 409M table entries.

## 6. d512 experiment plan (ready to launch)

- **Base:** candidate 6 (top-8) = `lc1-b83-base7` flags (3.0072 / 3.0109 at 2817 steps).
- **Commit:** branch `lc1-compressor`, code at `f601f20470`. It is based on `5eb9a5dd69`, the commit
  batch 87 ran on. The new flags are default-off, so base7 is a valid pair; an optional same-commit base pair is
  listed below.
- **Launch:** from the lc1 worktree with this branch checked out: `git -C /Users/larry/marin_wt_lc1 checkout
  lc1-compressor`, then `submit.sh` as usual. Equal steps (2817), 2 seeds each.

```bash
BASE="--model-set qk_mult=4.0 --model-set learnable_qk_mult=true --opt-set beta1=0.8 --model-set embed_gated_norm=false --model-set final_gated_norm=false --num-steps 2817 --model-set logit_soft_cap=10 --model-set second_embed=true --model-set second_embed_bigram=true --model-set embed2_rows=524288 --opt-set embed2_lr_mult=10 --model-set embed2_fsdp=true --model-set bigram_gate=true --model-set bigram_gate_rank=16 --model-set mla_key_offset=true --model-set mla_k_norm=true --model-set moe_ungated_relu2=true --model-set moe_ungated_kernel=true --model-set shared_ungated_relu2=true --model-set intermediate_dim=384 --model-set shared_expert_intermediate_dim=384 --opt-set muon_pre_norm=in --opt-set muon_bimaxwell=true --opt-set muonh_decay_power=0.7 --ema-beta 0.995 --ema-last-steps 1000 --ema-blend 0 --ema-blend 0.5 --ema-blend 0.75 --model-set mla_q_norm=true --opt-set adam_ademamix_alpha=5.0 --opt-set adam_ademamix_beta3=0.98"
STAT="--model-set ngram_stat_rows=2097152 --model-set ngram_stat_orders=2,3,4 --model-set ngram_stat_dim=64 --model-set ngram_stat_mlp_dim=512"
# BASE without the trained bigram table (for the replacement run S4)
NOE2=$(echo "$BASE" | sed -e 's/--model-set second_embed=true //' -e 's/--model-set second_embed_bigram=true //' \
  -e 's/--model-set embed2_rows=524288 //' -e 's/--opt-set embed2_lr_mult=10 //' -e 's/--model-set embed2_fsdp=true //' \
  -e 's/--model-set bigram_gate=true //' -e 's/--model-set bigram_gate_rank=16 //')
S=/private/tmp/claude-501/-Users-larry-marin/9e6a1ea3-cd87-457d-8f8e-b6ae301f2bee/scratchpad/submit.sh
# S0 (throughput probe first: 80 steps, no eval)
bash -c "$S lc1-scm-probe d512 $BASE $STAT --num-steps 80 --no-eval"
# S1: online stat 2/3/4 on top of the trained bigram (same data as the base)
bash -c "$S lc1-scm-s234 d512 $BASE $STAT"
bash -c "$S lc1-scm-s234-s1 d512 $BASE $STAT --seed 1"
# S2: + untimed pre-fill from 2x the run's tokens (5634 batches the run never trains on)
bash -c "$S lc1-scm-s234-p2x d512 $BASE $STAT --ngram-stat-prefill-batches 5634"
bash -c "$S lc1-scm-s234-p2x-s1 d512 $BASE $STAT --ngram-stat-prefill-batches 5634 --seed 1"
# S3: isolation, stat bigram only (same order as the trained table), online
bash -c "$S lc1-scm-s2 d512 $BASE --model-set ngram_stat_rows=2097152 --model-set ngram_stat_orders=2 --model-set ngram_stat_dim=64 --model-set ngram_stat_mlp_dim=512"
bash -c "$S lc1-scm-s2-s1 d512 $BASE --model-set ngram_stat_rows=2097152 --model-set ngram_stat_orders=2 --model-set ngram_stat_dim=64 --model-set ngram_stat_mlp_dim=512 --seed 1"
# S4: replacement, stat 2/3/4 pre-filled, no trained bigram table (tests stat-only + the table's speed cost)
bash -c "$S lc1-scm-s234-p2x-noe2 d512 $NOE2 $STAT --ngram-stat-prefill-batches 5634"
bash -c "$S lc1-scm-s234-p2x-noe2-s1 d512 $NOE2 $STAT --ngram-stat-prefill-batches 5634 --seed 1"
# S5: memory-size scaling, 8x pre-fill (~11.8B tokens the run never trains on)
bash -c "$S lc1-scm-s234-p8x d512 $BASE $STAT --ngram-stat-prefill-batches 22536"
bash -c "$S lc1-scm-s234-p8x-s1 d512 $BASE $STAT --ngram-stat-prefill-batches 22536 --seed 1"
# Optional: a same-commit base pair
bash -c "$S lc1-scm-base d512 $BASE"
bash -c "$S lc1-scm-base-s1 d512 $BASE --seed 1"
```

What to read off each run:

- **S0.**
  - Median step time vs base (~175 ms). The fixed-time rule: 1% step time ≈ 0.0017 loss.
  - The log line `ngram stat prefill done` does not apply to S0. For S2, read that line for pre-fill wall time
    and the fraction of rows filled.
  - Peak memory (`log_device_memory`).
- **S1 vs base7.** The same-data test of the idea. **Prediction: |Δ| < 0.003 (null)**, because the toy's
  same-data gain fades with tokens per row. A surprise win here would mean the rare 3/4-gram tail matters more at
  16k vocab than in the toy. A win at fixed time needs Δ < −0.0017 × (% slowdown).
- **S2 vs base7.** The extra-data test. **Prediction: Δ ≤ −0.005**, and S5 (8×) lower than S2.
  - This needs a rules call before it can count as a speedrun win. It reads 2× more of the (fixed) training
    data, untimed.
  - Without that call, it is still the measurement of "a memory that holds more data than the model has
    seen".
- **S3.** If S3 ≈ S1, the gain is the stable half at the same order. If S3 ≪ S1 in gain, it is the one-shot
  higher-order rows.
- **S4 vs base7.** The toy predicts a clear loss. The trained bigram table carries representation the
  statistics don't. A tie or win would mean the trained table's value at d512 was mostly statistics, and would
  free its −5% step time.
- **Watch in W&B.**
  - The AttnRes weights of the new source (the last extra source).
  - `attn_res_*` gate means; there is no stat-gate metric yet.
  - Train-vs-eval gap. Pre-fill should *not* widen it, because the pre-fill tokens are disjoint from the
    trained ones.

If S1 passes, the retention check is the same pair at d768 (baseline steps), as for every ladder ML change.

## 7. Open questions

1. **Rules.** Is an untimed pre-fill from more of the fixed training data allowed? It is a counting pass, not a
   training run, but it reads tokens beyond the fixed-time budget. The online variant (S1) is unambiguously
   legal.
2. **Fitted code.** The toy's PMI-SVD code beat a random code by 0.08. On GPU, the analogue is an SVD of the
   16k×16k bigram PMI from the pre-fill stream (randomized SVD, untimed). Alternatively, the lm_head of a
   finished d512 run: a "compressor trained outside the timed run", which is closest to the owner's framing but
   is a (tiny) teacher.
3. **Output-side reader.** kNN-LM / residual-n-gram-style logit interpolation from the statistics, in addition to
   the input-side source.
4. **Collisions.** Multi-head hashing (Engram) per order, so the reader can detect a collision when heads
   disagree, at the same total rows.
5. **Leave-one-out full-corpus memory** for hero runs, and host-DRAM offload of 10⁸–10⁹-row tables.
6. **Design B.** Soft (LSH) addresses from a frozen tiny encoder, to generalise past exact n-gram matches.
   Needs a distillation control.
7. **Why complementarity?** A probe: does the learned bigram row become more "representational" when stat
   tables are present? Measure its cosine to the reader's output for the same bigram over training.
