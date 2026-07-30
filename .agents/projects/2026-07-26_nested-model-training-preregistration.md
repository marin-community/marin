# Nested model training: research brief and preregistration

Status: revision 3; reviewed by Claude Fable

Experiment series: `NEST-MOE`

Date: 2026-07-26

Deadline: 2026-07-27 09:00 UTC

## TL;DR

The core idea has a strong dense-model precedent. Google trained the Gemma 3n
E2B submodel inside E4B with MatFormer and shipped both extracted checkpoints.
The original MatFormer result also found that nested 582M–850M decoder variants
matched or beat independently trained models. This is the most direct evidence
that a useful smaller model can be co-trained inside a larger model.

For Marin's 300B–700B MoE decision, the lowest-cost test is not a masked dense
matrix. A masked full-width forward still pays for the full matrix, while a
separate small forward has a different hidden-state trajectory and adds roughly
the small model's FLOPs. Instead, we can nest the *expert bank*. A fixed,
evenly interleaved subset of experts forms the small model; the full pool forms
the large model. Each training sequence is assigned to the subset or full
branch and keeps that assignment through every layer. Both branches run
together in one normal top-k MoE batch. Interleaving avoids concentrating the
small branch on only some expert-parallel ranks. The small checkpoint selects
and compacts the subset's expert weights and router columns.

The primary test compares four runs: conventional 256-expert large,
conventional 128-expert small, and 256-expert models with 25% or 50% of
sequences restricted to an interleaved set of 128 experts. All use the same top-k and
active width. The nested arms therefore target less than 10% measured training
overhead, not the roughly 100% cost of training a second model. We promote only
if the full model remains within 0.01 Paloma macro loss of the large control,
the extracted small model remains within 0.03 of the small control, and
throughput remains at least 90% of the large control at fixed tokens and
hardware.

This first experiment answers the *total parameter* choice. A 300B expert-bank
prefix inside a 700B MoE can have nearly the same active FLOPs as the 700B model
when top-k and expert width are unchanged. It does not by itself provide a
2.3× cheaper decoder. Width, depth, or active-expert nesting would be needed
for that second objective.

## Question

Can one pretraining process yield:

1. a full MoE that is competitive with a conventionally trained large model;
2. a fixed, extractable expert-subset model that is competitive with a
   conventionally trained smaller MoE; and
3. both checkpoints for less than 10% additional training cost over the large
   model alone?

The intended scale-up is a roughly 300B-total prefix inside a 700B-total expert
bank. The 24-hour study uses billion-parameter proxies and evaluates loss at
fixed measured training FLOPs.

## Background research brief

Effort: high.
Stop rule: stop when new primary sources no longer change the ranked
hypotheses or the four-arm matrix.
Internal-search caveat: Marin's Echo corpus returned a database authorization
error. Repository code, Git history, GitHub issues, W&B links in durable
artifacts, and published primary sources were still inspected.

### Direct evidence

#### MatFormer and Gemma 3n

MatFormer trains nested feed-forward widths with losses on multiple submodels.
Its 850M decoder contains variants from 582M to 850M; the paper reports better
validation loss and one-shot evaluations than separately trained counterparts.
The result is not limited to a paper prototype. Google states that Gemma 3n
simultaneously optimized its E2B submodel inside E4B, released both extracted
models, and supports intermediate "Mix-n-Match" models by slicing FFN widths
and skipping layers.

This is direct support for co-training and breakout. It is not direct support
for expert-prefix nesting, frontier-scale MoE, or a claim of zero overhead
relative to training only the largest model.

Sources:

- [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [MatFormer](https://arxiv.org/abs/2310.07707)

#### LayerSkip

Meta's LayerSkip applies increasing layer dropout and an early-exit loss during
pretraining, continual pretraining, and fine-tuning. It demonstrates that
intermediate depth can be made usable without a separate draft model and
reports inference speedups up to 2.16×. This is relevant if expert-prefix
nesting succeeds and Marin later wants a smaller *active-compute* model.
LayerSkip produces early exits, however, rather than a standalone transformer
whose later weights can be discarded without inference support.

Source: [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)

#### Flextron

NVIDIA's Flextron converts pretrained GPT-3 and Llama-2 family models into
nested elastic networks. The conversion uses 7.63% of the original pretraining
tokens and supports latency-targeted subnetworks without per-target
fine-tuning. This is evidence for a lower-risk post-hoc fallback if co-training
hurts the large model. It does not answer whether the submodel can be available
throughout pretraining.

Sources:

- [NVIDIA Flextron publication](https://research.nvidia.com/labs/lpr/publication/cai2024flextron/)
- [Flextron paper](https://arxiv.org/abs/2406.10260)

#### Mixture of Nested Experts and sparse upcycling

Google DeepMind's Mixture of Nested Experts trains nested expert widths for
vision and reports equivalent accuracy at more than 2× lower inference compute.
The modality and token-routing objective differ from language pretraining, so
this supports the mechanism but not the expected language-model effect size.

Sparse upcycling expands a trained dense model into an MoE. It supports the
opposite schedule: train the small model first, then expand it. That remains
the fallback if simultaneous nesting systematically undertrains the outer
experts or degrades the full model.

Sources:

- [Mixture of Nested Experts](https://deepmind.google/research/publications/108549/)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

### Lower-confidence MoE-specific leads

Recent Matryoshka/elastic MoE papers train across different active-expert counts
or nested expert widths. They are useful implementation leads, but they do not
yet carry the same production evidence as Gemma 3n and are not the basis for
the primary claim.

- [Matryoshka MoE](https://arxiv.org/abs/2509.26520)
- [Elastic MoE](https://arxiv.org/abs/2509.21892)
- [Mixture of Slimmable Experts](https://arxiv.org/abs/2602.06154)

### Marin context

Marin's current Grug MoE has a shared dense expert plus 256 routed experts,
top-4 routing, QB load balancing, and sigmoid combine weights. The June run
trained a 67.1B-total/2.01B-active model. Current B200 work has validated
single-rack 64-GPU execution, including a 512-expert/top-8 configuration at
about 14.4% bf16 MFU and 330k tokens/s. The existing GPU path and local
us-east object storage make a four-rack parallel comparison feasible.

Relevant durable references:

- [MoE architecture and scaling tracker](https://github.com/marin-community/marin/issues/6711)
- [67B/2B-active run](https://github.com/marin-community/marin/issues/6044)
- [single-rack B200 MoE work](https://github.com/marin-community/marin/issues/7012)
- [512-expert B200 result](https://github.com/marin-community/marin/issues/7332)
- `experiments/grug/moe/README.md`
- `experiments/grug/moe/model.py`
- `experiments/grug/moe/launch_cw_scale.py`

## Negative results and design constraints

1. **A full-matrix mask is not free.** Multiplying a large activation or weight
   by a mask retains the large GEMM unless the compiler proves and lowers a
   static slice. It also does not reproduce the small model's downstream
   activations unless the restricted branch is propagated through every layer.
2. **Two losses usually mean two forwards.** A simultaneous full and small loss
   on the same examples follows two hidden-state trajectories after the first
   differing layer. MatFormer pays for multiple nested paths or amortizes them
   against training several independent models; it is not evidence that an
   arbitrary small model is free relative to the large model alone.
3. **Token-level expert masks do not define a breakout model.** If a token can
   switch between small and large pools across layers, the resulting trajectory
   is not the extracted small checkpoint's trajectory. The branch assignment
   must be stable for a whole sequence through all layers.
4. **Random expert dropout does not define a fixed checkpoint.** A stable
   prefix is required. Random subsets are a regularizer, not a breakout model.
5. **Expert count is mostly a parameter-footprint lever.** With fixed top-k and
   expert width, an E128 and E256 MoE have similar model FLOPs per token. The
   smaller checkpoint saves parameter memory and may change communication, but
   does not halve arithmetic.
6. **The outer bank is at risk of undertraining.** When only full-branch
   sequences can use outer experts, their eligible token set is smaller.
   Marin's QB count equalization may compensate by routing more full-branch
   tokens to the outer bank. Routing counts and outer/core gradient norms are
   therefore preregistered diagnostics, not optional plots.

## Proposed architecture: branch-conditioned expert-bank nesting

Let the full routed expert bank contain `E_L` experts and the fixed small subset
contain `E_S` experts, where `E_S = E_L / 2` in the proxy experiment. The proxy
uses even-numbered experts, not a contiguous prefix. With four experts per
expert-parallel rank, this places two small-subset experts on every rank.
Extraction compacts the selected experts into a contiguous `E_S` bank. Both
models use the same shared expert, attention, embeddings, hidden width, depth,
expert width, and top-k.

For each packed training sequence, choose a branch bit `z`:

- `z = small`: set router logits outside the fixed interleaved subset to
  negative infinity at every MoE layer;
- `z = full`: leave all router logits available.

The bit is fixed for every token and every layer in that sequence. Small and
full sequences occupy different rows of one batch, so JAX executes one forward
and one backward with the same number of active experts per token. The small
model receives exact small-model trajectories. The large model contains all
parameters used by both branches.

QB statistics use actual eligibility. Outer experts can only receive full
sequences; count balancing is allowed to increase their share of those
sequences. This is the mechanism most closely aligned with the proposed
"remaining weights cover what the smaller model does not" effect. We will not
claim semantic novelty from routing counts alone; only loss and held-out
evaluation can support the usefulness claim.

Extraction selects and compacts the subset's expert weights and router columns.
The shared expert and dense backbone are unchanged. A contiguous prefix is
explicitly rejected for the proxy because expert-axis sharding would place it
on only half of the ranks and create a systems artifact.

## Preregistered hypotheses

### Primary hypothesis H1

At fixed training tokens, optimizer, data order, and hardware, a 25%-small
branch-conditioned E256 model will:

- retain at least 90% of the E256 control's tokens/s;
- finish within `+0.010` absolute Paloma macro loss of the E256 control when
  evaluated with the full expert pool;
- yield an E128 prefix within `+0.030` absolute Paloma macro loss of the
  independently trained E128 control; and
- keep both core and outer expert assignment CV at or below the E256 control's
  assignment CV plus `0.10`.

Passing all four thresholds promotes the method. A statistically tied or lower
full-model loss is the preferred outcome.

### Secondary hypothesis H2

A 50%-small schedule improves the extracted prefix relative to the 25% schedule
but is more likely to degrade the full model. We expect a Pareto tradeoff:
`small_loss_50 <= small_loss_25` and `large_loss_25 <= large_loss_50`.

H2 is descriptive. It does not override H1's absolute promotion thresholds.

### Mechanism hypothesis H3

QB compensation will keep total assignments per expert approximately balanced
while concentrating the outer bank's assignments on full-branch sequences.
H3 is falsified if outer experts receive less than 70% of the mean core-expert
assignment count after warmup, or if more than 1% of assignments overflow
capacity. H3 can explain an outcome but cannot promote a model without H1.

### Scale-up hypothesis H4

If H1 passes at both the first and final checkpoints without a widening
large-model loss gap, a 300B prefix inside a 700B expert bank is viable when:

- the 300B/700B difference is mainly routed-expert count;
- the same expert-parallel mesh can store and dispatch the prefix cleanly; and
- full-scale throughput overhead remains below 10%.

H4 is falsified for the intended run if proxy overhead exceeds 20%, the
large-model loss gap exceeds 0.02, or the small prefix remains more than 0.05
behind its standalone control after a 10%-compute breakout cooldown.

## Four-arm matrix

| ID | Total routed experts | Branch schedule | Role |
|---|---:|---|---|
| `NEST-MOE-001` | 256 | 100% full | conventional large control |
| `NEST-MOE-002` | 128 | 100% full | conventional small control |
| `NEST-MOE-003` | 256 | 25% prefix / 75% full | primary nested arm |
| `NEST-MOE-004` | 256 | 50% prefix / 50% full | small-quality tradeoff |

Fixed proxy shape:

- hidden dimension: 1280;
- layers: 13;
- routed expert intermediate dimension: 640;
- shared expert intermediate dimension: 1280;
- top-k: 4;
- sequence length: 8192;
- large total parameters: 8.64B;
- small total parameters: 4.54B;
- active parameters: approximately 0.6B, excluding the LM head;
- global batch: 256 sequences, or 2,097,152 tokens per step;
- common pretraining target: 5,275 steps, or 11.06B tokens per arm;
- one 64-GPU GB200 rack per arm, four arms concurrently, batch priority;
- project: `marin-community/marin_moe`;
- group: `NEST-MOE-20260726`.

The 20-step compile and throughput smoke can lower the common target if
observed throughput would not finish by 06:00 UTC, but it cannot increase it.
The final token target is the largest common target all four arms can reach
while reserving three hours for breakout cooldown, SFT/evaluation, analysis,
and termination. No arm receives a larger token budget because it started
earlier or ran faster.

## Gates

### Gate 0: implementation and numerical contract

Run local/small-device behavior tests before cluster launch:

- a zero subset fraction is exactly equal to the existing full router;
- an all-subset E256 forward matches an E128 model loaded from the same subset
  weights, within the existing bf16 numerical tolerance;
- the nested subset is evenly represented on every expert-parallel rank;
- outer-expert gradients are zero on an all-prefix batch;
- full-branch gradients remain nonzero in both core and outer banks;
- per-sequence branch assignments remain fixed through all layers;
- checkpoint extraction produces a loadable E128 checkpoint.

Any failure blocks launch.

### Gate 1: 20-step rack smoke

Launch all four arms for 20 optimizer steps on 64 GB200s each. Pass conditions:

- compilation and three steady-state steps complete;
- finite loss and gradients;
- no capacity overflow above 1%;
- nested throughput at least 85% of the large control;
- checkpoint save and resume succeed.

An arm that fails is fixed once and relaunched. A repeated architecture error
ends that arm. Scheduler preemption is retried from checkpoint and is not an
architecture failure.

### Gate 2: common-token pretraining

Run the four passing arms to a common token target. Record train loss,
validation loss, Paloma macro/domain losses, tokens/s, MFU, step time,
capacity overflow, routing entropy/counts, core/outer assignment CV, parameter
norms, and gradient norms.

Evaluate each E256 checkpoint twice: full mode and extracted-prefix mode.
Evaluate E128 once. Primary comparisons use the same checkpoint token count and
the measured training FLOPs through that checkpoint. Plot loss against tokens,
measured FLOPs, GPU-hours, and wall time.

Promote `NEST-MOE-003` only if H1 passes. Continue `NEST-MOE-004` to the common
target even if it trails at the first eval, unless its full loss is more than
0.05 worse or it is unstable. This preserves the requested four completed
experiments instead of replacing weak arms with a sweep.

### Gate 3: breakout cooldown

If either nested arm is within 0.05 of the standalone small control, extract its
E128 prefix and train it directly for up to 10% of Gate 2's per-arm FLOPs.
Evaluate at 2%, 5%, and 10% additional compute. The breakout succeeds if it
reaches the standalone E128 loss within that budget.

The direct cooldown is charged to nested-model cost. Cost comparisons report:

1. large model alone;
2. large plus independently trained small model;
3. nested pretraining alone; and
4. nested pretraining plus breakout cooldown.

### Gate 4: SFT and agentic smoke

The repository already contains suitable SFT infrastructure and pinned
instruction sources, including WildChat, SmolTalk/SmolTalk2, OpenHermes,
Tulu-3, and Nemotron science-reasoning data. If Gate 3 finishes by 07:00 UTC,
run the same short completions-only SFT schedule from the large control,
promoted full checkpoint, small control, and promoted extracted/cooldown
checkpoint. Use one existing pinned mixture and chat template unchanged.

Report SFT loss and a small fixed agentic/code evaluation bundle already
supported by Marin. This stage is a transfer check, not a basis for claiming
general agentic capability. If no pretraining arm passes H1 or the deadline
cannot accommodate a common SFT budget, skip SFT and record the reason.

## Evaluation and statistics

Primary metric: Paloma macro loss at the common final token/FLOP point.

Secondary metrics:

- Paloma domain losses;
- pretraining validation cross-entropy;
- training throughput and GPU-hours;
- routing balance and capacity overflow;
- SFT validation loss and the existing fixed agentic/code bundle, if run.

The experiment uses one seed per arm because the deadline favors four complete
arms over replications. Uncertainty is estimated by paired bootstrap over
Paloma documents/domains where raw examples are available and by checkpoint
trajectory consistency. A threshold crossing smaller than observed
checkpoint-to-checkpoint noise is reported as inconclusive. No p-value from
tokens within one run is treated as a model-seed replication.

The loss comparison is evaluated at:

- equal tokens;
- equal measured model FLOPs;
- equal GPU-hours; and
- equal wall-clock time.

Headline cost is `(nested GPU-hours + breakout GPU-hours) / large-control
GPU-hours - 1`. A result above 50% is treated as economically close to training
a separate model. A result below 10% with H1 passing is viable. Between 10% and
20% is conditional on the loss benefit. Above 20% does not promote.

## Deadline and operational plan

- Launch through `lib/iris/config/marin.yaml`; federation should place GB200
  gangs on `cw-us-east-08a`.
- Use `--priority batch`, 16 four-GPU nodes per arm, checkpointing, and
  automatic resume after preemption.
- Use only jobs launched by this experiment for stop/resubmit actions.
- Never restart or mutate an Iris cluster.
- At 08:15 UTC, stop new work and collect final checkpoints and metrics.
- At 09:00 UTC, terminate any experiment jobs still running, publish the final
  report and negative results, and close Weaver issue #652.

## Review record

The artifact was sent twice to `claude --model fable`: first with the relevant
Grug files and then with an artifact-only bounded prompt. Both processes
remained alive without returning review content and were stopped after bounded
waits. An asynchronous Loom launch with the same Fable model returned HTTP 405.
After the Loom ACP connection was repaired, Claude Fable completed the
code-grounded review and published Weaver artifact `review`. The review session
was archived after the artifact was fetched.

The reviewer agreed that changing capacity factor from 1.0 to 1.25 in every arm
is a valid common-system amendment for this proxy layout. The review added
three limits that constrain interpretation:

- the proxy has two nested and two outer experts on every expert-parallel rank,
  while a one-expert-per-rank production layout would require method-specific
  capacity headroom of at least `1 + nested_batch_fraction`;
- QB equal-count balancing deliberately redirects full-branch traffic toward
  the outer bank, with a distortion that grows with the nested fraction and is
  degenerate at 50%; loss from that arm tests this complete routing design, not
  parameter sharing in isolation;
- the fixed dispatch buffer makes capacity factor 1.25 approximately 25% more
  routed-expert work than ideal capacity in every proxy arm. Equal E256
  throughput does not establish less than 10% production overhead.

The final report must also treat routing counts as pre-capacity intent, state
that exact small trajectories hold only up to capacity contention, and limit
scale-up claims because 262M tokens substantially undertrain a 2B-parameter
proxy. These are review constraints, not post-hoc changes to the registered
loss comparisons.

## Protocol amendments

### Reference-attention data representation and proxy size

The first uniform reference-attention smoke produced a finite forward loss in
all four arms, but the first backward produced nonfinite gradients. Three arms
stopped cleanly at step 2; the slower E128 arm was stopped after the same
nonfinite-gradient signature was visible. The failure occurred in the E256 and
E128 controls as well as both treatments.

The reference backend had inherited `pack=1`, which was added only to satisfy
the THD FlashAttention metadata contract. Fixed-shape THD examples contain
fully masked padding query rows. The reference softmax evaluates those rows as
all-negative-infinity and produces `0/0`; zero token loss weights do not remove
the resulting backward NaNs. Reference-attention runs therefore return to the
ordinary causal example representation (`pack=None`). THD runs retain
`pack=1`. A materialized-config regression test covers both branches.

The d1280, length-8192 reference step measured about 262 seconds. It cannot
produce a useful loss trajectory before the deadline. The corrected shape is
d768, 8 layers, and length 2,048. The later fp32 feasibility amendment fixes
global batch 256, or 524,288 tokens per step. Its E256 and E128 models remain
approximately 2.0B and 1.1B parameters. Every scientific arm changes together.
The common final step count cannot exceed 500. This amendment trades proxy
scale for completed four-arm evidence; it does not relax any relative-loss or
overhead threshold.

### Finite router masking and proxy warmup

The first d768 smoke isolated two pre-optimization configuration faults. The
four-step smoke converted the default fractional 1% warmup to zero steps. Both
controls produced one finite gradient and then nonfinite weights after taking
the full learning rate immediately. The nested arms additionally had
nonfinite first gradients because their `-inf` router eligibility sentinel
entered QB subtraction and reduction arithmetic.

The eligibility sentinel is replaced with the finite fp32 value `-1e9`, which
has the same zero-probability and top-k behavior. A regression test now
requires every nested-router gradient leaf to be finite while outer-expert
gradients remain exactly zero. The proxy optimizer uses five explicit warmup
steps, equal to 1% of the already bounded 500-step production schedule. The
same value is used in smoke and production; it avoids changing the schedule
after Gate 1. All model arms are amended together. No usable validation
checkpoint was produced before this correction.

### Final fp32 feasibility gate

The finite-sentinel and warmup correction did not make the bf16 reference
backend stable. E128 produced a finite first gradient, but the E256 control and
both nested treatments produced nonfinite gradient norms; every final
validation loss was nonfinite. Iris recorded zero task failures and zero
preemptions.

One last feasibility gate changes no architecture treatment: run the E256
control and primary nested25 arm with batch 256 and
`params=float32,compute=float32,output=float32`. Both must produce three finite
updates, a finite Paloma macro loss, capacity overflow at or below 1%, and a
checkpoint. If either fails, Gate 1 closes as blocked and no production arms
launch. If both pass, the four production arms use this policy and batch size,
with a step count frozen before launch from the measured steady-state rate.

### User-directed discovery continuation after capacity overflow

The fp32 nested25 arm completed with finite gradients, a checkpoint, and a
`+0.00298` full-versus-nested Paloma gap, but dropped 5.93% of routed
assignments at capacity factor 1.0. Its E256 control failed during JAX gang
bootstrap before step 0. The registered gate therefore did not pass, but it
also did not produce a treatment-versus-control observation about overflow.

After seeing this result, the user directed the study to continue toward the
discovery objective rather than treat a common routing-system inefficiency as
an architecture rejection. This is an outcome-aware protocol amendment. It
does not retroactively pass the capacity-1.0 gate.

Rerun all four arms for 20 updates with batch 256, full-fp32 compute, and
capacity factor 1.25. The capacity change is identical across controls and
treatments; model topology, data, optimizer, seed, expert-parallel geometry,
and reference attention remain fixed. Interpret the result only if all four
arms produce finite checkpoints and less than 1% assignment overflow.
Throughput, loss, and routing comparisons then answer the original architecture
question under a usable common routing system.

## Decision rules

Promote:

- H1 passes at the common final point;
- the result is directionally consistent at the earlier eval;
- no capacity, stability, or extraction failure is hidden by the macro average.

Continue at larger proxy scale before a hero run:

- full and small losses pass, but overhead is 10–20%;
- the final point passes while the early point does not;
- a routing imbalance has a clear, bounded implementation fix.

Reject:

- full loss gap greater than 0.02;
- extracted small loss gap greater than 0.05 after breakout cooldown;
- overhead greater than 20%;
- outer experts remain materially undertrained;
- extraction changes small-mode numerics.

Even a pass does not authorize changing the 300B–700B architecture. It
authorizes a larger preregistered replication with the intended expert count,
parallelism, data mix, and at least two seeds.

## NEST-MOE-006 addendum: direct breakout and balanced complements

This addendum was registered after the matched 10B E256, E128, and
single-prefix runs completed, but before either follow-up produced an update.
It tests two explanations for the remaining E128 gap.

### Direct breakout

Use the terminal 10B E128-naive checkpoint at update 38,147 as the common
breakout point. Initialize two fresh-optimizer trainers from exactly those
weights:

1. an unrestricted E256 trainer retaining the checkpoint's full-bank QB state;
2. a physical E128 trainer containing experts 0--127, the corresponding router
   columns, and the E128 QB state.

Both branches use the terminal Datakit mixture throughout. MuonH and AdamH
restart at the parent schedule's terminal learning rates, respectively
`3.033425e-4` and `7.000212e-5`, with no warmup and linear decay. Evaluate
every 250 updates for at most 12,000 updates. The primary recovery events are
the first Paloma macro evaluations at or below `3.143486738` for E256 and
`3.181439161` for E128, the terminal losses of the matched standalone controls.
Uncheatable losses are secondary checks.

Report both total accelerator compute and parallel critical-path time. Total
breakout compute includes the observed 38,147-update joint prefix plus both
cooldown branches through their first recovery evaluations. The comparison
baseline is the sum of the completed E256 and E128 standalone runs. This
directly tests the report's extrapolated `34.0%` saving; that number is a
saving versus two separate runs, not overhead versus one E256 run.

### Balanced complements

Train one matched 10B E256 run from scratch with 50% full-bank sequences, 25%
restricted to experts 0--127, and 25% restricted to experts 128--255. Rotate
the row assignment by update while preserving exactly 16 full, eight lower,
and eight upper sequences in every batch of 32. Each routing bank has
independent QB state. The expected routed assignment rate of every expert is
therefore equal to the E256 control, unlike the single-prefix schedule.

At update 4,000, continue only if:

- full Paloma is at most `3.872742`, the matched control plus `0.10`;
- both E128 halves beat the matched control's fixed-half loss `4.023812`;
- median compiled-step overhead versus the matched E256 control is below 5%;
- training is finite and assignment overflow remains operationally comparable.

If the gate passes, continue to update 38,147. The primary endpoint compares
full E256 Paloma to the E256 control and both extracted halves to standalone
E128. A balanced-complements success requires the worse half to reduce the
single-prefix E128 gap without increasing the full-model loss penalty or
optimizer-step time.

## NEST-MOE-008 addendum: post-hoc-drop strongman recovery

This control is registered after the direct-breakout crossings and before
post-hoc E128 recovery training begins. It tests whether nesting beats an
unmodified E256 run followed by expert dropping and direct E128 cooldown.

Start from the completed E256 control checkpoint at update 38,147. Select 128
experts independently in each layer using the previously evaluated
`hybrid_greedy` rule: sum the standardized negative full-router QB bias and
standardized router-column norm, then retain the top 128 columns. This is the
best observed post-hoc Paloma selection at `3.539280891`; the expert sets are
fixed by its logged W&B artifact. Physically gather those expert weights,
router columns, full-router QB state, and pending QB state into an ordinary
E128 trainer.

Use the same direct-cooldown contract as nested breakout: terminal Datakit
phase, fresh optimizer state, MuonH learning rate `3.033425e-4`, AdamH learning
rate `7.000212e-5`, no warmup, a 12,000-update linear schedule, and evaluation
every 250 updates. The recovery event is the first Paloma macro evaluation at
or below the standalone E128 endpoint `3.181439161`.

The primary cost is E256-control optimizer time plus post-hoc E128 cooldown
through the first crossing. Compare it with the observed nested-prefix plus
two-branch breakout cost, `48.8628` H100-hours. At the measured E128 step time,
post-hoc recovery must cross in approximately 10,100 updates to beat nesting
in total accelerator compute. If it does not cross by 12,000 updates, report
the result as a lower bound and stop.
