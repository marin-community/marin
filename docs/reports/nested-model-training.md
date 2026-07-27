# Can one MoE pretraining run yield both a 300B and 700B model?

Experiment series: `NEST-MOE`

Date: 2026-07-27

## Summary

Yes, expert-bank nesting is a valid approach at proxy scale. It is not ready
to add unchanged to a 300B–700B hero run.

We trained a 2.039B-parameter, 256-expert MoE while restricting 25% of
sequences to a fixed, extractable 128-expert subset. The restriction is held
constant through every layer. Restricted and unrestricted sequences share one
forward, loss, and backward; there is no second small-model forward.

At 262.144M common training tokens:

- the full nested model finished at `6.15123` Paloma macro loss, `+0.00900`
  behind the conventional E256 control and inside the preregistered `+0.010`
  margin;
- the extracted E128 model finished at `6.18123`, `-0.03237` better than the
  independently trained E128 control, and won on all 16 Paloma domains;
- median training throughput was `100.31%` of the E256 control;
- core and outer assignment balance matched an untreated E256 expert bank;
- a 10%-FLOP E128 cooldown improved the extracted model to `6.13423`, another
  `-0.04700`, and finished `-0.07936` ahead of the standalone E128 control.

The 50%-restricted arm failed decisively. Its full-model loss was `+0.07298`
behind the E256 control. Under balanced routing, restricted rows already fill
the core bank's assignment share, forcing nearly every unrestricted assignment
to the outer bank. This is too aggressive for the proxy.

The primary full-model result is close to its threshold. A deterministic
control repeat finished at `6.13964`, which changes the full-model difference
to `+0.01159`. The small-model result is much stronger: evaluating the same
fixed half of an untreated E256 control gives `6.27433`, so co-training
improves that subset by `0.09311`. The correct decision is therefore
to run a larger replication before using the method in the hero run.

The economic result has two forms:

- without cooldown, one nested run produced both checkpoints for the same
  analytic model FLOPs as the large model alone; training E256 and E128
  independently would cost `1.9956×` as many analytic model FLOPs;
- a 50-step direct cooldown adds `9.956%` analytic model FLOPs. The observed
  proxy job cost was much worse, `+67.9%`, because compilation, checkpoint
  loading, and five evaluations dominate a 50-step job.

The unresolved production issue is routing capacity. All scientific arms used
capacity factor 1.25. That gives a fair architecture comparison, but it does
not prove that production nesting costs less than 10% if the large baseline
can run at 1.0. With one expert per expert-parallel rank, a 25%-restricted
branch can create `1.25×` mean load on core ranks before the router moves full
rows outward. A production implementation needs collocated core and outer
experts, a dropless or ragged dispatcher, or a measured method-specific
capacity charge.

## Prior evidence

The strongest precedent is Google's MatFormer work. MatFormer jointly trains
nested feed-forward widths and reports 582M–850M decoder submodels that match
or improve on independently trained counterparts. This is also a deployed
result: Google says Gemma 3n trained the E2B model embedded inside E4B and
released both extracted checkpoints.

Meta's LayerSkip is the strongest depth-specific result. Increasing layer
dropout and early-exit losses produce usable intermediate exits and enable
self-speculative decoding. It is relevant to active-compute reduction, though
it does not directly produce an ordinary shallower checkpoint.

NVIDIA's Flextron is the strongest post-hoc fallback. It converts pretrained
models into nested elastic networks using 7.63% of their original pretraining
tokens. This avoids risking the main pretraining run, but the small model is
not available during that run.

Google DeepMind's Mixture of Nested Experts demonstrates nested expert widths
in vision and reports equivalent accuracy at more than 2× lower inference
compute. Sparse upcycling demonstrates the reverse schedule: train the small
model first, expand it into an MoE, then continue training. These support the
mechanism and fallback schedule, not the expected effect size for a frontier
language MoE.

Primary sources:

- [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [MatFormer](https://arxiv.org/abs/2310.07707)
- [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [Flextron](https://research.nvidia.com/labs/lpr/publication/cai2024flextron/)
- [Mixture of Nested Experts](https://deepmind.google/research/publications/108549/)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

Together, these results support nested training as a real architecture family.
They do not establish that an expert-subset language model is free, that outer
experts learn semantically novel concepts, or that a billion-parameter proxy
transfers directly to 300B–700B.

## Architecture

### Why a dense mask is not enough

Multiplying a large dense matrix by a mask usually retains the large GEMM.
More importantly, small and large models follow different hidden-state
trajectories after their first differing layer. Computing exact losses for
both models on the same examples normally requires two forwards.

MoE routing gives a cheaper first experiment. Let the large bank contain
\(E_L\) experts and the small bank be a fixed subset of \(E_S\). Each sequence
receives one branch bit:

- `small`: router logits outside the fixed subset are ineligible in every
  layer;
- `full`: every expert remains eligible.

The bit is fixed across all tokens and layers in that sequence. Both branches
retain top-4 routing. They occupy different rows in one batch, so the model
executes one forward and one backward. The small rows follow exactly the path
available after extracting the fixed subset.

The subset contains even-numbered experts. In this proxy, every
expert-parallel rank owns two core and two outer experts. A contiguous E128
prefix would occupy only half of the 64 ranks and confound the architecture
with placement and communication. Extraction gathers the even experts and
compacts them into a conventional contiguous E128 checkpoint.

This design is a stochastic multi-architecture objective, not two simultaneous
losses on every sequence. Alternating entire small and full steps would
optimize the same family but make throughput and optimizer-state behavior more
dependent on the schedule.

### Variations

| ID | Expert bank | Restricted sequences | Purpose |
|---|---:|---:|---|
| `NEST-MOE-001` | E256 | 0% | large control |
| `NEST-MOE-002` | E128 | 0% | small control |
| `NEST-MOE-003` | E256 with fixed E128 subset | 25% | primary arm |
| `NEST-MOE-004` | E256 with fixed E128 subset | 50% | aggressive tradeoff |
| subset diagnostic | untreated E256 | evaluation only | fixed-half counterfactual |
| breakout | extracted E128 from `003` | direct E128 training | cooldown gate |

Randomly dropping half the experts was not tested because it does not define a
stable breakout checkpoint. The fixed E128 subset is the MoE-specific version
of "drop the second half of experts," corrected for expert-parallel placement.

## Preregistered hypothesis and gates

The primary hypothesis was frozen before the scientific runs:

> At common training tokens, the 25%-restricted E256 model retains at least
> 90% of E256-control throughput, finishes within +0.010 Paloma macro loss in
> full mode, yields an extracted E128 model within +0.030 of the standalone
> E128 control, and keeps core and outer assignment CV within +0.10 of an
> untreated E256 bank.

The complete preregistration and amendments are in
[`2026-07-26_nested-model-training-preregistration.md`](https://github.com/marin-community/marin/blob/main/.agents/projects/2026-07-26_nested-model-training-preregistration.md).

| Gate | Question | Decision |
|---|---|---|
| 0 | Is masked E128 evaluation equivalent to compact extraction? | pass |
| 1 | Do all four arms train, save, evaluate, and route stably for 20 steps? | pass at common capacity 1.25 |
| 2 | Does either nested schedule pass the 500-step quality and throughput margins? | 25% pass; 50% reject |
| 3 | Does up to 10% direct E128 cooldown recover the extracted model efficiently? | pass |
| 4 | Do promoted checkpoints accept a common assistant-masked SFT recipe? | completed; see SFT section |

Claude Fable reviewed the preregistration before the final analysis. Its main
findings are retained as scale-up conditions:

- common capacity 1.25 makes the proxy comparison valid but is not evidence of
  less than 10% production overhead;
- 50% restriction is structurally close to a degenerate routing schedule;
- exact small-model trajectories require no assignment overflow;
- the short proxy is intentionally undertrained;
- a one-expert-per-rank production layout loses the proxy's capacity
  neutrality.

## Experimental setup

### Model and training

| Property | Value |
|---|---:|
| Hidden dimension | 768 |
| Layers | 8 |
| Attention heads / KV heads | 6 / 1 |
| Sequence length | 2,048 |
| Global batch | 256 sequences |
| Tokens per update | 524,288 |
| Pretraining updates | 500 |
| Pretraining tokens | 262,144,000 |
| Routed expert intermediate dimension | 384 |
| Shared expert intermediate dimension | 768 |
| Active routed experts per token | 4 |
| Large / small routed experts | 256 / 128 |
| Large / small total parameters | 2.039B / 1.133B |
| Large / small analytic FLOPs per token | 357.728M / 356.155M |
| Expert-parallel axis | 64 |
| Dispatch capacity factor | 1.25 |
| Precision | fp32 parameters and compute |
| Hardware per arm | 64 NVIDIA GB200 GPUs |

The four primary arms ran concurrently on 256 GB200 GPUs on
`cw-us-east-08a`, submitted through the main Marin Iris controller at batch
priority. Training used the pinned SlimPajama-6B cache and Llama 3.1 tokenizer.
Validation used the pinned Paloma domain caches. All arms shared seed 0, data
order, optimizer, shared expert, attention, embeddings, hidden width, depth,
expert width, and top-k.

The SM100 FA4 backward kernel hung after successful forward execution in this
environment. Every scientific arm therefore used Levanter reference attention.
The initial bf16 reference runs were also numerically unstable, so all final
arms used fp32. These changes were common to every architecture. Loss and
relative proxy throughput remain comparable; absolute throughput is not a
production FA4 estimate.

No definitive run was lost to preemption. Earlier exit-137 and retry failures
were treated as infrastructure events and produced no model-quality evidence.
The stale JAX retry-coordinator failure and fix are recorded in
[`2026-07-26-jax-retry-stale-coordinator.md`](https://github.com/marin-community/marin/blob/main/.agents/ops/2026-07-26-jax-retry-stale-coordinator.md).

### Evaluation

Paloma macro loss was evaluated at approximately 52.4M, 104.9M, 157.3M,
209.7M, and 262.1M training tokens. Nested E256 checkpoints were evaluated
twice: once with the full bank and once with every row restricted to the fixed
E128 subset. Gate 0 tests established equivalence between masked evaluation
and the compacted checkpoint when no routes overflow.

The primary endpoint is final Paloma macro loss at common tokens. Secondary
checks are:

- all 16 Paloma domain losses;
- training loss trajectory;
- median and mean tokens/s after step 5;
- assignment overflow, routing entropy, assignment fraction, and core/outer
  coefficient of variation;
- successful child-task GPU-hours;
- direct E128 cooldown at 2%, 4%, 6%, 8%, and 10% of the pretraining step
  count.

There is one seed per arm. Domain results are paired diagnostics, not
independent model replications. The control repeat measures some
run-to-run sensitivity but does not replace a second seed.

## Results

### Training loss

![Training cross-entropy for the four 500-step arms. The 25% arm tracks the
E256 control; the 50% arm separates late in training.](assets/nested-model-training-pretraining-loss.png)

All four arms completed 500 finite updates and wrote checkpoints. The
25%-restricted arm follows the E256 control closely throughout training. The
50% arm separates progressively, which is also visible in every late Paloma
checkpoint.

### Final validation and throughput

| Arm | Full Paloma | Extracted E128 Paloma | Δ full vs E256 | Δ extracted vs E128 | Median tokens/s | vs E256 |
|---|---:|---:|---:|---:|---:|---:|
| E256 control | 6.14223 | — | — | — | 2.476M | 100.00% |
| E128 control | 6.21359 | — | — | — | 2.452M | 99.03% |
| nested 25% | 6.15123 | 6.18123 | +0.00900 | -0.03237 | 2.484M | 100.31% |
| nested 50% | 6.21521 | 6.20705 | +0.07298 | -0.00654 | 2.502M | 101.04% |

The 25% arm passes every preregistered primary threshold against the original
controls. Its full model is worse than the E256 control on 13 of 16 Paloma
domains, with a mean domain delta of `+0.00900`. Its extracted E128 is better
than the standalone E128 on all 16 domains, with domain deltas from `-0.08960`
to `-0.01395`.

The 50% arm is rejected. Equal or better throughput cannot compensate for a
`+0.07298` full-model loss penalty.

![Paloma trajectories for the full models, extracted E128, standalone E128,
and direct breakout cooldown.](assets/nested-model-training-paloma.png)

### Sensitivity and the untreated subset

The original E256 control repeat, augmented only with fixed-subset evaluation,
finished at `6.13964` full-mode Paloma. Relative to this repeat, nested25 is
`+0.01159`, missing the `+0.010` margin by `0.00159`. The preregistered
comparison passes, but the threshold decision is not robust to control-run
variation at this scale.

The same repeat's untreated fixed E128 half scores `6.27433`. The co-trained
E128 subset scores `6.18123`, an improvement of `0.09311`, and wins on 15 of
16 domains. This is the cleanest evidence that the nested objective improved
the breakout checkpoint rather than merely selecting a naturally strong half
of an ordinary E256 bank.

The result is asymmetric:

- the extracted-model benefit is large and consistent;
- the full-model cost is small but close to the acceptance boundary.

That is enough to establish viability, but not enough to select the method for
a 300B–700B run without a larger replication.

### Routing, the 1% gate, and the original 5.93%

Assignment overflow is the fraction of top-4 token-to-expert assignments
dropped because a rank's fixed dispatch buffer is full. It is not the
percentage of experts that failed. If 5% of assignments overflow, as many as
20% of tokens can lose one of their four expert contributions.

Overflow is critical for three reasons:

1. dropped assignments change the architecture being trained;
2. overloaded experts receive no gradient for those assignments;
3. throughput can look artificially better because the kernel performs less
   useful routed work.

The original capacity-1.0 nested canary ended at `5.93%` overflow. Its matched
control failed before step 0 for an independent retry-coordinator bug, so that
number could not be attributed to nesting. It was a routing-capacity
observation, not evidence about the research hypothesis. The common
capacity-1.25 rerun was therefore the correct discovery experiment.

At capacity 1.25, all four 20-step Gate 1 arms ended below `0.1%` overflow. In
the 500-step runs, mean overflow remained below 1% and every arm ended at zero:

| Arm | Mean overflow | Maximum startup overflow | Terminal overflow |
|---|---:|---:|---:|
| E256 control | 0.623% | 3.431% | 0% |
| E128 control | 0.707% | 3.973% | 0% |
| nested 25% | 0.565% | 3.617% | 0% |
| nested 50% | 0.384% | 2.546% | 0% |

The common control has the same transient behavior. The 5.93% canary was
therefore primarily a fixed-capacity/cold-router inefficiency, not an
architecture-specific finding. Correcting it does not contaminate the
discovery question; it makes the comparison interpretable.

The final untreated control has core/outer assignment CVs of `0.1037` and
`0.1012`. Nested25 has `0.1041` and `0.1005`, well inside the `+0.10` gate.
Nested25 sends `49.32%` of all assignments to the core bank. Since 25% of rows
are core-only, unrestricted rows send an inferred `32.43%` of assignments to
core and `67.57%` to outer experts.

This is almost exactly the balanced solution: the router uses the outer bank
disproportionately for full rows. It supports the proposed residual-capacity
mechanism. It does not show that the outer experts learned semantically novel
concepts; that requires capability or domain specialization evaluation.

At 50% restriction, the restricted rows alone account for the core bank's
balanced share. The final core assignment fraction is `50.03%`, implying that
nearly all unrestricted assignments go to outer experts. This explains why
the 50% schedule is structurally brittle and why it was rejected.

### Breakout cooldown

The fixed E128 subset was compacted from the nested25 step-500 checkpoint,
loaded into a conventional E128 model with a fresh optimizer, and trained for
50 more updates. The cooldown used 26.214M tokens and `9.956%` of the large
pretraining run's analytic model FLOPs.

| E128 cooldown updates | Added E128 tokens | Paloma macro loss |
|---:|---:|---:|
| 0 | 0 | 6.18123 |
| 10 | 5.243M | 6.18579 |
| 20 | 10.486M | 6.16572 |
| 30 | 15.729M | 6.15394 |
| 40 | 20.972M | 6.14163 |
| 50 | 26.214M | 6.13423 |

The small early regression is within the noise of a fresh optimizer and short
evaluation interval. By 20 updates the model improves, and by 50 it is
`0.04700` better than the extracted checkpoint, `0.07936` better than the
standalone E128 control, and `0.00800` better than the original E256 control.

Breakout works at an arbitrary checkpoint. A small model can ship immediately,
or direct cooldown can start when its quality becomes the priority.

### SFT and agentic transfer

The promoted checkpoints were initialized into a common Grug chat-SFT
pipeline:

- WildChat 385.7k, revision `46a5bb5`;
- Llama 3.1 instruct template;
- assistant-token-only loss and packed 2,048-token sequences;
- batch 256, eight updates, fresh optimizer, and identical data order;
- E256 control, E128 control, nested25 full, and cooled-down breakout arms.

The first matched execution exposed a pre-existing cache correctness bug:
Levanter warned that tokenizer and chat-template metadata differed, but reused
the old cache. Those loss values are excluded. The SFT launcher now isolates
step-count chat caches by tokenizer, template, packing mode, and cache version.
The corrected matched rerun is the source of the results below.

<!-- SFT_RESULTS_START -->
![Assistant-token loss for the four corrected WildChat SFT arms. The nested
full model tracks the E256 control; the cooled breakout remains below the
standalone E128 control.](assets/nested-model-training-sft-loss.png)

| Initialization | Mean loss, updates 2–7 | Final loss | Mean overflow | Median tokens/s, updates 3–7 |
|---|---:|---:|---:|---:|
| E256 control | 7.08179 | 7.10610 | 12.815% | 2.575M |
| E128 control | 7.15806 | 7.17838 | 11.060% | 2.556M |
| nested25 full | 7.08169 | 7.10641 | 11.927% | 2.518M |
| cooled E128 breakout | 7.03852 | 7.05814 | 10.794% | 2.552M |

On the same six logged batches, nested25 full differs from the E256 control by
`-0.00010` mean loss and `+0.00031` final loss. The cooled breakout is
`-0.11954` below the standalone E128 mean and `-0.12024` below its final loss.
Nested25 full retains `97.79%` of E256 median SFT throughput; breakout retains
`99.85%` of E128 throughput. All four jobs completed and saved checkpoints
without cache-metadata warnings.

Overflow is high in every arm, between 10.8% and 12.8% on average. This is a
cold-router response to the abrupt WildChat distribution shift, not evidence
that SFT is loss-safe at capacity 1.25. It does not prevent the narrower Gate 4
conclusion that the checkpoints load, optimize, and preserve their relative
short-horizon loss behavior. A real post-training run needs a loss-safe
dispatcher or more SFT routing headroom.
<!-- SFT_RESULTS_END -->

This stage is a transfer and trainability check. Eight updates of an
undertrained billion-parameter proxy cannot support a meaningful claim about
general agentic capability, so no agentic benchmark is reported.

## Cost

### Pretraining

Successful child-task durations give the following charged GPU-hours:

| Arm | GPU-hours | Ratio to E256 control | Analytic process-FLOP ratio |
|---|---:|---:|---:|
| E256 control | 7.011 | 1.000 | 1.0000 |
| E128 control | 6.654 | 0.949 | 0.9956 |
| nested 25% | 6.670 | 0.951 | 1.0000 |
| nested 50% | 7.282 | 1.039 | 1.0000 |
| E256 + independently trained E128 | 13.665 | 1.949 | 1.9956 |
| nested25 + 50-step E128 cooldown | 11.774 | 1.679 | 1.0996 |

The observed nested25 run is 4.9% cheaper than the E256 control. This is run
variance, not a claim that the mask speeds training up. The defensible result
is no measurable proxy overhead: both have the same shape, top-k, and analytic
model FLOPs.

The independent-model cost nearly doubles because E128 and E256 have almost
the same active FLOPs. One nested run avoids that second forward. The
50-step cooldown's observed `+67.9%` is not representative of a long run:
compilation, checkpoint loading, and five full evaluations dominate its 245
seconds of W&B runtime. If cooldown is appended in-process and amortized, its
model-FLOP charge is `+9.956%`. That production wall-clock result has not yet
been measured.

The untreated-subset diagnostic and four matched SFT arms are research
diagnostics, not part of the proposed pretraining method cost.

### The capacity-factor caveat

Capacity 1.25 allocates a 25% larger routed dispatch buffer. It was applied to
every proxy arm, so the architecture throughput comparison is fair. It may
still be a method-specific production cost.

The proxy places two core and two outer experts on every rank. Core pressure
and outer slack can cancel at the rank boundary. If the hero model uses one
expert per rank, a 25%-restricted schedule sends `1.25×` mean load to a core
rank before unrestricted routes rebalance outward. A fixed padded dispatcher
may therefore need capacity factor 1.25 for nesting even if the ordinary model
uses 1.0.

A 25% routed-compute or communication tax would erase the desired less-than-10%
economics. The next experiment must use the intended production sharding and
compare each method at its lowest loss-safe capacity. Dropless/ragged dispatch,
core-and-outer colocation, or explicit branch-balanced routing are viable
solutions.

## Viability at 300B–700B

### What scales directly

Expert-bank nesting is well matched when the 300B–700B difference is primarily
stored routed experts:

- it produces a fixed smaller checkpoint throughout pretraining;
- it keeps approximately the large model's active FLOPs instead of adding a
  second small-model forward;
- the subset can be sharded evenly;
- extraction reduces checkpoint and serving memory;
- direct cooldown can start at any checkpoint.

The proxy's E128 checkpoint has 55.5% of the E256 model's total parameters
because the dense backbone is shared. A 300B/700B target is 42.9%, so the exact
expert subset must be chosen after accounting for embeddings, attention,
shared experts, and other fixed parameters. The sequence restriction fraction
does not have to equal the parameter fraction; the proxy supports 25%, not
50%.

### What does not scale automatically

Expert-count nesting is a parameter-footprint lever, not an inference-FLOP
lever. The proxy E128 model uses 99.56% of E256 analytic FLOPs per token because
top-k, expert width, dense width, and depth are unchanged. If the intended
300B model must also decode much faster, expert-width, depth, or top-k nesting
is required.

The result also does not justify semantic-specialization claims. QB routing
pushes unrestricted rows toward outer experts, but route counts only show
residual utilization. Domain-specific routing, held-out capability deltas, or
expert interventions are needed to show that the outer bank covers novel
concepts.

### Recommendation

Do not add the current mechanism directly to the 300B–700B run. Promote one
25%-restricted arm to an intermediate replication with:

1. the intended expert-per-rank topology and production attention kernel;
2. at least two seeds;
3. a longer loss curve with enough tokens to separate a `0.01` effect;
4. the large control, small control, nested25, and one routing-capacity
   treatment—four complete arms, not a broad sweep;
5. each arm's lowest loss-safe dispatch capacity;
6. an appended in-process 10%-FLOP breakout cooldown;
7. fixed checkpoints for SFT and downstream capability evaluation.

Promote to the hero run only if the replicated full model is within `+0.01`,
the extracted model is no worse than the standalone small control, and total
measured overhead including capacity and cooldown is below 10%. A measured
overhead near 50% is not interesting; it is too close to training the smaller
model separately.

## Next architecture arms

The next ideas should remain gated and orthogonal:

1. MatFormer expert-width nesting: slice the shared and routed FFNs at
   static intermediate widths. This reduces active FLOPs and has the strongest
   shipped frontier-lab precedent.
2. LayerSkip-style depth nesting: train fixed early exits or retained-layer
   topologies. This reduces decoder latency and can support self-speculation.
3. Small-first sparse upcycling: train the small checkpoint conventionally,
   expand or duplicate experts, then continue as the large MoE. This protects
   small-model quality if simultaneous training remains too risky.
4. Elastic top-k: alternate active-expert counts while sharing the bank.
   This directly changes compute, but routing capacity and communication make
   it a later experiment.
5. Post-hoc elastic conversion: apply a Flextron-like cooldown to a
   completed large model when the pretraining schedule cannot accept
   architectural risk.

Width, depth, top-k, and expert-count nesting should not be composed in the
first scaled run. Sampling several axes independently creates submodels that
were rarely optimized together and makes a failure difficult to interpret.

## Reproducibility

The implementation and launchers are:

- [`experiments/grug/moe/model.py`](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/model.py)
- [`experiments/grug/moe/launch_nested_experts.py`](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/launch_nested_experts.py)
- [`experiments/grug/moe/launch_nested_sft.py`](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/launch_nested_sft.py)
- [`scripts/training/analyze_nested_moe.py`](https://github.com/marin-community/marin/blob/main/scripts/training/analyze_nested_moe.py)

Machine-readable results:

- [result JSON](assets/nested-model-training-results.json)
- [summary CSV](assets/nested-model-training-summary.csv)

Primary W&B runs:

- [E256 control](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-001-full-d768-s2048-e256-cf125-r15)
- [E128 control](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-002-full-d768-s2048-e128-cf125-r18)
- [nested25](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-003-full-d768-s2048-e256-cf125-r17)
- [nested50](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-004-full-d768-s2048-e256-cf125-r15)
- [untreated-subset diagnostic](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-001-full-d768-s2048-e256-subset-eval-cf125-r19)
- [E128 cooldown](https://wandb.ai/marin-community/marin_moe/runs/nest-moe-005-cooldown-d768-s2048-e128-cf125-r20)
- [E256 control SFT](https://wandb.ai/marin-community/marin_moe_sft/runs/nest-moe-sft-large-d768-s2048-r23)
- [E128 control SFT](https://wandb.ai/marin-community/marin_moe_sft/runs/nest-moe-sft-small-d768-s2048-r24)
- [nested25 full SFT](https://wandb.ai/marin-community/marin_moe_sft/runs/nest-moe-sft-nested_full-d768-s2048-r24)
- [cooled E128 breakout SFT](https://wandb.ai/marin-community/marin_moe_sft/runs/nest-moe-sft-breakout-d768-s2048-r24)

Checkpoints:

- nested25:
  `s3://marin-us-east-02a/marin/experiments/nested-moe/nest-moe-003-full-d768-s2048-e256-cf125-r17/2026.07.27/checkpoints/step-500`
- cooled E128:
  `s3://marin-us-east-02a/marin/experiments/nested-moe/nest-moe-005-cooldown-d768-s2048-e128-cf125-r20/2026.07.27/checkpoints/step-50`

The append-only task history is in
[`652-nested-model-training.md`](https://github.com/marin-community/marin/blob/main/.agents/logbooks/652-nested-model-training.md).
