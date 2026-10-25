# Policy learning

Levanter's policy learner applies a clipped PPO-style actor objective to batches
with precomputed advantages. GRPO supplies one recipe for those advantages and
objective weights. Value prediction, returns/GAE, rollout admission, and policy
publication remain caller responsibilities.

## Batch and objective

`levanter.policy_model.PolicyBatch` contains complete token sequences, attention
masks, explicit positions, response masks, advantages, policy/KL weights, raw
old-policy scores, and optional fixed-reference scores. Sequence fields use
`(batch, position)` axes; objective fields use `(batch, response)` axes aligned
with the final response positions. Tool observations remain context with zero
objective weight. Sampling-distribution behavior logprobs are separate from the
raw-policy scores used by this objective.

`levanter.policy.clipped_policy_loss` takes those scores and weights, with
`ClippedPolicyConfig` selecting asymmetric clipping widths and the k3
reference-KL coefficient. KL differentiates through current-policy scores.
Old scores, reference scores, advantages, and weights are detached.
A nonzero KL coefficient requires reference scores.

Compute advantages over complete admitted groups. Compute objective weights
for each complete optimizer-step batch before packing or execution slicing.
`levanter.grpo.grpo_advantages` implements group-relative outcome advantages;
`grpo_objective_weights` divides policy contributions by the step's total
response-token count. KL first averages within each nonempty response, then
across nonempty responses in that step. Empty responses contribute neither
loss nor denominator counts. Execution microbatch and data-parallel partitions
do not change these weights.

These helpers implement the outcome-advantage estimator, regular clipped PPO
with `token_mean` policy reduction, and k3 reference KL from
[MarinSkyRL revision 82396a73](https://github.com/marin-community/MarinSkyRL/commit/82396a73a31e776229f3240c3af2b50ad53c4336).
MarinSkyRL's default GRPO recipe selects `sequence_mean` policy reduction;
that recipe needs different policy weights. Other MSRL losses and corrections
remain separate integration work.
Other recipes may supply different advantages and weights to the same learner.

Gradients are averaged across execution microbatches. Set `accumulation_steps`
to the number of execution microbatches: the loss scales each contribution by
that count before the gradient mean, preserving the full batch's precomputed
weights. Metrics retain their own sum or maximum reductions.

## Scoring snapshots and updates

`levanter.policy_learner.PolicyLearner` has two operations:

- `score(state, batch)` returns a batch with raw old-policy scores from that state.
- `update(state, scored_batch)` performs one optimizer update and returns
  `PolicyStep(state, loss)` without replacing the old scores.

Score once at the start of an on-policy batch. Reuse the scored batch across
all update epochs so the ratio remains relative to the same policy snapshot.
For synchronous one-update GRPO, score once and update once. Advantages and
reference scores remain fixed across those updates too.

For SPMD execution, configure a `Trainer` with a partial of
`levanter.policy_model.policy_model_loss` binding `config`, `block_size`, and
`accumulation_steps`. The configured full batch size and microbatch size must
match the prepared batch. The learner uses the model's existing FSDP/EP
sharding and the Trainer's optimizer, hooks, and checkpoint state.

```python
from levanter.policy_learner import SpmdPolicyLearner

learner = SpmdPolicyLearner(trainer, block_size=1024)
scored_batch = learner.score(state, batch)
for _ in range(update_epochs):
    result = learner.update(state, scored_batch)
    state = result.state
```

Scoring and training use the same execution microbatch shape and disable
model dropout. Reference-policy scoring is supplied separately by the caller.

## Packing

`levanter.policy_model.pack_policy_batch` packs complete trajectories on the
host, preserving exact tokens, source positions, response masks, objective
weights, and optional references. It supplies explicit trajectory segment IDs
for attention. It rejects trajectories longer than the requested row.

Packed rows use response width `sequence_length - 1`. Prompt targets,
cross-segment targets, and padding have zero objective weight. Set
`batch_size_multiple` and `minimum_rows` for the desired execution shape before
configuring the learner. Packing preserves the prepared advantage and objective
weighting; it does not recompute either from packed rows.

## Pipeline execution

`levanter.policy_pipeline` implements the same `PolicyLearner` operations with
JaxPP. A model implements `PipelineStage` for embeddings, blocks, final
normalization, output weights, trainable selection, and routing counters. Install
the root `pipeline` extra for execution. The schedule is standard 1F1B;
in multihost execution, all devices belonging to a given stage must be owned
by the same process.
Expert and replica axes must divide the device count evenly.

Initialize stage-owned `PipelineState`, then construct the learner once for the
prepared execution shape:

```python
from levanter.policy_pipeline import make_pipeline_policy_learner

learner = make_pipeline_policy_learner(
    optimizer,
    mp_policy,
    state,
    batch,
    config=pipeline_config,
    objective=objective,
    mpmd_mesh=mpmd_mesh,
)
scored_batch = learner.score(state, batch)
for _ in range(update_epochs):
    result = learner.update(state, scored_batch)
    state = result.state
```

Both learner implementations accept `PolicyBatch`, including the shared packed
layout. `pipeline_batch` lowers named arrays into execution microbatches;
`packed_pipeline_batch` combines shared packing with that conversion.

The pipeline learner retains stage-local weights and optimizer moments. It
averages gradients after temporal accumulation and clips using the global norm
across stages before optimizer updates. FP32 and BF16 temporal gradient
accumulation are explicit choices. Trainable weights use the selected compute
precision; frozen tensors retain their dtype. The scorer preserves the training
cast boundary with an optimization barrier.

Scoring verifies exact repeatability on the first microbatch. Routing drops are
rejected by default; `RoutingDropPolicy.REPORT` permits explicit capacity
experiments and reports counts including padding. Rollout router replay remains
separate integration work. Pipeline checkpoint integration and distributed
model acceptance require validation with the chosen model and parallelism.
