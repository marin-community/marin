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
`ClippedPolicyConfig` selecting asymmetric clipping widths, the k3 reference-KL
coefficient, and `KlGradient`. Old scores, reference scores, advantages, and
weights are detached. `KlGradient.DETACHED` reports KL without a KL gradient;
`KlGradient.DIFFERENTIABLE` differentiates KL through current-policy scores.
A nonzero KL coefficient requires reference scores.

Compute advantages and objective weights over the complete admitted batch
before packing or execution slicing. `levanter.grpo.grpo_advantages` implements
group-relative outcome advantages; `grpo_objective_weights` preserves the
MarinSkyRL recipe's partition-weighted token means and sequence-weighted KL.
Each partition is an original learner microbatch on one data-parallel rank;
partitions contribute equally. Within a partition, policy loss averages over
response tokens and KL averages over sequences.
These recipe helpers follow [MarinSkyRL revision 8e33e017](https://github.com/marin-community/MarinSkyRL/commit/8e33e01707b7225ecde1d6b8ad172a3dd4dc8661).
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
