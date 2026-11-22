# Policy learning

Levanter's policy learner applies a clipped PPO-style actor objective to batches
with precomputed advantages. GRPO supplies one recipe for those advantages and
objective weights. Value prediction, returns/GAE, rollout admission, and policy
publication remain caller responsibilities.

## SkyRL quick loop

`experiments.post_training.inventory.train_quick_loop` accepts
`--learner megatron` (the default) and `--learner levanter`. Both use the same
pinned Snowball initializer, task selection, RolloutEngine/vLLM generation,
graders, evaluation, and coordinator-computed RLOO-Neutral (RLOO-N) advantages.
RLOO-N uses a leave-one-out reward baseline, excluding infrastructure failures
from both the baseline and gradient while retaining agent failures at zero reward. The native
adapter consumes SkyRL's finalized batch, including failed-sample masks and
fractional loss weights. It divides token loss weights by their sum over the
complete optimizer minibatch before device sharding or gradient accumulation.
Finalized advantages pass through unchanged; masked positions contribute zero
to the loss and denominator.

The Levanter strategy reserves the same four eight-GPU learner hosts and one
eight-GPU inference host as the Megatron quick loop. One JAX child process per
learner host owns native Snowball parameters and Adam state, using data4/expert8
sharding. Model-free parent actors stream bounded weight slices through the
existing expert-block receiver and byte verifier. The strategy requires
disaggregated roles, GPU-resident optimizer state, driver batch assembly,
regular clipped PPO with token-mean reduction, and disabled KL and entropy terms.

The Torch rollout runtime and JAX learner runtime use separate frozen Python
environments. The native profile explicitly selects attention/MoE kernels,
FP32 routing and selective attention operations, and BF16 projection logits
with FP32 softmax. Scoring and updates use the same precision settings.
Before each native BF16 scoring or update command, all learner hosts agree on
the execution signature and exact fixed XLA projection plan outside JIT tracing.
The immutable plan checks projection shapes, dtypes, and options locally inside
the traced loss. This keeps coordination independent of each host's trace cache.
The prepared path covers the matched Snowball BF16 profile; default FP32
projection retains its previous kernel selection. Prepared mode rejects
`LEVANTER_CE_XLA_FAST_BWD` overrides so all hosts use the covered XLA backward.
The learner overrides inherited CPU-coordinator JAX settings with
`JAX_PLATFORMS=cuda,cpu`, retaining Ray's `CUDA_VISIBLE_DEVICES` assignment.
CUDA must initialize successfully; the CPU backend supports checkpoint host
staging. Bootstrap verifies the isolated CUDA backend before starting Ray.
The quick-loop candidate enables BFC preallocation with the same `0.90`
device-memory fraction. This addresses segmented pool growth before the compiled
update without increasing the allocator limit. Native initialization receipts
record the resolved allocator policy and available device-limit counters;
successful updates and checkpoint continuation still require accelerator
acceptance. The worker's general `preallocate` default remains false.
Recovery checkpoints contain native learner state; terminal HF export is not
implemented for this strategy.

The initializer is `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`
at revision `fe666478454e79fe6027a0bab0388d9abd68f51a`. The default selection URI
points to `snowball-100k/2026.11.04`. Build without allocating GPUs,
supplying `--selection-uri` explicitly to keep both arms on that artifact:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --version YYYY.MM.DD --learner levanter --steps 25 \
  --selection-uri s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04
```

For a matched control, use `--learner megatron --verify-publication` with a
distinct artifact version and the same selection URI. Levanter always verifies
publication. CPU objective, native-model regression, and transport checks do not
establish full-model accelerator acceptance. Before a comparison, use the same
artifact entry point for a short `--steps 2 --validate-native-checkpoint` run,
following [RL launching](rl-launching.md). The validation option checks that the
first update changes parameters, saves complete native state, restarts the JAX
learner processes, and checks exact restored state fingerprints before training
continues. Parent actors and rollout engines remain in the same allocation.
Leave this option disabled for performance comparisons.
It also captures the first compiled train step's memory analysis, optimized HLO,
buffer assignment, and available device-memory counters before execution.
Bounded text and metadata artifacts upload to the checkpoint bucket's
`tmp/ttl=14d/xla-dumps/<run-id>/process-<rank>/pid-<pid>/` prefix; stderr's
`native_compiler_evidence` receipt gives the exact destination and upload status.
Artifacts also copy into `IRIS_OUTPUT_DIR/native-compile/` when that directory
is configured. Failure counters are saved separately, preserving the original
training exception if diagnostic collection or upload fails. These diagnostics
preserve the recipe and do not establish successful updates or peak-memory fit.
Require every learner process to report the pinned model revision and completed
optimizer steps. Receiver replay receipts must identify the installed version
and report zero byte mismatches against learner weights. Generation receipts
must contain nonempty returned tokens stamped with the updated policy step:
weight version `v` produces token stamps `v + 1`. Also require a terminal native checkpoint.
Checkpoint continuation additionally requires fresh learner processes and
matching model, Adam, RNG, step, and model-averaging fingerprints after restore,
followed by further optimizer updates. Full-model accelerator acceptance has not yet been
established by the regression tests on this branch.

`--coordinator-route regional` submits the coordinator directly through the
configured CoreWeave Kubernetes context when hub handoff is unavailable. It
preserves the training recipe and allocation. Before submitting a replacement,
cancel the previous coordinator and confirm it is terminal to prevent duplicate runs.

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

## Checkpoint continuation

For SPMD execution, save the Trainer's `state.saveable_state` together with
the scored `PolicyBatch` through `levanter.checkpoint.save_checkpoint`.
The batch preserves frozen old-policy log-probabilities, advantages, weights,
and fixed-reference scores for subsequent updates.

```python
from levanter.checkpoint import load_checkpoint, save_checkpoint

path = f"/tmp/policy-checkpoint/step-{int(state.step)}"
save_checkpoint({"state": state.saveable_state, "batch": scored_batch}, int(state.step), path)
restored = load_checkpoint(
    {"state": initial_state, "batch": batch_template},
    path,
    axis_mapping=trainer.parameter_axis_mapping,
    mesh=trainer.device_mesh,
)
```

`initial_state` and `batch_template` supply the destination's array shapes and
tree structure. Record the admitted-batch identity and the index of the next
update to execute on that batch, together with the resolved objective, model,
precision, and optimizer configuration. Save after a completed optimizer step;
these examples do not resume a partially accumulated step.

The bounded synthetic Llama smoke exercises group advantages, whole-step
weights, scoring, AdamW updates, and continuation in separate Python processes:

```bash
python -m levanter.testing.policy_learner_smoke --phase save --output-dir /tmp/policy-smoke
python -m levanter.testing.policy_learner_smoke --phase resume --output-dir /tmp/policy-smoke
```

Pass `--require-accelerator` for hardware validation and `--compute-dtype bfloat16`
for BF16 compute with FP32 parameters. The resume phase validates the saved
configuration and compares restored model, optimizer, step, RNG, and frozen
scores against the uninterrupted run. Receipts are written beside the
checkpoints. This smoke uses one process with all test devices visible and
synthetic batches; rollout admission and policy publication require separate
integration.

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
separate integration work.

For pipeline checkpoints, save the stage-owned state and scored batch together
through `levanter.mpmd_checkpoint.checkpoint_arrays` before passing the resulting
tree to `levanter.checkpoint.save_checkpoint`. The adapter exposes stage-local
buffers without gathering the model. Record the current admitted-batch identity
and the zero-based index of the next optimizer update on that batch in metadata
(for example, `batch_id` and `next_update_index`). Restore into the same stage layout with
`levanter.mpmd_checkpoint.restore_checkpoint`, supplying an exemplar payload
and its destination sharding tree.

The explicit GPU acceptance module is
`lib/levanter/tests/test_policy_pipeline_accelerator.py`. In a root `pipeline`
extra environment with pytest installed, run it on four GPUs with one worker:

```bash
python -m pytest -c pyproject.toml lib/levanter/tests/test_policy_pipeline_accelerator.py -n 0 -m 'manual and not slow and not integration and not data_integration and not cluster and not requires_cluster and not docker'
```

It compares two-stage pipeline updates, clipping norms, and AdamW state against
an independent SPMD calculation, then checks checkpoint reload and continuation.
FP32 compares PP2/data2 against SPMD/data4. BF16 compares PP2/data2 against
SPMD/data2 because its local gradient rounding depends on the spatial partition.
Two and four execution microbatches retain the same precomputed policy and KL
loss weights. The bounded test
uses one host and a synthetic stage model; multihost execution and real MoE/EP
model acceptance remain separate gates.
