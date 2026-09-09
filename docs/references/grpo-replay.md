# GRPO replay and offline learning

`levanter.grpo` implements the regular PPO surrogate used by the pinned
MarinSkyRL GRPO recipe. `marin.rl.grpo_replay` checks captured loss values and
logprob gradients. `marin.rl.train_grpo` trains a model on an ordered list of
completed captures; rollout generation and online policy publication are outside
this entrypoint.

## Objective and captures

Compute group advantages and objective weights over the complete admitted batch
before dividing it into execution microbatches. Advantages use group sample
standard deviation. Policy weights preserve the original objective partitions'
masked token means; KL weights preserve their sequence means. A partition is
an original oracle execution microbatch on one data-parallel rank. Partition
means receive equal weight; within each KL partition, sequences receive equal
weight. Empty responses count as sequences but contribute zero KL. Old-policy scores,
reference scores, advantages, and weights are detached. `KlGradient` explicitly
selects detached or differentiated reference KL; the pinned oracle uses detached
KL, with gradients through current-policy scores disabled. Reference scores are
always detached; `KlGradient.DIFFERENTIABLE` enables only the current-policy
contribution. `accumulation_steps` cancels Levanter's averaging over execution microbatches.

`marin.rl.grpo_artifact.GoldenRollout` stores complete token sequences, attention
masks, response-aligned objectives, group IDs, original objective-partition IDs,
and distinct raw-policy and sampling-policy scores. Its NPZ format embeds a JSON
manifest and loads with pickle disabled. `write_golden_rollout` and
`read_golden_rollout` validate the array shapes and masks.

Replay requires the recorded MarinSkyRL oracle revision
`8e33e01707b7225ecde1d6b8ad172a3dd4dc8661`, its configuration/provenance manifest,
and captured independent Torch logprob gradients:

```bash
uv run python -m marin.rl.grpo_replay /data/golden.npz \
  --output-uri /data/replay-result.json --atol 1e-5 --rtol 1e-5
```

The command rejects unsupported algorithm recipes and reports maximum/mean
absolute errors for advantages, loss, gradients, and PPO diagnostics. Gradient
comparisons divide by each positive policy weight before applying tolerances,
so a large batch cannot hide an error through a small reduction weight. This
checks the loss boundary; full-model optimizer parity requires separate evidence.

The supported surrogate takes the minimum of the unclipped ratio advantage and
the clipped-ratio advantage, with independently configured lower/upper clip
widths. Reference KL uses the clipped k3 estimator. Singleton reward groups
retain their reward; groups with equal rewards produce zero centered advantages.
Normalization adds epsilon `1e-6` to the sample standard deviation.

## Offline learner

Run `uv run python -m marin.rl.train_grpo --config_path CONFIG.yaml` with an
`OfflineGrpoConfig`. Required fields are `captures` (ordered NPZ paths),
`initial_model`, `tokenizer`, and `hf_save_path`. Set
`trainer.num_train_steps` to the number of captures and `trainer.train_batch_size`
to each capture's trajectory count. Each complete batch must divide evenly by
`trainer.microbatch_size` when configured. The model configuration defaults to
Qwen3 and can be selected through Levanter's registered model configs.

Each capture receives one update. Before that update, captured raw old-policy
scores are replaced by current-learner scores; captured tokens, rewards, groups,
partitions, and reference scores remain fixed. Sampling-policy scores are
retained for provenance and do not enter importance correction. This is offline
replay, not an off-policy-corrected training algorithm. Scoring and training use
the same trajectory count and padded sequence width within each capture. Every
capture must match the configured full batch count; a different sequence width
can trigger recompilation. It disables dropout and masks padding as a separate
attention segment. Every update saves through the existing Levanter checkpointer.
To interrupt deliberately, set `stop_after` to an absolute step; resume with the
same checkpointer path, run ID, ordered captures, and recipe, removing `stop_after`.
The saved contract hashes capture contents and rejects changed inputs or recipes.
At the final step the learner exports HF weights to `hf_save_path`.
This uses the existing single-learner checkpoint lifecycle; stage-local pipeline
checkpoint integration is separate.

## Pipeline execution

`levanter.grpo_pipeline` provides model-independent stage execution. A model
implements `PipelineStage` for embeddings, blocks, final normalization, output
weights, trainable selection, and routing counters. Install the root `pipeline`
extra for JaxPP execution. The schedule is standard 1F1B; multihost execution
requires one process to own all devices of each stage. Expert and replica axes
must divide the device count evenly.

`packed_pipeline_batch` uses Levanter packing to place complete trajectories in
fixed-length rows. It preserves source position IDs, segment boundaries, and
precomputed objective weights; it rejects trajectories longer than the row.
Cross-segment targets and padding have zero objective weight. Choose row length
and microbatch dimensions explicitly; packing does not split trajectories.

The pipeline scorer applies `jax.lax.optimization_barrier` after casting
trainable weights inside each stage. This blocks the cast fusion that caused the
observed June scorer/training discrepancy; other models and shapes still require
validation. Frozen tensors retain their dtype. The training path keeps
FP32 master parameters and exposes FP32 or BF16 temporal gradient accumulation;
global gradient clipping precedes the stage-local optimizer updates.
Routing drops are rejected by default. `RoutingDropPolicy.REPORT` explicitly
allows capacity experiments and reports assignment counts, including padding.

`log_ratio_abs_max` measures the largest absolute difference between current and
old logprobs where `policy_weights > 0`. PPO clipping-pressure metrics count
ratios outside the configured clip interval with policy weights. `policy_kl`
is the reference-KL estimator and is zero when its coefficient is zero; that
zero alone does not demonstrate scorer agreement.

The complete June integration, including separate model/kernel fixes, measured
`log_ratio_abs_max == 0` before parameter changes in each of three packed 67B
updates on 32 H100s with PP4/EP8 and capacity 16. This compares recomputed
old-policy scores with training-forward scores where `policy_weights > 0`.
Those results validate the combined integration; they are not an isolated
validation of these learner modules.
See the [numerical investigation](https://marina.oa.dev/echo/wiki/363) for the
configuration, artifacts, and validation limits. The June entrypoint and online
rollout lifecycle are separate integration work.
