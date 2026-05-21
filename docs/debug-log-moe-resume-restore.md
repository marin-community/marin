# Debugging log for MoE resume restore

Resume the Grug MoE 1e23 ragged EP8 production run from the latest complete
checkpoint using explicit launcher environment propagation and a fresh
timestamped run id. A complete checkpoint is a `step-*` directory with
`metadata.json`.

## Initial status

The restored experiment-control worktree had recreated launcher/controller
scripts and the base Grug MoE launcher had permanent checkpoint retention set to
every 1000 steps. No saved controller state existed in `scratch/`, so no active
submission had been recorded in this worktree.

## Ray helper auth import order

The local controller helper could submit a Ray job, but `status` and `logs`
failed with Ray token-auth errors. The root cause was import order:
`fray.v1.cluster.ray` imported Ray auth helpers before `marin.run.ray_run` set
`RAY_AUTH_MODE=token`, and Ray cached auth as disabled.

## Changes to make

Update `scripts/debug/manage_grug_moe_ep8_ragged_48l_fix.py` so it imports
`marin.run.ray_run` before Fray/Ray dashboard modules. Also make the
`checkpoints` subcommand report that no checkpoints exist yet instead of
raising during the first save interval.

## Results

After the import-order fix, `status` and `logs` successfully queried the Ray
dashboard. The first submitted job,
`ray-run-dlwh-moe-resume51351-20260503-222156`, had already failed during Ray
runtime-env setup before driver logs were emitted.

## Runtime-env missing find-links

The failed job metadata showed pip could not install `kitoken==0.10.2`.
`uv export` emitted the locked `kitoken` pin, but Fray's Ray dependency builder
did not copy `[tool.uv].find-links` from `pyproject.toml`, so pip did not see
the GitHub release page containing the `0.10.2` wheel.

## Changes to make

Update both Ray dependency builders:

- `lib/fray/src/fray/v1/cluster/ray/deps.py`
- `lib/fray/src/fray/v2/ray_backend/deps.py`

They now read `[tool.uv].find-links` from `pyproject.toml` and add each entry to
the generated requirements file.

## Results

The generated v1 and v2 Ray requirements now include both workspace find-links,
including the `kitoken-0.10.2` release page. Local validation confirmed the
generated requirements contain `kitoken==0.10.2` plus the required
`--find-links` entries.

Relaunched from:

`gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/checkpoints/step-51351`

Current successful submission state:

- Ray submission: `ray-run-dlwh-moe-resume51351-20260503-222738`
- Run id: `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51351_clip15_20260503_222738`
- Output root: `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51351_clip15_20260503_222738-dfadf3`
- W&B run: `https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51351_clip15_20260503_222738`
- Executor info: `gs://marin-us-central2/experiments/launch_grug_moe_ep8_ragged_48l_fix-7c130f.json`

Last observed state: Ray submission `RUNNING`; the executor reached the Grug
step, acquired the step lock, submitted the Fray training job, initialized W&B,
and workers were loading dataset cache ledgers.

## Future Work

- [ ] Check for the first checkpoint after the save interval.
- [ ] Confirm training has restored past step 51351 and is logging loss.
- [ ] Consider deduplicating the v1/v2 Ray dependency generation code after this
      production resume is stable.

## Babysitter

Started a detached babysitter for the running Ray submission.

- Script: `scripts/debug/babysit_grug_moe_ep8_ragged_48l_fix.py`
- Screen session: `moe-resume-babysitter`
- PID file: `scratch/grug_moe_ep8_ragged_48l_babysitter.pid`
- State: `scratch/grug_moe_ep8_ragged_48l_babysitter_state.json`
- Events: `scratch/grug_moe_ep8_ragged_48l_babysitter_events.jsonl`
- Stdout: `scratch/grug_moe_ep8_ragged_48l_babysitter_stdout.log`

It checks Ray status and checkpoint visibility on cadence, and on terminal
failure relaunches through `manage_grug_moe_ep8_ragged_48l_fix.py launch` with
an explicit `--initialize-from`. It uses the latest checkpoint for the active
run when one exists; otherwise it falls back to the saved `initialize_from`
checkpoint. It also verifies the `kitoken`/`find-links` runtime-env fix before
retrying runtime-env failures.

## Babysitter automation hardening

The first babysitter-managed run failed with a JAX coordination-service fatal:
the distributed service reported that another task likely died, the leader was
preempted/restarted, or networking broke. The babysitter relaunched from the
trusted `step-51351` checkpoint and the replacement Ray submission is running:

- Ray submission: `ray-run-dlwh-moe-resume51351-20260503-225653`
- Run id: `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51351_clip15_20260503_225653`
- W&B run: `https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51351_clip15_20260503_225653`

The automation also had two local problems. The relaunch path inherited the
manager subprocess output, which printed Ray environment variables into the
babysitter stdout log. It now captures and redacts manager output before
printing. The terminal-failure handler also treated every unknown failed Ray job
as recoverable; it now only relaunches on known recoverable signals, including
the observed JAX coordination-service/preemption class, and stops on
unclassified terminal failures.

## Step-51262 fallback

The `step-51351` prefix contains OCDBT checkpoint data but does not contain
`metadata.json`, so the current Grug `initialize_from` validation rejects it.
`step-51262` has the expected metadata marker and was selected for the current
resume attempt.

Stopped the bad `step-51351` Ray submission and relaunched from `step-51262`:

- Ray submission: `ray-run-dlwh-moe-resume51262-20260503-231119`
- Run id: `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51262_clip15_20260503_231119`
- W&B run: `https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51262_clip15_20260503_231119`
- Output root: `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51262_clip15_20260503_231119-498087`

Last observed state: Ray submission `RUNNING`; executor acquired the Grug step
lock, W&B initialized with a fresh run id, and workers were loading dataset
cache ledgers. The detached babysitter was restarted with a fresh state file for
this `step-51262` run.

The launcher, manager, and babysitter now resolve checkpoints from
`step-*/metadata.json` marker files. The manager's default launch target is the
latest complete checkpoint under the trusted source prefix. Once a run is
launched, the babysitter uses the explicit launch checkpoint until that run
writes a newer complete checkpoint of its own.

## Step-51000 restart

The `step-51262` relaunch was stopped because its loss was unacceptable. The
run had reached a complete checkpoint at `step-51521`, but that lineage was not
kept for recovery. Recovery policy was adjusted so an explicit downgrade uses
the requested launch checkpoint until the new run writes a newer complete
checkpoint of its own.

Stopped:

- Ray submission: `ray-run-dlwh-moe-resume51262-20260503-231119`
- Run id: `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51262_clip15_20260503_231119`

Relaunched from:

`gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/checkpoints/step-51000`

Current running submission:

- Ray submission: `ray-run-dlwh-moe-resume51000-20260504-011838`
- Run id: `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_011838`
- Output root: `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_011838-d916e4`
- W&B run: `https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_011838`

Last observed state: Ray submission `RUNNING`; executor acquired the Grug step
lock and W&B initialized. Detached babysitter `moe-resume-babysitter` was
restarted on this submission.

## Resume determinism parity with classic Levanter

Classic Levanter checkpoints the per-step training RNG state. `TrainerState`
contains `training_key`, `Trainer._train_step` splits it and passes the split
key to the loss, and `TrainerState.saveable_state` explicitly marks
`training_key` saveable. The custom Grug train states do not carry a
`training_key`; `experiments/grug/moe/train.py` only checkpoints `step`,
`params`, `opt_state`, optional `ema_params`, and `pending_qb_betas`.

Verified the actual `step-51000` OCDBT key set:

- `training_key`: 0 keys
- `pending_qb_betas`: present
- `opt_state`: present
- `step`: present

This is a real parity gap with classic Levanter and would break bitwise resume
as soon as the Grug loss/model/optimizer path consumes per-step randomness.
For the current MoE code path, the model search found `jax.random` only in
initialization and data construction. The training loss calls
`next_token_loss(...)` without a key, and `next_token_loss`/router code does
not consume RNG at train time. The missing `training_key` is therefore a
checkpoint-state bug to fix, but it is not yet enough by itself to explain the
observed high-loss resume unless there is stochasticity outside the inspected
model/loss path.

Other state checked:

- QB router state is carried by `pending_qb_betas` and is saved in the
  checkpoint. This should preserve the first post-resume router-bias update.
- Data loading resumes from `train_loader.iter_from_step(int(state.step))`.
  Mixture sampling derives assignments from the fixed data seed and block
  index, so there is no mutable data RNG cursor to checkpoint for this loader.
- `allow_partial_checkpoint` defaults to false, so missing checkpoint leaves
  should fail instead of being silently initialized in the current config.

## Checkpoint leaf and source-bundle comparison

Compared OCDBT `zarr.json` manifests for these complete checkpoints:

- `resume50000` complete checkpoint:
  `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50000_clip15_20260502-faf9c1/checkpoints/step-50654`
- `resume50654` complete checkpoints:
  `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/checkpoints/step-51000`
  and `.../step-51262`

Each checkpoint has 90 tensor leaves and 9745 OCDBT keys. The leaf path,
shape, dtype, and chunk shape sets match exactly across `step-50654`,
`step-51000`, and `step-51262`. The critical MoE leaves are present in all
three:

- `pending_qb_betas`: shape `(48, 64)`, dtype `float32`, chunks `(48, 64)`
- `params/stacked_blocks/stacked/mlp/router_bias`: shape `(48, 64)`, dtype
  `float32`, chunks `(48, 64)`

`training_key` and `ema_params` are absent from all three. The absent
`training_key` remains a bitwise-resume parity gap with classic Levanter, but
the identical leaf/shape comparison means the current high-loss resume is not
explained by a missing or differently-shaped saved tensor leaf between these
checkpoints.

Ray job records still had the source packages for the earlier runs. Copied
them locally for comparison:

- `resume50000`: `gcs://_ray_pkg_cd6cd7365c4c4083.zip`, copied to
  `scratch/source_bundles/resume50000__ray_pkg_cd6cd7365c4c4083`
- `resume50654`: `gcs://_ray_pkg_7dff3fd4a63a36f2.zip`, copied to
  `scratch/source_bundles/resume50654__ray_pkg_7dff3fd4a63a36f2`
- bad `resume51000`: `gcs://_ray_pkg_74c1eec45cd9acf0.zip`, copied to
  `scratch/source_bundles/resume51000__ray_pkg_74c1eec45cd9acf0`

The MoE experiment files under `experiments/grug/moe` are identical across
these three bundles, ignoring Python bytecode. The important source drift is in
`lib/levanter/src/levanter/grug/grug_moe.py`. The earlier `resume50000` and
`resume50654` bundles computed `ragged_all_to_all` output offsets as
sender-side offsets:

```python
sender_output_offsets = jnp.cumsum(shard_counts, axis=0, dtype=shard_counts.dtype) - shard_counts
output_offsets = sender_output_offsets[shard_id]
```

The bad `resume51000` bundle had reverted that to receiver-local prefix sums:

```python
recv_sizes = shard_counts[:, shard_id]
output_offsets = jnp.cumsum(jnp.concatenate((jnp.array([0], dtype=recv_sizes.dtype), recv_sizes[:-1])))
```

JAX `ragged_all_to_all` expects output offsets describing where this sender's
slices should land on each receiver; JAX then exchanges those offsets
internally. The bad bundle therefore changed expert-parallel communication
semantics without changing checkpoint shape or W&B config. Since the one-off
launcher explicitly sets `moe_implementation="ragged_all_to_all"`, this is the
strongest current explanation for the immediate train-loss jump after resume.

## Ragged offset fix and corrected step-51000 relaunch

Restored the known-good sender-side offset implementation in
`lib/levanter/src/levanter/grug/grug_moe.py` and restored the regression
coverage in `lib/levanter/tests/grug/test_grugformer_moe.py`:

- `_shard_a2a_params` now returns sender-side output offsets again.
- `test_shard_a2a_params_uses_sender_side_output_offsets` checks the exact
  offset vector.
- `test_moe_mlp_ragged_matches_ring_with_ep_axis_when_available` is restored;
  it is skipped locally on CPU but runs where `ragged_all_to_all` exists.

Validation:

- `uv run --project lib/levanter --group test pytest tests/grug/test_grugformer_moe.py -k shard_a2a_params`: passed.
- `uv run --project lib/levanter --group test pytest tests/grug/test_grugformer_moe.py -k "shard_a2a_params or ragged_matches"`: one passed, one skipped on CPU.
- `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/tests/grug/test_grugformer_moe.py`: passed.

Stopped the bad `step-51000` resume job and relaunched from the same complete
checkpoint using the corrected source:

- Stopped Ray submission:
  `ray-run-dlwh-moe-resume51000-20260504-011838`
- Corrected Ray submission:
  `ray-run-dlwh-moe-resume51000-20260504-014844`
- Corrected run id:
  `moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_014844`
- `initialize_from`:
  `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/checkpoints/step-51000`
- Corrected Ray source package:
  `gcs://_ray_pkg_55c26154ca7c9975.zip`
- W&B:
  `https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_014844`

Last observed state after relaunch: Ray submission is `RUNNING`, W&B has
initialized, workers restored `step-51000`, TensorStore error checks completed
successfully, and the run reached fused cross-entropy setup. No first
`global_step`/loss or complete checkpoint had been logged yet.

## Overnight heartbeat: first good loss

At `2026-05-04T09:30Z`, the corrected `step-51000` run was still `RUNNING` and
had logged the expected loss scale:

- W&B `global_step`: `51093`
- W&B `train/loss`: `2.1476118564605713`
- log progress loss: around `2.12` to `2.16`
- first complete checkpoint under the corrected run:
  `gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_resume51000_clip15_20260504_014844-3cdab2/checkpoints/step-51076`

The detached babysitter process was alive but had not recorded a check after
`2026-05-04T09:09Z`, so it was restarted locally without touching the Ray job.
The fresh babysitter immediately checked the corrected run, saw status
`RUNNING`, and recorded checkpoint `step-51076`.
