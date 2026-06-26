# Codex fixes for Delphi midtraining

Date: 2026-05-16

Scope: audit and repair the post-redesign Delphi midtraining code after the
3e18 K=0.20 sweep showed WSD learning-rate curves in W&B.

## Current status

As of this update, rerunning the 3e18 K=0.20 sweep from the current code will
render the intended triangular LR schedule:

```text
warmup 0.1
decay None
stable_steps 0
```

The first a001 3e18 sweep remains WSD because those jobs already had
`decay: 0.2` baked into `train_lm_config.yaml` on GCS. These fixes only affect
freshly rendered launches, e.g. a002 or later.

Latest local verification:

```bash
uv run pytest \
  tests/midtraining/test_preflight.py \
  tests/midtraining/test_watch.py \
  tests/midtraining/test_schema.py \
  tests/midtraining/test_val_set_equivalence.py \
  tests/midtraining/test_levanter_config.py \
  tests/midtraining/test_spec_validators.py -q
```

Result: `77 passed in 3.27s`.

Repo gate on touched files:

```bash
./infra/pre-commit.py --fix \
  lib/marin/src/marin/midtraining/modes.py \
  lib/marin/src/marin/midtraining/watch.py \
  lib/marin/src/marin/midtraining/preflight.py \
  lib/marin/src/marin/midtraining/schema.py \
  lib/marin/src/marin/midtraining/launch.py \
  lib/marin/src/marin/midtraining/spec.py \
  lib/marin/src/marin/midtraining/__init__.py \
  experiments/midtrain_specs/delphi_small_cpt_k020.py \
  tests/midtraining/test_watch.py \
  tests/midtraining/test_preflight.py \
  tests/midtraining/test_schema.py \
  tests/midtraining/test_val_set_equivalence.py \
  tests/midtraining/test_spec_validators.py \
  .agents/logbooks/midtraining_redesign.md \
  .agents/logbooks/codex_fixes_midtraining.md
```

Result: `OK`.

## What was wrong

The 3e18 launcher inherited `base.decay_fraction = 0.20` and passed it to
`AdamHConfig.decay`. In Levanter, `warmup=0.10, decay=0.20` does **not** mean
"decay after warmup"; it means 10% warmup, 70% stable plateau, 20% decay. That
is WSD and does not match the legacy Delphi CPT runs.

The legacy launcher (`experiments/exp_delphi_math_10b_midtrain.py`) used:

```python
decay_steps = num_train_steps - warmup_steps
```

So the intended CPT shape is triangular: warmup, then immediate linear decay
over the full post-warmup remainder.

There was a second subtle trap: using `decay=0.90` is close but not exact for
all run lengths, because Levanter converts each fractional schedule value with
`int(frac * num_train_steps)`. For step counts that are not divisible by 10,
independently flooring 10% warmup and 90% decay can leave a one-step stable
plateau. The correct Levanter representation is `decay=None`, the scheduler
sentinel for full remainder decay.

## Code changed

- `lib/marin/src/marin/midtraining/modes.py`
  - Added canonical CPT schedule constants:
    - `CPT_DEFAULT_WARMUP_FRACTION = 0.10`
    - `CPT_DEFAULT_DECAY = None`
  - Documented why `None`, not `0.90`, is the safe triangular-decay value.
  - Removed unused `CptMode` optimizer knobs (`lr_factor`, `lr_multiplier`,
    `min_lr_ratio`, `warmup_tokens`, `warmup_fraction`). They were no-ops:
    callers could set them and the rendered config would not change.
  - Removed unused `ComputeProfile` fields (`tensor_parallel_size`,
    `priority`, `launch_spacing_seconds`). They were validated but never
    propagated to Fray/Iris or the training config.

- `lib/marin/src/marin/midtraining/__init__.py`
  - Re-exported `CPT_DEFAULT_DECAY` and `CPT_DEFAULT_WARMUP_FRACTION`.

- `experiments/midtrain_specs/delphi_small_cpt_k020.py`
  - Switched the optimizer builder to use the CPT schedule constants instead
    of `DelphiModel.warmup_fraction` / `DelphiModel.decay_fraction`.
  - Removed the dead `lr_factor` argument from `CptMode`; LR scaling remains
    in the rendered `optimizer_config`, where it actually takes effect.
  - Made the coordinator wait path return the child terminal status instead
    of always looking successful. If `wait()` unexpectedly returns a
    non-terminal status or raises transiently, the parent now sleeps and
    retries instead of exiting early and letting Iris cascade-kill the child.

- `lib/marin/src/marin/midtraining/launch.py`
  - `LaunchResult.wait()` now returns the backend job status instead of
    discarding it.

- `tests/midtraining/test_val_set_equivalence.py`
  - Added a regression test that renders the 3e18 K=0.20 config and asserts:
    - `warmup == 0.10`
    - `decay is None`
    - `lr_schedule == "linear"`
    - `num_train_steps == 7400`

- `.agents/logbooks/midtraining_redesign.md`
  - Updated the top policy callout and affected snippets to record the
    current policy: fractional warmup and Levanter full-remainder decay.
  - Added a note that optimizer knobs do not belong on `CptMode` unless the
    renderer consumes them.

## Verification

Focused tests:

```bash
uv run pytest tests/midtraining/test_val_set_equivalence.py tests/midtraining/test_levanter_config.py tests/midtraining/test_spec_validators.py -q
```

Result: `49 passed in 3.50s`.

Rendered 3e18 config check:

```text
num_train_steps 7400
warmup 0.1
decay None
lr_schedule linear
warmup_steps 740
decay_steps 6660
stable_steps 0
```

## Operational notes

The in-flight 3e18 K=0.20 sweep was not modified by these code changes. Those
jobs were launched with the old WSD config. Decide separately whether to keep
them as known-WSD data, kill/relaunch, or relaunch only selected cells.

The local Codex monitor loop was stopped during this audit. No Iris training
jobs were killed.

## Follow-ups not fixed here

- Refresh `experiments/throughput_stats.py` once completed W&B runs are
  available; the seeded 3e18 throughput anchor is still a mid-flight estimate.
- The redesign logbook is still partly historical and contains old sketches.
  I corrected the load-bearing policy text and misleading `CptMode` example,
  but did not rewrite the whole design document.
- If future launchers need priority control, add it end-to-end through Fray
  and Iris. Do not re-add a `ComputeProfile.priority` field unless it is
  actually propagated.

## Second audit pass against the redesign doc

Started after the schedule fix, at the user's request, to look for additional
Claude-introduced launch gotchas.

### Finding: TPU allowlist only ran in the CLI

The redesign doc treats per-base TPU allowlists as a spec-level guard:
unknown TPU profiles should fail before launch. The current
`experiments/midtrain_specs/delphi_small_cpt_k020.py` implementation checked
`ALLOWED_TPUS_PER_BASE` only in `main()`. Any Python driver importing
`build_spec()` directly could bypass the allowlist and render a launch with an
oversized or undersized TPU.

Fix applied:

- `build_spec()` now calls `_check_tpu_allowed()` immediately after resolving
  the requested/default TPU.
- The CLI no longer carries a separate, bypassable allowlist check.
- The stale comment claiming v5p-8 "fits all of them" was corrected.
- Added `test_small_base_build_spec_rejects_disallowed_tpu()` so imported
  Python drivers are covered, not just CLI execution.

### Finding: fresh CPT preflight ignored temporary checkpoints

The redesign doc says fresh launches must refuse both permanent and temporary
checkpoint namespaces. `preflight()` checked permanent checkpoints and the run
manifest, but did not check the run-scoped temporary checkpoint path unless the
spec was already marked as a resume. That leaves a narrow but real footgun:
if a prior attempt wrote temp checkpoints but the manifest/permanent path was
missing or cleaned inconsistently, a fresh launch could accidentally resume or
collide with the temp namespace.

Fix applied:

- `_check_run_namespace()` now checks the run-scoped temp checkpoint root for
  fresh CPT launches.
- Fresh CPT now fails if either permanent or temp checkpoints already exist.
- Added `test_fresh_cpt_refuses_existing_temp_checkpoint_namespace()`.

### Finding: resume checkpoint floor could ignore newer temp checkpoints

The resume path computed `permanent_latest or temp_latest`. If permanent
checkpoints existed at a low step and temporary checkpoints existed at a higher
step, the low permanent value won and preflight could incorrectly fail a valid
resume. Worse, future recovery code could reason from the wrong checkpoint
floor. The AGENTS.md resume rule requires checking both permanent and
temporary checkpoints.

Fix applied:

- Added `_latest_step_across()` and used it for CPT and cooldown resume floor
  checks.
- Added `test_resume_uses_latest_across_permanent_and_temp_checkpoints()`.

### Finding: startup proof was static and wrong for HF CPT / natural resume

The watcher used static expected/forbidden lines from the mode class. That
misses two real Levanter behaviors:

- The 3e18 sweep uses `CheckpointSourceKind.HF_WEIGHTS`. Levanter logs
  `No training checkpoint found. Initializing model from HF checkpoint`, not
  `Loading checkpoint from ...`. The old CPT startup proof would fail the
  live small-base launcher path.
- Natural resume calls Levanter's checkpoint loader and logs
  `Loading checkpoint from ...`. Cooldown mode was forbidding that phrase,
  which would mark a healthy natural resume as unhealthy.

Planned fix: make startup expectations depend on the resolved spec: CPT fresh
from HF expects the HF init line; CPT fresh from native Levanter expects
`Loading checkpoint from`; any resume/cooldown expects discovered checkpoint +
resume step and does not forbid generic checkpoint loading.

Fix applied:

- `watch.evaluate_startup()` now derives expected and forbidden lines from the
  resolved spec.
- Removed the stale static startup-line methods from `CptMode` and
  `CooldownMode`.
- Added coverage for HF CPT startup, CPT resume startup, and cooldown natural
  resume with Levanter's `Loading checkpoint from ...` line.
- Corrected the redesign doc startup-proof section so it distinguishes native
  CPT init, HF CPT init, and natural resume.

### Finding: run manifest omitted launch compute fields

The redesign doc says the manifest should record the compute profile and
`max_task_failures`, because those choices matter for replaying or explaining a
run. `RunManifestRow` recorded model/data/token facts but not the TPU type,
train batch size, per-device parallelism, or max task failures. Those values
were only recoverable indirectly from W&B tags or the rendered YAML.

Fix applied:

- `RunManifestRow` now requires `tpu_type`, `train_batch_size`,
  `per_device_parallelism`, and `max_task_failures`.
- `build_manifest_row()` writes those fields from `spec.compute`.
- `SCHEMA_VERSION` was bumped to 2 because the manifest contract changed.
- `tests/midtraining/test_schema.py` was updated so schema roundtrips cover the
  new required manifest fields.

### Finding: model-shape validation skipped max sequence length

The redesign doc requires shape metadata checks before launch. The validator
checked `hidden_dim` and `num_layers`, but not the rendered model
`max_seq_len`. A spec could carry `seq_len=4096` while rendering a model config
with a different max context length.

Fix applied:

- `_assert_model_config_matches_base()` now validates `max_seq_len` when the
  rendered model config carries that key.
- Added `test_model_config_max_seq_len_must_match_spec_seq_len()`.

### Finding: selector validation lived mostly in argparse

The CLI uses argparse choices, but `build_spec()` is the reusable cell-author
surface. Imported drivers could pass an unknown base, mix, or LR and get a
generic `KeyError` or a later failure instead of the explicit selector failure
the redesign doc calls for.

Fix applied:

- Added `_check_selectors()` and call it at the start of `build_spec()`.
- Added parameterized tests for bad base, mix, and LR selector values.

### Remaining design-doc gaps not fixed in this pass

These are real gaps, but I did not patch them because they need live-service
plumbing or a larger operator workflow change:

- **W&B existence preflight**: the design doc says fresh launch should refuse
  if the target W&B run id already exists. Current preflight catches W&B step
  regression from startup logs, but it does not query W&B before launch.
- **Startup proof is not wired into the CLI wait loop**: `watch.evaluate_startup()`
  is now less wrong, but `delphi_small_cpt_k020.py` still submits and waits for
  terminal status; it does not stream startup logs and kill early on bad proof.
- **`launch_command.txt` is not written**: `RunIdentity` and schema constants
  name this artifact, but the one-cell script does not write the actual launch
  command. That weakens later forensic replay.
- **The redesign doc still contains historical sketches**: I corrected the
  load-bearing policy and startup-proof sections touched by this audit, but
  the doc still has older pseudocode blocks that are not the current API.

## 2026-05-16 a002 relaunch failure audit

The a002 triangular relaunch did not fail because "v5p cannot launch many jobs
in parallel." It failed because the outer CPU coordinators were submitted with
`--preemptible`, so Iris packed all twelve coordinator tasks onto the same
preemptible TPU worker:

```text
marin-tpu-v5p-preemptible-8-us-east5-a-20260516-0924-89d80509-worker-0
```

When that one worker missed ping and was marked failed, all twelve parent
coordinators were retried. The retry then hit our fresh-launch preflight guard:

```text
Fresh CPT launch refused: manifest already at
gs://marin-us-east5/checkpoints/...-a002/midtrain_manifest.json
```

Iris evidence checked:

- `job_config.submit_argv_json` for `/ahmed/aa-d3e18-p33m67-lr33-a002-1778963071`
  shows the parent was CPU-only but explicitly submitted with `--preemptible`.
- `job_config.constraints_json` for that parent includes
  `preemptible=true`.
- `task_attempts` for every a002 parent show attempt 0 as
  `WORKER_FAILED` on the same preemptible v5p worker, then attempt 1 as
  `FAILED exit=1`.
- Parent logs show attempt 0 submitted the child, then attempt 1 restarted
  the launcher and failed on the already-written run manifest.
- Six child jobs also hit a transient startup failure:
  `TypeError: the JSON object must be str, bytes or bytearray, not NoneType`
  from the GCS/gcsfs path. That is real, but secondary; the sweep-wide kill
  came from parent failure and cascade-kill.

Why this did not affect unrelated jobs: the failure was concentrated by our
launch topology. We placed twelve long-lived parents on one preemptible worker;
when it failed, all children lost their parent at once. Other users' training
jobs are not all children of that same single CPU worker.

Fixes applied:

- `experiments/midtrain_specs/delphi_small_cpt_k020.py` now refuses to launch
  a non-dry-run coordinator if Iris says the parent is constrained to
  `preemptible=true` or the current worker id contains `-preemptible-`.
  The error tells the operator to launch the outer job as CPU-only without
  `--preemptible` (or with `--no-preemptible`). The nested TPU child remains
  preemptible.
- `preflight()` now has an explicit
  `allow_existing_matching_manifest` path for **same Iris task attempt
  retries only**. It reads the existing manifest, compares the run identity
  and resolved launch fields, and accepts the namespace only if they match.
  Normal fresh launches still reject an existing manifest/checkpoint
  namespace.
- The one-cell launcher passes `allow_existing_matching_manifest=True` only
  when `IRIS_TASK_ID` resolves to retry attempt id `> 0`; a new top-level
  rerun of the same `--attempt` still fails and must bump the attempt.
- Verified Iris/Fray existing-job semantics: Fray submits child jobs with
  `EXISTING_JOB_POLICY_KEEP`; Iris keeps only non-terminal existing jobs and
  replaces finished/killed ones. So a same-parent retry can either adopt a
  still-running child or recreate a child that was killed by the parent
  preemption.
- `ComputeProfile.max_retries_failure` now defaults to `3` and is propagated
  into the Fray/Iris child `JobRequest`. This gives transient startup errors
  like the observed gcsfs JSON `None` failure a small retry budget without
  changing the training config.

Relaunch guidance after this incident:

- Do not reuse a002 for the real sweep. The a002 run ids have failed W&B/GCS
  artifacts; launch a003.
- Do not delete the successful a001 WSD outputs unless doing a deliberate
  cleanup pass; they are now historical bad-policy reference runs.
- Launch each outer coordinator as stable CPU, not preemptible CPU:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-east5 \
  --priority interactive \
  --no-preemptible \
  --job-name aa-d3e18-<mix>-lr<lr>-a003-<timestamp> \
  --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -- python experiments/midtrain_specs/delphi_small_cpt_k020.py \
    --base 3e18 --mix <mix> --lr <lr> --attempt 3
```

Verification for this patch:

```bash
uv run pytest \
  tests/midtraining/test_preflight.py \
  tests/midtraining/test_spec_validators.py \
  tests/midtraining/test_watch.py \
  tests/midtraining/test_schema.py \
  tests/midtraining/test_val_set_equivalence.py \
  tests/midtraining/test_levanter_config.py -q
```

Result: `82 passed in 3.14s`.

## 2026-05-16 a003 relaunch by Codex

Fresh `a003` launch was done directly after validating the a003 namespaces
were empty:

```bash
for mix in p33m67 p50m50 p67m33; do
  for lr in 0.33 0.50 0.67 0.83; do
    uv run python experiments/midtrain_specs/delphi_small_cpt_k020.py \
      --base 3e18 --mix "$mix" --lr "$lr" --attempt 3 --dry-run
  done
done
```

Result: all 12 dry-runs passed before submitting any Iris job.

Launch policy:

- Fresh attempt: `--attempt 3`; no attempt to resume the poisoned a002
  namespace.
- Parent coordinators: CPU-only Iris jobs with `--no-preemptible`.
- TPU children: still use the configured v5p preemptible training pool.
- Submission spacing: 20 seconds between cells to avoid another simultaneous
  GCS config-fetch burst.
- Monitoring state file:
  `scratch/20260516-1424_delphi_3e18_a003_monitoring_state.json`.

Submitted parent jobs:

- `/ahmed/aa-d3e18-p33m67-lr33-a003-1778966347`
- `/ahmed/aa-d3e18-p33m67-lr50-a003-1778966347`
- `/ahmed/aa-d3e18-p33m67-lr67-a003-1778966347`
- `/ahmed/aa-d3e18-p33m67-lr83-a003-1778966347`
- `/ahmed/aa-d3e18-p50m50-lr33-a003-1778966347`
- `/ahmed/aa-d3e18-p50m50-lr50-a003-1778966347`
- `/ahmed/aa-d3e18-p50m50-lr67-a003-1778966347`
- `/ahmed/aa-d3e18-p50m50-lr83-a003-1778966347`
- `/ahmed/aa-d3e18-p67m33-lr33-a003-1778966347`
- `/ahmed/aa-d3e18-p67m33-lr50-a003-1778966347`
- `/ahmed/aa-d3e18-p67m33-lr67-a003-1778966347`
- `/ahmed/aa-d3e18-p67m33-lr83-a003-1778966347`

Immediate post-launch checks still need to verify:

- parents are not on preemptible CPU;
- children were created for all cells;
- rendered `train_lm_config.yaml` has `optimizer.decay: null`;
- early logs reach config fetch/training startup rather than preflight
  rejection.

Immediate post-launch check at about 21:27Z:

- `8/12` parents running; `4/12` parents pending on on-demand CPU capacity.
- `5/12` children running; `3/12` children pending on v5p capacity; `4/12`
  children not yet created because their parents were still pending.
- No a003 parent or child failures observed.
- The first running child
  (`delphi-3e18-p33m67-k0p20-lr33-a003`) rendered:
  `optimizer.warmup: 0.1`, `optimizer.decay: null`,
  `optimizer.min_lr_ratio: 0.0`, `num_train_steps: 7400`.
- Logs for that child reached HF checkpoint load, train-step tracing, the
  first train step, and step-0 eval. This is past the manifest/preflight
  failure point that killed a002. It had data-loader stalls during eval, so
  continue watching throughput rather than assuming healthy training speed.

Five-minute follow-up at about 21:33Z:

- Still no a003 failures or killed jobs.
- State remained `8/12` parents running, `4/12` parents pending on CPU,
  `5/12` children running, `3/12` children pending on v5p.
- The p33m67/lr33 child completed step-0 eval, logged `eval loss: 2.869`,
  completed `First train step ... (step 0)`, and reported train progress at
  step 2 with loss 2.82. That confirms the run is past JIT/eval startup and
  into actual training, though eval/data loading remains slow.

Status check at about 21:42Z:

- No a003 parent or child is failed/killed.
- Parent coordinators:
  - Running: all four `p33m67`, all four `p50m50`, and `p67m33/lr33`
    (`9/12`).
  - Pending on on-demand CPU: `p67m33/lr50`, `p67m33/lr67`,
    `p67m33/lr83` (`3/12`).
- Child jobs:
  - Actively training: `p33m67/lr33`, `p33m67/lr50`, `p33m67/lr67`,
    `p33m67/lr83`.
  - Waiting for TPU capacity: `p50m50/lr50`, `p50m50/lr67`,
    `p50m50/lr83`, `p67m33/lr33`.
  - Created but waiting after TPU preemption: `p50m50/lr33`; its child
    reached startup, then lost the v5p worker before a first permanent
    checkpoint. The parent remained alive, so this is normal TPU churn, not
    the a002 parent-preemption failure mode.
  - Not yet created: `p67m33/lr50`, `p67m33/lr67`, `p67m33/lr83`, because
    their parents are still pending CPU.
- Progress from logs:
  - `p33m67/lr33`: step `557/7400`, loss `2.45`; past first checkpoint.
  - `p33m67/lr50`: step `742/7400`, loss `2.05`; past first checkpoint.
  - `p33m67/lr67`: step `2/7400`, loss `2.82`; through step-0 eval.
  - `p33m67/lr83`: step `2/7400`, loss `2.82`; through step-0 eval.
  - `p50m50/lr33`: preempted during step-0 eval/startup, currently waiting
    on a replacement v5p worker.

Follow-up on W&B "crashed" label for `delphi-3e18-p50m50-k0p20-lr33-a003`:

- Iris child summary showed `State: running`, `preemptions=1`.
- The logs show the first worker started W&B at `21:26Z`, reached first train
  progress, then lost the v5p worker before a permanent checkpoint.
- A replacement worker started at `21:44Z` and W&B logged:
  `Resuming run delphi-3e18-p50m50-k0p20-lr33-a003`.
- W&B API lookup returned `state=running` after the retry, so the UI
  "crashed" label was the transient state of the preempted first process
  attempt, not the current Iris job state.

## 2026-05-16 throughput registry populated from W&B

Updated `experiments/throughput_stats.py` from the W&B mining pass. The
registry now has 8 anchors:

- `3e18` on `v5p-8`
- canonical true-midtrain `1e20` on `v5p-32`
- canonical true-midtrain `1e21` on `v5p-64`
- legacy `1e21-v5` on `v5p-64` and `v5p-256`
- legacy `1e22-v5` on `v5p-64`, `v5p-256`, and `v5p-512`

The contaminated legacy `1e20-iso-d2048-L21` run was intentionally not added
as a canonical anchor. Notes in the registry spell out the Iris-vs-W&B v5p
count convention: e.g. Iris `v5p-512` appears as 256 W&B accelerators.

Verification:

```bash
uv run python experiments/throughput_stats.py --list
uv run python experiments/throughput_stats.py \
  --model 1e22-v5 --tpu v5p-512 --steps 2384 \
  --train-batch-size 1024 --seq-len 4096
./infra/pre-commit.py --fix experiments/throughput_stats.py \
  .agents/logbooks/codex_fixes_midtraining.md
```

Result: all commands passed.

## 2026-05-16 20-step Delphi hardware probes

Goal: get cheap throughput/HBM signal for `p67m33/lr50` at the smaller
Delphi scales across `v5p-8`, `v6e-4`, and `v6e-8`. These are **not**
quality-comparable midtraining cells; each run is fixed at 20 train steps and
is visibly named/tagged as a probe.

Code changes:

- `experiments/midtrain_specs/delphi_small_cpt_k020.py`
  - Added `--probe-steps`, implemented as `BudgetPolicy.fixed_steps(...)`.
  - Requires `--run-suffix probe-*` whenever probe mode is used.
  - Adds W&B tags: `probe:throughput-hbm`, `probe_steps:<n>`, and
    `do_not_compare:quality`.
  - Extended the TPU allowlist only for the probe-relevant cases:
    `2e19` on `v6e-{4,8}`, `3e19` on `v6e-{4,8}`, and `9e19` on `v6e-8`.
- `tests/midtraining/test_val_set_equivalence.py`
  - Added coverage that probe mode renders `num_train_steps: 20` and carries
    the explicit probe tags.
  - Added coverage that `--probe-steps` refuses non-`probe-*` suffixes.

Verification:

```bash
uv run pytest tests/midtraining/test_val_set_equivalence.py -q
./infra/pre-commit.py --fix \
  experiments/midtrain_specs/delphi_small_cpt_k020.py \
  tests/midtraining/test_val_set_equivalence.py
```

Result: `36 passed`; pre-commit passed.

Launch path note:

- First tried stable CPU coordinator jobs, matching the a003 sweep safety
  pattern. Those queued on scarce on-demand CPU (`need 1 cores`, then
  `need 0.1 cores`) and were killed before they ran.
- Final launch bypassed the coordinator layer: configs/manifests were rendered
  locally with the same validation/preflight/write path, then the Levanter
  training command was submitted directly as top-level TPU Iris jobs with
  `--priority interactive`, `--enable-extra-resources`, `--tpu <type>`, and
  `--extra marin:tpu`.

Final submitted probe jobs (`a003`):

| run id | tpu | Iris job |
| --- | --- | --- |
| `delphi-2e19-p67m33-k0p20-lr50-probe-v5p8-20s-a003` | `v5p-8` | `/ahmed/midtrain-delphi-2e19-p67m33-k0p20-lr50-probe-v5p8-20s-a003` |
| `delphi-2e19-p67m33-k0p20-lr50-probe-v6e4-20s-a003` | `v6e-4` | `/ahmed/midtrain-delphi-2e19-p67m33-k0p20-lr50-probe-v6e4-20s-a003` |
| `delphi-2e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` | `v6e-8` | `/ahmed/midtrain-delphi-2e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` |
| `delphi-3e19-p67m33-k0p20-lr50-probe-v5p8-20s-a003` | `v5p-8` | `/ahmed/midtrain-delphi-3e19-p67m33-k0p20-lr50-probe-v5p8-20s-a003` |
| `delphi-3e19-p67m33-k0p20-lr50-probe-v6e4-20s-a003` | `v6e-4` | `/ahmed/midtrain-delphi-3e19-p67m33-k0p20-lr50-probe-v6e4-20s-a003` |
| `delphi-3e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` | `v6e-8` | `/ahmed/midtrain-delphi-3e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` |
| `delphi-9e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` | `v6e-8` | `/ahmed/midtrain-delphi-9e19-p67m33-k0p20-lr50-probe-v6e8-20s-a003` |

Initial status after submission:

- All seven direct TPU jobs were accepted by Iris at interactive priority.
- All seven were initially pending on TPU availability, not CPU
  coordinators:
  - `v5p-8`: pending, `need 4, available 0`.
  - `v6e-4`: pending, `need 4, available 0`.
  - `v6e-8`: pending, `need 8, available 0`.

Follow-up: the direct TPU `a003` submissions used Iris's default task memory
(`1GB`) because the coordinator path had previously hidden that detail. The
first two scheduled `v6e-4` tasks failed with:

```text
Exit code 137: OOM killed (container exceeded memory limit)
Container was OOM killed by the kernel
```

This was host/container RAM, **not TPU HBM**. There were no XLA/JAX HBM
messages such as `RESOURCE_EXHAUSTED`, `Program hbm requirement`, or
`Largest program allocations in hbm`.

Relaunch correction:

- Stopped the remaining pending `a003` direct TPU probe jobs.
- Relaunched all seven as `a004` with `--memory 16GB --disk 20GB`.
- `2e19/v6e-4/a004` survived startup and remained running; logs showed W&B
  initialization and cache loading with no HBM/OOM error.
- `2e19/v5p-8/a004`, `3e19/v5p-8/a004`, and `3e19/v6e-4/a004` still hit
  container-RAM exit 137 at 16GB.
- Stopped the pending `v6e-8/a004` jobs before they could hit the same false
  host-RAM limit.
- Relaunched every not-yet-useful probe as `a005` with
  `--memory 64GB --disk 50GB`:
  - `delphi-2e19-p67m33-k0p20-lr50-probe-v5p8-20s-a005`
  - `delphi-2e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005`
  - `delphi-3e19-p67m33-k0p20-lr50-probe-v5p8-20s-a005`
  - `delphi-3e19-p67m33-k0p20-lr50-probe-v6e4-20s-a005`
  - `delphi-3e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005`
  - `delphi-9e19-p67m33-k0p20-lr50-probe-v6e8-20s-a005`

Active monitor set at `2026-05-16T23:03Z`:

- Running: `delphi-2e19-p67m33-k0p20-lr50-probe-v6e4-20s-a004`.
- Pending on TPU/worker availability: all six `a005` replacements above.
- No real HBM signal yet; all observed failures so far were container RAM.

Further correction:

- `2e19/v6e-4/a004` also eventually failed with container-RAM exit 137
  while reading the 3.35 GB HF safetensors file.
- Relaunched it as
  `delphi-2e19-p67m33-k0p20-lr50-probe-v6e4-20s-a005` with
  `--memory 64GB --disk 50GB`.
- At `2026-05-16T23:18Z`, `3e19/v6e-4/a005` reached first train step,
  logged eval losses, reached train step 19, and saved a checkpoint at
  `.../probe-v6e4-20s-a005/checkpoints/step-19` with no HBM error. This is
  the first useful v6e probe datapoint from the batch.
- `9e19/v6e-8/a005` loaded both safetensor shards and started JIT, so it
  cleared the previous host-RAM failure point; final fit/throughput still
  pending as of this note.

Throughput extraction at `2026-05-16T23:31Z`:

- `3e19/v6e-4/a005` finished successfully. W&B has 20 history rows and a
  stable `throughput/duration` median of `1.300s` for `global_step >= 5`
  (`train_batch_size=32`, `seq_len=4096`, about `100.8k tok/s`).
- `9e19/v6e-8/a005` reached the step-19 Levanter checkpoint, then failed
  during HF export with container-RAM exit 137 while saving a 4.97 GB shard.
  The training window is still valid: W&B stable median is `1.988s/step`
  (`train_batch_size=64`, `seq_len=4096`, about `131.8k tok/s`). This is not
  a TPU-HBM failure.
- `2e19/v5p-8/a005` logged 19 train-step metric rows before a preemption.
  W&B stable median is `0.511s/step` (`train_batch_size=16`,
  `seq_len=4096`, about `128.2k tok/s`). Treat it as a measured but
  preempted probe anchor.
- No usable throughput rows were present yet for `2e19/v6e-4`,
  `2e19/v6e-8`, `3e19/v5p-8`, or `3e19/v6e-8` at this checkpoint.
- Added the three measured probe anchors to
  `experiments/throughput_stats.py` with caveats in the notes field.

Probe status check at `2026-05-16T23:37Z`:

- `2e19/v6e-8/a005` has since succeeded. W&B stable median is
  `0.345s/step` (`train_batch_size=16`, `seq_len=4096`, about
  `189.7k tok/s`). Added this as a fourth probe anchor in
  `experiments/throughput_stats.py`.
- `2e19/v6e-4/a005` is running again after one preemption. Logs show W&B
  resumed the same run and Levanter is in the startup eval after the first
  train step (`eval 1.57k/1.79k` at `23:36Z`), so the hold-up is eval
  overhead plus preemption, not HBM.
- A follow-up W&B read found `2e19/v6e-4/a005` had emitted 19 train-step
  metric rows. Stable median is `0.573s/step` (`train_batch_size=16`,
  `seq_len=4096`, about `114.3k tok/s`), so this is now also recorded as a
  probe anchor even if the Iris job is still doing final eval/export.
- `3e19/v6e-8/a005` is also running again after one preemption. Logs show it
  resumed the same run, loaded eval caches, and selected the fused
  cross-entropy implementation; it had not yet emitted throughput rows at
  this checkpoint.
- `2e19/v5p-8/a005` is pending after being preempted, but it already logged
  a useful throughput window before the preemption.
- `3e19/v5p-8/a005` is pending after preemption and has no useful
  throughput rows yet.
- Operational lesson: these 20-step probes are still paying full startup
  eval and final HF-export costs. For future probes, disable eval and HF
  export; otherwise wall-clock is dominated by non-training work and the
  large 9e19 probe can fail after training during HF export even though TPU
  HBM fit and train throughput were fine.

Probe + 3e18 sweep checkpoint at `2026-05-16T23:40Z`:

- `experiments/throughput_stats.py` now contains every probe datapoint that
  has a usable stable train-window metric:
  - `2e19/v5p-8`: `0.511s/step`, `128.2k tok/s` (preempted later).
  - `2e19/v6e-4`: `0.573s/step`, `114.3k tok/s` (running through
    eval/export after useful train rows).
  - `2e19/v6e-8`: `0.345s/step`, `189.7k tok/s` (succeeded).
  - `3e19/v6e-4`: `1.300s/step`, `100.8k tok/s` (succeeded).
  - `9e19/v6e-8`: `1.988s/step`, `131.8k tok/s` (trained to step 19, then
    container-RAM OOM during HF export).
- Not added because there are no usable train-window rows yet:
  `3e19/v5p-8` and `3e19/v6e-8`.
- Follow-up check found `3e19/v6e-8/a005` had emitted 19 train-step metric
  rows. Stable median is `0.770s/step` (`train_batch_size=32`,
  `seq_len=4096`, about `170.2k tok/s`). Added it to
  `experiments/throughput_stats.py` as a measured train-throughput anchor;
  Iris was still marking the job running afterward, so this is not an
  end-to-end completion anchor.
- 3e18 triangular a003 sweep is **not finished**. W&B progress by cell:

| cell | max train step / 7400 | W&B state |
| --- | ---: | --- |
| `p33m67-lr33` | 4440 | running |
| `p33m67-lr50` | 4440 | running |
| `p33m67-lr67` | 3700 | running |
| `p33m67-lr83` | 3700 | running |
| `p50m50-lr33` | 3145 | running |
| `p50m50-lr50` | 1295 | running |
| `p50m50-lr67` | 1295 | running |
| `p50m50-lr83` | 1110 | running |
| `p67m33-lr33` | 2590 | running |
| `p67m33-lr50` | 1110 | running |
| `p67m33-lr67` | 555 | running |
| `p67m33-lr83` | no W&B run yet | parent pending on CPU |

Iris confirms the first 11 a003 child train jobs are still `running`; the
`p67m33-lr83` parent is still `pending` with insufficient CPU. The sweep is
making progress but is nowhere near complete.

Full 9e18 + 2e19 launch checkpoint at `2026-05-17T00:22Z`:

- User requested a full K=0.20 sweep for two more isoflop budgets with batch
  priority: `9e18` on `v6e-4`, and `2e19` on `v6e-8`.
- Rendered and preflighted all 24 cells via
  `experiments/midtrain_specs/delphi_small_cpt_k020.py` before submission:
  three mixes (`p33m67`, `p50m50`, `p67m33`) by four LR factors
  (`0.33`, `0.50`, `0.67`, `0.83`) for each base.
- Wrote manifests and train configs to canonical a001 output paths under
  `gs://marin-us-east5/checkpoints/`:
  - `delphi-9e18-<mix>-k0p20-lr<factor>-a001`, `num_train_steps=8819`,
    `tpu_type=v6e-4`.
  - `delphi-2e19-<mix>-k0p20-lr<factor>-a001`, `num_train_steps=10983`,
    `tpu_type=v6e-8`.
- Submitted the 24 training jobs directly as top-level Iris TPU jobs rather
  than CPU-coordinator jobs, so the actual training jobs carry
  `--priority batch`.
- Submit log: `/var/folders/47/lpjssw6541b61gw4dj1645z80000gn/T/tmp.6vyGaBydZt`.
- Iris verification immediately after submission:
  - `9e18/v6e-4`: 12 jobs present; 2 running
    (`p33m67-lr33`, `p33m67-lr50`) and 10 pending with
    `Scheduler: Insufficient TPUs (need 4, available 0)`.
  - `2e19/v6e-8`: 12 jobs present; all 12 pending with
    `Scheduler: Insufficient TPUs (need 8, available 0)`.
- Operational caveat: these were submitted under batch priority as requested,
  so queueing behind existing interactive work is expected. They are launched
  but not expected to all start concurrently unless v6e capacity frees up.

Monitoring start at `2026-05-17T00:23Z`:

- Started a Codex-owned monitor for the canonical full-sweep jobs under
  `/ahmed/midtrain-delphi-9e18` and `/ahmed/midtrain-delphi-2e19`.
- State file:
  `scratch/20260516-1723_midtrain_sweep_monitoring_state.json`.
- Cadence is 180 seconds. The monitor reports summary counts and flags any
  terminal non-success state. It does not stop or resubmit jobs without
  explicit user approval.

Monitoring correction at `2026-05-17T00:45Z`:

- A state-only Iris monitor hid useful information: `running` jobs can still
  have been preempted and restarted under the same job id.
- Sampled summaries for the three running `9e18` jobs:
  - `p33m67-lr67` is healthy and has saved a temp checkpoint at step 401;
    latest sampled progress was at least step 602.
  - `p33m67-lr33` and `p33m67-lr50` each show `preemptions=1` and are
    restarting from dependency sync/build rather than continuing from a
    useful checkpoint.
- Replaced the monitor with a preemption-aware heartbeat that still reports
  state counts but also summarizes preempted running jobs.

Heartbeat at `2026-05-17T00:45Z`:

- v6e capacity opened up after the initial batch queue.
- Iris state counts:
  - `9e18/v6e-4`: 12 running, 0 pending, 0 failed/killed.
  - `2e19/v6e-8`: 8 running, 4 pending, 0 failed/killed.

Heartbeat/log sample at `2026-05-17T00:52Z`:

- State counts remained stable: `9e18` all 12 running; `2e19` 8 running and
  4 pending; no failed/killed jobs.
- Sampled `2e19/v6e-8` startup logs for representative cells. Two had W&B
  runs and reached the first train step. No sampled HBM/OOM/resource errors.

W&B progress check at `2026-05-17T00:55Z`:

- `9e18` has train rows for the three earliest-running cells:
  `p33m67-lr33` at 126/8819, `p33m67-lr50` at 92/8819, and
  `p33m67-lr67` at 999/8819. Other `9e18` cells had W&B runs but no train
  rows yet.
- `2e19` is still mostly in startup. Sampled W&B rows were only tiny early
  progress (`p50m50-lr67` at step 0, `p50m50-lr83` at step 3); several
  pending cells had no W&B run object yet.

W&B/Iris mismatch at `2026-05-17T01:07Z`:

- W&B marked `delphi-2e19-p50m50-k0p20-lr83-a001` as `crashed` after only
  3 train rows.
- Iris summary shows the real cause is a TPU preemption, not HBM/OOM or a
  code crash: job state is still `running`, `preemptions=1`, and the task is
  restarting from dependency sync under the same Iris job id.
- Because this happened before a useful checkpoint, it will re-run startup
  work; no manual resubmit is warranted unless Iris later reaches a terminal
  non-success state or repeatedly preempts before checkpointing.

Targeted preemption pass at `2026-05-17T01:13Z`:

- Iris job state still has no terminal failures, but TPU preemptions are
  widespread under the batch-launched v6e jobs.
- `9e18` jobs with observed preemptions:
  - `p67m33-lr83`: 1
  - `p67m33-lr67`: 1
  - `p67m33-lr50`: 1
  - `p67m33-lr33`: 2
  - `p50m50-lr83`: 1
  - `p50m50-lr67`: 2
  - `p50m50-lr50`: 2
  - `p50m50-lr33`: 1
  - `p33m67-lr83`: 1
  - `p33m67-lr50`: 2
  - `p33m67-lr33`: 2
- `2e19` active jobs with observed preemptions:
  - `p50m50-lr83`: 2
  - `p50m50-lr67`: 1
  - `p50m50-lr50`: 1
  - `p50m50-lr33`: 1
  - `p33m67-lr83`: 1
  - `p33m67-lr67`: 1
  - `p33m67-lr50`: 1
  - `p33m67-lr33`: 1
- Interpretation: these are still live Iris jobs, not terminal crashes, but
  several cells are likely re-paying startup work before their first useful
  checkpoint. Continue monitoring for cells that fail to cross the checkpoint
  window after repeated retries.

Monitor adjustment at `2026-05-17T01:16Z`:

- Reverted the automatic heartbeat to state counts only. That signal is
  reliable for terminal/non-terminal state.
- Preemptions are now checked with explicit targeted summary passes, because
  the aggregate heartbeat can race against restarts and produce stale-looking
  preemption subfields.
- Current state-only heartbeat: `9e18` 12 running / 0 pending; `2e19` 8
  running / 4 pending; no terminal failures.

W&B progress recovery at `2026-05-17T01:22Z`:

- The preempted cells are now making real train progress.
- `9e18`: all 12 cells have W&B train rows. Most are at or past step 599;
  `p33m67-lr67` is furthest at 2199/8819. Current laggards are
  `p50m50-lr50` and `p50m50-lr67` at 199/8819, and `p67m33-lr33` at
  399/8819.
- `2e19`: the 8 allocated cells have W&B train rows. Seven are at
  599/10983; `p50m50-lr83` is at 199/10983 after its preemption restarts.
  The four pending `p67m33-*` cells have no W&B run yet.

3e18 triangular a003 status at `2026-05-17T01:28Z`:

- The `3e18` a003 sweep is **not complete**.
- W&B progress:
  - `p33m67-lr33`: finished at 7399/7400.
  - `p33m67-lr50`: finished at 7399/7400.
  - `p33m67-lr67`: running at 7214/7400.
  - `p33m67-lr83`: running at 6659/7400.
  - `p50m50-lr33`: running at 6104/7400.
  - `p50m50-lr50`: running at 5549/7400.
  - `p50m50-lr67`: running at 4254/7400.
  - `p50m50-lr83`: running at 4254/7400.
  - `p67m33-lr33`: running at 5734/7400.
  - `p67m33-lr50`: running at 3884/7400.
  - `p67m33-lr67`: running at 4439/7400.
  - `p67m33-lr83`: W&B run exists but no train row yet.
- Iris confirms all remaining a003 jobs are running, with no terminal
  failures. The laggard `p67m33-lr83` is newly allocated on v5p-8, has
  `preemptions=0`, and is loading the HF checkpoint after eval/cache setup.

Post-run status at `2026-05-18T17:41Z`:

- Stopped the stale background monitor for the `9e18`/`2e19` batch sweeps
  because those jobs have all reached terminal states.
- `3e18` triangular `a003`: all 12 cells succeeded in Iris and W&B shows
  `finished:7399/7400` for every cell.
- `9e18` full sweep on `v6e-4`: all 12 jobs failed with exit 137
  (`OOM killed (container exceeded memory limit)`). W&B train progress before
  failure:
  - `p33m67`: `lr33=4399/8819`, `lr50=4399/8819`,
    `lr67=4399/8819`, `lr83=8799/8819`.
  - `p50m50`: `lr33=4403/8819`, `lr50=4404/8819`,
    `lr67=4404/8819`, `lr83=4404/8819`.
  - `p67m33`: `lr33=4404/8819`, `lr50=4399/8819`,
    `lr67=4399/8819`, `lr83=4404/8819`.
- `2e19` full sweep on `v6e-8`: all 12 jobs failed with exit 137
  (`OOM killed (container exceeded memory limit)`). W&B train progress before
  failure:
  - `p33m67`: `lr33=4391/10983`, `lr50=3293/10983`,
    `lr67=4391/10983`, `lr83=4386/10983`.
  - `p50m50`: all four LRs reached `3293/10983`.
  - `p67m33`: `lr33=2178/10983`, `lr50=3293/10983`,
    `lr67=3293/10983`, `lr83=3291/10983`.
- Representative Iris summaries show the exit 137 happened while saving
  HuggingFace `model.safetensors` shards, e.g. `9e18-p50m50-lr67` failed
  while saving `hf/step-4405`, and `2e19-p50m50-lr50` failed while saving a
  3.35 GB shard. This is host/container RAM during HF checkpoint/export, not
  TPU HBM.

Container-memory OOM investigation at `2026-05-18T18:45Z`:

- Concrete failed job investigated:
  `/ahmed/midtrain-delphi-2e19-p50m50-k0p20-lr50-a001`.
- Iris bug report resource envelope: `32 cpu, 64 GiB memory, 50 GiB disk,
  v6e-8`.
- Rendered config:
  - `hf_save_path=gs://marin-us-east5/checkpoints/delphi-2e19-p50m50-k0p20-lr50-a001/hf`
  - `hf_save_steps=1098`
  - no `hf_save_dtype`, so Levanter did not downcast for HF export.
  - permanent checkpoint keep interval also landed on `step-3294`.
- Failure window:
  - `02:40:03`: permanent checkpoint starts at
    `checkpoints/step-3294`.
  - `02:40:08`: JAX checkpoint commit starts.
  - `02:40:08`: HF export starts concurrently at
    `hf/step-3294`.
  - Levanter logs `Checkpoint size -/3348030976`,
    `Will save 1 shards with max size 3.35 GB`, and
    `Saving shard model.safetensors (3.35 GB, 100.00% of model)`.
  - `02:40:23`: kernel OOM kills the container.
- Code path:
  - `train_lm.py` registers `save_hf_checkpoint_callback(...,
    every=config.hf_save_steps)`.
  - `HFCheckpointConverter.save_pretrained` computes
    `state_dict_shape`, shards under `DEFAULT_MAX_SHARD_SIZE=int(5e9)`, then
    for each shard calls `_to_state_dict_with_dtype(...)` followed by
    `shard_numpy = {k: np.asarray(v) ...}` and `save_state_dict(...)`.
  - For GCS paths, `temp_dir_before_upload` writes into a local `/tmp/...`
    directory first, then uploads with fsspec.
  - `_to_state_dict_with_dtype` applies a no-op dtype path here
    (`dtype=None`) and deshards with `PartitionSpec()`, so the full HF shard
    is materialized on the host.
- Storage accounting for the failed `2e19` job:
  - Expected HF export size from log: `3,348,030,976` bytes =
    `3.348 GB` = `3.118 GiB`. This is effectively fp32 weights for the
    837M-param model.
  - Completed HF exports already on GCS:
    `hf/step-1098 = 3.13 GiB`, `hf/step-2196 = 3.13 GiB`.
  - Failed HF export: no `hf/step-3294` objects landed.
  - Complete permanent checkpoints:
    `step-1098 = 9.35 GiB`, `step-2196 = 9.35 GiB`.
  - Failed permanent checkpoint: `step-3294 = 56.74 MiB` partial object set.
  - Remaining temp checkpoint: `step-3001 = 9.35 GiB`.
  - Persisted GCS usage under the run prefix: `25.03 GiB`; additional temp
    checkpoint prefix: `9.35 GiB`.
- Comparison with `9e18-p50m50-lr67`:
  - Resource envelope was the same `64 GiB memory, 50 GiB disk`, on
    `v6e-4`.
  - Complete HF exports at steps `881`, `1762`, `2643`, `3524` are each
    `2.07 GiB`.
  - Complete permanent checkpoints at the same steps are each `6.15 GiB`.
  - The failed `step-4405` permanent checkpoint is only `75.96 MiB`, and
    the HF export died before landing.
- Interpretation:
  - Disk/storage was not the bottleneck. The local HF temp shard is only
    about `2.05 GiB` for `9e18` or `3.12 GiB` for `2e19`, well below the
    `50 GiB` disk request.
  - The cgroup/RSS memory limit was the bottleneck. Iris did not record peak
    memory (`memory_peak_mb=0` in summary JSON), but exit 137 means RSS
    exceeded the requested `64 GiB`.
  - The avoidable spike is HF export materializing a full fp32 model shard on
    host while a JAX/Orbax checkpoint commit is also in flight. Earlier save
    points survived; later ones died after accumulated runtime/JAX/data/eval
    memory plus concurrent checkpoint/export crossed the cgroup limit.
- Iris resource-history check at `2026-05-18T19:06Z`:
  - The local Iris code still contains `task_resource_history` and
    `worker_resource_history` references, but the live Marin controller DB no
    longer has those tables.
  - `schema_migrations` on the live controller includes
    `0040_drop_resource_history_tables.py`, and
    `SELECT name FROM sqlite_master WHERE type='table' AND name LIKE
    '%resource%'` returned no rows.
  - `job summary --json` for
    `/ahmed/midtrain-delphi-2e19-p50m50-k0p20-lr50-a001` reports
    `memory_mb=0`, `memory_peak_mb=0`, `cpu_millicores=0`, and `disk_mb=0`.
  - `task_attempts` still records the two attempts: attempt 0 was a worker
    preemption, and attempt 1 failed with exit 137 while saving
    `model.safetensors`. It does not retain the worker resource time series.
  - Conclusion: Iris can confirm the terminal OOM from the controller DB, but
    the controller DB cannot reconstruct the RAM curve. The resource history
    was moved to finelog stats namespaces.
  - Provenance for the dropped tables:
    - Original commit:
      `527c2e44eaf415dcdbb2d060a9ab0efabeb1ee24`
      (`[iris] drop worker_resource_history / task_resource_history tables`,
      `2026-05-01T20:09:55Z`).
    - Merged PR:
      `#5370` (`[iris] move per-tick resource stats out of controller DB`),
      merge commit `865e7cb41985a4b421772fc3b4d041662e4b6519`,
      merged `2026-05-03T23:13:19Z`.
    - Live Marin controller applied migration
      `0040_drop_resource_history_tables.py` at
      `2026-05-04T22:34:28.627Z` according to `schema_migrations`.
    - The closest issue is `#5072` (`iris - structured logging`), which
      proposed moving worker/task resource history out of controller SQLite.
      PR `#5370` itself has no closing issue references.
  - Finelog storage root for Marin:
    `gs://marin-us-central2/finelog/marin/`.
  - Namespaces present:
    `log/`, `iris.profile/`, `iris.task/`, and `iris.worker/`.
  - Queried `iris.task` directly with DuckDB over GCS using a
    `TYPE GCS, bearer_token` secret. For
    `/ahmed/midtrain-delphi-2e19-p50m50-k0p20-lr50-a001/0`:
    - `915` task-stat rows from `2026-05-17T00:45:02.918Z` to
      `2026-05-17T02:40:18.258Z`.
    - attempt 0: `55` rows, max sampled memory `20,633 MB`, then worker
      preemption.
    - attempt 1: `860` rows, max sampled memory `59,791 MB`, max sampled
      disk `4,691 MB`, then exit 137 during HF export.
    - The sampled task peak first hit `59,791 MB` at
      `2026-05-17T02:11:07.395Z`. Around final HF save, samples were
      `42,209 MB` at `02:40:05`, `19,363 MB` at `02:40:11`, and
      `7,024 MB` at `02:40:18`; the fatal cgroup spike was not captured by
      the ~6.5s sampler.
  - Queried `iris.worker` for the final worker
    (`marin-tpu-v6e-preemptible-8-us-east5-b-20260517-0056-66f2039c-worker-0`):
    host memory was about `59.6-61.1 GB` just before OOM and dropped to
    about `16 GB` once the task died. The TPU VM had about `1.52 TB` host
    RAM, so this was the task/container cgroup limit, not host exhaustion.
- Recovery implications:
  - Relaunch with the exact old output path if resuming. For the investigated
    `2e19` job, the best complete recovery point is the temp checkpoint at
    `step-3001`; the permanent `step-3294` is partial.
  - Fix options before relaunch: request substantially more container memory
    for direct TPU jobs, set `hf_save_dtype: bfloat16`, disable/reduce
    intermediate HF exports, or stagger HF export so it does not overlap the
    permanent checkpoint commit.

## 2026-05-18T19:50Z — Midtraining Replay Mix Includes Tiny StarCoder/ProofPile Tails

Question: the current `p33m67` / `p50m50` / `p67m33` data sections list
`proofpile_2` and `starcoderdata`; is that a new inconsistency, or did the
legacy/reference Delphi midtraining mix already include them?

Findings:

- The small-base launcher uses `data_section_override=_data_section_for_mix(mix)`
  in `experiments/midtrain_specs/delphi_small_cpt_k020.py`, which loads the
  frozen JSONs under `experiments/midtrain_specs/data_sections/`.
- Those JSONs were added by commit `bbd4d5bb26` (`[midtrain] Add small-base
  K=0.20 CPT sweep with val-set bit-equivalence`, `2026-05-16`) as captured
  `data:` blocks from canonical 1e21 K=0.20 reference runs:
  - `p33m67`: `legacy:delphi-1e21-p33m67-9p25b-lr0.5-efbc63`
  - `p50m50`: `legacy:delphi-1e21-p50m50-9p25b-lr0.5-973c46`
  - `p67m33`: `legacy:delphi-1e21-p67m33-9p25b-lr0.5-114e49`
- The replay base is `experiments.pretraining_datasets.nemotron.nemotron_mix`,
  not pure Nemotron-CC. `nemotron_mix` has included `starcoderdata` and
  `proofpile_2` since commit `2983e3f578` (`NanoChat and Hparam Sweep
  References (#2432)`, `2026-03-04`, William Held):
  - Nemotron-CC split weights sum to `10.70716`.
  - `starcoderdata` raw weight is `0.25`.
  - `proofpile_2` raw weight is `0.055`.
- After normalization inside the replay mix, the extra tails are small:
  - `starcoderdata`: `2.2702%` of replay.
  - `proofpile_2`: `0.4994%` of replay.
- Effective total-run weights in the frozen K=0.20 data sections:
  - `p33m67`: `starcoderdata=0.0074917`, `proofpile_2=0.0016482`.
  - `p50m50`: `starcoderdata=0.0113511`, `proofpile_2=0.0024972`.
  - `p67m33`: `starcoderdata=0.0152105`, `proofpile_2=0.0033463`.

Conclusion:

- This is consistent with the legacy 1e21 reference data sections and the
  existing `nemotron_mix` replay definition. It is not caused by the failed
  `9e18` / `2e19` cleanup, and it was not introduced by the v6/v5 probe code.
- The naming is the only misleading part: `p33m67` means 33% pretrain replay
  plus 67% math, and the pretrain replay is `nemotron_mix`, which is mostly
  Nemotron-CC but includes tiny StarCoder/ProofPile tails.
- Keep future comparisons on the same frozen `data_sections/<mix>.json` files
  unless deliberately starting a new condition named something like
  `pure_nemotron_cc_p33m67` so it cannot be confused with the legacy-compatible
  Delphi K=0.20 replay mix.

## 2026-05-18T20:15Z — Relaunch 9e18/2e19 a002 With 2x Container RAM

Goal: relaunch the failed full `9e18` and `2e19` K=0.20 sweeps after deleting
their failed `a001` GCS artifacts and user-deleting the W&B runs.

Changes before launch:

- Added `ComputeProfile.ram` and threaded it through
  `lib/marin/src/marin/midtraining/launch.py` into
  `ResourceConfig.with_tpu(..., ram=...)`.
- Set `experiments/midtrain_specs/delphi_small_cpt_k020.py` default
  container RAM to `256g`, doubling Fray's default TPU-task RAM of `128g`.
- Dry-run checks passed:
  - `9e18 / p33m67 / lr0.5 / v6e-4 / a002`: `8819` steps, `ram=256g`.
  - `2e19 / p33m67 / lr0.5 / v6e-8 / a002`: `10983` steps, `ram=256g`.

Launch plan:

- `9e18`: full 12-cell sweep (`p33m67`, `p50m50`, `p67m33` x
  `lr0.33/0.5/0.67/0.83`) on `v6e-4`, attempt `a002`, Iris priority
  `batch`, direct TPU jobs with `--memory 256g`.
- `2e19`: same 12 cells on `v6e-8`, attempt `a002`, Iris priority `batch`,
  direct TPU jobs with `--memory 256g`.
- Direct `iris job run` is used for this relaunch so the actual TPU training
  job, not just a coordinator, receives Iris `--priority batch`.

Submission result:

- Materialized `24` `a002` train configs/manifests to GCS.
  Launch manifest: `scratch/20260518T2015_delphi_9e18_2e19_a002_launch.tsv`.
- Submitted all `24` direct TPU jobs by `2026-05-18T20:29Z`.
  Submit log: `scratch/20260518T2015_delphi_9e18_2e19_a002_submit.log`.
- Each submitted job used:
  - `--priority batch`
  - `--memory 256g`
  - `--disk 50g`
  - `--cpu 32`
  - `--region us-east5`
  - `--max-retries 3`
  - `--extra marin:tpu`
- Immediate Iris status after submission:
  - `9e18/v6e-4`: `6` running, `6` pending for v6e-4 capacity.
  - `2e19/v6e-8`: `12` pending for v6e-8 capacity.
  - No immediate failed jobs at this check.
