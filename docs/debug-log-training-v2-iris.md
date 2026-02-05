# Debugging log for training-v2-iris

Validate fray v2 migration for Levanter training and confirm Iris submission paths (smoke -> CPU tutorial -> TPU tutorial) work with correct TPU resource requests.

## Initial status

User requested fray v2 migration for training and Iris test runs using `uv run iris job run`, with correct TPU type for TPU jobs.

## Hypothesis 1

If Iris jobs fail, the most likely causes are missing extras (CPU vs TPU deps) or incorrect TPU type passed to `iris job run`.

## Changes to make

- Update Levanter tutorial scripts to import `ResourceConfig` from `fray.v2`.
- Update training design doc with spiral plan and Iris run steps.
- Use `uv run iris --config ... job run` for smoke/CPU/TPU jobs with explicit `--tpu` for TPU.

## Future Work

- [ ] Confirm `default_train` works on Iris with fray v2 end-to-end.
- [ ] Align remaining tutorial scripts with `fray.v2` if needed.
- [ ] Add a minimal Iris job wrapper for Levanter tests if repeated manual runs are common.

## Results

Smoke job (local Iris cluster) succeeded using:

- `uv run iris --config lib/iris/examples/local.yaml job run --extra marin:cpu -- python -c "print('smoke')"`

Observed autoscaler scale-up, worker registration, and job completion with log output `smoke`.

Attempted Levanter CPU tutorial (local Iris) without MARIN_PREFIX and saw:

- `ValueError: Must specify a prefix or set the MARIN_PREFIX environment variable`

Retried with MARIN_PREFIX, but tokenization step failed due to request timeouts
from Zephyr actor RPCs (Iris local cluster). The job ended in failed state after
timeouts. A follow-up run with `ZEPHYR_NUM_WORKERS=4` reduced worker fan-out but
still stalled with repeated retries/timeouts; I terminated the run to avoid a
runaway local controller.

Retried again with `ZEPHYR_NUM_WORKERS=1`. Zephyr worker/coordinator jobs still
cycled with retries and were repeatedly relaunched. I terminated the run to
avoid a runaway controller.

Next: add targeted logging in Iris task lifecycle and local container startup to
understand why local CPU runs churn. Remove ad-hoc timeout overrides and rely
on Iris job metadata.

## Changes to make

- Add lifecycle logging in `lib/iris/src/iris/cluster/worker/task_attempt.py`
  to see bundle, build, container, and exit timing.
- Add local container startup logging in
  `lib/iris/src/iris/cluster/vm/local_platform.py`.
- Remove ad-hoc actor timeout overrides.

## Results

Logging added to Iris task attempts and local VM container start.
Ad-hoc actor timeout overrides removed.

Rerun local CPU tutorial:

- Command: `uv run iris --config lib/iris/examples/local.yaml job run --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -e ZEPHYR_NUM_WORKERS 1 -- python -m experiments.tutorials.train_tiny_model_cpu`
- Iris controller registered worker and launched job quickly (<1s).
- Executor submitted nested `/train_lm` job to Iris (as expected for fray v2).
- No training logs observed in parent stream.
- Checkpoints directory created at `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc` with executor metadata files only.

Next: improve visibility into nested Iris jobs (child log streaming) to determine whether training is stalled or logs are simply not visible in the parent log stream.

## Changes to make

- Add child log streaming support in Iris client wait loop.
- Add CLI flag to include child logs during `iris job run`.

## Results

Added `--include-children-logs` to `iris job run` and child-log streaming in `Job.wait()`.

Rerun local CPU tutorial with fresh checkpoint path:

- Removed `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc` to force rerun.
- Command: `uv run iris --config lib/iris/examples/local.yaml job run --include-children-logs --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -e ZEPHYR_NUM_WORKERS 1 -- python -m experiments.tutorials.train_tiny_model_cpu`
- Nested `/train_lm` job logs visible (wandb init, training progress, eval, checkpoint save).
- Training completed successfully (100 steps, eval loss ~10.029, HF checkpoint saved to `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc/hf/step-99`).
- Top-level Iris job completed successfully.
- One warning observed: `iris.managed_thread` reported `on_stop callback for task-/.../train_lm/0 did not complete` after task exit.

Next: decide whether to address the on_stop callback warning or treat as benign; proceed to TPU job once desired.

## Hypothesis 2

TPU runs on the eu-west4 Iris cluster were failing due to a controller restart
race in `ManagedVm._run`, combined with stale CLI tooling that masked autoscaler
status and worker visibility. Fixing the stop-event wiring and CLI status
formatting should make it possible to see TPU job progress and logs.

## Changes to make

- Fix `ManagedVm._run` to pass the thread stop event directly into
  `wait_for_connection`.
- Update CLI debug/status commands to use proto timestamps instead of deleted
  `*_ms` fields.
- Re-run TPU tutorial job and inspect autoscaler + worker status while it builds.

## Results

- Restarted eu-west4 cluster and confirmed controller health at
  `http://10.164.0.63:10000`.
- TPU worker slice(s) came up cleanly; workers registered as healthy.
- TPU tutorial job submitted with `--tpu v5litepod-16` and `--extra marin:tpu`.
- Top-level job completed image build and launched nested `/train_lm` job.
- Nested job currently building (installing deps, Rust toolchain). No training
  step logs yet; will continue until build finishes and training starts.
- CLI debug/status commands now show autoscaler status, VM status, and worker
  heartbeats without crashing.

Follow-ups:
- Removed `iris cluster debug show-task-logs` (out of date with RPC schema).
- Updated TPU tutorial resources to request 4 replicas and larger CPU/RAM so
  the Iris UI reflects the actual TPU slice layout.
- Renamed `iris run` to `iris job run` and added `iris job stop`/`iris job logs`.

## Latest status

- Previous TPU attempt failed (`/iris-run-power-job-20260205-013042` failed with
  nested `/train_lm` exit code 137). Jobs are now complete (no running match).
- New TPU run submitted with `iris job run` after updating resource defaults
  (disk set to 50g). Monitoring in progress.
- An intermediate run failed because `ResourceConfig.with_tpu("v4-128", slice_count=4)`
  conflicted with the new slice-count assertion. Scoped the assertion to fray v2
  only to avoid breaking fray v1 experiment configs.
- Current TPU job: `/iris-run-power-job-20260205-015604` (building image; logs streaming).
- TPU run failed with `libcublas.so` missing inside the TPU job; nested
  `/train_lm` failed with GPU library probe error and killed other replicas.
  Added TPU-specific env overrides (`JAX_PLATFORMS=tpu`, `JAX_PLATFORM_NAME=tpu`,
  `CUDA_VISIBLE_DEVICES=""`) in `run_levanter_train_lm` to prevent CUDA probing.
- Follow-up TPU run still hit torch CUDA dependency (`libcudart.so.12`) via
  transformers import. Added `TRANSFORMERS_NO_TORCH=1` and `USE_TORCH=0` to the
  TPU env to skip torch import on TPU.
- Torch still attempted to load CUDA deps; added `TORCH_DISABLE_GLOBAL_DEPS=1`
  to bypass global CUDA library loading in torch.
- Latest TPU run (`/iris-run-power-job-20260205-021248/train_lm`) still failed
  with `libcudart.so`/`libcublas.so` missing via `torchvision` import from
  `transformers.image_utils`. Added `TRANSFORMERS_NO_TORCHVISION=1` to the TPU
  env to skip torchvision imports.
- Correction: the TPU tutorial entrypoint is a CPU parent job that submits a
  TPU child job from `default_train` (via `ResourceConfig.with_tpu`).
  The correct launch command is:
  `uv run iris --config lib/iris/examples/eu-west4.yaml job run --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -- python -m experiments.tutorials.train_tiny_model_tpu`
  The nested TPU job should request the `marin:tpu` extra when it builds
  (the parent should not pass `--tpu`).
- Migrated `marin.execution.executor` to fray v2 client submission. First run
  failed because Iris job names cannot contain `/`; executor step names include
  slashes. Added `_sanitize_job_name` to replace `/` and spaces before submit.
- `iris job logs --follow --include-children` crashed with `ConnectError: Task has no assigned worker`
  while the job was still waiting for scheduling. Added a guard in
  `lib/iris/src/iris/cli/job.py` to ignore this transient condition.
- Tokenize step stayed pending on TPU-only cluster because default executor
  resources forced `preemptible=False`, which cannot schedule on the
  preemptible-only TPU scale group. Defaulted CPU steps to `preemptible=True`.
- 2026-02-05: Restarted eu-west4 cluster after restart request. `cluster restart`
  timed out during controller teardown, so reran `cluster start` (build + push
  images) which created a new controller VM. Controller bootstrap running; CLI
  `cluster status` shows tunnel up but RPC resets while controller finishes
  startup. Will recheck and proceed once controller RPC is healthy.
- 2026-02-05: Reran `cluster start` with longer timeout; controller VM bootstrapped
  successfully, pulled `iris-controller` image, started the container, and
  health checks passed. Controller now at `http://10.164.0.5:10000`.
- 2026-02-05: Started CPU parent job:
  `uv run iris --config lib/iris/examples/eu-west4.yaml job run --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -- python -m experiments.tutorials.train_tiny_model_tpu`
  Job `/iris-run-power-job-20260205-031244` submitted. Autoscaler began creating
  TPU slice `iris-tpu_v5e_16-1770261135359`; worker bootstrap logs show container
  pulls and worker registration. Job currently in `building` state (image build
  still running).
- 2026-02-05: Parent job failed because the tokenize child job produced an
  invalid Docker tag: `__tokenize:tokenized-roneneldan-TinyStories` includes a
  colon. Updated `_sanitize_job_name` to lower-case and replace any non
  `[a-z0-9_.-]` character with `-`, ensuring Docker tags are valid.
- 2026-02-05: Rerun `/iris-run-power-job-20260205-032117` launched sanitized
  child job `tokenize-tokenized-roneneldan-tinystories`, but it failed with
  `ValueError: libcublas.so.* not found` from JAX CUDA probing. Added
  resource-aware env vars in executor to force `JAX_PLATFORMS=cpu` for CPU steps
  and `JAX_PLATFORMS=tpu` for TPU steps (plus `CUDA_VISIBLE_DEVICES=""`) so
  tokenization stays CPU-only on TPU hosts.
- 2026-02-05: Tokenization still failed, now from `transformers` importing
  `torch` and attempting to load CUDA libs (`libcudart.so.12`). Added
  `env_vars` to `ExecutorStep` and set CPU-safe defaults for `default_tokenize`
  (`TRANSFORMERS_NO_TORCH=1`, `USE_TORCH=0`, `TORCH_DISABLE_GLOBAL_DEPS=1`,
  `TRANSFORMERS_NO_TORCHVISION=1`) to avoid torch import during tokenization.
- 2026-02-05: Tokenization still imported torch before env vars applied; added
  `os.environ.setdefault(...)` guards in `marin.processing.tokenize.data_configs`
  before importing `transformers` to ensure torch is skipped in tokenization
  modules even on TPU workers.
- 2026-02-05: Checked controller `list-tasks`; no running jobs. Latest CPU parent
  `/iris-run-power-job-20260205-034928` failed after tokenization child exited
  137. `iris job logs` returned no task logs for the window; need to re-run with
  a fresh job after updating env propagation and/or tokenization env controls.
- 2026-02-05: Implemented default env inheritance for nested Iris jobs by
  merging parent `os.environ` (excluding `IRIS_*`) into child `EnvironmentConfig`
  in `RemoteClusterClient.submit_job`. Removed resource-based env injection from
  `Executor` to keep env control in step definitions.
- 2026-02-05: Removed `os.environ` mutation from tokenization module; env vars are
  now only set on job submission (tokenize ExecutorStep). Stopped and reran the
  parent job as `/iris-run-power-job-20260205-155616`.
- 2026-02-05: Fixed NameError by restoring `import os` in
  `marin.processing.tokenize.data_configs`. Added `/tmp/run_iris_job_with_logs.sh`
  to submit job + capture logs for 15 minutes. Launched job
  `/iris-run-power-job-20260205-161127` via the script; image build and install
  were in progress.
- 2026-02-05: Tokenize child job `/iris-run-power-job-20260205-161127/tokenize-tokenized-roneneldan-tinystories`
  failed with exit code 137 after image build. Added explicit resources to
  `default_tokenize`: `ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g")`.
- 2026-02-05: Tokenize child job for `/iris-run-power-job-20260205-161659` still
  failed with `libcublas.so` missing, indicating JAX GPU probe. Added
  `JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=cpu`, and `CUDA_VISIBLE_DEVICES=""` to
  the tokenization job env vars.
- 2026-02-05: Tokenize child job was still pulling GPU-capable deps because its
  Dockerfile used plain `uv sync` (no extras). Added
  `pip_dependency_groups=["marin:cpu"]` to `default_tokenize` so the child job
  builds with CPU extras and avoids CUDA libs.

## Hypothesis 3

Tokenization failures on TPU hosts persist because child jobs are not building
with CPU extras and/or are not inheriting the parent env vars that disable CUDA
probing. We need to validate the docker build flags for child jobs and confirm
env propagation works as intended.

## Changes to make

- Re-run the TPU tutorial after cluster restart to validate the docker build.
- Inspect child job build logs to confirm `uv sync` includes `--extra cpu`.
- If extras still missing, trace fray v2 -> iris environment propagation.

## Results

- 2026-02-05: TPU run failed during docker build: `libtorch_cpu.so: No space left
  on device` while installing `torch-2.9.0+cpu`. Triggered cluster restart to
  clear disk/cache before rerunning.
- 2026-02-05: `uv run iris --config lib/iris/examples/eu-west4.yaml cluster restart`
  hung during controller teardown; reran `cluster start` and brought up a fresh
  controller VM. Health checks passed and controller came back online.
- 2026-02-05: New CPU parent job `iris-run-power-job-20260205-164351` failed.
  Tokenize child job failed with `connectrpc.errors.ConnectError: Request timed out`.
  Worker logs show some zephyr worker tasks failed to build because the build
  context could not find `Dockerfile.iris`:
  `failed to read dockerfile: open Dockerfile.iris: no such file or directory`.
  Other workers built successfully, so this looks like a bundle/worker build
  context issue rather than a systemic Docker failure.
- 2026-02-05: Root cause identified: concurrent workers write/remove the same
  `Dockerfile.iris` in a shared bundle context. One worker deletes the file
  while another build is starting, causing the missing dockerfile error.
  Fixed by writing a unique dockerfile name per build in
  `lib/iris/src/iris/cluster/worker/docker.py`.
 - 2026-02-05: Relaunched CPU parent job `iris-run-power-job-20260205-165903`.
   Job is running and currently in BUILDING state (image build in progress).
 - 2026-02-05: Checked cluster status (controller healthy, no TPU demand, no jobs).
 - 2026-02-05: Submitted new CPU parent job via `/tmp/run_iris_job_with_logs.sh`
   (job `iris-run-power-job-20260205-164351`) and started streaming logs for 15
   minutes to `/tmp/iris-job-logs/iris-run-power-job-20260205-084346.log`.
 - 2026-02-05: Filed GitHub issue for env propagation defaults:
   https://github.com/marin-community/marin/issues/2672.
