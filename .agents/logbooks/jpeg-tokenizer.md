# JPEG Tokenizer: Research Logbook

## Scope

- Goal: determine whether simple autoregressive modeling over canonical JPEG-derived token streams is viable, starting with a bounded `K=4` coefficient baseline.
- Primary metric(s): train/eval NLL, sequence-length feasibility, tokenizer determinism, and basic reconstruction quality for lossy coefficient streams.
- Constraints: keep storage localized to `eu-west4`, prefer small reproducible artifacts, and use `marin-community/tokexplore` for W&B.

## Baseline

- Date: 2026-03-08
- Code refs:
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/jpeg_codecs.py`
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/data.py`
  - `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/launch.py`
- Baseline numbers:
  - Imagenette `320px` train split: `9469` examples
  - K=4 coefficient store: `4096` tokens/example, configured vocab `4095`, observed vocab `1796`
  - K=4 reconstruction: mean SSIM `0.7164`, mean PSNR `24.65 dB`

## Experiment Log

### 2026-03-08 20:00 - Phase 0 finalized locally

- Hypothesis: a fixed-length K=4 coefficient stream is a clean first training target because it avoids the sequence-length variance of bytes and symbols.
- Command:
  - `uv run python scripts/jpeg_tokenizer/inspect_representations.py --output-dir ...`
  - `uv run python scripts/jpeg_tokenizer/evaluate_coeff_reconstruction.py --output-dir ...`
- Config: Imagenette `320px`, canonical `256x256` luma JPEG, quality `95`, `K=4`.
- Result: deterministic preprocessing passed on all examples; coefficient sequences are exactly length `4096`; K=4 is lossy but structurally coherent.
- Interpretation: Phase 1 should begin with coefficient-only training, while bytes and symbols need explicit windowing or a different compute budget.
- Next action: materialize a reusable token store and move the first TPU run to `eu-west4`.

### 2026-03-08 21:10 - Token store built and mirrored

- Hypothesis: a precomputed store is the lowest-risk input path for the first TPU-backed run.
- Command:
  - `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- Config: local build plus regional GCS mirror.
- Result: train/validation matrices and manifests were written locally and mirrored to `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`.
- Interpretation: the first cluster run can stream a small fixed artifact instead of rebuilding data remotely.
- Next action: wire the launch step directly to the mirrored store.

### 2026-03-08 21:35 - Launch path hardening

- Hypothesis: the first Ray launch should use a small smoke step on `v6e-8` before attempting the longer baseline.
- Command:
  - `uv run python experiments/jpeg_tokenizer/base/launch.py --help`
- Config: `tokexplore/jpeg-tokenizer-k4-smoke` and `tokexplore/jpeg-tokenizer-k4-trial`, both reading the regional GCS store.
- Result: discovered and fixed a script-entrypoint import hazard where local `tokenizers.py` shadowed the third-party `tokenizers` package; renamed the module to `jpeg_codecs.py`.
- Interpretation: the launch surface is now safe to execute as a standalone script under Ray runtime packaging.
- Next action: validate tests/pre-commit, commit the launch milestone, then submit the smoke run to `marin-eu-west4-a`.

### 2026-03-08 22:30 - Cluster bring-up failures and fixes

- Hypothesis: once the launch surface is clean, the K=4 smoke run should at least reach TPU-local trainer initialization.
- Command:
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -- python experiments/jpeg_tokenizer/base/launch.py ...`
- Config: `tokexplore/jpeg-tokenizer-k4-smoke`, `v6e-8`, regional `gs://marin-eu-west4` prefix, mirrored token store.
- Result:
  - first submit failed with Ray request `413` because local `artifacts/` were being uploaded with the working directory
  - added `.rayignore` for `artifacts/`
  - second submit failed before execution because `--run_only` must be passed as a list value, not a bare string
  - corrected submit syntax and reached executor startup
  - third run failed during TPU dispatch because `DirectDatasetComponent` datasets were embedded in the Ray payload and were not serializable
  - refactored JPEG training config so the token-store path is serialized, while `LmDataConfig` is built inside the TPU-local entrypoint
  - fourth run reached the TPU-local entrypoint but failed because token-store materialization happened before `trainer.initialize()`, which tripped JAX distributed startup ordering
  - refactored the shared grug local runner so JPEG can initialize the trainer first, then build its data config, then enter the common training loop
  - fifth run reached evaluator setup and exposed that `compute_bpb=True` is invalid for the passthrough JPEG tokenizer because the generic bytes-per-token probe tokenizes `"."` as text
  - disabled `compute_bpb` for the JPEG smoke/trial eval configs
- Interpretation: the pipeline now matches the actual execution boundary. The remaining test is a fresh cluster smoke submit from the worker-side token-store path.
- Next action: commit the worker-side materialization fix and rerun `tokexplore/jpeg-tokenizer-k4-smoke`.

### 2026-03-08 23:00 - Smoke run succeeded on eu-west4

- Hypothesis: after fixing worker-side token-store loading and disabling `compute_bpb`, the `K=4` smoke run should complete without TPU/runtime bring-up failures.
- Command:
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k4-smoke"]'`
- Config: `v6e-8`, batch size `256`, `128` train steps, eval every `64` steps, `compute_bpb=False`.
- Result:
  - Ray job: `ray-run-dlwh-launch-20260308-084426`
  - W&B run: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k4-smoke`
  - eval loss: `8.508 -> 5.118 -> 4.694`
  - train progress snapshots: step `11/128` loss `6.93`, step `106/128` loss `5.25`, final step `128/128` loss `4.85`
  - checkpoint written to `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-smoke-3cabe7/checkpoints/step-128`
- Interpretation: the basic K=4 coefficient pipeline is now healthy on the production path. Remaining warning is the non-fatal `draccus` config-artifact dump issue in tracker setup.
- Next action: launch the `tokexplore/jpeg-tokenizer-k4-trial` baseline and monitor early progress.

### 2026-03-09 00:00 - Trial baseline launched and training

- Hypothesis: the longer `K=4` baseline should follow the same path as the smoke run and reach stable optimization without additional infra-specific fixes.
- Command:
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k4-trial"]'`
- Config: `v6e-8`, batch size `512`, `2000` train steps, eval every `1000` steps, `compute_bpb=False`.
- Result so far:
  - Ray job: `ray-run-dlwh-launch-20260308-085237`
  - W&B run: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k4-trial`
  - initial eval loss: `8.476`
  - train progress snapshots:
    - `3/2000`, loss `8.48`
    - `49/2000`, loss `6.01`
    - `96/2000`, loss `5.62`
    - `143/2000`, loss `5.31`
- Interpretation: the longer baseline is past bring-up and optimizing normally. The remaining known issue is still the non-fatal tracker config-artifact dump warning.
- Next action: leave the trial running and inspect the first scheduled eval at step `1000`.

### 2026-03-09 01:45 - K=4 baseline completed successfully

- Hypothesis: the full `K=4` baseline should reach terminal success on `v6e-8` without new infrastructure fixes, giving a usable first comparison point for future coefficient ablations.
- Command:
  - monitoring only via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
- Config: `tokexplore/jpeg-tokenizer-k4-trial`, `v6e-8`, batch size `512`, `2000` train steps, eval every `1000`, token store `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`.
- Result:
  - Ray job: `ray-run-dlwh-launch-20260308-085237`
  - W&B run: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k4-trial`
  - terminal status: `SUCCEEDED`
  - eval loss: `8.476` at startup, `4.376` at step `1000`, `4.417` at step `2000`
  - checkpoints written at steps `421`, `887`, `1000`, `1464`, `1928`, and final `2000`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-trial-603f95/checkpoints/step-2000`
  - wall-clock executor time: `2778.14s` (`44.3` minutes)
- Interpretation:
  - the `K=4` coefficient baseline is now proven end to end on the production Ray + TPU path
  - optimization is stable and the final eval is close to the best mid-run eval, so this config is good enough to use as the reference point for the next ablation
  - there is still background Ray worker churn on unrelated nodes plus two non-fatal warnings: the `draccus` config-artifact dump failure and a final W&B `BrokenPipeError` during process teardown
- Next action: build and mirror the `K=8` coefficient store, then launch the smallest safe `K=8` smoke run on `v6e-8`.

### 2026-03-09 01:55 - K=8 smoke succeeded on v6e-8

- Hypothesis: doubling coefficient retention to `K=8` should still fit and train on `v6e-8` if the batch size is reduced enough.
- Command:
  - `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --k 8 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0`
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-smoke"]'`
- Config:
  - token store: `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0`
  - model max sequence length: `8192`
  - batch size: `128`
  - steps: `96`
  - eval every `48` steps
- Result:
  - Ray job: `ray-run-dlwh-launch-20260308-094432`
  - W&B run: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-smoke`
  - terminal status: `SUCCEEDED`
  - eval loss: `8.549 -> 4.446 -> 4.124`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-smoke-c2bc8c/checkpoints/step-96`
  - executor wall time: `321.72s`
- Interpretation:
  - `K=8` is operationally viable on the same TPU shape without any new infrastructure work
  - the safe batch point is at least `128`, which is enough to proceed to a longer run and compare learning dynamics against the completed `K=4` baseline
  - the same two non-fatal warnings remain: `draccus` config-artifact serialization and W&B teardown `BrokenPipeError`
- Next action: launch a longer `K=8` trial with the same batch shape and compare the `1000`-step eval against the `K=4` reference.

### 2026-03-09 03:10 - K=8 trial recovered after one preemption

- Hypothesis: the active `K=8` trial should resume cleanly after a single TPU slice preemption, so the right response is to monitor rather than resubmit.
- Command:
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a list-jobs`
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a job-logs -n 260 ray-run-dlwh-launch-20260308-095441`
- Config: `tokexplore/jpeg-tokenizer-k8-trial`, `v6e-8`, batch size `128`, output path `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-69fd6d`.
- Result:
  - Ray job `ray-run-dlwh-launch-20260308-095441` remained `RUNNING`
  - one TPU worker on `10.164.3.247` died and Fray reported `Preempted 1 times. Continuing to retry.`
  - the run resumed on a new worker `10.164.2.175` under the same W&B run `jpeg-tokenizer-k8-trial`
  - resumed progress reached startup eval loss `8.547` and train step `81/2000` with loss `4.98`
- Interpretation:
  - the retry path is good enough for this scale of experiment, so an immediate resubmit would only waste time and compute
  - the more actionable cleanup item was the repeated config-artifact warning, not the transient preemption itself
- Next action: fix the tracker config-artifact dump so subsequent failures are easier to read, then prepare the `K=16` rung.

### 2026-03-09 03:20 - Tracker config-artifact dump fixed locally

- Hypothesis: the repeated `draccus` warning comes from dataclasses defined with postponed annotations, so dumping the already-materialized hyperparameter dict to YAML should preserve the artifact without touching run behavior.
- Command:
  - `uv run --with pytest python -m pytest -o addopts='' lib/levanter/tests/test_tracker.py`
  - `uv run --with pytest python -m pytest -o addopts='' tests/test_jpeg_tokenizer_scaffold.py`
- Config: switched `levanter.tracker.log_configuration(...)` artifact serialization to YAML over `hparams_to_dict(...)`, and added a regression test for a dataclass created under `from __future__ import annotations`.
- Result:
  - local tests passed
  - the config artifact path is now independent of `draccus.dump(...)` and no longer depends on resolved dataclass field types
- Interpretation:
  - future tokenizer runs should stop emitting the false-positive config-artifact stack trace
  - this is a global cleanup in Levanter, not just a JPEG-specific workaround
- Next action: commit the tracker fix, then build the `K=16` coefficient store and add a smoke step only if the artifact size still looks modest.

### 2026-03-09 03:30 - K=16 store mirrored and smoke launched

- Hypothesis: `K=16` should still be operationally feasible on `v6e-8` if the smoke run uses batch size `64`, which keeps tokens per update roughly aligned with the successful `K=8` smoke.
- Command:
  - `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --k 16 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k16-smoke"]'`
- Config:
  - token store: `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`
  - sequence length: `16384`
  - artifact size: `839 MiB`
  - smoke batch size: `64`
  - smoke steps: `64`
  - eval every `32` steps with eval batch size `16`
- Result so far:
  - local store built successfully and mirrored to regional GCS
  - launch config now includes `tokexplore/jpeg-tokenizer-k16-smoke`
  - Ray job submitted: `ray-run-dlwh-launch-20260308-101439`
  - executor step output path: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795`
  - status reached executor launch and Fray TPU dispatch for `grug-train-jpeg-tokenizer-k16-smoke`
- Interpretation:
  - `K=16` is now staged end to end with bounded additional storage cost
  - the next meaningful checkpoint is whether the worker reaches W&B/trainer startup and first eval, not whether packaging works
- Next action: monitor the `K=16` smoke through trainer bring-up, then decide whether the next commit should include only launch/staging or also a completed smoke result.

### 2026-03-09 03:45 - Monitoring update and executor-log cleanup

- Hypothesis: the next useful improvement is reducing executor-side serialization noise while the active runs continue in the background; this improves monitoring without changing experiment behavior.
- Command:
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a job-logs -n 220 ray-run-dlwh-launch-20260308-095441`
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a job-logs -n 220 ray-run-dlwh-launch-20260308-101439`
  - `uv run --with pytest python -m pytest -o addopts='' tests/execution/test_json_encoder.py tests/execution/test_executor.py`
- Config:
  - `CustomJsonEncoder` now serializes dataclasses structurally and `PartitionSpec` values as readable strings
  - added a regression test covering a nested dataclass payload with `PartitionSpec`
- Result:
  - `K=8` trial `ray-run-dlwh-launch-20260308-095441` is still running after a second TPU preemption; latest observed progress was step `498/2000` with loss `4.08`
  - `K=16` smoke `ray-run-dlwh-launch-20260308-101439` is still running but has only reached executor launch and Fray dispatch so far
  - executor serialization warnings for dataclass configs and `PartitionSpec` are fixed locally for future runs
- Interpretation:
  - the active `K=8` trial is robust enough to keep running despite repeated preemptions
  - `K=16` has not yet proven trainer bring-up, so launching a full `K=16` trial now would be premature
  - the next submitted runs should have much cleaner executor logs because the local encoder fix is independent of model code
- Next action: commit the encoder cleanup, then keep monitoring the two active jobs rather than adding more parallel experiments.

### 2026-03-09 04:00 - K=16 smoke succeeded; K=8 trial needs denser checkpoints

- Hypothesis: `K=16` only needed a smoke to establish operational feasibility, while the `K=8` trial needs more frequent checkpoints because preemptions are erasing useful progress.
- Command:
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
  - local validation:
    - `uv run --with pytest python -m pytest -o addopts='' lib/levanter/tests/test_tracker.py`
    - `uv run --with pytest python -m pytest -o addopts='' tests/test_jpeg_tokenizer_scaffold.py tests/execution/test_json_encoder.py`
- Result:
  - `K=16` smoke `ray-run-dlwh-launch-20260308-101439` finished successfully
  - W&B run: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k16-smoke`
  - eval loss trajectory: `8.620 -> 3.612 -> 3.435`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795/checkpoints/step-64`
  - `K=8` trial `ray-run-dlwh-launch-20260308-095441` suffered another preemption, resumed on a new worker, and restarted from scratch because no checkpoint was available
  - latest observed `K=8` resume reached step `81/2000` again with the same early loss pattern
- Interpretation:
  - `K=16` is clearly operational on `v6e-8` at batch size `64`
- the current `K=8` trial configuration is not resilient enough for preemptible slices; continuing unchanged is likely to waste more TPU time
- the right immediate fix is denser checkpointing rather than changing model hyperparameters
- Next action: commit the tracker+checkpointing hardening, stop the degraded `K=8` trial, and relaunch a fresh retry with 2-minute checkpoints under a new run id.

### 2026-03-09 04:10 - K=8 retry is checkpointing; K=16 trial staged

- Hypothesis: once the `K=8` retry proves that 2-minute checkpointing actually lands durable state, the next bounded rung should be a full `K=16` trial rather than more `K=8` infrastructure churn.
- Command:
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a job-logs -n 400 ray-run-dlwh-launch-20260308-103217`
  - local edit of `experiments/jpeg_tokenizer/base/launch.py`
- Config:
  - active run: `tokexplore/jpeg-tokenizer-k8-trial-r2`
  - staged run: `tokexplore/jpeg-tokenizer-k16-trial`
  - checkpoint policy: every `2` minutes, keep every `500` steps
- Result:
  - the `K=8` retry wrote durable checkpoints at steps `79`, `218`, and `358` to `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints`
  - latest observed `K=8` progress reached step `429/2000` with train loss `4.18`
  - launch config now includes a full `K=16` trial at sequence length `16384`, batch size `64`, `2000` steps, and eval every `1000` steps
- Interpretation:
  - the earlier `K=8` failure mode is resolved: preemptions should no longer erase the whole run
  - `K=16` is ready for the same end-to-end comparison treatment as `K=4` and `K=8`
- Next action: validate the new launch surface, commit the milestone, and submit `tokexplore/jpeg-tokenizer-k16-trial` on `marin-eu-west4-a`.

### 2026-03-09 04:25 - K=16 trial submitted; K=8 retry remains healthy

- Hypothesis: with `K=8` now writing durable checkpoints every few minutes, the best parallel use of cluster time is to leave it running and queue the full `K=16` trial on the same regional cluster.
- Command:
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k16-trial"]'`
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
- Config:
  - new run: `tokexplore/jpeg-tokenizer-k16-trial`
  - output path: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-trial-a39c10`
  - batch size: `64`
  - steps: `2000`
  - checkpoint policy: every `2` minutes, keep every `500` steps
- Result:
  - submitted Ray job: `ray-run-dlwh-launch-20260308-104918`
  - the executor and Fray dispatch path completed cleanly for `grug-train-jpeg-tokenizer-k16-trial`
  - as of the latest check, the job is controller-`RUNNING` but has not yet reached trainer/W&B startup, which points to scheduler/capacity delay rather than a code failure
  - meanwhile the active `K=8` retry advanced through steps `635`, `703`, and `772`, with durable checkpoints at steps `639` and `777`
- Interpretation:
  - `K=8` is now behaving like a real baseline instead of a brittle smoke continuation
  - `K=16` launch code is good; the remaining variable is TPU allocation latency on `marin-eu-west4-a`
- Next action: leave both jobs in place and continue monitoring for either `K=8` eval at step `1000` or `K=16` trainer startup.

### 2026-03-09 05:05 - K=8 resume bug found; K16 de-prioritized

- Hypothesis: once the `K=8` retry survives to a durable `step-1000` checkpoint, the remaining risk shifts from preemption loss to correctness of the resume path itself.
- Command:
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
  - local fix + validation in `experiments/grug/base/train.py`
- Result:
  - `K=8` reached `step-1000`, wrote `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints/step-1000`, and recorded eval loss `3.428`
  - after a TPU preemption, the replacement worker on `10.164.0.3` discovered `step-1000` and logged `Loading checkpoint from .../step-1000`
  - the same worker then incorrectly logged `Checkpoint not found at ... Starting from scratch.`
  - root cause: the grug resume path was swallowing any downstream `FileNotFoundError` from `load_checkpoint(...)` and treating it as "no checkpoint exists", even after checkpoint discovery had succeeded
  - local fix: split checkpoint discovery from restore in `experiments/grug/base/train.py` so only the no-checkpoint case is soft-failed; restore-time file errors now propagate
  - validation: targeted pytest passed for the new regression covering this exact failure mode
  - to reduce contention while fixing the real blocker, the queued `K=16` trial `ray-run-dlwh-launch-20260308-104918` was intentionally stopped
- Interpretation:
  - the checkpoint cadence problem is solved
  - the next blocker is a real resume bug, now fixed locally
  - `K=16` should stay paused until `K=8` resumes cleanly from the committed fix
- Next action: commit the resume fix, stop the buggy live `K=8` retry, and relaunch it from the saved `step-1000` checkpoint.
