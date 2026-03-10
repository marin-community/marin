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

### 2026-03-09 11:10 - SWA head-to-head staged

- Hypothesis: the cleanest next comparison is whole-image bytes with `SWA=4096` versus exact-libjpeg `K=8` coefficients with the same `SWA=4096`, shared optimizer settings, and roughly matched tokens per step.
- Command:
  - `uv run python experiments/jpeg_tokenizer/base/launch.py --help`
- Config:
  - added `tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-smoke`
  - added `tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-trial`
  - added `tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial`
  - chose `K=8` batch size `56` so `8192 * 56 = 458,752` tokens/step, close to whole-image bytes `54,656 * 8 = 437,248`
- Result: launch surface now has a fairer SWA comparison path instead of mixing full-attention coeff runs with sliding-window byte runs.
- Interpretation: `K=8` remains the practical coefficient baseline, and exact-libjpeg avoids ambiguity about whether the coefficient path is merely a reference approximation.
- Next action: validate locally, run the `K=8` SWA smoke, and only then launch the paired trials.

### 2026-03-09 14:50 - Exact `K=8` SWA smoke succeeded

- Hypothesis: exact-libjpeg `K=8` should still optimize normally when switched from full attention to `SWA=4096`, which is the last check needed before the paired whole-image-byte comparison.
- Command:
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-smoke"]'`
- Config:
  - exact-libjpeg `K=8`
  - `seq_len=8192`
  - `sliding_window=4096`
  - batch size `56`
  - steps `96`
- Result:
  - Ray job: `ray-run-dlwh-launch-20260309-183657`
  - terminal status: `SUCCEEDED`
  - W&B: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-libjpeg-swa4096-smoke`
  - eval loss: `8.554 -> 4.335 -> 4.115`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-smoke-8e1097/checkpoints/step-96`
- Interpretation: the coeff side now has the same SWA architecture as the whole-image byte baseline, so the longer head-to-head trials are worth running.
- Next action: launch `tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-trial` and `tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial`.

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

### 2026-03-09 05:12 - Fixed K=8 retry relaunched; cluster is capacity-bound

- Hypothesis: with the resume bug fixed in `198523e81`, the correct next move is to replace the buggy live retry with a fresh submission using the same run id and checkpoint path.
- Command:
  - `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a stop-job ray-run-dlwh-launch-20260308-103217`
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-trial-r2"]'`
- Result:
  - old buggy retry stop requested: `ray-run-dlwh-launch-20260308-103217`
  - new fixed retry submitted: `ray-run-dlwh-launch-20260308-110836`
  - executor step reused the same output path `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd`
  - current cluster signal is not another resume failure; it is scheduler pressure:
    `No available node types can fulfill resource requests ... TPU-v6e-8-head`
- Interpretation:
  - the next gating factor is TPU availability on `marin-eu-west4-a`, not checkpoint correctness
  - further stop/resubmit churn is unlikely to improve anything while the cluster is resource-constrained
- Next action: leave the fixed `K=8` retry queued, keep `K=16` stopped, and resume monitoring once capacity is available.

### 2026-03-08 17:47 - K=8 baseline completed; K=16 trial relaunched

- Hypothesis: once the fixed `K=8` retry reaches a clean terminal success, the next bounded rung is to relaunch the
  staged `K=16` trial and use the successful `K=8` run as the comparison anchor.
- Command:
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k16-trial"]'`
- Result:
  - the fixed `K=8` retry job `ray-run-dlwh-launch-20260308-180311` finished with controller status `SUCCEEDED`
  - final `K=8` checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints/step-2000`
  - final `K=8` eval loss: `3.253`
  - final `K=8` output path remained `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd`
  - a new `K=16` trial submission was created as `ray-run-dlwh-launch-20260309-003238`
  - early `K=16` controller status is `PENDING` with runtime-env setup in progress and no trainer/W&B lines yet
- Interpretation:
  - `K=8` is now a complete baseline rather than a half-resumed infrastructure recovery run
  - the next research question is whether the fidelity gain from `K=16` beats the sequence-length cost strongly enough to justify the larger context
  - no new code issue is indicated by the early `K=16` submission state
- Next action: monitor `ray-run-dlwh-launch-20260309-003238` through trainer startup and the first eval/checkpoint boundary.

### 2026-03-08 21:08 - K=16 baseline completed cleanly

- Hypothesis: if `K=16` runs to completion without new infra bugs, the first coefficient ladder on Imagenette is complete
  enough to compare sequence-length tradeoffs directly.
- Command:
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
- Result:
  - `K=16` job `ray-run-dlwh-launch-20260309-003238` finished with controller status `SUCCEEDED`
  - final `K=16` checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-trial-a39c10/checkpoints/step-2000`
  - final `K=16` eval loss: `2.668`
  - executor marked `tokexplore/jpeg-tokenizer-k16-trial_166a0b35` succeeded
  - W&B run synced successfully at `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k16-trial`
  - as with the successful `K=8` finish, wandb emitted an ignored shutdown-time `BrokenPipeError`, but the Ray submission still ended `SUCCEEDED`
- Interpretation:
  - the first coefficient ladder is now complete on Imagenette:
    `K=4 -> 4.417`, `K=8 -> 3.253`, `K=16 -> 2.668`
  - quality keeps improving with larger `K`, but each rung doubles sequence length, so the next useful work is analysis rather than more launch plumbing
  - there is no remaining evidence of a resume/checkpoint correctness bug on the current path
- Next action: write up the K-rung comparison and decide whether the next experiment should be `K=16` follow-up training analysis, a bytes/symbols baseline, or a WebVision-style robustness pass.

### 2026-03-08 21:54 - Matched-budget K4 rerun staged and submitted

- Hypothesis: the original `K=4` baseline is directionally useful but not compute-matched, because it trained at
  `4096 * 512 = 2,097,152` tokens/step while `K=8` and `K=16` both trained at `1,048,576` tokens/step. A rerun at
  batch size `256` should give the apples-to-apples coefficient comparison we actually want.
- Command:
  - local edit of `experiments/jpeg_tokenizer/base/launch.py`
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k4-trial-matched"]'`
- Config:
  - new step: `tokexplore/jpeg-tokenizer-k4-trial-matched`
  - run id: `jpeg-tokenizer-k4-trial-matched`
  - sequence length: `4096`
  - batch size: `256`
  - tokens per step: `1,048,576`
  - token store: `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
  - W&B group/tags: `tokexplore-jpeg-tokenizer-k4-matched`, `["jpeg-tokenizer", "coeff-k4", "matched-budget"]`
- Result:
  - launch config committed as `d6c58996b` (`Add matched-budget K4 JPEG trial`)
  - new Ray submission: `ray-run-dlwh-launch-20260309-045325`
  - immediate controller state: `PENDING`
  - immediate message: `Job has not started yet. It may be waiting for the runtime environment to be set up.`
- Interpretation:
  - the next experiment is now exactly the one we wanted: same model family, same dataset, same training length, same
    token budget per step as `K=8/K=16`, but shorter `K=4` sequences
  - no new code-level issue is visible yet; early state looks like ordinary runtime-env setup
- Next action: monitor `ray-run-dlwh-launch-20260309-045325` through executor startup and the first training/checkpoint lines.

### 2026-03-08 22:20 - Matched-budget K4 completed

- Hypothesis: the matched-budget `K=4` rerun is the missing apples-to-apples baseline for the coefficient ladder.
- Command:
  - monitoring via `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a ...`
- Result:
  - `ray-run-dlwh-launch-20260309-045325` finished `SUCCEEDED`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-trial-matched-13c1dd/checkpoints/step-2000`
  - final eval loss: `4.340`
  - the step-level training loop completed cleanly; the shutdown-time wandb `BrokenPipeError` matched the already-known benign pattern
- Interpretation:
  - matching the token budget did not change the overall picture: `K=4` remains materially worse than `K=8` and `K=16`
  - the fair ladder is now `K=4 -> 4.340`, `K=8 -> 3.253`, `K=16 -> 2.668`
- Next action: evaluate the completed checkpoints on sequence-level and shared-prefix losses to separate "better representation" from "easier extra tokens".

### 2026-03-08 22:48 - Sequence-level coefficient sweep completed on dev TPU

- Hypothesis: mean token loss alone is ambiguous because larger `K` adds extra targets that may simply be easier. The right check is image-level NLL plus shared-prefix losses on identical target subsets.
- Command:
  - initial Ray job (head-node placement, failed): `ray-run-dlwh-evaluate_coefficient_sweep-20260309-052440`
  - TPU-reserved Ray retry (capacity-bound, stopped): `ray-run-dlwh-evaluate_coefficient_sweep-20260309-052657`
  - dev TPU fallback:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-eu-west4-a.yaml --tpu-name dlwh-jpeg-seqeval-0527 allocate --tpu-type v6e-8`
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-eu-west4-a.yaml --tpu-name dlwh-jpeg-seqeval-0527 execute --no-sync -- uv run python scripts/jpeg_tokenizer/evaluate_coefficient_sweep.py ...`
- Result:
  - added direct-GCS output support to the evaluator and committed it as `253e4dc31`
  - found and fixed a real tail-batch bug in the evaluator: the final validation batch could be smaller than the `8`-way data sharding, which broke `device_put(...)` on TPU
  - local fix validated with targeted pytest and a padding regression test
  - final sweep output written to:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-coeff-sequence-eval-253e4dc31-r4`
  - shared-prefix results:
    - `prefix_4` mean bits/image: `K=4 25430.55`, `K=8 24672.92`, `K=16 24282.01`
    - `prefix_8` mean bits/image: `K=8 44845.61`, `K=16 44167.21`
  - whole-sequence results:
    - `sequence` mean bits/image: `K=4 25430.55`, `K=8 44845.61`, `K=16 75156.11`
- Interpretation:
  - the whole-sequence bits/image numbers rise with `K` largely because the representation itself is longer, so they are not the right quality metric across different `K`
  - the shared-prefix numbers are the useful answer, and they move in the same direction as the training losses: larger `K` improves modeling of the shared early coefficients, not just the appended tail
  - that makes the "extra coefficients are merely easier" explanation incomplete; at least on Imagenette, the richer representation is helping prediction on the common prefix too
- Next action: commit the evaluator fix and write the Phase 1 note framing the K sweep around shared-prefix quality rather than mean token loss alone.

### 2026-03-08 23:06 - Prefix-only context ablation reverses the K advantage

- Hypothesis: the earlier prefix win for larger `K` may be coming from extra tail coefficients being useful *context*, not from a fundamentally better model of the low-frequency prefix in isolation.
- Command:
  - local evaluator update adding `--context-ablation-prefixes`
  - dev TPU run:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-eu-west4-a.yaml --tpu-name dlwh-jpeg-ablate-0550 allocate --tpu-type v6e-8`
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-eu-west4-a.yaml --tpu-name dlwh-jpeg-ablate-0550 execute --no-sync -- uv run python scripts/jpeg_tokenizer/evaluate_coefficient_sweep.py ... --context-ablation-prefixes 4,8`
- Result:
  - output written to:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-coeff-sequence-eval-ablate-0179b3388`
  - shared-prefix results with tail context removed:
    - `prefix_4_context_prefix_only` mean bits/image:
      `K=4 25430.55`, `K=8 27249.77`, `K=16 28640.63`
    - `prefix_8_context_prefix_only` mean bits/image:
      `K=8 44845.61`, `K=16 51906.86`
- Interpretation:
  - once coefficients beyond the shared prefix are removed from context, the larger-`K` models lose their prefix advantage and in fact become worse
  - that means the earlier `prefix_4` and `prefix_8` improvements were driven by extra retained coefficients being useful autoregressive context for later low-frequency targets
  - the more surprising "larger K intrinsically models the first coefficients better in isolation" hypothesis is not supported by this ablation
- Next action: update the Phase 1 report so the coefficient sweep conclusion is framed as a context-vs-length tradeoff, not a pure low-frequency representation win.

### 2026-03-08 23:22 - Byte-window baseline staged and launched

- Hypothesis: the next useful non-coefficient baseline is canonical JPEG bytes with an explicit fixed-window policy, so we can test whether raw byte syntax competes with the coefficient stream at the same token budget.
- Command:
  - local build:
    `uv run python scripts/jpeg_tokenizer/build_byte_token_store.py --log-every 1000 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`
  - regional mirror:
    `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`
  - smoke launch:
    `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-bytes-w8192-smoke"]'`
- Result:
  - added a frozen byte-window tokenizer config and helper in `experiments/jpeg_tokenizer/base/jpeg_codecs.py`
  - added `scripts/jpeg_tokenizer/build_byte_token_store.py`
  - added byte smoke/trial launch steps in `experiments/jpeg_tokenizer/base/launch.py`
  - local byte-window store built at:
    `artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`
  - regional mirror completed at:
    `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0`
  - store stats:
    - sequence length: `8192`
    - vocab size: `257` (`0..255` byte ids plus `256` as EOS/PAD)
    - train windows: `34191`
    - validation windows: `14264`
    - on-disk size: about `1.5 GiB`
  - smoke submission:
    `ray-run-dlwh-launch-20260309-061949`
  - executor step output:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-w8192-smoke-c5f441`
  - latest observed status:
    controller `RUNNING`, executor launched cleanly, Fray dispatch submitted `grug-train-jpeg-tokenizer-bytes-w8192-smoke`, but no trainer/W&B lines yet
- Interpretation:
  - the byte baseline is now concrete and cheap enough to run at the same `8192 * 128 = 1,048,576` tokens/step budget as the `K=8` coefficient baseline
  - the main remaining question is empirical: whether the byte stream can train cleanly and where its eval loss lands relative to `K=8`
- Next action: commit the byte builder/launch surface and keep an eye on `ray-run-dlwh-launch-20260309-061949` until trainer startup or a concrete failure signal appears.

### 2026-03-09 00:40 - Byte smoke succeeded; exact libjpeg K=8 baseline staged

- Hypothesis: the first useful pair of non-reference baselines is full canonical JPEG bytes and an exact libjpeg-backed
  coefficient stream at the same `K=8` budget.
- Command:
  - byte smoke monitoring:
    `uv run scripts/ray/cluster.py --cluster marin-eu-west4-a job-logs -n 200 ray-run-dlwh-launch-20260309-061949`
  - byte trial launch:
    `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-bytes-w8192-trial"]'`
  - exact coeff sanity:
    `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --source libjpeg --k 8 --max-train-examples 16 --max-validation-examples 8 ...`
  - exact coeff full build:
    `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --source libjpeg --k 8 --log-every 1000 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`
  - exact coeff mirror:
    `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`
  - exact coeff smoke launch:
    `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-libjpeg-smoke"]'`
- Result:
  - byte smoke completed successfully:
    - Ray job: `ray-run-dlwh-launch-20260309-061949`
    - W&B: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-bytes-w8192-smoke`
    - eval loss: `5.657 -> 4.610 -> 4.546`
    - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-w8192-smoke-c5f441/checkpoints/step-96`
  - byte trial submitted:
    - Ray job: `ray-run-dlwh-launch-20260309-073427`
  - added an explicit libjpeg-backed coefficient path via `CoefficientTokenSource.LIBJPEG`
  - tiny `K=8` exact-vs-reference sanity on `16` train images:
    - sequence shapes matched exactly: `(16, 8192)`
    - exact and reference rows differed on every example, but token equality was still `94.688%`
    - every mismatch was only `±1`, which is consistent with real libjpeg quantization differing slightly from the
      floating-point reference DCT
  - full exact `K=8` store mirrored to:
    `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0`
    - size: `420.76 MiB`
    - train examples: `9469`
    - validation examples: `3925`
    - sequence length: `8192`
    - vocab size: `4095`
  - first exact-coeff smoke submission failed immediately:
    - Ray job: `ray-run-dlwh-launch-20260309-074034`
    - failure: `ModuleNotFoundError: No module named 'jpeglib'`
    - cause: the Ray job was submitted from the pre-change commit, so the worker runtime env did not include the new
      dependency yet
- Interpretation:
  - the byte baseline is healthy enough to justify the full trial
  - the exact libjpeg coefficient path is now implemented, validated locally, and cheap enough to compare directly to
    the existing reference `K=8` rung
  - the libjpeg smoke failure is packaging hygiene, not a modeling or infra issue
- Next action: commit the new dependency + libjpeg launch surface, then relaunch `tokexplore/jpeg-tokenizer-k8-libjpeg-smoke`
  from the updated commit.

### 2026-03-09 01:10 - Exact libjpeg smoke fixed on the worker runtime

- Hypothesis: the exact `K=8` libjpeg smoke only needs explicit `pip_packages` on the Fray job request because the
  worker runtime env is exported from the `marin` package rather than the repository root lockfile.
- Command:
  - `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-libjpeg-smoke"]'`
- Config:
  - threaded `pip_packages=("jpeglib>=1.0.2",)` from `JpegTokenizerLaunchConfig` through the JPEG train config into
    `dispatch_grug_training_run(...)`
  - moved the `jpeglib` import in `jpeg_codecs.py` inside the exact-libjpeg extraction helper so non-libjpeg runs do
    not require the package at module import time
- Result:
  - the old exact smoke failed on worker import with `ModuleNotFoundError: No module named 'jpeglib'`
  - after the runtime-env fix, `ray-run-dlwh-launch-20260309-080827` completed successfully
  - executor step `tokexplore/jpeg-tokenizer-k8-libjpeg-smoke_1fb75879` reached terminal `SUCCEEDED`
- Interpretation:
  - the exact JPEG coefficient path is now operational on the production Ray + TPU stack rather than just locally
  - the next meaningful comparison is no longer a packaging check; it is the full exact `K=8` baseline versus the
    earlier reference `K=8` baseline
- Next action: launch `tokexplore/jpeg-tokenizer-k8-libjpeg-trial` from the fixed commit and monitor it through the
  first eval/checkpoint boundary.

### 2026-03-09 01:47 - Exact libjpeg K=8 trial completed

- Hypothesis: once `jpeglib` is installed through the Fray job request, the full exact `K=8` baseline should behave
  like the earlier reference `K=8` run rather than exposing a new runtime-only failure.
- Command:
  - submit:
    `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-k8-libjpeg-trial"]'`
  - monitor:
    followed `.agents/docs/job-monitoring-loop.md` on Ray track for `ray-run-dlwh-launch-20260309-081237`
- Result:
  - Ray job: `ray-run-dlwh-launch-20260309-081237`
  - W&B: `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-libjpeg-trial`
  - output path: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-trial-cdae37`
  - worker runtime confirmed: `Building environment with ['jpeglib>=1.0.2'], extras ['tpu']`
  - startup eval loss: `8.544`
  - final eval loss at step `2000`: `3.263`
  - final checkpoint: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-trial-cdae37/checkpoints/step-2000`
  - controller status: `SUCCEEDED`
- Interpretation:
  - the exact libjpeg-backed `K=8` baseline is now complete, so the representation comparison can move beyond the
    floating-point reference coefficient path
  - exact libjpeg lands essentially on top of the earlier reference `K=8` result (`3.263` vs `3.253`), which suggests
    the reference coefficient stream was already close enough for the first-order modeling conclusions
  - the shutdown-time W&B `BrokenPipeError` remains non-fatal and does not change the successful Ray terminal state
- Next action: update the Phase 1 comparison note to include exact-libjpeg `K=8`, then decide whether the next
  comparison should focus on bytes vs coeffs or on a sharper JPEG-symbol baseline.

### 2026-03-09 10:08 - Whole-image byte baseline prepared, but not launched

- Hypothesis: the honest byte-vs-coeff comparison should use one whole canonical JPEG per example rather than
  fixed `8192`-byte windows.
- Command:
  - added `scripts/jpeg_tokenizer/build_whole_image_byte_token_store.py`
  - scanned full Imagenette lengths with:
    `uv run python - <<'PY' ... canonicalize_image(...) ... whole_image_byte_length(...) ... PY`
- Result:
  - added a whole-image byte tokenizer path with distinct `EOS=256` and `PAD=257`
  - the passthrough data path now reads `loss_mask_ignore_id` from token-store metadata and masks pad tails correctly
  - full Imagenette whole-image canonical JPEG lengths came out to:
    - max length: `54544`
    - mean length: `25524.86`
    - examples: `13394`
  - a small whole-image builder smoke succeeded locally with `seq_len=28826` on a `4+2` example slice
- Interpretation:
  - the data path for a true whole-image byte baseline is now ready
  - but the current JPEG model is still the plain full-attention grug transformer, so a `54544`-token whole-image run
    is not a credible TPU experiment under the current architecture
  - the next fair byte-vs-coeff comparison requires an attention/windowing decision that can be applied consistently
    across both representations, rather than just swapping the token store
- Next action: keep the whole-image byte store path ready, but do not launch it until the attention regime for the
  comparison is made explicit.

### 2026-03-09 22:13 - Exact whole-image JPEG symbol baseline staged

- Hypothesis: the right follow-on to exact libjpeg coefficients is an exact JPEG symbol-stream baseline that keeps the
  same whole-image SWA setup while moving one level closer to the codec syntax.
- Command:
  - added `scripts/jpeg_tokenizer/build_whole_image_symbol_token_store.py`
  - built the local store with:
    `uv run python scripts/jpeg_tokenizer/build_whole_image_symbol_token_store.py --pad-to-multiple 128 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0 --log-every 2000`
  - mirrored it with:
    `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0`
  - submitted the smoke through a direct `JobSubmissionClient` on the forwarded dashboard:
    `raysubmit_XNjfpJ91vCFmCTs5`
- Result:
  - the symbol tokenizer now derives its run-length/category/value events from exact libjpeg quantized luma blocks via
    `CoefficientTokenSource.LIBJPEG`
  - whole-image symbol-store metadata:
    - `seq_len=58240`
    - `vocab_size=36835`
    - `eos_token_id=36833`
    - `pad_token_id=36834`
    - `loss_mask_ignore_id=36834`
  - observed maximum pre-pad symbol length on Imagenette: `58126`
  - mirrored store path:
    `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0`
  - smoke launch surface added:
    - `tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke`
    - `tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial`
  - live smoke status at log time:
    - Ray submit id: `raysubmit_XNjfpJ91vCFmCTs5`
    - controller state: `RUNNING`
    - executor and Fray dispatch completed cleanly to
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke-ee1834`
- Interpretation:
  - the exact symbol-stream baseline is now staged end-to-end with the same whole-image SWA regime as the byte run
  - this is the right comparison to answer whether codec-structured syntax tokens still beat raw canonical JPEG bytes
    when both are modeled as whole-image sequences
  - the open question is now purely runtime/training behavior, not data preparation or representation plumbing
- Next action: monitor the symbol smoke through trainer startup; if it clears, submit the full exact-symbol SWA trial.

### 2026-03-09 22:47 - Exact whole-image JPEG symbol smoke succeeded and trial launched

- Hypothesis: if the exact symbol stream trains cleanly under the same whole-image `SWA=4096` setup as bytes, it is
  worth launching the full head-to-head trial immediately rather than adding another intermediate debug pass.
- Command:
  - monitored the smoke via a single `JobSubmissionClient` + W&B polling loop for `raysubmit_XNjfpJ91vCFmCTs5`
  - submitted the full trial through the forwarded dashboard with:
    `raysubmit_Yi8WRcbCjnP3qEsp`
- Result:
  - smoke submit id: `raysubmit_XNjfpJ91vCFmCTs5`
  - smoke W&B:
    `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke`
  - smoke output path:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke-ee1834`
  - smoke terminal state: `SUCCEEDED`
  - smoke eval trajectory:
    - startup eval loss `10.662`
    - final eval loss `4.01261`
  - smoke final checkpoint:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke-ee1834/checkpoints/step-32`
  - full trial submit id: `raysubmit_Yi8WRcbCjnP3qEsp`
  - full trial initial controller state: `PENDING`
- Interpretation:
  - the exact symbol representation is now validated end-to-end on the production TPU path rather than just as a local
    token-store artifact
  - the smoke result is strong enough to justify the real run without additional launch-surface work
  - the next meaningful comparison is the completed symbol trial versus whole-image bytes and exact `K=8` coefficients
    under matched `SWA=4096`
- Next action: monitor `raysubmit_Yi8WRcbCjnP3qEsp` through startup and the first eval boundary.
