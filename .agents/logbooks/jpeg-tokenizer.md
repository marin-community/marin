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

### 2026-03-10 17:35 - Middle-ground JPEG smokes split cleanly

- Hypothesis: a payload-only byte baseline and a decoded Huffman-event baseline should tell us whether the full-byte
  weakness is mostly container noise or entropy-coding state.
- Command:
  - `uv run python scripts/jpeg_tokenizer/build_whole_image_scan_byte_token_store.py --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_scan_bytes_whole_v0`
  - `uv run python scripts/jpeg_tokenizer/build_whole_image_huffman_event_token_store.py --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_huffman_events_whole_libjpeg_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_scan_bytes_whole_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_scan_bytes_whole_v0`
  - `gsutil -m rsync -r artifacts/jpeg_tokenizer/token_store/imagenette_huffman_events_whole_libjpeg_v0 gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_huffman_events_whole_libjpeg_v0`
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-scan-bytes-whole-swa4096-smoke"]'`
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-smoke"]'`
- Config:
  - `scan_payload_bytes`: `seq_len=53760`, `vocab_size=258`
  - `huffman_events`: `seq_len=115840`, `vocab_size=2224`
  - both ran as whole-image `SWA=4096` smokes on `v6e-8`
- Result:
  - scan bytes smoke `ray-run-dlwh-launch-20260310-171904` `SUCCEEDED`, final eval loss `5.518`
  - huffman-event smoke `ray-run-dlwh-launch-20260310-172146` `SUCCEEDED`, final eval loss `2.276`
- Interpretation:
  - stripping container bytes is not enough; payload-only bytes still behave much more like raw bytes than like the
    structured JPEG baselines
  - decoded Huffman events are much stronger than expected and immediately deserve a long run
- Next action: launch both full trials so the control baseline exists, but prioritize monitoring the Huffman-event run.

### 2026-03-10 17:45 - Huffman-event trial launched

- Hypothesis: the strong `2.276` smoke should hold up over a full `2000`-step run and tell us whether decoded
  entropy events can beat the exact symbol stream.
- Command:
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial"]'`
- Config:
  - whole-image `huffman_events`
  - `seq_len=115840`
  - vocab `2224`
  - `SWA=4096`
  - batch size `8`
- Result so far:
  - Ray job: `ray-run-dlwh-launch-20260310-173123`
  - controller status: `RUNNING`
  - W&B run started successfully at `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial`
  - trainer reached `Starting from scratch` and entered the `2000`-step loop
- Interpretation: despite the very long sequence, the current TPU path is healthy enough to test the full decoded
  Huffman-event baseline directly.
- Next action: add the matching `scan_payload_bytes` full trial and monitor both to terminal state.

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

### 2026-03-09 23:04 - Exact whole-image JPEG symbol trial timed out before worker startup

- Hypothesis: the newly launched exact-symbol trial is healthy at the code/config level and only needs the normal
  startup stabilization check before it proceeds into training.
- Command:
  - monitored `raysubmit_Yi8WRcbCjnP3qEsp` with `JobSubmissionClient.get_job_status(...)`,
    `JobSubmissionClient.get_job_info(...)`, and W&B run lookup for
    `jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial`
- Result:
  - controller state: `FAILED`
  - W&B runs created: `0`
  - Ray error type: `JOB_SUPERVISOR_ACTOR_START_TIMEOUT`
  - Ray message:
    `Job supervisor actor failed to start within 900.0 seconds. This timeout can be configured by setting the environment variable RAY_JOB_START_TIMEOUT_SECONDS.`
  - there was no trainer traceback and no worker-side log output for the trial
- Interpretation:
  - this failure happened before the actual training job came up, so it does not invalidate the symbol smoke result
    or indicate a modeling/code regression in the symbol path itself
  - the immediate blocker is Ray job-supervisor startup latency / cluster pressure rather than the symbol tokenizer
    implementation
- Next action: resubmit the exact-symbol trial when capacity looks better, or raise the Ray job start timeout if this
  continues to recur on otherwise healthy large runtime-env submissions.

### 2026-03-10 06:25 - Exact whole-image JPEG symbol trial completed on the standard submit path

- Hypothesis: the repeated symbol-trial startup failures are submit-path / cluster-pressure issues, not a real problem
  with the symbol tokenizer or training config, so the standard `ray_run.py` path should still get the exact-symbol
  baseline through if we keep retrying.
- Command:
  - two direct-dashboard retries both failed before worker startup with
    `JOB_SUPERVISOR_ACTOR_START_TIMEOUT`:
    - `raysubmit_Yi8WRcbCjnP3qEsp`
    - `raysubmit_L6djGtQv6BgXFUF2`
  - resubmitted via:
    `RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$HOME/.ray/auth_token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial"]'`
  - successful Ray job:
    `ray-run-dlwh-launch-20260310-111410`
- Result:
  - exact-symbol W&B:
    `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial`
  - final output path:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3`
  - final checkpoint:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints/step-2000`
  - executor terminal status:
    `SUCCESS`
  - final eval loss:
    `2.885793924331665`
  - validation mean sequence lengths:
    - symbols: `32598.44`
    - bytes: `25662.39`
    - exact `K=8` coeffs: `8192.00`
  - coarse implied bits/image from final token-NLLs:
    - symbols: `135717.76`
    - bytes: `155903.86`
    - exact `K=8` coeffs: `38552.14`
- Interpretation:
  - exact JPEG symbols are now the strongest token-level baseline on the current JPEG ladder:
    - symbols: `2.886`
    - exact `K=8` coeffs: `3.262`
    - whole-image bytes: `4.211`
  - symbols beat bytes by a wide margin and also beat the exact coefficient stream even though the symbol sequence is
    much longer on average
  - coefficients still win on total representation compactness, which means the representation story now has a real
    two-axis tradeoff:
    - symbols win predictability
    - coefficients win compactness
- Next action: write the exact three-way SWA comparison into the Phase 1 report and then prefer evaluation/analysis of
  whole-sequence metrics over launching more JPEG variants immediately.

### 2026-03-10 08:35 - Exact sequence-level JPEG evaluator completed

- Hypothesis: the JPEG head-to-head should be written up from exact per-image sequence losses, not from rough
  back-of-the-envelope conversions from mean token loss, because the whole-image byte and symbol runs have variable
  sequence lengths and masked pad tails.
- Command:
  - first submit failed because the evaluator itself needed a TPU and I had not reserved one:
    `ray-run-dlwh-evaluate_representation_head2head-20260310-160818`
  - successful retry:
    `RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$HOME/.ray/auth_token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a --tpu auto -e WANDB_API_KEY=$WANDB_API_KEY -- python scripts/jpeg_tokenizer/evaluate_representation_head2head.py --run-spec name=coeff_k8_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-trial-392707/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0,sliding_window=4096,unit_name=block,unit_count=1024 --run-spec name=bytes_whole,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial-7cc718/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0,sliding_window=4096 --run-spec name=symbols_whole_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096 --output-dir gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-ebf28526a-r2`
  - successful Ray job:
    `ray-run-dlwh-evaluate_representation_head2head-20260310-162115`
- Result:
  - summary:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-ebf28526a-r2/summary.md`
  - json:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-ebf28526a-r2/representation_eval.json`
  - exact `coeff_k8` metrics:
    - mean actual tokens/image: `8192.00`
    - mean bits/image: `44928.74`
    - mean bits/pixel: `0.6856`
    - mean bits/modeled-token: `5.4851`
    - mean bits/block: `43.8757`
  - exact `bytes_whole` metrics:
    - mean actual tokens/image: `25662.39`
    - mean bits/image: `159685.81`
    - mean bits/pixel: `2.4366`
    - mean bits/modeled-token: `6.1948`
  - exact `symbols_whole_exact` metrics:
    - mean actual tokens/image: `32598.44`
    - mean bits/image: `145094.24`
    - mean bits/pixel: `2.2140`
    - mean bits/modeled-token: `4.3496`
- Interpretation:
  - the coarse earlier estimates were directionally right, but the exact evaluator makes the conclusion much firmer:
    - symbols beat bytes on both token-level predictability and total bits/image
    - coeffs remain far more compact than either whole-image syntax stream
    - bytes are worst on both axes the current evaluator measures well
  - the JPEG thread now has a stable baseline story:
    - symbols = best predictability
    - coeff `K=8` = best compactness
    - bytes = weakest representation of the three
- Next action: treat JPEG as baseline-complete and move the next mechanism work to the gzip/reset thread rather than
  launching more JPEG training variants.

### 2026-03-10 09:58 - Two middle-ground JPEG baselines staged without launching TPU runs

- Hypothesis: there should be at least one useful representation between full canonical JPEG bytes and the current
  pre-Huffman symbol stream, but it is not obvious whether the best middle ground is "less container noise" or
  "less collapsed syntax."
- Changes:
  - added `encode_jpeg_scan_bytes(...)` in
    `experiments/jpeg_tokenizer/base/jpeg_codecs.py`
    to extract only entropy-coded scan payload bytes from canonical JPEGs
  - added `encode_jpeg_huffman_events(...)` in
    `experiments/jpeg_tokenizer/base/jpeg_codecs.py`
    to emit split entropy events: event ids plus separate amplitude payload tokens
  - added builders:
    - `scripts/jpeg_tokenizer/build_whole_image_scan_byte_token_store.py`
    - `scripts/jpeg_tokenizer/build_whole_image_huffman_event_token_store.py`
  - added focused regression coverage in `tests/test_jpeg_tokenizer_scaffold.py`
- Validation:
  - targeted pytest over the new codec helpers passed
  - `py_compile` over the new codec and builder paths passed
  - tiny local smoke stores on `2` train + `2` validation examples succeeded:
    - `scan_payload_bytes`: `seq_len=26055`, `vocab_size=258`
    - `huffman_events`: `seq_len=71211`, `vocab_size=2224`
- Interpretation:
  - `scan_payload_bytes` is the cleaner immediate next JPEG baseline because it isolates "bytes minus container noise"
    without changing the sequence regime too radically
  - `huffman_events` is a semantic middle ground, but not a length middle ground; it is substantially longer than the
    existing exact symbol stream and would need a different architecture or packing story before it is an efficient TPU
    baseline
- Next action: keep the gzip/reset thread as the main follow-up, and if we return to JPEG first, prioritize
  `scan_payload_bytes` over `huffman_events`.

### 2026-03-11 20:45 - Exact `K=64` SWA baseline staged

- Hypothesis: `K=8` is dramatically more compact, but that result is confounded by heavy coefficient truncation. A full
  `K=64` exact-libjpeg coefficient run should tell us where the coefficient family lands once it is brought much closer
  to the information content of the whole-image syntax streams.
- Changes:
  - added `tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-smoke` and
    `tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial` in
    `experiments/jpeg_tokenizer/base/launch.py`
  - both use exact libjpeg coefficients, `max_seq_len=65536`, `sliding_window=4096`, and `batch_size=8`
- Data:
  - built full store locally with:
    `uv run python scripts/jpeg_tokenizer/build_coeff_token_store.py --k 64 --source libjpeg --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0`
  - mirrored to:
    `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0`
  - store stats:
    - local size: `3.3G`
    - `seq_len=65536`
    - `vocab_size=4095`
    - train examples: `9469`
    - validation examples: `3925`
- Interpretation:
  - `K=64` is now in the same rough whole-image sequence regime as the syntax streams:
    - `coeff_k64`: `65536`
    - `huffman_events`: mean `63173.26`
    - `symbols`: mean `32598.44`
    - `bytes`: mean `25662.39`
  - the main remaining question is whether full coefficients retain enough structure to beat the near-lossless syntax
    streams once the `K=8` compactness advantage is gone
- Next action: launch the `K=64` exact-libjpeg `SWA=4096` smoke on `marin-eu-west4-a`, then decide whether the full
  trial is worth the spend from the smoke behavior.

### 2026-03-11 22:13 - Exact `K=64` whole-image result recorded

- Outcome:
  - `K=64` smoke succeeded:
    - Ray job:
      `ray-run-dlwh-launch-20260312-034553`
    - eval loss:
      `8.891 -> 1.654 -> 1.573`
  - `K=64` full trial succeeded:
    - Ray job:
      `ray-run-dlwh-launch-20260312-035554`
    - final checkpoint:
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial-7e3e81/checkpoints/step-2000`
    - final eval loss:
      `1.078`
- Follow-up:
  - the first attempt at a six-way exact whole-image evaluator was broader than necessary and took too long to be the
    right move for answering the immediate question
  - launched a focused exact evaluator for `coeff_k64_exact` only:
    - Ray job:
      `ray-run-dlwh-evaluate_representation_head2head-20260312-050919`
    - summary:
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-k64-only-r1/summary.md`
- Exact sequence-level result:
  - `coeff_k64_exact`
    - mean actual tokens/image: `65536.00`
    - mean bits/image: `138557.72`
    - mean bits/pixel: `2.1142`
    - mean bits/modeled-token: `2.1143`
    - mean bits/block: `135.3103`
- Combined whole-image table:
  - `coeff_k8_exact`: `44928.74` bits/image
  - `coeff_k64_exact`: `138557.72`
  - `symbols_whole_exact`: `145094.24`
  - `huffman_events`: `147539.84`
  - `scan_payload_bytes`: `158185.08`
  - `bytes_whole`: `159685.81`
- Interpretation:
  - `K=8` remains the compact lossy coefficient baseline
  - `K=64` is the first fair whole-image coefficient comparison, and it still beats the near-lossless syntax streams on
    total bits/image
  - exact symbols remain the best non-coefficient whole-image JPEG representation
  - bytes are still worst, and removing headers still does not matter

### 2026-03-11 22:23 - Longer-run and larger-model JPEG sweeps launched

- Goal:
  - first test whether the current ordering survives more training at the same small model
  - then test whether the ordering is just a small-model artifact by staging a modestly larger SWA model on the same
    three representations
- Code:
  - commit:
    `d9be3644f` (`Stage JPEG long-run and large-model sweeps`)
  - added a larger baseline model in
    `experiments/jpeg_tokenizer/base/model.py`:
    `hidden_dim=768`, `intermediate_dim=2688`, `num_layers=8`, `num_heads=12`
  - added long-run resume steps and larger-model smoke steps in
    `experiments/jpeg_tokenizer/base/launch.py`
- Current long-run jobs:
  - `ray-run-dlwh-launch-20260312-052159`
    - `tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-long`
    - resumes from
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial-7e3e81/checkpoints`
    - target steps:
      `8000`
  - `ray-run-dlwh-launch-20260312-052212`
    - `tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long`
    - resumes from
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints`
    - target steps:
      `8000`
  - `ray-run-dlwh-launch-20260312-052225`
    - `tokexplore/jpeg-tokenizer-bytes-whole-swa4096-long`
    - resumes from
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial-7cc718/checkpoints`
    - target steps:
      `8000`
- Current larger-model smokes:
  - `ray-run-dlwh-launch-20260312-052055`
    - `tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-smoke`
  - `ray-run-dlwh-launch-20260312-052237`
    - `tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-smoke`
  - `ray-run-dlwh-launch-20260312-052250`
    - `tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-smoke`
- Initial cluster check:
  - all six are controller-`RUNNING`
  - I only submitted the larger-model variants as smokes because batch must stay divisible by the 8-way TPU mesh and
    the larger model may still hit memory limits at these sequence lengths

### 2026-03-11 22:34 - Large-model smoke outcomes and promoted trials

- `symbols` larger-model smoke:
  - Ray job:
    `ray-run-dlwh-launch-20260312-052055`
  - output:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-smoke-1be3e2`
  - W&B:
    `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-smoke`
  - final eval loss:
    `4.066`
  - status:
    succeeded
- `bytes` larger-model smoke:
  - Ray job:
    `ray-run-dlwh-launch-20260312-052250`
  - output:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-smoke-8f5fa3`
  - W&B:
    `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-bytes-whole-large-swa4096-smoke`
  - final eval loss:
    `5.458`
  - status:
    succeeded
- `K=64` larger-model smoke:
  - Ray job:
    `ray-run-dlwh-launch-20260312-052237`
  - output:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-smoke-7603d1`
  - current state:
    controller `RUNNING`, executor launched, Fray training job submitted, but no worker-side W&B or checkpoint signal yet
  - interpretation:
    looks like delayed worker allocation / TPU scheduling rather than an immediate model crash
- Promoted trials launched from the larger-model smoke results:
  - `symbols` larger-model trial:
    - Ray job:
      `ray-run-dlwh-launch-20260312-052950`
    - step:
      `tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-trial`
  - `bytes` larger-model trial:
    - Ray job:
      `ray-run-dlwh-launch-20260312-053012`
    - step:
      `tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-trial`
  - `K=64` larger-model trial:
    - Ray job:
      `ray-run-dlwh-launch-20260312-055501`
    - step:
      `tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-trial`
    - note:
      launched despite the smoke still looking scheduler-delayed, because the remaining risk is queue waste rather than
      an identified model/config bug
- Long same-model resumes still active:
  - `ray-run-dlwh-launch-20260312-052159` (`K=64` long)
  - `ray-run-dlwh-launch-20260312-052212` (`symbols` long)
  - `ray-run-dlwh-launch-20260312-052225` (`bytes` long)

### 2026-03-12 11:36 - Long-run and larger-model trials completed

- Goal:
  - close out the launched longer/smaller-model and larger-model sweeps and check whether the representation ordering
    persists under more training and more capacity
- Result summary:
  - all targeted submissions reached terminal `SUCCEEDED`
  - long small-model (`step 8000`) final eval losses:
    - `K=64` long (`ray-run-dlwh-launch-20260312-052159`): `1.013`
    - `symbols` long (`ray-run-dlwh-launch-20260312-052212`): `2.673`
    - `bytes` long (`ray-run-dlwh-launch-20260312-052225`): `3.930`
  - larger-model (`8L/768d`, `step 2000`) final eval losses:
    - `K=64` trial (`ray-run-dlwh-launch-20260312-055501`): `1.054`
    - `symbols` trial (`ray-run-dlwh-launch-20260312-052950`): `2.795`
    - `bytes` trial (`ray-run-dlwh-launch-20260312-053012`): `4.078`
  - previously ambiguous `K=64` large smoke now confirmed succeeded:
    - `ray-run-dlwh-launch-20260312-052237`
    - final smoke eval loss: `1.905`
- Key output paths:
  - `K=64` long:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-long-5272ec/checkpoints/step-8000`
  - `symbols` long:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long-b4aa28/checkpoints/step-8000`
  - `bytes` long:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-long-64db87/checkpoints/step-8000`
  - `K=64` large trial:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-trial-de16b2/checkpoints/step-2000`
  - `symbols` large trial:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-trial-4b09ce/checkpoints/step-2000`
  - `bytes` large trial:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-trial-f64948/checkpoints/step-2000`
- Interpretation:
  - the core ordering is stable under both longer training and the larger model:
    `K=64` best, `symbols` second, `bytes` last
  - this reduces the chance that earlier ordering was a short-training or too-small-model artifact

### 2026-03-12 23:10 - Representation eval relaunch succeeded (whole-image metrics)

- Goal:
  - complete the `long-r3` and `large-r3` whole-image representation evaluations and report only whole-sequence loss metrics
- Command:
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a --submission-id ray-run-dlwh-launch-20260313-054701 -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-representation-eval-long-r3"]'`
  - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a --submission-id ray-run-dlwh-launch-20260313-054708 -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-representation-eval-large-r3"]'`
- Fix applied before relaunch:
  - removed a circular import by making `evaluate_representation_head2head` a lazy import inside
    `experiments/jpeg_tokenizer/base/launch.py::_run_representation_eval_local`
  - prior failed submissions (`ray-run-dlwh-launch-20260313-053321`, `...053348`) died with
    `ImportError: cannot import name 'main' from partially initialized module ...`
- Result:
  - relaunch jobs:
    - `ray-run-dlwh-launch-20260313-054701`: `SUCCEEDED`
    - `ray-run-dlwh-launch-20260313-054708`: `SUCCEEDED`
  - whole-image sequence loss (`mean bits/image`) from summaries:
    - long (`step 8000`):
      - `coeff_k64_long`: `133,249.02`
      - `symbols_long`: `137,400.13`
      - `bytes_long`: `152,544.14`
    - large (`8L/768d`, `step 2000`):
      - `coeff_k64_large`: `166,994.45`
      - `symbols_large`: `171,295.41`
      - `bytes_large`: `190,202.97`
  - output summaries:
    - `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-long-r3/summary.md`
    - `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-large-r3/summary.md`
- Interpretation:
  - under whole-image sequence loss, ordering is unchanged in both regimes:
    `coeff K=64` best, `symbols` second, `bytes` worst
  - this resolves the previously pending long/large whole-image head-to-head with terminal successful runs.

### 2026-03-13 11:06 - Iris relaunch in `europe-west4-a` completed

- Goal:
  - relaunch the long whole-image representation eval using Iris in `europe-west4-a` and verify region-local execution
- Commands:
  - initial Iris submit path (executor step):
    - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v6e-8 --region europe-west4 --zone europe-west4-a --job-name jpeg-tokenizer-representation-eval-long-r3-iris3 -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-representation-eval-long-r3"]'`
  - effective direct relaunch (to bypass executor dedupe):
    - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 1.0 --memory 16GB --extra marin:tpu --tpu v6e-8 --region europe-west4 --zone europe-west4-a --job-name jpeg-tokenizer-representation-eval-long-direct-iris2 -- python scripts/jpeg_tokenizer/evaluate_representation_head2head.py --run-spec name=coeff_k64_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-long-5272ec/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0,sliding_window=4096,unit_name=block,unit_count=1024 --run-spec name=symbols_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long-b4aa28/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096 --run-spec name=bytes_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-long-64db87/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0,sliding_window=4096 --output-dir gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-long-r4-iris2`
- Notable Iris issues and fixes:
  - `iris job run` initially stalled because workspace bundling included large untracked `artifacts/` files (`git ls-files --others` path); fixed by adding `artifacts/` to local git excludes (`.git/info/exclude`) for this workspace.
  - first direct Iris run (`...direct-iris1`) failed with container OOM (`exit 137`) under default `memory=1GB`; relaunch with `--memory 16GB --cpu 1.0` succeeded.
  - executor-based Iris relaunches were terminal-success but executed `0` steps due existing completed step status for that executor step.
- Result:
  - terminal Iris job: `/dlwh/jpeg-tokenizer-representation-eval-long-direct-iris2` -> `JOB_STATE_SUCCEEDED`
  - worker: `marin-tpu_v6e_8-europe-west4-a-20260313-1731-e4119b4d-worker-0`
  - output: `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-long-r4-iris2`
  - summary metrics (`mean bits/image`):
    - `coeff_k64_long`: `133,249.02`
    - `symbols_long`: `137,400.13`
    - `bytes_long`: `152,544.19`
- Interpretation:
  - Iris region-constrained run reproduces prior long-eval ordering with equivalent whole-image loss numbers.

### 2026-03-13 15:52 - AC-dense context-ablation stream staged and smoke launched

- Goal:
  - isolate the effect of symbol run-length structure by keeping JPEG coefficient semantics but removing variable-length
    AC run/EOB encoding
- Changes:
  - added a new codec path in `experiments/jpeg_tokenizer/base/jpeg_codecs.py`:
    - `AcDenseTokenizerConfig`
    - `encode_jpeg_ac_dense_tokens(...)` (`DC delta` + `63` dense AC tokens per block)
    - `ac_dense_vocab_size(...)`
  - added store builder:
    `scripts/jpeg_tokenizer/build_whole_image_ac_dense_token_store.py`
  - added launch steps in `experiments/jpeg_tokenizer/base/launch.py`:
    - `tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke`
    - `tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-trial`
  - added deterministic codec coverage in `tests/test_jpeg_tokenizer_scaffold.py`
- Data:
  - full store build command:
    `uv run python scripts/jpeg_tokenizer/build_whole_image_ac_dense_token_store.py --source libjpeg --log-every 1000 --output-dir artifacts/jpeg_tokenizer/token_store/imagenette_ac_dense_whole_libjpeg_v0`
  - mirrored to:
    `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_ac_dense_whole_libjpeg_v0`
  - store stats:
    - local size: `3.3G`
    - `seq_len=65536`
    - `vocab_size=6142`
    - train examples: `9469`
    - validation examples: `3925`
- Launch:
  - submitted smoke run:
    `ray-run-dlwh-launch-20260313-224035`
  - command:
    `RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster marin-eu-west4-a -e WANDB_API_KEY=$WANDB_API_KEY -- python experiments/jpeg_tokenizer/base/launch.py --prefix gs://marin-eu-west4 --executor_info_base_path gs://marin-eu-west4/experiments --run_only '["tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke"]'`
- Current status:
  - executor status is `RUNNING`; training has been dispatched via Fray but is still in TPU scheduling wait (no
    checkpoint directory yet at
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke-666ade`)
- Next action:
  - continue monitoring to terminal state; if smoke succeeds, run the `ac-dense` trial and then add whole-image
    comparison against `symbols` and `coeff_k64`.

### 2026-03-13 22:55 - Switched stalled AC-dense smoke from Ray to Iris; trial launched

- Trigger:
  - user requested kill + switch to Iris after prolonged Ray scheduler stall
- Actions:
  - stopped Ray smoke job:
    `ray-run-dlwh-launch-20260313-224035` (`STOPPED`)
  - launched Iris smoke in `europe-west4-a`:
    `/dlwh/jpeg-tokenizer-ac-dense-smoke-iris1`
    with `JPEG_TOKENIZER_RUN_ID=jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke-iris1`
  - Iris smoke took over stale executor state by lock takeover and dispatched:
    `grug-train-jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke-iris1`
- Smoke outcome:
  - terminal status in executor output:
    `SUCCESS`
  - W&B run:
    `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke-iris1`
  - key logs:
    - eval loss `8.296 -> 1.848 -> 1.717`
    - checkpoint saved at
      `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-smoke-666ade/checkpoints/step-32`
- Follow-on launch:
  - submitted Iris trial:
    `/dlwh/jpeg-tokenizer-ac-dense-trial-iris1`
    with `JPEG_TOKENIZER_RUN_ID=jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-trial-iris1`
  - command path:
    `... --run_only '["tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-trial"]'`
- Next action:
  - monitor `/dlwh/jpeg-tokenizer-ac-dense-trial-iris1` to terminal; then run whole-image representation eval with
    AC-dense included.

### 2026-03-14 13:52 - Perturbation-sensitivity head-to-head succeeded (whole-image deltas)

- Goal:
  - strengthen the "token meaning stability" claim with a direct local-corruption test, not just clean NLL ordering
- Changes:
  - added evaluator:
    `scripts/jpeg_tokenizer/evaluate_representation_perturbation.py`
  - bug fix after first failed submit:
    - set `use_explicit_mesh_axes=True` and call `trainer.initialize()` in the evaluator's `TrainerConfig` path
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 1.0 --memory 24GB --extra marin:tpu --tpu v6e-8 --region europe-west4 --zone europe-west4-a --job-name jpeg-tokenizer-perturbation-r1-iris1 -- uv run python scripts/jpeg_tokenizer/evaluate_representation_perturbation.py --run-spec name=coeff_k64_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial-7e3e81/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0,sliding_window=4096 --run-spec name=ac_dense_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-ac-dense-whole-libjpeg-swa4096-trial-97d827/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_ac_dense_whole_libjpeg_v0,sliding_window=4096 --run-spec name=symbols_whole_exact,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096 --batch-size 8 --max-examples 512 --perturb-fractions 0.5 --horizons 1,64,512,4096 --output-dir gs://marin-eu-west4/tokexplore/jpeg-tokenizer-perturbation-r1-iris1`
- Result:
  - Iris job: `/dlwh/jpeg-tokenizer-perturbation-r1-iris1` -> `JOB_STATE_SUCCEEDED`
  - output:
    `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-perturbation-r1-iris1/{summary.md,perturbation_eval.json}`
  - clean whole-image loss (`mean bits/image`, 512 validation examples):
    - `coeff_k64_exact`: `121,938.50`
    - `ac_dense_exact`: `122,106.54`
    - `symbols_whole_exact`: `128,364.00`
  - single-token corruption at fraction `0.5` (`mean delta bits/image`):
    - total delta:
      - `coeff_k64_exact`: `13.86`
      - `ac_dense_exact`: `13.62`
      - `symbols_whole_exact`: `4.04`
    - immediate delta (`h1`):
      - `coeff_k64_exact`: `0.03`
      - `ac_dense_exact`: `0.03`
      - `symbols_whole_exact`: `0.29`
    - short-horizon tail-only delta (`h64`, excluding immediate):
      - `coeff_k64_exact`: `0.11`
      - `ac_dense_exact`: `0.02`
      - `symbols_whole_exact`: `0.72`
- Interpretation:
  - symbols are much more locally brittle to a one-token corruption than either coefficient stream (about `10x` immediate
    and `6x-30x` short-tail amplification), which supports the "context-dependent token semantics hurt local robustness"
    mechanism.
  - `ac_dense` remains the most locally stable stream among the two JPEG-syntax-adjacent baselines tested (`ac_dense` vs
    symbols) while preserving near-lossless clean whole-image loss.
  - total-delta ordering differs (`coeff/ac_dense > symbols`), suggesting weaker but more diffuse long-tail coupling in
    coefficient streams; this is a follow-up axis rather than a contradiction.
- Next action:
  - rerun perturbation with multiple positions (`0.25,0.5,0.75`) and token-type-conditioned corruption (especially
    symbol control-token hits) to separate immediate desync brittleness from long-tail coupling.
