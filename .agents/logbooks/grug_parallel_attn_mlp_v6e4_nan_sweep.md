# Grug Parallel-Attn-MLP v6e-4 NaN Sweep: Research Logbook

## Scope
- Goal: Find non-NaN hyperparameters for `experiments/grug/parallel_attn_mlp` on `v6e-4`.
- Primary metric(s):
  - `train/loss` remains finite through >=400 steps.
  - `grad/norm/total` remains finite at watch intervals.
- Constraints:
  - Run on `v6e-4` for consistent hardware.
  - Use one-axis sweeps first.
  - Keep run commands reproducible and logged.

## Metadata
- Experiment issue: https://github.com/marin-community/marin/issues/3316
- Branch: `codex/research-grug-parallel-nan-sweep-v6e4`
- Experiment ID prefix: `GRUG-NAN`

## Baseline
- Date: 2026-03-05
- Code refs:
  - `experiments/grug/parallel_attn_mlp/launch.py`
  - `experiments/grug/parallel_attn_mlp/train.py`
- Baseline config (current defaults):
  - `lr=3e-3`
  - `warmup=1000`
  - `min_lr_ratio=0.1`
  - `z_loss_weight=1e-4`
  - `max_grad_norm=1.0`
- Known behavior before this sweep:
  - recurrent NaN onset in early training for parallel variant.

## Initial Matrix
- `GRUG-NAN-001`: Baseline on `v6e-4` (confirm failure signature/time).
- `GRUG-NAN-002`: Lower peak LR to `2e-3`.
- `GRUG-NAN-003`: Lower peak LR to `1e-3`.
- `GRUG-NAN-004`: Lower peak LR to `5e-4`.
- Follow-up axis (after LR): warmup (`1000/2000/4000`) around best LR.
- Follow-up axis (if needed): `z_loss_weight`, `max_grad_norm`.

## Stop Criteria
- Stop and recommend candidate when at least one config is non-NaN through >=400 steps and replicated >=2 times.
- Stop and escalate to model-level changes if all one-axis sweeps fail.

## Experiment Log
### 2026-03-05 18:42 - Kickoff (`GRUG-NAN-000`)
- Hypothesis: NaNs are optimizer/schedule-instability driven and can be removed by lower LR + adjusted warmup.
- Command: N/A (kickoff)
- Config: N/A
- Result: Research thread initialized (issue + branch + logbook).
- Interpretation: Ready to launch v6e-4 one-axis LR sweep.
- Next action: Add lightweight env overrides for launch parameters; submit first batch.

### 2026-03-05 23:55 - Region-pinned LR sweep relaunch (`GRUG-NAN-014..017`)
- Hypothesis: Prior crash/kill behavior was capacity/worker-path specific; pinning `us-east1` and reducing loader pressure allows stable v6e-4 screening.
- Command:
  - `GRUG_RUN_ID=grug-parallel-v6e4-GRUG-NAN-014-20260305-235544 GRUG_WANDB_GROUP=grug-parallel-attn-v6e4-us-east1-nan-sweep GRUG_TPU_VARIANT=v6e-4 GRUG_TPU_REGION=us-east1 GRUG_STEPS=400 GRUG_DISABLE_EVAL=1 GRUG_PREFETCH_SIZE=4 GRUG_MAX_BUFFERED_BATCHES=16 GRUG_LR=0.003 uv run iris --config lib/iris/examples/marin.yaml job run --extra marin:tpu --reserve=v6e-4 --region us-east1 -- python experiments/grug/parallel_attn_mlp/launch.py`
  - `GRUG_RUN_ID=grug-parallel-v6e4-GRUG-NAN-015-20260305-235550 GRUG_WANDB_GROUP=grug-parallel-attn-v6e4-us-east1-nan-sweep GRUG_TPU_VARIANT=v6e-4 GRUG_TPU_REGION=us-east1 GRUG_STEPS=400 GRUG_DISABLE_EVAL=1 GRUG_PREFETCH_SIZE=4 GRUG_MAX_BUFFERED_BATCHES=16 GRUG_LR=0.002 uv run iris --config lib/iris/examples/marin.yaml job run --extra marin:tpu --reserve=v6e-4 --region us-east1 -- python experiments/grug/parallel_attn_mlp/launch.py`
  - `GRUG_RUN_ID=grug-parallel-v6e4-GRUG-NAN-016-20260305-235558 GRUG_WANDB_GROUP=grug-parallel-attn-v6e4-us-east1-nan-sweep GRUG_TPU_VARIANT=v6e-4 GRUG_TPU_REGION=us-east1 GRUG_STEPS=400 GRUG_DISABLE_EVAL=1 GRUG_PREFETCH_SIZE=4 GRUG_MAX_BUFFERED_BATCHES=16 GRUG_LR=0.001 uv run iris --config lib/iris/examples/marin.yaml job run --extra marin:tpu --reserve=v6e-4 --region us-east1 -- python experiments/grug/parallel_attn_mlp/launch.py`
  - `GRUG_RUN_ID=grug-parallel-v6e4-GRUG-NAN-017-20260305-235606 GRUG_WANDB_GROUP=grug-parallel-attn-v6e4-us-east1-nan-sweep GRUG_TPU_VARIANT=v6e-4 GRUG_TPU_REGION=us-east1 GRUG_STEPS=400 GRUG_DISABLE_EVAL=1 GRUG_PREFETCH_SIZE=4 GRUG_MAX_BUFFERED_BATCHES=16 GRUG_LR=0.0005 uv run iris --config lib/iris/examples/marin.yaml job run --extra marin:tpu --reserve=v6e-4 --region us-east1 -- python experiments/grug/parallel_attn_mlp/launch.py`
- Config:
  - LR axis: `3e-3`, `2e-3`, `1e-3`, `5e-4`.
  - Common: `warmup=1000`, `max_grad_norm=1.0`, `z_loss_weight=1e-4`, eval disabled.
- Result:
  - Active Iris child jobs:
    - `/dlwh/iris-run-launch-20260306-075548/grug-train-grug-parallel-v6e4-GRUG-NAN-014-20260305-235544` (`RUNNING`)
    - `/dlwh/iris-run-launch-20260306-075556/grug-train-grug-parallel-v6e4-GRUG-NAN-015-20260305-235550` (`RUNNING`)
    - `/dlwh/iris-run-launch-20260306-075604/grug-train-grug-parallel-v6e4-GRUG-NAN-016-20260305-235558` (`PENDING`, capacity)
    - `/dlwh/iris-run-launch-20260306-075610/grug-train-grug-parallel-v6e4-GRUG-NAN-017-20260305-235606` (`PENDING`, capacity)
  - W&B:
    - `014`: https://wandb.ai/marin-community/marin/runs/grug-parallel-v6e4-GRUG-NAN-014-20260305-235544
    - `015`: https://wandb.ai/marin-community/marin/runs/grug-parallel-v6e4-GRUG-NAN-015-20260305-235550
  - Early metrics (finite):
    - `014 (lr=3e-3)`: step 14, `train/loss=10.8231`
    - `015 (lr=2e-3)`: step 29, `train/loss=9.9624`
- Interpretation:
  - Region pinning and loader tuning moved runs into steady training.
  - No NaN observed so far for the two highest-LR configs in this batch.
- Next action:
  - Continue monitoring `014/015` for NaN/divergence.
  - Start `016/017` when v6e-4 capacity frees up; compare stability and loss slope.

### 2026-03-06 00:09 - Live monitoring checkpoint (`GRUG-NAN-014/015`)
- Hypothesis: `lr=2e-3` and `lr=3e-3` may both be viable on v6e-4 if they survive early preemption/restarts.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs <child-job-id> --since-seconds 600`
  - `uv run python - <<'PY' ... wandb.Api().run(...).history(...) ... PY`
- Config: unchanged from `GRUG-NAN-014..017`.
- Result:
  - `015` is steadily training:
    - Iris progress: `36/400`, `loss=9.51` (then `42` in W&B).
    - W&B latest: `step=42`, `train/loss=8.7628` (finite).
  - `014` was preempted once and restarted on a different worker:
    - preemption count now `1`.
    - restart observed (`worker=10.142.0.48`) with same W&B run ID.
    - latest completed loss sample before restart remains finite (`step=14`, `train/loss=10.8231`).
  - `016` and `017` still pending due `tpu_v6e_4-us-east1-d` capacity.
- Interpretation:
  - `lr=2e-3` currently has strongest non-NaN signal and improving loss.
  - `lr=3e-3` remains inconclusive post-restart (not failed, but resumed).
- Next action:
  - Keep monitoring for NaN/terminal states.
  - Let `016/017` start automatically once capacity frees.

### 2026-03-06 00:16 - Queue retarget + launch reliability check (`GRUG-NAN-018..024`)
- Hypothesis: With `015` strongest early, warmup-axis follow-ups around `lr=2e-3` would likely produce better 400-step loss.
- Command:
  - stop pending low-LR queue:
    - `uv run iris --config lib/iris/examples/marin.yaml job stop /dlwh/iris-run-launch-20260306-075604`
    - `uv run iris --config lib/iris/examples/marin.yaml job stop /dlwh/iris-run-launch-20260306-075610`
  - attempted follow-up launches (`018..024`) using both:
    - attached mode (`job run ...`) and
    - detached mode (`job run --no-wait` / `--no-terminate-on-exit`)
  - attempted constraints and retries:
    - `--region us-east1`, `--zone us-east1-d`, fresh run IDs, repeated submission loop.
- Config:
  - targeted candidates around current best:
    - `lr=2e-3`, `warmup=400`
    - `lr=2e-3`, `warmup=200`
    - one lower-LR follow-up (`lr=1.5e-3`, `warmup=200`)
- Result:
  - New submissions repeatedly died in the launch phase with immediate `JOB_STATE_KILLED` before useful child progress.
  - Failures often landed on known-bad launch worker `10.164.0.38`; retries remained unstable.
  - No reliable additional comparative run was obtained from `018..024`.
- Interpretation:
  - Launch-path reliability (not model instability) became the limiting factor for additional sweep breadth in this window.
  - Best evidence remained from stable long runs `014` and `015`.
- Next action:
  - Continue `014`/`015` to completion and select best by non-NaN + final/min loss.

### 2026-03-06 01:33 - 400-step comparison outcome (`GRUG-NAN-014` vs `GRUG-NAN-015`)
- Hypothesis: One of `lr=3e-3` or `lr=2e-3` (both with `warmup=1000`) would emerge as best stable 400-step config.
- Command:
  - continuous monitoring loop:
    - `uv run iris --config lib/iris/examples/marin.yaml job list --json`
    - `uv run python - <<'PY' ... wandb.Api().run(...).history(...) ... PY`
  - cleanup:
    - `uv run iris --config lib/iris/examples/marin.yaml job stop /dlwh/iris-run-launch-20260306-075556`
    - `uv run iris --config lib/iris/examples/marin.yaml job stop /dlwh/iris-run-launch-20260306-075548`
- Config:
  - `GRUG-NAN-014`: `lr=3e-3`, `warmup=1000`
  - `GRUG-NAN-015`: `lr=2e-3`, `warmup=1000`
  - common: `steps=400`, `GRUG_DISABLE_EVAL=1`, `GRUG_PREFETCH_SIZE=4`, `GRUG_MAX_BUFFERED_BATCHES=16`
- Result:
  - `014` (W&B): https://wandb.ai/marin-community/marin/runs/grug-parallel-v6e4-GRUG-NAN-014-20260305-235544
    - finished at `step=399` (non-NaN)
    - `last_loss=4.8022`
    - `min_loss=4.7555` (best observed)
  - `015` (W&B): https://wandb.ai/marin-community/marin/runs/grug-parallel-v6e4-GRUG-NAN-015-20260305-235550
    - finished at `step=399` (non-NaN)
    - `last_loss=4.9243`
    - `min_loss=4.8561`
  - Iris quirk observed post-completion:
    - completed child could restart/resume and re-enter startup/load loop; manually stopped after metrics were sealed.
- Interpretation:
  - Best config from this sweep window: **`lr=3e-3`, `warmup=1000`** (`GRUG-NAN-014`), with lower final and minimum loss than `lr=2e-3`.
  - Both configs were non-NaN through 400 steps.
- Next action:
  - Treat `lr=3e-3,warmup=1000` as current winner on v6e-4.
  - Resume warmup-axis refinement once launch-path flake is mitigated.

### 2026-03-06 12:16 - v6e-32 confirmation run to 500 (`grug-parallel-v6e32-500-20260306-120353`)
- Hypothesis: Current parallel-attn-mlp implementation and launch wiring can run stably past the earlier quick-check horizon when pushed to ~500 steps on v6e-32.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --extra marin:tpu --reserve=v6e-32 --region europe-west4 -e GRUG_RUN_ID grug-parallel-v6e32-500-20260306-120353 -e GRUG_WANDB_GROUP grug-parallel-attn-v6e32-500 -e GRUG_TPU_VARIANT v6e-32 -e GRUG_STEPS 500 -e GRUG_DISABLE_EVAL 1 -e GRUG_PREFETCH_SIZE 4 -e GRUG_MAX_BUFFERED_BATCHES 16 -e GRUG_LR 0.002 -e GRUG_WARMUP 1000 -- uv run python experiments/grug/parallel_attn_mlp/launch.py`
- Config:
  - `steps=500`, `lr=2e-3`, `warmup=1000`, `disable_eval=1`, `prefetch_size=4`, `max_buffered_batches=16`
  - TPU: `v6e-32`, region `europe-west4`
- Result:
  - Parent job: `/dlwh/iris-run-launch-20260306-200425` -> `JOB_STATE_SUCCEEDED`
  - Child train job: `/dlwh/iris-run-launch-20260306-200425/grug-train-grug-parallel-v6e32-500-20260306-120353` -> `JOB_STATE_SUCCEEDED`
  - Training progressed through `500/500` (last logged train postfix `loss=4.87`).
  - Final checkpoint persisted at `gs://marin-eu-west4/grug/parallel-attn-mlp-trial-e9ee7a/checkpoints/step-500`.
  - Experiment metadata: `gs://marin-eu-west4/experiments/launch-ac3dbc.json`
- Interpretation:
  - The run is stable to 500 steps on v6e-32 with the current parallel block implementation and Fray-dispatched training path.
  - Earlier config-dump `TypeError: cannot create weak reference to 'str' object` warnings remained non-fatal and did not block completion.
- Next action:
  - Treat this as a successful extended confirmation for the variant.
  - Use this run as the reference baseline when following up on restart/fault-path NaN investigations.
