# Grug MoE v5p-64 Perf Issue 3096: Research Logbook

## Scope
- Goal: Bring up a short profiled performance run for a Qwen3-shaped Grug MoE on `v5p-64` and determine whether the configuration runs without HBM/OOM failure.
- Primary metric(s): successful training progress through the profile window, steady-state step time, and profiler trace dump.
- Constraints: preemptible TPU availability, enough HBM for FSDP+EP, and current Iris worker/runtime behavior.
- Experiment issue: https://github.com/marin-community/marin/issues/3357
- GitHub issue: https://github.com/marin-community/marin/issues/3096
- Related issues: https://github.com/marin-community/marin/issues/2704, https://github.com/marin-community/marin/issues/2710, https://github.com/marin-community/marin/issues/2851, https://github.com/marin-community/marin/issues/3341
- Branch: `codex/grug-moe-v5p64-perf`
- Experiment ID prefix: `GRUG-PERF`

## Baseline
- Date: 2026-03-06
- Code refs:
  - `experiments/grug/moe/launch_qwen3_32b_a4b_perf.py`
  - `experiments/grug/moe/train.py`
  - `lib/iris/src/iris/cluster/runtime/docker.py`
- Baseline numbers:
  - No prior successful run for this exact `32B-A4B` Grug MoE shape on `v5p-64` in this thread.
  - Capacity choice was `v5p-64` in `us-central1-a` based on current TPU observability signals and HBM requirements.

## Experiment Log
### 2026-03-06 14:xx PT - GRUG-PERF-001 initial v5p-64 launch blocked by Iris Docker API mismatch
- Hypothesis: the Qwen3-shaped Grug MoE config should at least launch on `v5p-64`, and any first failure is more likely to be infra or HBM than config syntax.
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-perf -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-perf -- python experiments/grug/moe/launch_qwen3_32b_a4b_perf.py
```
- Config:
  - Model shape: `~31B total / ~4B active`
  - `hidden_dim=2048`
  - `intermediate_dim=768`
  - `shared_expert_intermediate_dim=2048`
  - `num_experts=128`
  - `num_experts_per_token=8`
  - `num_layers=48`
  - `num_heads=32`
  - `num_kv_heads=4`
  - `head_dim=128`
  - `max_seq_len=4096`
  - `batch_size=32`
  - `steps=30`
  - Profiling window: `start_step=8`, `num_steps=15`, `perfetto_link=False`
  - Mesh: `data=-1, replica=1, model=1, expert=8`
- Result:
  - Launcher job `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-perf` failed before user code started.
  - Worker image pull failed with Docker API mismatch: `client version 1.53 is too new. Maximum supported API version is 1.43`.
  - Filed follow-up bug: https://github.com/marin-community/marin/issues/3341
- Interpretation:
  - This was an Iris worker/runtime failure, not an HBM or model-shape failure.
  - The perf experiment remained viable if the worker-side Docker env could be forced to `DOCKER_API_VERSION=1.43`.
- Next action:
  - Patch Iris locally and/or restart the remote `iris-worker` containers with `DOCKER_API_VERSION=1.43`, then relaunch the same run unchanged.

### 2026-03-06 15:23 PT - GRUG-PERF-002 successful relaunch, profiler dumped, no HBM failure through profile window
- Hypothesis: with the Iris Docker pull issue worked around, the same config should train and survive the profiling window on `v5p-64` without HBM/OOM failure.
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-perf -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-perf -- python experiments/grug/moe/launch_qwen3_32b_a4b_perf.py
```
- Config:
  - Same config as `GRUG-PERF-001`
  - Ready slice used: `v5p-64` in `us-central1-a`
  - Relaunch submitted at `2026-03-06 15:23:12 PT` (`2026-03-06 23:23:12 UTC`)
  - W&B run: https://wandb.ai/marin-community/marin/runs/grug-moe-qwen3-32b-a4b-v5p64-perf
  - Job IDs:
    - launcher: `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-perf`
    - child: `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-perf/grug-train-grug-moe-qwen3-32b-a4b-v5p64-perf`
  - Output root observed from logs: `gs://marin-us-central1/grug/moe-qwen3-32b-a4b-v5p64-perf-f1c3f2`
- Result:
  - All `8/8` tasks reached `JOB_STATE_RUNNING`.
  - First visible training progress:
    - `2026-03-06 23:34:14 UTC`: `1.00it/30.0it`, `413.7s/it`
    - `2026-03-06 23:40:36 UTC`: `2.00it/30.0it`, `395.3s/it`, `loss=19.3`
    - `2026-03-06 23:41:51 UTC`: `11.0it/30.0it`, `24.4s/it`, `loss=15.8`
  - Profiler lifecycle:
    - `2026-03-06 23:40:59 UTC`: all 8 ranks logged `Starting profiler until step 23.`
    - `2026-03-06 23:42:37 UTC`: all 8 ranks logged `Stopping profiler.`
  - Confirmed dumped trace on worker 0 task container:
    - `/app/logs/grug-moe-qwen3-32b-a4b-v5p64-perf/profiler/plugins/profile/2026_03_06_23_43_29/t1v-n-5ac65cca-w-0.xplane.pb`
    - size: `1,152,221,848` bytes
  - No HBM/OOM-class failure observed before or through the profiler dump.
  - Nonfatal logging issue persisted:
    - `TypeError: cannot create weak reference to 'str' object`
    - source path in traceback: `levanter.tracker.tracker_fns` via `draccus.dump(...)`
- Interpretation:
  - The selected `v5p-64` slice is sufficient for this short `32B-A4B` bring-up and profile capture.
  - Compile/startup cost dominates the first two visible progress reports; steady-state inside the profile window was materially faster (`24.4s/it` by step 11).
  - The main blocker encountered was infrastructure-related Iris Docker behavior, not model HBM pressure.
- Next action:
  - Pull the profile artifacts and inspect the hottest kernels/collectives.
  - Decide whether the next perf iteration should stay on `v5p-64` or move to a larger slice for more representative scaling behavior.
  - File or fix the nonfatal `draccus.dump(...)` config-artifact logging bug separately if it starts obscuring run metadata.

### 2026-03-06 16:10 PT - GRUG-PERF-003 submitted `batch_size=512` no-profile rerun to test MFU hypothesis
- Hypothesis: the previous MFU was depressed primarily by an undersized global batch, so increasing from `32` to `512` should improve steady-state utilization if HBM remains sufficient.
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs512 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs512 -- python experiments/grug/moe/launch_qwen3_32b_a4b_perf.py
```
- Config:
  - Same model shape and mesh as `GRUG-PERF-002`
  - `batch_size=512`
  - Profiler disabled for this iteration
  - Default run id: `grug-moe-qwen3-32b-a4b-v5p64-bs512`
  - Launcher job id: `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs512`
- Result:
  - Submitted successfully at `2026-03-06 16:10:08 PT`.
  - As of `2026-03-06 16:12 PT`, the job is still pending at reservation time:
    - `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs512/:reservation:`
    - pending reason: `Waiting for workers in scale group 'tpu_v5p_64-us-central1-a' to become ready`
  - Scale-group snapshot at the same time:
    - `tpu_v5p_64-us-central1-a`: `0/1 ready`, `Demand: 1`
- Interpretation:
  - The next experimental axis is queued correctly, but there is not yet a model-side result because the TPU slice has not become ready.
  - The Docker API workaround is still included in the submission so the expected next blocker, if any, should be model/HBM rather than the known worker pull bug.
- Next action:
  - Wait for the `v5p-64` reservation to become ready.
  - Once the child starts, check immediately for either HBM/OOM failure or improved first-step/steady-state throughput relative to `GRUG-PERF-002`.

### 2026-03-06 16:59 PT - GRUG-PERF-004 submitted parallel `v6e-64` hedge at `batch_size=512`
- Hypothesis: if `v5p-64` remains trapped in preemptible churn, a `v6e-64` run can provide a second hardware datapoint with enough aggregate HBM to keep the same `32B-A4B` shape and batch size.
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v6e-64 --job-name grug-moe-qwen3-32b-a4b-v6e64-bs512 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v6e64-bs512 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v6e64_bs512.py
```
- Config:
  - Same model shape, step count, and `batch_size=512` target as `GRUG-PERF-003`
  - Accelerator target: `v6e-64`
  - Default run id: `grug-moe-qwen3-32b-a4b-v6e64-bs512`
  - Launcher job id: `/dlwh/grug-moe-qwen3-32b-a4b-v6e64-bs512`
- Result:
  - Submitted successfully.
  - Current state remains pending:
    - launcher: `JOB_STATE_PENDING`
    - reservation: `JOB_STATE_PENDING`
  - Current cluster snapshot shows no ready `v6e-64` slices in configured zones.
- Interpretation:
  - This hedge is correctly queued but has not yet converted into actual reservation demand or worker startup.
  - Until a `v6e-64` slice is assigned, there is no model-side signal on fit or throughput.
- Next action:
  - Keep polling for a reservation transition.
  - If a fresh `v6e-64` slice comes up, be ready to reapply the `iris-worker` Docker API workaround before task image pull.

- Terminal update:
  - The run later reached live `v6e-64` reservation and started the training child plus Zephyr cache-build subjobs.
  - Cache build eventually made partial progress (`35/2755` completed at best in observed logs) but never reached training steps.
  - Launcher terminated at `2026-03-06 17:39:50 PT` with:
    - `RuntimeError: Step failed: grug/moe-qwen3-32b-a4b-v6e64-bs512_541cde78`
    - `RuntimeError: 1 step(s) failed`
  - Final job states:
    - launcher `JOB_STATE_FAILED`
    - train child `JOB_STATE_KILLED`
    - reservation `JOB_STATE_KILLED`
  - No HBM/OOM signature was observed before termination.

### 2026-03-06 17:10 PT - GRUG-PERF-005 submitted parallel `v6e-32` hedge at `batch_size=512`
- Hypothesis: a `v6e-32` request may clear capacity faster than `v6e-64`, and even a failure would be informative about the minimum viable HBM envelope for this config.
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v6e-32 --job-name grug-moe-qwen3-32b-a4b-v6e32-bs512 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v6e32-bs512 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v6e32_bs512.py
```
- Config:
  - Same model shape, step count, and `batch_size=512` target as `GRUG-PERF-003`
  - Accelerator target: `v6e-32`
  - Default run id: `grug-moe-qwen3-32b-a4b-v6e32-bs512`
  - Launcher job id: `/dlwh/grug-moe-qwen3-32b-a4b-v6e32-bs512`
  - Code ref: `experiments/grug/moe/launch_qwen3_32b_a4b_v6e32_bs512.py`
- Result:
  - Launch file validated with `py_compile`.
  - Submitted successfully.
  - Current state remains pending:
    - launcher: `JOB_STATE_PENDING`
    - reservation: `JOB_STATE_PENDING`
  - Current cluster snapshot shows no ready `v6e-32` slices in configured zones.
- Interpretation:
  - This is the fastest additional hedge against `v6e-64` capacity starvation.
  - If it starts, it will answer both capacity and likely HBM headroom much sooner than waiting on `v6e-64` alone.
- Next action:
  - Monitor `v6e-32`, `v6e-64`, and `v5p-64` in parallel.
  - Promote whichever non-`v5p` run gets a reservation first into the main monitoring loop.

- Terminal update:
  - The run later reached live `v6e-32` reservation and started the training child plus Zephyr cache-build subjobs.
  - Cache build never made actual shard progress in the observed window; coordinator remained pinned around:
    - `0/2755 complete`
    - `59/128 workers alive`
    - `69 dead`
  - Launcher terminated at `2026-03-06 17:39:28 PT` with:
    - `RuntimeError: Step failed: grug/moe-qwen3-32b-a4b-v6e32-bs512_ed6c5397`
    - `RuntimeError: 1 step(s) failed`
  - Final job states:
    - launcher `JOB_STATE_FAILED`
    - train child `JOB_STATE_KILLED`
    - reservation `JOB_STATE_KILLED`
  - No HBM/OOM signature was observed before termination.

### 2026-03-06 17:39 PT - GRUG-PERF-006 terminal outcome for `v5p-64 bs512`
- Observed behavior:
  - `v5p-64` never reached user-code startup for the `batch_size=512` rerun.
  - It remained trapped in reservation/recovery churn with child tasks stuck pending and reservation `preemption_count=101`.
- Terminal result:
  - Launcher terminated at `2026-03-06 17:39:47 PT` with:
    - `RuntimeError: Step failed: grug/moe-qwen3-32b-a4b-v5p64-bs512_b39566cf`
    - `RuntimeError: 1 step(s) failed`
  - Final job states:
    - launcher `JOB_STATE_FAILED`
    - train child `JOB_STATE_KILLED`
    - reservation `JOB_STATE_KILLED`
- Interpretation:
  - This run did not produce a model-fit or MFU signal.
  - The failure mode remained infrastructure/scheduling instability rather than HBM pressure.

### 2026-03-06 17:42 PT - GRUG-PERF-007 profile-driven readout from successful `v5p-64` trace
- Inputs:
  - W&B profiler artifact: `marin-community/marin/run-grug-moe-qwen3-32b-a4b-v5p64-perf-profiler:v0`
  - Local summary: `scratch/profile_summary_grug_moe_v5p64_perf.json`
  - Local report: `scratch/profile_report_grug_moe_v5p64_perf.md`
- Summary:
  - Profile quality warning: exported trace appears truncated at exactly `1,000,000` complete events, so hotspot attribution is directionally useful but not perfect.
  - Steady-state summary from the ingested trace:
    - median step: `148.867`
    - p90 step: `969.623`
    - p90 / median: `6.51`
  - Global share:
    - compute: `84.99%`
    - communication: `12.16%`
    - stall: `2.85%`
  - Communication is not the dominant bucket; the run is primarily compute-bound in the captured window.
- Main hotspots:
  - Large MoE gather region:
    - `train_step=>Transformer=>Block=>MoEMLP=>moe_mlp=>gather=>_take=>gather`
    - `26.83%` inclusive / exclusive share of total traced time
  - Fused cross-entropy backward matmul/fusion family also appears repeatedly near the top.
  - Largest single collective by exclusive op time:
    - `psum.802`
    - `6.8 ms` average
    - source: `levanter/grug/loss.py:131`
- Main bottlenecks from the structured report:

### 2026-03-08 20:08 PT - GRUG-PERF-XXX `bs384` ceiling result on Ray with `offload_mlp`
- Hypothesis:
  - The newer `EP=4 / topk=4 / cf=1.25 / matched-active / synthetic / offload_mlp` path might buy enough headroom to push beyond the previously best stable `bs320`.
- Config:
  - Accelerator: `v5p-64` on Ray (`us-east5-a`)
  - `batch_size=384`
  - `expert_axis_size=4`
  - `num_experts_per_token=4`
  - `capacity_factor=1.25`
  - `match_activated_params=True`
  - `block_remat=offload_mlp`
  - synthetic input enabled
  - host-offload TPU flags enabled via `LIBTPU_INIT_ARGS`
- Result:
  - The retried Ray run reached a terminal model-side failure rather than infra churn.
  - Terminal error:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED`
    - `XLA:TPU compile permanent error. Ran out of memory in memory space host.`
    - `Used 154.41G of 128.00G host. Exceeded host capacity by 26.41G.`
  - Step id:
    - `grug/moe-q32a4b-v5p64-bs384-ep4-cf1p25-k4-ma-rom-pf32-buf64-syn-prof_72e03e8f`
- Interpretation:
  - `bs384` is beyond the host-memory compile limit for the current best-known path.
  - `bs320` remains the highest confirmed working batch on this path.
  - `bs352` remains the only plausible probe left if tighter ceiling estimation is needed, but expected perf upside is likely modest.
  - Large pre-op idle gaps before `copy.1`:
    - total gap: `1,094,689.423`
    - max single gap: `136,841.520`
    - occurrences: `8`
  - High step jitter:
    - steady-state p90 / median `6.51`
- Interpretation:
  - The low MFU from the profiled `batch_size=32` run is not explained by communication alone.
  - Two stronger suspects are:
    - underfilled compute around MoE gather / dispatch-related work,
    - intermittent long waits before large `copy.*` regions, consistent with synchronization / input / dispatch stalls.
  - This supports trying a larger effective batch, but only after eliminating infrastructure-side noise from cache-building and preemptible churn.
- Next action:
  - Re-run the larger-batch experiment against prebuilt token caches so the job measures training rather than Zephyr cache construction.
  - If that succeeds, capture a second profile and compare `before/after` using `profile_summary.py compare`.

### 2026-03-06 17:45 PT - GRUG-PERF-008 relaunch `v6e-{32,64}` with tokenized data pinned to `marin-us-central1`
- Hypothesis:
  - The failed `v6e` bring-up was dominated by missing regional token caches, not model fit.
  - Reusing the known-good `gs://marin-us-central1/tokenized/...` caches should bypass Zephyr cache build and let us test actual training startup at `batch_size=512`.
- Code change:
  - Added `experiments/grug/moe/data.py` with `qwen3_moe_perf_mix_us_central1()`
  - Remapped Nemotron, StarCoder, and ProofPile token caches to the existing `marin-us-central1` paths
  - Updated:
    - `experiments/grug/moe/launch_qwen3_32b_a4b_v6e32_bs512.py`
    - `experiments/grug/moe/launch_qwen3_32b_a4b_v6e64_bs512.py`
- Validation:
  - Confirmed central cache ledgers exist:
    - `gs://marin-us-central1/tokenized/nemotron_cc/hq_actual-5af4cc/train/shard_ledger.json`
    - `gs://marin-us-central1/tokenized/starcoderdata-12f018/train/shard_ledger.json`
    - `gs://marin-us-central1/tokenized/proofpile_2-4a35c7/train/shard_ledger.json`
  - Confirmed east1 Nemotron ledger does not exist:
    - `gs://marin-us-east1/tokenized/nemotron_cc/hq_actual-5af4cc/train/shard_ledger.json`
- Result:
  - Relaunched both `v6e-32` and `v6e-64` with the same `bs512` target and cache-pin change.
  - Current state at time of writing:
    - both jobs are alive and pending
    - autoscaler demand continues to shift across `us-east1-d`, `us-east5-b`, and `europe-west4-a`
    - no new training or failure signal yet after the cache-pin relaunch

### 2026-03-06 18:05 PT - GRUG-PERF-009 launch `v5p-64 bs256` as profile-driven hedge
- Hypothesis:
  - The successful `bs32` profile points to underfilled compute and MoE gather overhead rather than a primarily communication-bound regime.
  - `batch_size=512` is still worth chasing, but current `v6e` attempts are capacity-blocked and the prior `v5p-64 bs512` run never reached training due to preemptible churn.
  - A `v5p-64` rerun at `batch_size=256` is the fastest hedge that can produce a real training datapoint on a known-good topology while the `v6e` reservations wait.
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs256.py`
  - Uses the same Qwen3-shaped Grug MoE config and `30` steps
  - Disables profiling for throughput focus
  - Pins data to `qwen3_moe_perf_mix_us_central1()` to reuse prebuilt token caches
- Command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs256 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs256 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs256.py
```
- Validation:
  - Launch file passed `uv run python -m py_compile`.
- Result:
  - Submitted successfully as `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs256`.
  - Initial reservation state:
    - launcher: `JOB_STATE_PENDING`
    - reservation: `JOB_STATE_PENDING`
    - pending reason: `Waiting for worker scale-up in scale group 'tpu_v5p_64-us-central1-a' (1 slice(s) requested)`
- Next action:
  - Monitor `v5p-64 bs256` alongside the pending `v6e-32 bs512` and `v6e-64 bs512` relaunches.
  - If `v5p-64 bs256` reaches training first, use it as the next real post-profile throughput datapoint and only fall further on batch size if it trips HBM/XLA allocation errors.

### 2026-03-06 18:11 PT - GRUG-PERF-010 `v6e-32 bs512` fails on compile HBM; fallback to `bs256`
- Observed failure:
  - The cache-pinned `v6e-32 bs512` run reached train child startup (`8/8 running`) and then failed during JAX compile with:
    - `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error`
    - `Ran out of memory in memory space hbm. Used 45.02G of 31.25G hbm. Exceeded hbm capacity by 13.78G.`
    - `Program hbm requirement 45.02G`
- Immediate action:
  - Stopped the `bs512` job and launched `v6e-32 bs256` with the same code path and central-cache pin.
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v6e32_bs256.py`.
- Result:
  - `bs256` reproduced the exact same HBM signature (`45.02G` requirement on `31.25G` HBM), indicating the limiting footprint is not meaningfully reduced by this batch cut.
  - Stopped `v6e-32 bs256` after confirming the repeat failure.
- Interpretation:
  - Under the current Grug MoE sharding/state layout, `v6e-32` is not a viable bring-up target for this `32B-A4B` config.
  - The failure appears structural (model/init-state footprint) rather than batch-size-limited activation pressure.

### 2026-03-06 18:13 PT - GRUG-PERF-011 `v6e-64` also fails on `jit__init_state`; batch cuts do not help
- Observed failure on `v6e-64 bs512`:
  - Train child reached `16/16 running` and then failed loading `jit__init_state` with:
    - `RESOURCE_EXHAUSTED: Error loading program 'jit__init_state'`
    - `Attempting to reserve 27.02G at the bottom of memory`
    - `There are 12.22G free, 0B reserved, and 12.22G reservable`
- Immediate action:
  - Stopped the `bs512` run and launched `v6e-64 bs256`.
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v6e64_bs256.py`.
- Result:
  - `v6e-64 bs256` reproduced the same `jit__init_state` reservation failure on multiple ranks.
  - Stopped the run after confirming the repeat failure.
- Interpretation:
  - `v6e-64` is also not viable for this config under the current sharding/state placement, and batch reduction to `256` does not address the limiting memory requirement.
  - Current `v6e` outcome is therefore a negative result for this exact setup, not a capacity issue.

### 2026-03-06 18:40 PT - GRUG-PERF-012 successful `v5p-64 bs256` throughput run
- Observed behavior:
  - `v5p-64 bs256` avoided the earlier `bs512` preemption churn and did not hit HBM or TPU runtime failures.
  - First train progress arrived at `2026-03-07 02:20:45 UTC`:
    - `1/30`
    - compile-dominated `431.4s/it`
  - The run then converged steadily:
    - `5/30` at `2026-03-07 02:28:38 UTC`, `106.5s/it`, `loss=19.3`
    - `8/30` at `2026-03-07 02:29:53 UTC`, `49.8s/it`, `loss=17.1`
    - `11/30` at `2026-03-07 02:31:54 UTC`, `47.0s/it`, `loss=15.5`
    - `14/30` at `2026-03-07 02:33:09 UTC`, `32.4s/it`, `loss=14.7`
    - `17/30` at `2026-03-07 02:34:26 UTC`, `27.8s/it`, `loss=14.1`
    - `20/30` at `2026-03-07 02:35:40 UTC`, `25.8s/it`, `loss=13.7`
    - `23/30` at `2026-03-07 02:36:56 UTC`, `25.3s/it`, `loss=13.4`
    - `26/30` at `2026-03-07 02:38:10 UTC`, `25.0s/it`, `loss=13.3`
    - `29/30` at `2026-03-07 02:39:24 UTC`, `24.8s/it`, `loss=13.1`
- Interpretation:
  - The steady-state throughput signal on `v5p-64` is about `25s/it` at `batch_size=256` for this Qwen3-shaped Grug MoE config.
  - This is the first clean non-profiled perf datapoint after moving off the obviously underfilled `bs32` run.
  - The `v6e` failures indicate that, for this config, `v5p` is the viable perf platform without changing sharding/model-state strategy.
- Next action:
  - If the goal is to hill-climb MFU on viable hardware, continue from this `v5p-64 bs256` baseline.
  - Candidate next moves are:
    - retry `v5p-64 bs512` when slice churn is lower,
    - or try an intermediate `v5p-64 bs384` if you want a narrower search step than jumping back to `512`.

### 2026-03-06 18:46 PT - GRUG-PERF-013 launch `v5p-64 bs384` from the successful `bs256` baseline
- Motivation:
  - `v5p-64 bs256` produced the first clean steady-state datapoint for this config, converging to about `24.8-25.0s/it` by `26-29/30`.
  - `v6e-32` and `v6e-64` are ruled out for the current config due to reproducible model/init-state memory failures that did not improve with smaller batch.
  - The next viable hill-climb move is therefore an intermediate `v5p-64 bs384` run rather than jumping directly back to the earlier infra-blocked `bs512`.
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs384.py`
  - Same model/data/resources path as `bs256`
  - `batch_size=384`
  - profiling disabled
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs384 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs384 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs384.py
```
- Operational note:
  - Stopped the nearly finished `bs256` run after it had already yielded the steady-state datapoint, to reuse the same `v5p-64` slice for `bs384`.
- Current state:
  - launcher `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs384`: `JOB_STATE_RUNNING`
  - reservation `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs384/:reservation:`: `JOB_STATE_RUNNING`
  - both are in the early `assigned` state and have not yet emitted user-code logs
- Next action:
  - Continue monitoring for train child startup, then either:
    - first-step progress and a new steady-state throughput datapoint, or
    - the next limiting failure signal (HBM, TPU init, or scheduler churn).

### 2026-03-06 19:00 PT - GRUG-PERF-014 `v5p-64 bs384` stalls before step 1; batch midpoint fallback
- Observed behavior:
  - `v5p-64 bs384` did reach live train startup:
    - child `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs384/grug-train-grug-moe-qwen3-32b-a4b-v5p64-bs384`
    - `8/8` tasks running
  - It cleared the usual early startup sequence:
    - JAX distributed init
    - W&B init
    - central-cache loading
    - fused cross-entropy kernel selection
  - It also emitted the first train progress marker:
    - `2026-03-07 02:55:48 UTC`: `Progress on:train -/30`
  - But it never completed step 1 and stopped producing new child logs after:
    - `2026-03-07 02:56:08 UTC`: `Data loading is taking a long time: 10.0 seconds. Waiting for 1536 items.`
  - The job remained nominally `running` for several more minutes with no additional log movement or failure transition.
- Interpretation:
  - This did not look like an HBM or `jit__init_state` failure; instead it looked like a nonproductive stall around first-step compile and/or input pipeline pressure at this larger batch.
  - The strongest live signal was data starvation (`waiting for 1536 items`) rather than TPU memory exhaustion.
- Action taken:
  - Stopped the stalled `bs384` run to avoid burning more `v5p-64` time without a new datapoint.
  - Selected `batch_size=320` as the next midpoint between the working `256` and the stalled `384`.

### 2026-03-06 19:01 PT - GRUG-PERF-015 launch `v5p-64 bs320`
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320.py`
  - Same Qwen3-shaped model, same central-cache pin, same `30`-step short run
  - `batch_size=320`
  - profiling disabled
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs320 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs320 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320.py
```
- Rationale:
  - `bs256` is the current known-good point at about `25s/it`.
  - `bs384` did not yield a real throughput datapoint and instead stalled before step 1.
  - `bs320` is the next narrow hill-climb step that may retain most of the batch-size gain without reproducing the `bs384` startup/input stall.

### 2026-03-06 19:37 PT - GRUG-PERF-016 `v5p-64 bs320` trains cleanly but underperforms `bs256`
- Observed behavior:
  - `v5p-64 bs320` was a real train run, not a repeat of the `bs384` dead stall.
  - It showed the same early loader warnings seen on smaller batches:
    - `Prefetch wasn't fast enough`
    - `Data loading is taking a long time: 10.0 seconds. Waiting for 1280 items.`
  - It then advanced through training without HBM or TPU runtime errors:
    - `1/30` at `2026-03-07 03:13:21 UTC`, `458.9s/it`
    - `10/30` at `2026-03-07 03:24:33 UTC`, `43.7s/it`, `loss=15.9`
    - `13/30` at `2026-03-07 03:26:50 UTC`, `41.7s/it`, `loss=15.0`
    - `19/30` at `2026-03-07 03:29:56 UTC`, `32.0-32.1s/it`, `loss=13.8`
    - `21/30` at `2026-03-07 03:30:57 UTC`, `31.3s/it`, `loss=13.5`
    - `23/30` at `2026-03-07 03:31:58 UTC`, `31.0s/it`, `loss=13.4`
    - `25/30` at `2026-03-07 03:33:00 UTC`, `30.8s/it`, `loss=13.2`
    - `27/30` at `2026-03-07 03:34:01 UTC`, `30.7s/it`, `loss=13.1`
    - `29/30` at `2026-03-07 03:35:02 UTC`, `30.6s/it`, `loss=13.0`
- Interpretation:
  - `bs320` is viable on `v5p-64`, but the throughput curve is clearly worse than the `bs256` baseline once startup is amortized.
  - Relative to `bs256`, the run handled more samples per optimizer step but not enough more to overcome the slower step time; this makes `320` a negative result for throughput hill-climbing.
  - There was still no sign that HBM was the limiting factor on `v5p`; the practical issue here is reduced throughput, likely with input pressure contributing.
- Decision:
  - Do not continue upward from `320`.
  - Probe a smaller upward step at `bs288` to see whether the optimum lies between `256` and `320`.

### 2026-03-06 19:36 PT - GRUG-PERF-017 launch `v5p-64 bs288`
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs288.py`
  - Same model/data/resources path as the successful `bs256` and viable-but-slower `bs320` runs
  - `batch_size=288`
  - profiling disabled
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs288.py`
  - `./infra/pre-commit.py --all-files`
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs288 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs288 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs288.py
```
- Immediate result:
  - Submitted successfully as `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs288`.
  - Initial state at submission:
    - launcher: `running`
    - reservation: `pending`
    - message: `Waiting for workers in scale group 'tpu_v5p_64-us-central1-a'`
- Rationale:
  - `bs256` is still the best measured point.
  - `bs320` moved far enough upward to rule out that part of the sweep.
  - `bs288` is the next bounded test before concluding the optimum is effectively at `256` under the current input/runtime setup.

### 2026-03-06 20:12 PT - GRUG-PERF-018 switch sweep metric to average `throughput/tokens_per_second`
- Correction:
  - Earlier `bs320` interpretation was based on `sec/it`, which is not the right hill-climb metric for this sweep because batch size changes across runs.
  - The requested optimization target is average throughput, so comparisons should use W&B `throughput/tokens_per_second` (equivalently `batch_size * 4096 / sec_per_it` when computed from logs).
- Current ranking on that metric:
  - `v5p-64 bs256`
    - W&B state: `crashed`
    - `_step=28`
    - `throughput/tokens_per_second = 42661.58093787186`
    - `throughput/examples_per_second = 10.415425033660124`
  - `v5p-64 bs320`
    - W&B state: `crashed`
    - `_step=28`
    - `throughput/tokens_per_second = 43210.65659968317`
    - `throughput/examples_per_second = 10.549476708907024`
  - `v5p-64 bs288`
    - W&B state at query time: `running`
    - `_step=13`
    - `throughput/tokens_per_second = 42484.16373605434`
    - `throughput/examples_per_second = 10.37211028712264`
- Interpretation:
  - On average token throughput, `bs320` is ahead of `bs256`, even though the larger batch has slower `sec/it`.
  - The sweep direction therefore remains upward from `320`, bounded by the earlier `bs384` startup stall.

### 2026-03-06 20:14 PT - GRUG-PERF-019 `v5p-64 bs288` terminates without beating `bs320`
- Observed result:
  - W&B later reported `v5p-64 bs288` as terminal (`state=crashed`) while retaining the same latest summary point:
    - `_step=13`
    - `throughput/tokens_per_second = 42484.16373605434`
    - `throughput/examples_per_second = 10.372110287122641`
    - `train/loss = 14.431426048278809`
- Interpretation:
  - `bs288` did not improve throughput over either `bs256` or `bs320`.
  - Because `320` remains the best measured point and `384` previously stalled, the next bounded search point is `bs352`.

### 2026-03-06 20:25 PT - GRUG-PERF-020 launch `v5p-64 bs352`
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs352.py`
  - Same model/data/resources path as the other `v5p-64` runs
  - `batch_size=352`
  - profiling disabled
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs352.py`
  - `./infra/pre-commit.py --all-files`
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs352 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs352 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs352.py
```
- Immediate result:
  - Submitted successfully as `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs352`.
- Rationale:
  - `bs320` is the current best measured throughput point on `v5p-64`.
  - `bs288` moved downward and lost throughput.
  - `bs352` is the next bounded probe between the current leader and the earlier `bs384` stall.

### 2026-03-06 21:36 PT - GRUG-PERF-021 `v5p-64 bs352` is infra-borked; stop the sweep at `bs320`
- Observed failure mode:
  - `bs352` never progressed past:
    - launcher `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs352`: `pending`
    - reservation `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs352/:reservation:`: `running`
  - No train child was ever created.
  - Launcher log fetches produced no user-code output.
- Infrastructure inspection:
  - Queried the live `v5p-64` slice in `us-central1-a`:
    - `ray-marin-us-central1-worker-b17974ff-tpu`
  - SSH to all 8 workers succeeded.
  - `docker ps` on every worker showed only system containers:
    - `monitoringagent`
    - `ray_docker`
    - `google-runtime-monitor`
    - `healthagent`
    - `google-collectd`
    - `vbarcontrolagent`
    - `tpu-runtime`
    - `instance_agent`
  - This container listing is from the Ray-managed TPU VMs and does not, by itself, prove an Iris worker bootstrap failure.
  - The supported conclusion from this run is narrower: the reservation stayed live while the Iris launcher never progressed to child creation, so the run yielded no usable performance datapoint.
- Action taken:
  - Stopped `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs352` and its reservation.
- Conclusion:
  - The batch sweep is no longer blocked by uncertainty about model fit or throughput trend; it is blocked by Iris/infra startup failures on fresh slices.
  - Best measured throughput result remains:
    - `v5p-64 bs320`
    - W&B `_step=28`
    - `throughput/tokens_per_second = 43210.65659968317`
  - `bs256` is the next-best clean point at `42661.58093787186 tok/s`.
  - `bs288` underperformed and terminated early.
  - `bs352` yielded no usable datapoint because the launcher never progressed into a train child.

### 2026-03-06 21:43 PT - GRUG-PERF-022 loader-tuned retry at the current winner (`v5p-64 bs320`)
- Profiling-guided hypothesis:
  - The existing profile is compute-dominated (~85%) rather than communication-dominated (~12%), so collective tuning is not first priority.
  - The strongest non-kernel signals are:
    - large idle gap before `copy.1`
    - very high step jitter (`p90/median ~= 6.5`)
    - repeated loader starvation warnings on the batch sweep runs
  - Average `throughput/loading_time` in W&B is tiny, so the likely issue is intermittent starvation/jitter rather than steady-state mean loader cost.
- Code change:
  - Added explicit loader knobs to `experiments/grug/moe/train.py`:
    - `GrugTrainerConfig.loader_max_buffered_batches`
    - `GrugTrainerConfig.loader_prefetch_size`
  - `build_train_loader(...)` now threads those values into the underlying `DataLoader`.
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_pf64_buf256.py`.
- Experimental design:
  - Keep the best current batch size (`320`) fixed.
  - Change only loader buffering:
    - `prefetch_size=64` (from 32)
    - `max_buffered_batches=256` (from 64)
  - This is a one-axis retry targeted at the observed jitter/starvation behavior rather than another batch-size sweep.
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_pf64_buf256.py`
  - `./infra/pre-commit.py --all-files`
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs320-pf64-buf256 -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs320-pf64-buf256 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_pf64_buf256.py
```
- Operational note:
  - The first submit attempt was interrupted by an Iris controller restart (`iris-controller-marin` restarted at `2026-03-06T21:39:44-08:00`).
  - After the controller came back and active jobs were empty, the same job was resubmitted successfully as `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs320-pf64-buf256`.

### 2026-03-06 21:53 PT - GRUG-PERF-023 pivot from loader knobs to block shuffle
- New hypothesis:
  - The user suggested trying block shuffle rather than only prefetch/buffer tuning for the data-loader path.
  - The reference branch `codex/block-shuffle-defaults` uses:
    - `io_block_size=256`
    - `window_blocks=512`
    - `perm_type="feistel"`
- Code change:
  - Added a perf-mix variant in `experiments/grug/moe/data.py`:
    - `qwen3_moe_perf_mix_us_central1_block_shuffle()`
  - It preserves the same central-cache-pinned token data and changes only the mixture shuffle policy.
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_block_shuffle.py`.
- Experimental design:
  - Keep the current winning batch size fixed at `320`.
  - Change only the shuffle policy to the branch’s block-shuffle defaults.
  - This is a cleaner one-axis comparison than combining shuffle and loader-prefetch changes in the same run.
- Operational action:
  - Stopped trying to queue the still-pending loader-prefetch run (`bs320-pf64-buf256`); no running job matched at stop time.
  - Submitted the block-shuffle run as:
    - `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle`
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_block_shuffle.py
```

### 2026-03-06 22:03 PT - GRUG-PERF-024 relaunch block shuffle with profiling enabled
- Motivation:
  - The user requested an actual profile on the current tuned path.
  - The Grug MoE loop already has the profile callback wired; the missing piece was a launch with `ProfilerConfig(enabled=True, start_step=8, num_steps=15)`.
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_block_shuffle_profile.py`
  - Same shape, batch size, and block-shuffle data config as `GRUG-PERF-023`
  - profiling enabled for the middle 15 steps (`8-22` inclusive stop-on-step semantics)
- Operational action:
  - Stopped the unprofiled `bs320-block-shuffle` launcher before it accumulated more unprofiled runtime.
  - Submitted the profiled replacement as:
    - `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle-profile`
- Launch command:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 4 --memory 16GB --extra marin:tpu --reserve=v5p-64 --job-name grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle-profile -e DOCKER_API_VERSION 1.43 -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -e GRUG_RUN_ID grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle-profile -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_bs320_block_shuffle_profile.py
```

### 2026-03-06 22:10 PT - GRUG-PERF-025 profiled block-shuffle rerun is blocked by broken east5 slices
- Observed failure:
  - The profiled block-shuffle run reached reservation and child creation, but the child failed before user code:
    - `RuntimeError: Failed to pull image ... iris-task:2de7ce035`
    - `client version 1.53 is too new. Maximum supported API version is 1.43`
  - So no new `jax_profile` artifact was produced from this rerun.
- Infrastructure inspection:
  - The reservation path was landing in `us-east5-a`.
  - Inspected the newest likely `v5p-64` slice:
    - `ray-marin-us-east5-a-worker-d37fa85c-tpu`
  - Inspected an older ready `v5p-64` slice as a sanity check:
    - `ray-marin-us-east5-a-worker-e7b7696d-tpu`
  - On both slices, all 8 TPU VMs were reachable over SSH, but `docker ps` showed only system containers:
    - `ray_docker`
    - `google-runtime-monitor`
    - `monitoringagent`
    - `healthagent`
    - `google-collectd`
    - `vbarcontrolagent`
    - `tpu-runtime`
    - `instance_agent`
  - As above, this container listing is from Ray-managed TPU VMs and is not sufficient evidence, by itself, to conclude that Iris worker bootstrap is absent on those slices.
- Conclusion:
  - The proven blocker for fresh profile capture on the block-shuffle rerun is the Iris task runtime's Docker API mismatch during worker-side image pull.
  - We still have the earlier valid profile artifact from the successful `v5p-64` run, but we cannot currently produce a fresh profile on the block-shuffle path until that Docker runtime issue is fixed or worked around on the new slice.

### 2026-03-06 22:27 PT - GRUG-PERF-026 pivot to Ray EP sweep on east5-a
- Motivation:
  - The user asked to stop leaning on Iris for this thread and instead use the available Ray `v5p-64` capacity in `us-east5-a`.
  - New focus: one-axis sweep of expert parallelism (`EP=1,2,4,8`) with short profiles for each run.
- Experimental design:
  - Hold workload constant at the best measured throughput point so far:
    - shape: Qwen3-inspired `32B-A4B`
    - hardware: `v5p-64`
    - batch size: `320`
    - data: `qwen3_moe_perf_mix_us_central1()` (same central-cache-pinned mix as the successful throughput runs)
  - Change only the expert mesh axis:
    - `expert=1`
    - `expert=2`
    - `expert=4`
    - `expert=8`
  - Keep profile duration short:
    - total steps: `18`
    - profile window: steps `8-12` (`num_steps=5`)
- Code change:
  - Added `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
  - The entrypoint is parameterized by:
    - `--expert-axis-size`
    - `--batch-size`
    - `--steps`
    - `--profiler-start-step`
    - `--profiler-num-steps`
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
- Next action:
  - Submit four Ray jobs on `us-east5-a`, one per EP value, then monitor until each profile artifact is produced or a terminal blocker appears.

### 2026-03-06 22:31 PT - GRUG-PERF-027 Ray EP sweep first submit failed on launcher arg handling
- Observed failure:
  - Initial Ray submissions for `EP=1,2,4,8` all failed before user code.
  - The outer Ray job logs showed:
    - `launch_qwen3_32b_a4b_v5p64_ep_profile.py: error: unrecognized arguments: --expert-axis-size ...`
- Root cause:
  - The new parameterized launcher parsed its own CLI flags, but then `executor_main(...)` re-read `sys.argv` and rejected those same flags.
  - This was a local launcher bug, not an HBM/compiler/runtime failure in the actual training run.
- Code fix:
  - Updated `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py` so `parse_args()` uses `parse_known_args()` and rewrites `sys.argv` to leave only executor-owned arguments.
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
- Relaunch:
  - Submitted fresh Ray retry jobs:
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r1-20260306`
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep2-profile-r1-20260306`
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep4-profile-r1-20260306`
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-profile-r1-20260306`
- Current status:
  - All four retry jobs are `RUNNING` at the outer Ray level.
  - Per-step output paths exist and `.executor_status` is `RUNNING` for at least `EP=1` and `EP=8`.
  - No W&B runs or trainer-side files have appeared yet; the visible activity is repeated TPU autoscaler demand in the outer logs, so the active blocker is still startup/capacity rather than model fit.

### 2026-03-06 22:56 PT - GRUG-PERF-028 collapse Ray EP sweep to a single EP=1 run
- Observed degradation:
  - With all four `EP` values active in parallel, outer Ray jobs stayed `RUNNING` for ~20 minutes without creating any W&B runs or trainer-side files.
  - Outer logs for `EP=1` and `EP=8` repeatedly showed:
    - `Adding 4 node(s) of type tpu_slice_v5p_64`
    - no trainer progress, profiler, or loss lines
- Intervention:
  - Stopped the parallel jobs:
    - `EP=2`: `STOPPED`
    - `EP=4`: `STOPPED`
    - `EP=8`: `STOPPED`
  - Stopped the stale `EP=1` run as well and retried it cleanly.
  - First clean retry (`r2`) failed before job creation because the Ray dashboard package upload path returned `Connection refused` on `localhost:8265`.
  - Second clean retry succeeded:
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r3-20260306`
- New signal:
  - On `r3`, the autoscaler demand changed materially:
    - previous parallel state: repeated `Adding 4 node(s) of type tpu_slice_v5p_64`
    - current single-run state: `Adding 1 node(s) of type tpu_slice_v5p_64`
  - That suggests the parallel sweep itself was inflating or fragmenting demand, and the single-run relaunch is a cleaner shot at getting the first EP profile through.
- Current status:
  - `EP=1 r3` is `RUNNING`
  - still no W&B run yet
  - still no trainer-side files beyond `.executor_info` / `.executor_status`
  - but the scheduler signal is healthier than the earlier parallel attempt

### 2026-03-06 23:47 PT - GRUG-PERF-029 resubmit EP=1 on the correct east5-a Ray cluster
- What changed:
  - The previous monitoring checks mixed up two distinct Ray clusters:
    - `infra/marin-us-east5.yaml`
    - `infra/marin-us-east5-a.yaml`
  - The live Grug EP run is on the zonal `us-east5-a` cluster, not the broader `us-east5` cluster.
- Relaunch:
  - Submitted a fresh EP=1 profile run against `us-east5-a`:
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r4-20260306`
  - Command:
    - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster us-east5-a --submission-id ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r4-20260306 --env_vars LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py --expert-axis-size 1 --batch-size 320 --steps 18 --profiler-start-step 8 --profiler-num-steps 5`
- Monitoring correction:
  - `us-east5-a` dashboard requires Ray bearer-token auth.
  - After querying the correct dashboard with the local auth token, the job is visible and healthy:
    - submission id: `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r4-20260306`
    - status: `RUNNING`
    - message: `Job is currently running.`
- Earliest launcher-side logs:
  - `Runtime env is setting up.`
  - `Running entrypoint for job ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r4-20260306`
  - `### Inspecting the 1 provided steps ###`
- Important startup signal:
  - The launcher has entered user code and is resolving the central-cache-pinned dataset mix.
  - Current warnings are cache-path override notices (east5 defaults vs pinned central1 overrides), not HBM/OOM or TPU-compile failures.
- Next action:
  - Keep monitoring `r4` until it either:
    - spawns the train child and reaches trainer logs/profiler start, or
    - fails with a concrete startup/runtime error worth fixing before trying `EP=2`.

### 2026-03-06 23:52 PT - GRUG-PERF-030 EP=1 completed cleanly; EP=2 relaunched
- `EP=1` result:
  - Submission: `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-r4-20260306`
  - Final outer Ray status: `SUCCEEDED`
  - Executor log tail confirms the Grug step was not just queued:
    - `Status grug/moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile_9148edb5: RUNNING`
    - `Step grug/moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile_9148edb5 has already succeeded.`
    - `Executor run took 194.78s`
  - Output path:
    - `gs://marin-us-east5/grug/moe-qwen3-32b-a4b-v5p64-bs320-ep1-profile-ce7217`
  - Observed artifact:
    - checkpoint written at `checkpoints/step-18`
  - Open follow-up:
    - the profiler dump is not copied into the step root automatically, so the next metadata lookup still needs to identify the worker-side or W&B artifact location for the trace.
- `EP=2` state:
  - Old `EP=2 r1` turned out to be a stale stopped run from the earlier collapsed sweep:
    - status: `STOPPED`
    - message: `Job was intentionally stopped.`
  - Fresh relaunch submitted successfully:
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep2-profile-r2-20260306`
- Next action:
  - Monitor `EP=2 r2` until it either reproduces the same clean `EP=1` path or hits a distinct startup/runtime failure.

### 2026-03-07 00:18 PT - GRUG-PERF-031 apply ring-EP fix and recover EP=8 on Ray
- Code change:
  - Cherry-picked the production ring-EP optimization commit from `codex/grug-moe-ring-ep-opt`:
    - `5ad7652c9a98852856e3898c31cb02e9356cbe07 Optimize Grug MoE ring EP compaction`
  - Validation on this branch:
    - `./infra/pre-commit.py --all-files`
    - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`
- First optimized Ray attempt:
  - Submission: `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs512-ep8-profile-ring-r1-20260306`
  - It reached outer Fray dispatch but stalled on an unsatisfiable slice-local resource:
    - `No available node types can fulfill resource requests {'ray-worker-manual-zhacenfn': 1.0}*1`
  - Recovery action:
    - stopped `r1`
    - deleted the poisoned manual TPU slice in `us-east5-a`:
      - `ray-worker-manual-zhacenfn`
- Second optimized Ray attempt:
  - Submission: `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs512-ep8-profile-ring-r2-20260307`
  - This recovered into user code and created the W&B run:
    - `grug-moe-qwen3-32b-a4b-v5p64-bs512-ep8-profile`
  - Nonfatal repeated traceback remained the known config artifact bug:
    - `TypeError: cannot create weak reference to 'str' object`
  - Useful startup signals:
    - central-cache loading succeeded
    - checkpoint check fell through to scratch start
    - trainer entered `Progress on:train -/18`
  - Blocking signal:
    - repeated loader starvation before first step:
      - `Prefetch wasn't fast enough`
      - `Data loading is taking a long time: 10.0 seconds. Waiting for 2048 items.`

### 2026-03-07 00:18 PT - GRUG-PERF-032 degrade bs512, relaunch EP=8 at bs320 with unique run suffix
- Why `bs512` was abandoned:
  - `EP=8 bs512 r2` never logged a train step or throughput row to W&B.
  - After multiple monitor intervals, Ray started retrying `run_on_pod_ray` on dead nodes:
    - `Task run_on_pod_ray failed. There are infinite retries remaining...`
    - repeated node-death messages across the east5-a cluster
  - Conclusion:
    - `bs512` is not a stable profiling point for the optimized EP sweep under current east5-a conditions.
- Small launcher fix:
  - Added `--run-suffix` support to `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
  - Purpose:
    - avoid W&B/run-id collisions with the earlier pre-fix EP profile runs when relaunching optimized variants at the same `bs`/`EP`
  - Validation:
    - `python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
    - `./infra/pre-commit.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
- Current active retry:
  - Stopped `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs512-ep8-profile-ring-r2-20260307`
  - Submitted:
    - `ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-profile-ringopt-r1-20260307`
  - Command:
    - `uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster us-east5-a --submission-id ray-run-grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-profile-ringopt-r1-20260307 --env_vars LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py --expert-axis-size 8 --batch-size 320 --steps 18 --profiler-start-step 8 --profiler-num-steps 5 --run-suffix ringopt-r1`
- Next action:
  - Monitor `EP=8 bs320 ringopt-r1` to first step or profile artifact.
  - If that stabilizes, carry the same `bs320` setting through optimized `EP=4` and `EP=2` for comparable short profiles.

### 2026-03-07 00:52 PT - GRUG-PERF-033 merge origin/main and retry Iris on current code
- Branch maintenance:
  - fetched and merged `origin/main` into `codex/grug-moe-v5p64-perf`
  - the merge pulled in a substantial Iris update across controller, scheduler, config, and worker code
- New Iris surface:
  - `iris job run` now supports native TPU requests via `--tpu`
  - current `marin.yaml` still advertises `tpu_v5p_64-us-east5-a`
- First retry on merged main:
  - job: `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-iris-r1`
  - command used:
    - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --tpu v5p-64 --zone us-east5-a --cpu 4 --memory 16GB --extra marin:tpu --job-name grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-iris-r1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -e LIBTPU_INIT_ARGS "--xla_tpu_scoped_vmem_limit_kib=50000" -- python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py --expert-axis-size 8 --batch-size 320 --steps 18 --profiler-start-step 8 --profiler-num-steps 5 --run-suffix iris-r1`
  - result:
    - submission was clean
    - but the job stayed pending with:
      - `coschedule_mismatch: job needs 8 tasks coscheduled but matching groups have num_vms=[16]`
  - interpretation:
    - the merged CLI auto-detected `replicas=8` for `v5p-64`, but the current Iris scale group in `marin.yaml` expects `16` workers for `tpu_v5p_64-us-east5-a`
- Workaround retry:
  - stopped `iris-r1`
  - resubmitted with explicit `--replicas 16`:
    - `/dlwh/grug-moe-qwen3-32b-a4b-v5p64-bs320-ep8-iris-r2`
  - current status after startup stabilization:
    - `JOB_STATE_PENDING`
    - `task_state_counts.pending = 16`
    - `pending_reason = Waiting for workers in scale group 'tpu_v5p_64-us-east5-a' to become ready`
  - this is the first healthy Iris pending state on current main; no Docker API pull failure has appeared so far.

### 2026-03-09 01:20 PT - GRUG-PERF-034 restore legacy `offload_mlp` semantics and re-establish Ray controls
- Root-cause fix:
  - Restored legacy block-only `save_mlp*` / `offload_mlp*` semantics in `experiments/grug/moe/model.py`.
  - Removed all `MOE_*_CKPT` names from the `save_mlp*` and `offload_mlp*` policy families.
  - Kept explicit MoE-only policy families instead:
    - `save_moe_input`
    - `save_moe_hidden`
    - `save_moe_output`
    - `save_moe_inputs_outputs`
    - `save_moe`
    - `offload_moe_input`
    - `offload_moe_hidden`
    - `offload_moe_output`
    - `offload_moe_inputs_outputs`
    - `offload_moe`
  - Updated `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py` to expose the revised mode names and keep short run-name slugs.
- Validation:
  - `python -m py_compile experiments/grug/moe/model.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py lib/levanter/src/levanter/grug/grug_moe.py`
  - `./infra/pre-commit.py experiments/grug/moe/model.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py lib/levanter/src/levanter/grug/grug_moe.py`
- Control rerun 1: `e128` with restored legacy `offload_mlp`
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romfix-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_mlp`
  - Terminal status:
    - `SUCCEEDED`
  - Key signals:
    - `2026-03-09 00:36:12 UTC` first step:
      - `Progress on:train 1.00it/18.0it rate:472.0s/it`
    - `2026-03-09 00:44:40 UTC` profiler start:
      - `Starting profiler until step 13.`
    - `2026-03-09 00:46:25 UTC` profiler stop:
      - `Stopping profiler.`
    - `2026-03-09 00:52:26 UTC` completion:
      - `Progress on:train 18.0it/18.0it ... loss=15.1`
  - Interpretation:
    - The apparent `offload_mlp` regression was indeed policy drift; the restored legacy block-only `offload_mlp` path is healthy again on `e128`.
- Control rerun 2: `e64` with restored legacy `offload_mlp`
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e64-k4-cf1p25-romfix-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=64 / matched-active / synthetic / offload_mlp`
  - Terminal status:
    - `SUCCEEDED`
  - Key signals:
    - `2026-03-09 00:56:20 UTC` entered first-step compile:
      - `Progress on:train -/18`
    - `2026-03-09 01:03:16 UTC` first step:
      - `Progress on:train 1.00it/18.0it rate:416.6s/it`
    - `2026-03-09 01:10:45 UTC` profiler start:
      - `Starting profiler until step 13.`
    - `2026-03-09 01:12:23 UTC` profiler stop:
      - `Stopping profiler.`
    - `2026-03-09 01:17:14 UTC` completion:
      - `Progress on:train 18.0it/18.0it ... loss=14.7`
  - Interpretation:
    - The earlier `e64` OOM also appears attributable to the broadened `offload_mlp` semantics rather than the `num_experts=64` change itself.

### 2026-03-09 01:18 PT - GRUG-PERF-035 start explicit MoE-only offload A/B on `e128`
- First explicit MoE-only variant:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romei-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_input`
  - Purpose:
    - isolate offloading only `MOE_DISPATCH_INPUT_CKPT` after restoring legacy block-only `offload_mlp`
  - Status at launch:
    - `RUNNING`
    - submission accepted cleanly on Ray `us-east5-a`

### 2026-03-09 01:32 PT - GRUG-PERF-036 `offload_moe_input` fails cleanly; advance to `offload_moe_output`
- `offload_moe_input` result:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romei-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_input`
  - First-step compile began:
    - `2026-03-09 01:22:05 UTC` `Progress on:train -/18`
  - Primary failure:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to reserve 51.43G at the bottom of memory. That was not possible. There are 46.61G free, 0B reserved, and 46.61G reservable.`
  - Interpretation:
    - offloading only `MOE_DISPATCH_INPUT_CKPT` is not viable on this shape; it increases load-time TPU memory pressure enough to fail before step 1.
  - Operational note:
    - Ray kept the submission nominally `RUNNING` while retrying around the same nonrecoverable OOM, so the job was stopped manually after the primary failure was confirmed.
- Next A/B entry launched:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeo-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_output`
  - Status at launch:
    - accepted cleanly on Ray `us-east5-a`

### 2026-03-09 01:47 PT - GRUG-PERF-037 `offload_moe_output` also OOMs; advance to `offload_moe_hidden`
- `offload_moe_output` result:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeo-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_output`
  - First-step compile began:
    - `2026-03-09 01:35:45 UTC` `Progress on:train -/18`
  - Primary failure:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to reserve 51.22G at the bottom of memory. That was not possible. There are 46.61G free, 0B reserved, and 46.61G reservable.`
  - Interpretation:
    - offloading only `MOE_DISPATCH_OUTPUT_CKPT` is also not viable on this shape; it fails before step 1 with essentially the same TPU load-time memory wall as `offload_moe_input`.
  - Operational note:
    - Ray kept retrying around the same nonrecoverable OOM, so the job was stopped manually after the primary failure was confirmed.
- Next A/B entry launched:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeh2-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_hidden`
  - Status at launch:
    - accepted cleanly on Ray `us-east5-a`

### 2026-03-09 02:01 PT - GRUG-PERF-038 `offload_moe_hidden` OOMs even harder; advance to `offload_moe_inputs_outputs`
- `offload_moe_hidden` result:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeh2-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_hidden`
  - First-step compile began:
    - `2026-03-09 01:51:02 UTC` `Progress on:train -/18`
  - Primary failure:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to reserve 57.26G at the bottom of memory. That was not possible. There are 46.61G free, 0B reserved, and 46.61G reservable.`
  - Interpretation:
    - offloading only `MOE_EXPERT_HIDDEN_CKPT` is the worst single MoE-only offload so far; it fails before step 1 with the largest TPU load-time reservation among the explicit single-tensor variants.
  - Operational note:
    - Ray again stayed nominally alive while retrying around the same nonrecoverable OOM, so the job was stopped manually after confirming the primary failure.
- Next A/B entry launched:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeio-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_inputs_outputs`
  - Status at launch:
    - accepted cleanly on Ray `us-east5-a`

### 2026-03-09 02:14 PT - GRUG-PERF-039 `offload_moe_inputs_outputs` also OOMs; advance to full `offload_moe`
- `offload_moe_inputs_outputs` result:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-romeio-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe_inputs_outputs`
  - First-step compile began:
    - `2026-03-09 02:04:52 UTC` `Progress on:train -/18`
  - Primary failure:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to reserve 58.81G at the bottom of memory. That was not possible. There are 46.61G free, 0B reserved, and 46.61G reservable.`
  - Interpretation:
    - combining dispatch input and output offload is worse than either single dispatch tensor and slightly worse than `offload_moe_hidden`.
  - Operational note:
    - Ray again retried around the nonrecoverable OOM, so the job was stopped manually after confirming the primary failure.
- Next A/B entry launched:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-rome-r2-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe`
  - Status at launch:
    - accepted cleanly on Ray `us-east5-a`

### 2026-03-09 02:25 PT - GRUG-PERF-040 full `offload_moe` fails in host compile memory; move to shared-expert-width trial
- `offload_moe` result:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-rome-r2-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / matched-active / synthetic / offload_moe`
  - First-step compile began:
    - `2026-03-09 02:18:06 UTC` `Progress on:train -/18`
  - Primary failure:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space host. Used 129.30G of 128.00G host. Exceeded host capacity by 1.30G.`
  - Interpretation:
    - full MoE offload is strictly worse than legacy `offload_mlp`; unlike the single- or dual-tensor MoE-only variants, it pushes the compile over the host-memory limit instead of failing in TPU HBM.
  - Comparison summary at fixed `e128 / bs320 / EP4 / topk4 / cf1.25 / synthetic`:
    - legacy `offload_mlp`: `SUCCEEDED`
    - `offload_moe_input`: TPU load-time OOM at `51.43G`
    - `offload_moe_output`: TPU load-time OOM at `51.22G`
    - `offload_moe_hidden`: TPU load-time OOM at `57.26G`
    - `offload_moe_inputs_outputs`: TPU load-time OOM at `58.81G`
    - `offload_moe`: host compile OOM at `129.30G / 128.00G`
  - Decision:
    - keep legacy `offload_mlp` as the default; none of the explicit MoE-only offload variants are viable on this shape.
- Next planned knob launched:
  - Submission:
    - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-sx4096-mf-rom-r1-20260309`
  - Shape:
    - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / shared_expert_intermediate_dim=4096 / match_total_active_flops / synthetic / legacy offload_mlp`
  - Derived routed expert width:
    - `intermediate_dim=1024`
  - Purpose:
    - test whether moving more active FFN budget into the shared expert improves the perf/memory tradeoff while keeping total active FLOPs matched

### 2026-03-09 02:52 PT - GRUG-PERF-041 shared-expert-width trial succeeds with matched active FLOPs
- Submission:
  - `ray-run-grug-moe-q32a4b-v5p64-bs320-ep4-e128-k4-cf1p25-sx4096-mf-rom-r1-20260309`
- Shape:
  - `bs320 / EP4 / topk4 / cf1.25 / num_experts=128 / shared_expert_intermediate_dim=4096 / intermediate_dim=1024 / match_total_active_flops / synthetic / legacy offload_mlp`
- Terminal status:
  - `SUCCEEDED`
- Key signals:
  - `2026-03-09 02:28:49 UTC` entered first-step compile:
    - `Progress on:train -/18`
  - `2026-03-09 02:36:53 UTC` first step:
    - `Progress on:train 1.00it/18.0it rate:483.9s/it`
  - `2026-03-09 02:45:16 UTC` profiler start:
    - `Starting profiler until step 13.`
  - `2026-03-09 02:46:56 UTC` profiler stop:
    - `Stopping profiler.`
  - `2026-03-09 02:46:35 UTC` step `11`:
    - `Progress on:train 11.0it/18.0it rate:34.5s/it remaining:04:01 elapsed:17:46 postfix:loss=15.5`
  - `2026-03-09 02:49:42 UTC` final checkpoint:
    - `Saving checkpoint at step 18`
- Immediate comparison against the restored legacy baseline:
  - baseline `shared_expert_intermediate_dim=2048 / intermediate_dim=1536 / offload_mlp`:
    - succeeded
    - step `11` at `35.1s/it`
  - matched-flops shared-heavy variant `shared_expert_intermediate_dim=4096 / intermediate_dim=1024 / offload_mlp`:
    - succeeded
    - step `11` at `34.5s/it`
- Interpretation:
  - moving more active FFN budget into the shared expert while matching total active FLOPs is viable on this shape and looks marginally faster than the restored baseline, though the delta is small enough that the profile should be inspected before drawing a strong conclusion.

### 2026-03-09 03:29 PT - GRUG-PERF-042 shorten run ids for W&B profile persistence; launch short-name reruns
- Launcher cleanup:
  - Shortened generated run ids and groups in `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`.
  - Main change:
    - old base stem: `grug-moe-q32a4b-v5p64-bs...`
    - new base stem: `gq32-v5p64-b...`
  - Also dropped the default loader suffix for the common `pf32 / buf64` case to keep profiled run ids shorter.
  - Validation:
    - `python -m py_compile experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
    - `./infra/pre-commit.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
- New short-name profiled reruns launched:
  - Shared-heavy matched-FLOPs variant:
    - submission: `ray-run-gq32-sx4096-mf-rom-r3-20260309`
    - run suffix: `s4r3`
  - Fastest `64/128` baseline control:
    - submission: `ray-run-gq32-e64-rom-r2-20260309`
    - run suffix: `e64r2`
- Purpose:
  - ensure W&B profile artifacts persist under shorter run ids for both:
    - the next logical shared-expert-width trial
    - the fastest `64/128` baseline control for direct profile comparison

### 2026-03-09 12:26 PT - GRUG-PERF-043 add minimal Pallas CE reproducer and relaunch with halved `v_block_size`
- Motivation:
  - The explicit `pallas_tpu,xla` CE runs on the hybrid `E64 / topk4 / shared2048 / cf1.1` shape failed in the Pallas CE kernel with scoped-VMEM exhaustion before any fallback to XLA.
  - We want a minimal reproducer outside the full Grug stack and one controlled relaunch that only changes `v_block_size`.
- Reproducer:
  - Added `lib/levanter/scripts/bench/repro_pallas_ce_vmem.py`.
  - Default reproducer shape matches the failing CE call:
    - `x: [40960, 2048]`
    - `labels: [40960]`
    - `w: [2048, 128256]`
    - `implementation=pallas_tpu`
    - `compute_dtype=float32`
  - It prints the inferred/tuned block sizes and then JITs a single fused CE call.
- Launcher/code changes:
  - Threaded explicit CE `BlockSizes` through:
    - `lib/levanter/src/levanter/grug/loss.py`
    - `experiments/grug/moe/model.py`
    - `experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py`
  - Added launcher arg:
    - `--cross-entropy-v-block-divisor`
  - The launcher now infers the per-device CE shape for `v5p-64`, keeps inferred `b_block_size` and `h_block_size`, and shrinks only `v_block_size`.
- Exact inferred CE block sizes for the failing hybrid shape:
  - local CE batch tokens: `320 * 4096 / 32 = 40960`
  - inferred block sizes for `[40960, 2048] x [2048, 128256]`: `BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)`
  - tuned-table match: `False`
  - relaunch override: `v_block_size=512`
- Validation:
  - `uv run python -m py_compile experiments/grug/moe/model.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py lib/levanter/src/levanter/grug/loss.py lib/levanter/scripts/bench/repro_pallas_ce_vmem.py`
  - `./infra/pre-commit.py experiments/grug/moe/model.py experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py lib/levanter/src/levanter/grug/loss.py lib/levanter/scripts/bench/repro_pallas_ce_vmem.py`
- Relaunch:
  - submission: `ray-run-gq32-e64h-cev2-r1-20260309`
  - shape:
    - `bs320 / EP4 / cf1.1 / E64 / topk4 / shared2048 / match_activated_params / synthetic / offload_mlp`
    - `cross_entropy_implementation=pallas_tpu,xla`
    - `cross_entropy_v_block_divisor=2`

### 2026-03-09 12:49 PT - GRUG-PERF-044 retry Pallas CE with higher scoped VMEM and autotune-on-miss
- Observation:
  - The explicit `cross_entropy_v_block_divisor=2` retry bypasses the fused-CE autotune-on-miss path because explicit `BlockSizes` short-circuit candidate selection.
  - The earlier scoped-vmem failure margin was narrow:
    - scoped allocation `52.48M`
    - scoped limit `48.83M`
    - over by `3.65M`
- Decision:
  - Retry the same hybrid shape without explicit CE block-size override so Pallas can sweep candidate block sizes on miss.
  - Raise `--xla_tpu_scoped_vmem_limit_kib` from `50000` to `56000` to test whether the failure is simply clipping against the rounded default budget.
- Relaunch:
  - submission: `ray-run-gq32-e64h-ceat-r1-20260309`
  - env:
    - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=56000`
    - `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=1`
  - shape:
    - `bs320 / EP4 / cf1.1 / E64 / topk4 / shared2048 / match_activated_params / synthetic / offload_mlp`
    - `cross_entropy_implementation=pallas_tpu,xla`
    - no explicit CE `BlockSizes`

### 2026-03-11 21:40 PT - GRUG-PERF-045 target-shape recommendation after corrected shared-width FLOPs accounting
- Observation:
  - Correcting `lm_flops_per_token(...)` to account for actual `shared_expert_intermediate_dim` materially changed the ranking of the geometry sweep.
  - The earlier "no shared wins" conclusion was wrong under the buggy accounting.
- Best corrected overall candidate:
  - `es3r2`
  - W&B: `gq32-v5p64-b320-e4-c1p25-e64-k4-ix1024-sx1024-h4096-l27-rom-cex-syn-p-es3r2`
  - Meaning:
    - `hidden_dim=4096`
    - `num_layers=27`
    - `EP=4`
    - `num_experts=64`
    - `topk=4`
    - routed expert width `intermediate_dim=1024`
    - shared expert width `shared_expert_intermediate_dim=1024`
  - Result:
    - max `throughput/tokens_per_second`: `139,044.68`
    - corrected `flops/token`: `8.323e9`
    - corrected model FLOPs/s: `1.157e15`
    - final short-run loss: `13.9`
- Best simpler companion candidate:
  - `es2`
  - W&B: `gq32-v5p64-b320-e4-c1p25-e64-k4-ix1536-sx1536-h3072-l32-rom-cex-syn-p-es2`
  - Meaning:
    - `hidden_dim=3072`
    - `num_layers=32`
    - `EP=4`
    - `num_experts=64`
    - `topk=4`
    - routed expert width `intermediate_dim=1536`
    - shared expert width `shared_expert_intermediate_dim=1536`
  - Result:
    - max `throughput/tokens_per_second`: `130,273.02`
    - corrected `flops/token`: `8.360e9`
    - corrected model FLOPs/s: `1.089e15`
- Interpretation:
  - `es3r2` is the current "best really-MoE-like" target: it keeps shared width modest and equal to routed width while beating the other equal-shared variants.
  - `es2` is the cleaner, simpler fallback target in the same equal-shared-width design family.
  - These are the two target shapes to carry into the longer real-data compare.

### 2026-03-12 10:18 PT - GRUG-PERF-046 seal snapshot for issue #3528
- Scope closeout:
  - Sealing the current research thread state after geometry sweeps, corrected FLOPs accounting, and scale-up planning notes.
  - Issue: `#3528` (long-run compare thread).
- Final target-shape recommendation for the next long run:
  - primary: `es3r2`
    - `hidden_dim=4096 / layers=27 / E=64 / EP=4 / topk=4 / routed=1024 / shared=1024 / cf=1.25`
  - fallback: `es2`
    - `hidden_dim=3072 / layers=32 / E=64 / EP=4 / topk=4 / routed=1536 / shared=1536 / cf=1.25`
- Additional scale-up follow-ups captured before close:
  - Re-test MaxText MoE XLA perf flag bundle at higher EP:
    - `MOE_VMEM_LIMIT_FLAG + CF_FOR_ALL_GATHER + DATA_PARALLEL_OVERLAP`
  - Re-evaluate dispatch crossover at high-order EP (`ragged_all_to_all` vs ring path), cross-referencing `#2710` low-EP ring results.
- Seal artifact intent:
  - Include launch/config scripts used for v5p64 and v4 sizing checks.
  - Tag snapshot for reproducibility and issue permalinking.
