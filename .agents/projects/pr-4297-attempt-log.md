# PR 4297 CoreWeave H100x8 Attempt Log

## Context

- Worktree: `/home/ubuntu/dev/marin-wt/research-pr-4297-followup`
- Branch: `research/pr-4297-followup`
- Goal:
  - verify the real CoreWeave CI submission path for `experiments/grug/moe/launch_h100_pr4297.py`
  - run pinned-`jax==0.8.0` A/B with `RAGGED_DOT_IMPL=xla|triton`
  - fill `.agents/projects/pr-4297-gh-comment-draft.md` with measured results

## Verified Submission Path

The working path keeps the outer Iris job CPU-only and lets `executor_main`
submit the real H100x8 child job on CoreWeave CI:

```bash
export KUBECONFIG=~/.kube/coreweave-iris
source ~/.config/yoblin/env >/dev/null 2>&1 || true

uv run iris --config=lib/iris/examples/coreweave-ci.yaml job run \
  --no-wait \
  --job-name "<job-name>" \
  --cpu 1 --memory 16G --disk 16G --extra cpu \
  -e MARIN_PREFIX "s3://marin-na/marin/" \
  -e GRUG_RUN_ID "<run-id>" \
  -e RAGGED_DOT_IMPL "<xla|triton>" \
  -e WANDB_ENTITY "marin-community" \
  -e WANDB_PROJECT "romain-dev" \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e HF_TOKEN "${HF_TOKEN}" \
  -e R2_ACCESS_KEY_ID "${R2_ACCESS_KEY_ID}" \
  -e R2_SECRET_ACCESS_KEY "${R2_SECRET_ACCESS_KEY}" \
  -e AWS_ACCESS_KEY_ID "${R2_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY "${R2_SECRET_ACCESS_KEY}" \
  -e AWS_ENDPOINT_URL "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com" \
  -e FSSPEC_S3 '{"endpoint_url":"https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com","client_kwargs":{"region_name":"auto"}}' \
  -- python -m experiments.grug.moe.launch_h100_pr4297 --force_run_failed false
```

Why the extra env is required:

- the CPU-only parent uses `executor_main`
- `executor_main` writes metadata to `s3://marin-na/marin/...`
- without the R2/AWS-compatible env, the parent fails before it can submit the
  child GPU job

## Harness Changes in This Worktree

`experiments/grug/moe/launch_h100_pr4297.py` was adjusted to use only the
prebuilt caches that are actually loadable from R2 on CoreWeave CI:

- kept:
  - `nemotron_cc/hq_actual`
  - `nemotron_cc/medium_high`
  - `nemotron_cc/low_actual`
  - `nemotron_cc/low_synth`
  - `starcoderdata`
- dropped as broken or incomplete:
  - `nemotron_cc/hq_synth`
  - `nemotron_cc/medium`
  - `nemotron_cc/medium_low`
  - `proofpile_2`

Additional harness fixes:

- switched to cache-only `DatasetComponent`s with explicit `s3://marin-na/marin/...` cache paths
- set `auto_build_caches=False` so the run does not try to rebuild tokenized data
- made the executor step name depend on both `RAGGED_DOT_IMPL` and `GRUG_RUN_ID`
  so repeated A/B pairs get fresh executor output paths and checkpoint
  namespaces

## Run Chronology

### 2026-04-02: corrected XLA run

- parent job: `/ubuntu/pr4297-xla-20260402-024606`
- child job: `/ubuntu/pr4297-xla-20260402-024606/grug-train-pr4297-grug-moe-256m-xla-20260402-024606`
- verified child resources:
  - `cpu_millicores=32000`
  - `memory_bytes=274877906944`
  - `device.gpu.variant=H100`
  - `device.gpu.count=8`
- child logs confirmed the pinned stack and GPU path were active

Observed behavior:

- training progressed and logged metrics through W&B step `94`
- after that, the data loader stalled while waiting for `1024` items
- Iris still showed the job as running, but no forward progress resumed
- because the cluster had only one schedulable H100x8+RDMA slot, the hung XLA
  job was stopped to free capacity for Triton

Known XLA metrics already visible in W&B summary:

- run id: `pr4297-grug-moe-256m-xla-20260402-024606`
- state at observation time: `running`
- `global_step=94`
- `throughput/examples_per_second=108.96959521894227`
- `throughput/tokens_per_second=446339.46201678755`
- `throughput/mfu=5.343554568058925`
- `throughput/mean_mfu=5.329041455395002`
- `train/loss=7.66891622543335`

Median XLA metrics over the usable comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=114.65729441567908`
- `throughput/tokens_per_second=469636.2779266215`
- `throughput/mfu=5.6224629274357145`
- `throughput/duration=0.2790925790031906`
- `train/loss=9.364686012268066`

### 2026-04-02: sequential Triton relaunch

- parent job: `/ubuntu/pr4297-triton-20260402-030112`
- run id: `pr4297-grug-moe-256m-triton-20260402-030112`
- reason for sequential execution:
- the CoreWeave CI pool only exposed one schedulable H100x8+RDMA slot
- concurrent Triton launch previously stayed pending with `Insufficient rdma/ib`
  and `Insufficient nvidia.com/gpu`

Observed failure on the first sequential relaunch:

- the CPU-only parent started correctly
- `executor_main` then refused to relaunch the Triton step because the earlier
  aborted Triton attempt had left the same executor step marked `FAILED`
- error:
  - `PreviousTaskFailedError: Step grug/pr4297_h100_repro_triton_e83940e9 failed previously. Status: FAILED`

### 2026-04-02: forced Triton rerun after stale failed status

- parent job: `/ubuntu/pr4297-triton-rerun-20260402-030301`
- run id: `pr4297-grug-moe-256m-triton-20260402-030301`
- launch arg change:
  - `--force_run_failed true`

Verified behavior so far:

- the CPU-only parent job is running
- the parent submitted the GPU child job successfully
- child job:
  - `/ubuntu/pr4297-triton-rerun-20260402-030301/grug-train-pr4297-grug-moe-256m-triton-20260402-030301`
- requested child resources:
  - `cpu_millicores=32000`
  - `memory_bytes=274877906944`
  - `device.gpu.variant=H100`
  - `device.gpu.count=8`

Final outcome:

- child Iris state: `JOB_STATE_SUCCEEDED`
- parent Iris state: `JOB_STATE_SUCCEEDED`
- W&B state: `finished`
- `global_step=99`

Median Triton metrics over the shared comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=123.69937739935658`
- `throughput/tokens_per_second=506672.64982776454`
- `throughput/mfu=6.065860590197692`
- `throughput/duration=0.2586916819855105`
- `train/loss=9.36465072631836`

Paired deltas versus XLA over the same `20-94` window:

- examples/s: `+7.886182060860691%`
- tokens/s: `+7.886182060860691%`
- MFU: `+0.4433976627619778` points
- duration: `-7.309723924062805%`
- loss delta: `-3.528594970703125e-05`

Interpretation:

- the end-to-end CoreWeave CI submission path is now verified for both variants
- Triton improved throughput and MFU on the pinned `jax==0.8.0` stack for this
  workload
- the measured uplift is materially smaller than the PR body's claimed `20%`
  speedup
- this is still only one paired run on a cache-only data subset, so the result
  should be described as exploratory

### 2026-04-02: repeat-pair harness fix

Problem:

- repeated submissions of the same variant reused the same executor step name
  and output path
- `StepRunner` always skips `STATUS_SUCCESS`, so additional A/B pairs would not
  have produced fresh runs

Fix on the test branch:

- salt the executor step name with `GRUG_RUN_ID`
- this gives each submission a fresh executor output path and checkpoint
  namespace while keeping the external launch path unchanged

Next:

- commit and push the rerun-enabler on `research/pr-4297-followup`
- run two additional sequential XLA/Triton pairs
- aggregate per-run medians across all three pairs

Next checks:

- confirm the Triton child job is created under the CPU-only parent
- confirm the child lands on H100x8 resources
- monitor whether Triton completes or hits the same late data-loader stall
- compute median comparison metrics from W&B history and update the GitHub draft

### 2026-04-02: pair 2 XLA

- parent job: `/ubuntu/pr4297-p2-xla-20260402-032629`
- child job:
  - `/ubuntu/pr4297-p2-xla-20260402-032629/grug-train-pr4297-grug-moe-256m-xla-p2-20260402-032629`
- run id: `pr4297-grug-moe-256m-xla-p2-20260402-032629`

Final outcome:

- child Iris state: `JOB_STATE_SUCCEEDED`
- parent Iris state: `JOB_STATE_SUCCEEDED`
- W&B state: `finished`
- `global_step=99`

Median XLA metrics over the shared comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=112.89190327040627`
- `throughput/tokens_per_second=462405.2357955841`
- `throughput/mfu=5.535893238892966`
- `throughput/duration=0.2834569980041124`
- `train/loss=9.364654541015625`

### 2026-04-02: pair 2 Triton

- parent job: `/ubuntu/pr4297-p2-triton-20260402-033540`
- child job:
  - `/ubuntu/pr4297-p2-triton-20260402-033540/grug-train-pr4297-grug-moe-256m-triton-p2-20260402-033540`
- run id: `pr4297-grug-moe-256m-triton-p2-20260402-033540`

Final outcome:

- child Iris state: `JOB_STATE_SUCCEEDED`
- parent Iris state: `JOB_STATE_SUCCEEDED`
- W&B state: `finished`
- `global_step=99`

Median Triton metrics over the shared comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=122.84021376358888`
- `throughput/tokens_per_second=503153.51557566004`
- `throughput/mfu=6.023729684219815`
- `throughput/duration=0.26050101200235076`
- `train/loss=9.364668846130371`

Paired deltas versus XLA over the same `20-94` window:

- examples/s: `+8.812244461282347%`
- tokens/s: `+8.812244461282347%`
- MFU: `+0.4878364453268495` points
- duration: `-8.09826442163529%`
- loss delta: `+1.430511474609375e-05`

### 2026-04-02: pair 3 XLA

- parent job: `/ubuntu/pr4297-p3-xla-20260402-034620`
- child job:
  - `/ubuntu/pr4297-p3-xla-20260402-034620/grug-train-pr4297-grug-moe-256m-xla-p3-20260402-034620`
- run id: `pr4297-grug-moe-256m-xla-p3-20260402-034620`

Final outcome:

- child Iris state: `JOB_STATE_SUCCEEDED`
- parent Iris state: `JOB_STATE_SUCCEEDED`
- W&B state: `finished`
- `global_step=99`

Median XLA metrics over the shared comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=113.19352972341481`
- `throughput/tokens_per_second=463640.6977471071`
- `throughput/mfu=5.550684129944574`
- `throughput/duration=0.2827016710070893`
- `train/loss=9.364670753479004`

### 2026-04-02: pair 3 Triton

- parent job: `/ubuntu/pr4297-p3-triton-20260402-035616`
- child job:
  - `/ubuntu/pr4297-p3-triton-20260402-035616/grug-train-pr4297-grug-moe-256m-triton-p3-20260402-035616`
- run id: `pr4297-grug-moe-256m-triton-p3-20260402-035616`

Observed behavior:

- the child hit one worker-level retry late in the first attempt
- Iris kept the job alive and the retried task resumed the same W&B run id
- the task-local `.wandb` files were split across attempts, so the final
  pair-3 Triton metrics were read from the merged W&B run history instead of a
  single local fragment

Final outcome:

- child Iris state: `JOB_STATE_SUCCEEDED`
- parent Iris state: `JOB_STATE_SUCCEEDED`
- child `failure_count=1`
- W&B state: `finished`
- `global_step=99`

Median Triton metrics over the shared comparison window (`global_step` `20-94`):

- rows in window: `75`
- `throughput/examples_per_second=123.14440004864048`
- `throughput/tokens_per_second=504399.4625992314`
- `throughput/mfu=6.038646102049601`
- `throughput/duration=0.2598575330048334`
- `train/loss=9.364619255065918`

Paired deltas versus XLA over the same `20-94` window:

- examples/s: `+8.791023965362976%`
- tokens/s: `+8.791023965362976%`
- MFU: `+0.487961972105027` points
- duration: `-8.08065191863796%`
- loss delta: `-5.14984130859375e-05`

### 2026-04-02: aggregated 3-pair result

Aggregate method:

- compute per-run medians over the shared step window `20-94`
- then take the median across the `3` pair-level medians for each variant

Aggregate median-of-pair-medians:

- `xla`
  - `throughput/examples_per_second=113.19352972341481`
  - `throughput/tokens_per_second=463640.6977471071`
  - `throughput/mfu=5.550684129944574`
  - `throughput/duration=0.2827016710070893`
  - `train/loss=9.364670753479004`
- `triton`
  - `throughput/examples_per_second=123.14440004864048`
  - `throughput/tokens_per_second=504399.4625992314`
  - `throughput/mfu=6.038646102049601`
  - `throughput/duration=0.2598575330048334`
  - `train/loss=9.36465072631836`

Aggregate deltas:

- examples/s: `+8.791023965362976%`
- tokens/s: `+8.791023965362976%`
- MFU: `+0.487961972105027` points
- duration: `-8.08065191863796%`
- loss delta: `-2.002716064453125e-05`

Interpretation after all 3 pairs:

- Triton beat XLA in all `3` paired runs on the same nearby geometry
- the pinned-`jax==0.8.0` validation now has repeated, not just exploratory,
  support for Triton on this workload
- the end-to-end uplift remained in the `8-9%` range, which is still
  materially below the PR body's `20%` headline

### 2026-04-02: reproducible metrics script

- added `experiments/grug/moe/summarize_h100_pr4297.py`
- the script queries the `3` paired W&B runs directly and prints:
  - per-run coverage with `state`, `summary_global_step`, and `window_rows`
  - the aggregate markdown table used in the draft comment
- the script hard-codes the validated run ids and defaults to the shared
  comparison window `20-94`
- this also makes the step-`94` behavior explicit:
  - only pair-1 XLA stopped at `94`
  - pair-2 XLA and pair-3 XLA both reached `99`
- live verification from W&B matched the published table:
  - pair-1 XLA state `crashed`, `summary_global_step=94`
  - every other run state `finished`, `summary_global_step=99`
