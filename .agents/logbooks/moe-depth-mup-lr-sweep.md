# MoE Depth MuP LR Sweep: Research Logbook

## Scope

- Goal: test whether depth MuP residual scaling makes the current Grug MoE
  recipe less learning-rate-sensitive across model scale.
- Primary metrics: final `eval/paloma/macro_loss`, LR optimum curvature per
  scale.
- Secondary metrics: `throughput/tokens_per_second`,
  `throughput/total_tokens`, effective speedup versus the compute-optimal MoE
  baseline.
- Issue: https://github.com/marin-community/marin/issues/5178
- Prior LR sweep: https://github.com/marin-community/marin/issues/4225

## Baseline

- Date: 2026-04-25
- Code refs:
  - `experiments/grug/moe/README.md`
  - `experiments/grug/moe/model.py`
  - `experiments/grug/moe/heuristic.py`
- Baseline numbers: compute-optimal d512, d768, d1024, and d1280 table in
  `experiments/grug/moe/README.md`.

## Experiment Log

### 2026-04-25 11:25 - Kickoff

- Hypothesis: scaling each layer's residual updates by `1 / sqrt(num_layers)`
  will reduce depth-dependent residual growth and flatten the LR optimum across
  MoE scales.
- Command: local implementation only so far.
- Config:
  - architecture: current `experiments/grug/moe` recipe
  - intervention: depth MuP residual update scale
  - sweep scales: d512, d768, d1024, d1280 compute-optimal budgets from the
    README
  - LR multipliers: planned around the existing v16 LR formula
- Result: pending.
- Interpretation: pending.
- Next action: add tests for explicit residual scale behavior and sweep step
  construction, then implement the model and launch wiring.

### 2026-04-25 11:45 - Implementation scaffold

- Hypothesis: the smallest safe implementation is an opt-in model config flag,
  keeping the baseline recipe unchanged while the sweep enables depth MuP.
- Command:
  - `uv run --with pytest --with pytest-timeout python -m pytest tests/test_grug_moe_depth_mup.py`
  - `uv run --with pytest --with pytest-timeout python -m pytest tests/test_grug_variant_contracts.py -k moe`
  - `uv run --with pyrefly pyrefly check experiments/grug/moe/model.py experiments/grug/moe/heuristic.py experiments/grug/moe/depth_mup_lr_sweep.py tests/test_grug_moe_depth_mup.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/model.py experiments/grug/moe/heuristic.py experiments/grug/moe/depth_mup_lr_sweep.py experiments/grug/moe/README.md experiments/grug/moe/agent.md tests/test_grug_moe_depth_mup.py .agents/logbooks/moe-depth-mup-lr-sweep.md`
- Config:
  - `GrugModelConfig.depth_mup_residual_scaling=True`
  - residual update scale: `1 / sqrt(num_layers)`
  - sweep module: `experiments/grug/moe/depth_mup_lr_sweep.py`
  - sweep grid: d512, d768, d1024, d1280 x 9 LR multipliers
- Result:
  - red phase: 3 expected failures for missing config field, block scale, and
    sweep module
  - green phase: 3 depth-MuP tests passed after implementation
  - MoE contract slice: 3 passed
  - targeted Pyrefly: 0 errors
  - targeted pre-commit: OK
- Interpretation: local behavior is wired correctly; no TPU jobs launched yet.
- Next action: run broader contract checks, then decide whether to submit the
  sweep jobs.

### 2026-04-25 11:33 - Iris submission blocked

- Hypothesis: the depth MuP LR sweep can be submitted through the standard MoE
  Iris command once the branch is pushed.
- Command:
  - `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.depth_mup_lr_sweep`
  - `.venv/bin/iris --config lib/iris/examples/marin-dev.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.depth_mup_lr_sweep`
- Config:
  - active GCP account: `kaiyuew@stanford.edu`
  - GCP project: `hai-gcp-models`
  - `WANDB_API_KEY`: present
  - controller tunnel: none found on `localhost:10000`
- Result: both production and dev configs failed before creating a job:
  `GCP API error 403: Required 'compute.instances.list' permission for
  'projects/hai-gcp-models'`.
- Interpretation: the run submission is blocked by local GCP permissions, not
  by the experiment code or Iris job configuration.
- Next action: retry submission after authenticating with an account that can
  list controller VMs in `hai-gcp-models`, or provide an explicit
  `--controller-url` for an existing Iris tunnel.

### 2026-04-25 11:47 - Iris submission accepted

- Hypothesis: after switching local GCP/ADC auth to `kaiyuewen3@gmail.com`,
  the production Marin Iris controller can accept the depth MuP sweep.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 1 --memory 2G --extra cpu -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.depth_mup_lr_sweep`
- Config:
  - submission worktree: `/tmp/marin-depth-mup-run-94fb32af1`
  - code commit: `94fb32af1`
  - coordinator resources: CPU-only (`cpu=1`, `memory=2G`, `extra=cpu`)
  - child step resources: `ResourceConfig.with_tpu("v5p-8")`
  - sweep grid: 36 steps (4 scales x 9 LR multipliers)
- Result: Iris accepted coordinator job
  `/kaiyue/iris-run-job-20260425-184727`.
- Interpretation: the prior GCP permission/controller discovery blocker is
  cleared. The experiment is now in the Iris startup window.
- Next action: perform the 120-second startup check, then monitor on the normal
  babysit cadence.

### 2026-04-25 11:51 - Startup check

- Hypothesis: transient TPU init failures should be recoverable by Iris task
  retries if subsequent attempts get a usable v5p-8 worker.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run iris --config lib/iris/examples/marin.yaml job summary --json /kaiyue/iris-run-job-20260425-184727/grug-train-moe-depth-mup-lr-d512-lr1x`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 500 /kaiyue/iris-run-job-20260425-184727/grug-train-moe-depth-mup-lr-d512-lr1x`
- Result:
  - coordinator: `JOB_STATE_RUNNING`
  - d512 children visible: 8, all `JOB_STATE_RUNNING`
  - sampled d512 `lr1x` child: 2 recovered preemptions from TPU init failures
    (`Device or resource busy` / `No accelerator found`), then a running
    attempt loading caches
  - sample W&B run:
    `https://wandb.ai/understanding-sam/marin_moe/runs/moe-depth-mup-lr-d512-lr1x`
- Interpretation: Iris recovered from the initial bad-node symptoms without
  job-level intervention. No training loss yet; cache/eval setup is still in
  progress.
- Next action: switch to the normal babysit cadence and watch for first loss
  lines, terminal states, or repeated TPU bad-node errors.

### 2026-04-25 12:04 - First progress check

- Hypothesis: once the d512 children pass cache setup and first eval, the run
  should produce comparable early loss/progress lines without more bad-node
  intervention.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 1200 --tail /kaiyue/iris-run-job-20260425-184727`
- Result:
  - 9 jobs visible: coordinator + 8 d512 children, all `JOB_STATE_RUNNING`
  - each visible child still has `preemption_count=2` from the recovered TPU
    init failures
  - first eval/progress logs are present; sampled values:
    - d512 `lr1x`: Paloma macro 4.830 at first post-start eval, train progress
      ~1.35k / 6.39k steps with loss ~4.23
    - d512 `lr0.5x`: Paloma macro 5.137, train progress ~1.35k / 6.39k steps
      with loss ~4.38
    - d512 `lr2.83x`: Paloma macro 5.147, train progress ~1.35k / 6.39k
      steps with loss ~4.75
- Interpretation: the sweep is now doing real training. Early d512 logs make
  `lr1x` look better than the sampled lower/higher LR points, but this is still
  too early to draw conclusions.
- Next action: continue normal cadence until the d512 slice finishes and the
  executor advances to the next scale.

### 2026-04-25 12:42 - d512 partial completion

- Hypothesis: final d512 W&B summaries should identify whether depth MuP shifts
  the LR optimum away from the baseline formula before larger scales finish.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - 8 d512 runs finished; delayed `d512-lr4x` is still running.
  - 7 d768 runs are running.
  - Completed d512 W&B summaries:

    | LR multiplier | State | Paloma macro | Tok/s | Tokens |
    | --- | --- | ---: | ---: | ---: |
    | 0.25x | finished | 4.0652 | 406,349 | 837,156,864 |
    | 0.354x | finished | 3.9475 | 406,406 | 837,156,864 |
    | 0.5x | finished | 3.8671 | 407,291 | 837,156,864 |
    | 0.707x | finished | 3.8302 | 405,839 | 837,156,864 |
    | 1x | finished | 3.8188 | 406,669 | 837,156,864 |
    | 1.41x | finished | 3.8284 | 406,452 | 837,156,864 |
    | 2x | finished | 3.8695 | 405,423 | 837,156,864 |
    | 2.83x | finished | 3.9301 | 406,235 | 837,156,864 |
    | 4x | running | n/a | n/a | n/a |

- Interpretation: among completed points, d512 is best at `1x` LR with Paloma
  macro 3.8188. The compute-optimal baseline d512 macro is 3.8104, so depth MuP
  is roughly neutral/slightly worse at this first scale before the delayed 4x
  point finishes.
- Next action: continue monitoring `d512-lr4x` and d768 startup; wait for d768
  first eval before making scale-sensitivity claims.

### 2026-04-25 13:14 - d768 first meaningful eval

- Hypothesis: d768 should reveal whether depth MuP keeps the LR optimum near
  the same multiplier as d512 after the initial placeholder evals disappear.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris still shows 8 succeeded jobs and 9 running jobs; no terminal failures.
  - `d512-lr4x` remains running at W&B step 4042 with Paloma macro 4.7823.
  - Seven d768 runs are running around W&B steps 1366-1391. The two highest LR
    points, `2.83x` and `4x`, have not launched yet.
  - Current d768 W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Tok/s |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 0.25x | running | 1374 | 5.6296 | 4.4109 | 275,963 |
    | 0.354x | running | 1382 | 5.2119 | 4.0573 | 275,795 |
    | 0.5x | running | 1391 | 4.8381 | 4.0311 | 276,564 |
    | 0.707x | running | 1371 | 4.6411 | 3.9799 | 274,922 |
    | 1x | running | 1374 | 4.5468 | 3.9933 | 275,879 |
    | 1.41x | running | 1367 | 4.5297 | 3.9370 | 274,950 |
    | 2x | running | 1366 | 4.5989 | 4.0871 | 275,213 |
    | 2.83x | missing | n/a | n/a | n/a | n/a |
    | 4x | missing | n/a | n/a | n/a | n/a |

- Interpretation: at this early d768 point, the best observed LR multiplier is
  `1.41x`, with `1x` close behind and `2x` already slightly worse. This is not
  final, but the usable LR band is centered near the d512 optimum rather than
  moving dramatically.
- Next action: continue normal cadence until `d512-lr4x` finishes and the
  remaining d768 LR points launch.
