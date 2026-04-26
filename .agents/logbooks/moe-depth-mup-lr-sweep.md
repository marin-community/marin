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

### 2026-04-25 13:35 - d512 complete

- Hypothesis: including the delayed `4x` run should sharpen the d512 LR curve
  and confirm whether the first scale is centered near the baseline multiplier.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 9 succeeded jobs and 9 running jobs; no terminal failures.
  - All d512 runs are finished.
  - d768 `2.83x` has launched and is running; d768 `4x` has not launched yet.
  - Completed d512 W&B summaries:

    | LR multiplier | Paloma macro | Train loss | Tok/s | Tokens |
    | --- | ---: | ---: | ---: | ---: |
    | 0.25x | 4.0652 | 3.7375 | 406,349 | 837,156,864 |
    | 0.354x | 3.9475 | 3.6286 | 406,406 | 837,156,864 |
    | 0.5x | 3.8671 | 3.5483 | 407,291 | 837,156,864 |
    | 0.707x | 3.8302 | 3.5066 | 405,839 | 837,156,864 |
    | 1x | 3.8188 | 3.4932 | 406,669 | 837,156,864 |
    | 1.41x | 3.8284 | 3.5006 | 406,452 | 837,156,864 |
    | 2x | 3.8695 | 3.5429 | 405,423 | 837,156,864 |
    | 2.83x | 3.9301 | 3.6019 | 406,235 | 837,156,864 |
    | 4x | 4.0568 | 3.7285 | 406,238 | 837,156,864 |

- Interpretation: the completed d512 sweep is best at `1x`, with a relatively
  shallow basin around `0.707x`-`1.41x` and clear degradation at both extremes.
  Compared with the README baseline macro 3.8104, depth MuP at d512 is neutral
  to slightly worse but does not require a shifted LR at this scale.
- Next action: continue monitoring d768 through final summaries and the launch
  of `d768-lr4x`.

### 2026-04-25 16:09 - d768 partial completion and d1024 startup

- Hypothesis: the first finished d768 points should show whether the LR optimum
  shifts materially from the d512 optimum before the delayed high-LR points
  finish.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 16 succeeded jobs and 9 running jobs; no terminal failures.
  - Seven d768 low-to-2x jobs are finished.
  - d768 `2.83x` is still running at W&B step 8160 with Paloma macro 3.8212.
  - d768 `4x` has launched and is running at W&B step 47.
  - The first six d1024 jobs have launched (`0.25x` through `1.41x`), with no
    W&B training summaries yet.
  - Finished d768 W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Tok/s |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 0.25x | finished | 10342 | 3.5921 | 3.3211 | 273,923 |
    | 0.354x | finished | 10342 | 3.5244 | 3.2554 | 275,041 |
    | 0.5x | finished | 10342 | 3.4701 | 3.2025 | 275,812 |
    | 0.707x | finished | 10342 | 3.4387 | 3.1675 | 274,990 |
    | 1x | finished | 10342 | 3.4279 | 3.1535 | 274,617 |
    | 1.41x | finished | 10342 | 3.4295 | 3.1581 | 273,881 |
    | 2x | finished | 10342 | 3.4724 | 3.1922 | 274,267 |
    | 2.83x | running | 8160 | 3.8212 | 3.4842 | 272,446 |
    | 4x | running | 47 | 11.7911 | 10.9101 | 277,573 |

- Interpretation: among finished d768 points, the optimum remains at `1x`, with
  `1.41x` effectively tied and `0.707x` close. This matches the d512 optimum
  more than it suggests a scale-dependent LR shift, but the high-LR tail is not
  finished yet.
- Next action: continue monitoring d768 `2.83x`/`4x` and d1024 startup.

### 2026-04-25 17:12 - d1024 first meaningful eval

- Hypothesis: early d1024 evals should show whether the LR basin remains near
  the d512/d768 optimum as model scale increases.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 17 succeeded jobs and 9 running jobs; no terminal failures.
  - d768 `2.83x` finished with Paloma macro 3.5249; d768 `4x` remains running
    at W&B step 3545 with Paloma macro 4.7201.
  - d1024 `0.25x` through `2x` are running. The first six d1024 runs have
    moved beyond placeholder evals; `2x` has just launched and still has only
    the initial placeholder eval.
  - d1024 W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Tok/s |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 0.25x | running | 1111 | 5.2304 | 4.4190 | 177,844 |
    | 0.354x | running | 1127 | 4.7424 | 4.0443 | 178,745 |
    | 0.5x | running | 1121 | 4.4787 | 4.0458 | 177,431 |
    | 0.707x | running | 1114 | 4.2972 | 3.8251 | 177,874 |
    | 1x | running | 1084 | 4.1957 | 3.7822 | 177,617 |
    | 1.41x | running | 1091 | 4.1650 | 3.7849 | 177,541 |
    | 2x | running | 134 | 11.7908 | 8.6933 | 179,158 |
    | 2.83x | missing | n/a | n/a | n/a | n/a |
    | 4x | missing | n/a | n/a | n/a | n/a |

- Interpretation: early d1024 is best at `1.41x`, with `1x` close behind and
  the lower LR points clearly worse. This is still early, but the best point is
  within one sweep step of the d512/d768 optimum rather than moving far away.
- Next action: continue monitoring d1024 through later evals and wait for
  d768 `4x` to finish.

### 2026-04-25 18:05 - d1024 second eval snapshot

- Hypothesis: the next d1024 eval should clarify whether the early best point
  remains at the high side of the basin or moves back toward the d512/d768
  optimum.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 1500 --tail /kaiyue/iris-run-job-20260425-184727`
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 17 succeeded jobs and 9 running jobs; no terminal failures.
  - d768 `4x` is still running at W&B step 6453 with Paloma macro 4.3554.
  - d1024 `0.25x` through `2x` are running. `2.83x`, `4x`, and all d1280
    runs have not launched yet.
  - d1024 W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Tok/s |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 0.25x | running | 2086 | 4.0084 | 3.6512 | 177,663 |
    | 0.354x | running | 2103 | 3.8979 | 3.4492 | 178,420 |
    | 0.5x | running | 2093 | 3.8318 | 3.5104 | 176,989 |
    | 0.707x | running | 2088 | 3.8059 | 3.4784 | 177,492 |
    | 1x | running | 2060 | 3.8305 | 3.4729 | 177,034 |
    | 1.41x | running | 2066 | 3.9059 | 3.5336 | 177,487 |
    | 2x | running | 1122 | 4.1707 | 3.7606 | 177,537 |
    | 2.83x | missing | n/a | n/a | n/a | n/a |
    | 4x | missing | n/a | n/a | n/a | n/a |

- Interpretation: by the second d1024 eval, the best observed point has moved
  to `0.707x`, with `0.5x` and `1x` essentially close behind. The basin still
  overlaps the d512/d768 optimum, but the early `1.41x` advantage did not hold
  at this checkpoint.
- Next action: continue the normal babysit cadence until d768 `4x` finishes or
  the next d1024 launches/evals create a clearer scale trend.

### 2026-04-25 18:59 - d1024 third eval snapshot

- Hypothesis: the ~3k-step d1024 eval should test whether the `0.707x` best
  point from the previous checkpoint is stable or whether the basin shifts back
  toward `1x`.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 1500 --tail /kaiyue/iris-run-job-20260425-184727`
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260425-184727`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 17 succeeded jobs and 9 running jobs; no terminal failures.
  - d768 `4x` is still running at W&B step 8864 with Paloma macro 4.0186 after
    one preemption.
  - d1024 `0.5x` also picked up one preemption and has not posted its ~3k-step
    eval yet.
  - d1024 W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Tok/s |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 0.25x | running | 3081 | 3.7652 | 3.3393 | 177,525 |
    | 0.354x | running | 3105 | 3.6933 | 3.3829 | 178,385 |
    | 0.5x | running | 2860 | 3.8318 | 3.3356 | 176,744 |
    | 0.707x | running | 3085 | 3.6647 | 3.3696 | 177,297 |
    | 1x | running | 3053 | 3.7080 | 3.3664 | 177,281 |
    | 1.41x | running | 3061 | 3.8062 | 3.4856 | 177,299 |
    | 2x | running | 2116 | 4.0351 | 3.7342 | 177,468 |
    | 2.83x | missing | n/a | n/a | n/a | n/a |
    | 4x | missing | n/a | n/a | n/a | n/a |

- Interpretation: the third d1024 checkpoint keeps `0.707x` as the best
  observed multiplier, with `0.354x` and `1x` close but worse. The missing
  `0.5x` third eval makes the center of the basin incomplete, but the optimum
  still appears to overlap the d512/d768 `0.707x`-`1x` region rather than
  moving far with scale.
- Next action: continue monitoring until d1024 `0.5x` catches up, d768 `4x`
  finishes, or the next d1024 jobs launch.

### 2026-04-25 19:10 - interruption recovery

- Context: while responding to a corrected instruction, the original Iris
  parent job `/kaiyue/iris-run-job-20260425-184727` was stopped after it had
  completed 17 children and still had 8 active children.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job stop /kaiyue/iris-run-job-20260425-184727`
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 1 --memory 2G --extra cpu -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.depth_mup_lr_sweep`
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Replacement parent job: `/kaiyue/iris-run-job-20260426-020352`.
  - The executor relaunched only the 8 unfinished children plus the parent, so
    the completed d512 and d768 low-to-2.83x runs were not rerun.
  - Several early replacement attempts hit TPU device-busy startup errors
    (`No accelerator found` / `Device or resource busy`), but Iris retried.
  - At the 19:10 check all 8 unfinished children were running in Iris, and W&B
    had recovered the checked run IDs to `running` with
    `model.depth_mup_residual_scaling=True`.
  - Recovery checkpoint W&B summaries:

    | Run | State | Step | Paloma macro | Train loss |
    | --- | --- | ---: | ---: | ---: |
    | d768 4x | running | 8999 | 4.0186 | 3.4979 |
    | d1024 0.25x | running | 3155 | 3.7652 | 3.4442 |
    | d1024 0.354x | running | 3179 | 3.6933 | 3.4104 |
    | d1024 0.5x | running | 2900 | 3.8318 | 3.3211 |
    | d1024 0.707x | running | 3155 | 3.6647 | 3.3581 |
    | d1024 1x | running | 3132 | 3.7080 | 3.3675 |
    | d1024 1.41x | running | 3136 | 3.8062 | 3.4573 |
    | d1024 2x | running | 2190 | 4.0351 | 3.6537 |

- Interpretation: the depth MuP sweep remains valid and continuing after
  recovery. The replacement job is a continuation of unfinished work, not a
  full restart of completed points.
- Next action: continue short-cadence monitoring until the restarted tasks show
  sustained post-retry training progress, then return to the normal cadence.

### 2026-04-25 19:26 - recovery heartbeat and CI closure

- Hypothesis: after the replacement launch stabilizes, the sweep should behave
  like a normal executor run with 8 concurrent child steps.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 1800 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
  - `curl -fsS ... /repos/marin-community/marin/commits/117f0f414/check-runs`
- Result:
  - Iris shows 8 active child runs plus the parent, all `JOB_STATE_RUNNING`.
  - No active child has nonzero `failure_count` or a `pending_reason`.
  - Recent logs show d768 `4x` around 9.66k/10.3k steps after Paloma macro
    3.8426, so it should finish soon.
  - d1024 children are making post-retry progress: `0.354x` is around
    3.35k/12.6k, `0.5x` around 3.00k/12.6k, and `2x` around 2.35k/12.6k.
  - W&B still reports `model.depth_mup_residual_scaling=True` for every
    started run.
  - GitHub checks on `117f0f414` are complete: relevant checks passed and the
    rest were skipped.
- Interpretation: no intervention is needed. The missing d1024 `2.83x`/`4x`
  and d1280 runs are expected because `StepRunner` defaults to 8 concurrent
  steps; they should enqueue after current active children finish or skip.
- Next action: return to the normal babysit cadence and wait for d768 `4x` to
  finish or for the next executor wave to launch.

### 2026-04-25 19:41 - d768 sweep complete

- Hypothesis: if depth MuP reduces LR sensitivity across small MoE scales, the
  d512 and d768 LR optima should stay near the same multiplier, even if the
  quality delta versus baseline is small.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 420 --max-lines 1200 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
  - `uv run python - <<'PY' ... effective_speedup(...) ... PY`
- Result:
  - d768 `4x` finished successfully with Paloma macro 3.6372 and 274,573 tok/s.
  - The executor immediately launched d1024 `2.83x`.
  - The completed d768 LR curve is:

    | LR multiplier | Paloma macro | Tok/s |
    | --- | ---: | ---: |
    | 0.25x | 3.5921 | 273,923 |
    | 0.354x | 3.5244 | 275,041 |
    | 0.5x | 3.4701 | 275,812 |
    | 0.707x | 3.4387 | 274,990 |
    | 1x | 3.4279 | 274,617 |
    | 1.41x | 3.4295 | 273,881 |
    | 2x | 3.4724 | 274,267 |
    | 2.83x | 3.5249 | 273,856 |
    | 4x | 3.6372 | 274,573 |

  - Effective-speedup comparison against README baselines:

    | Scale | Best depth-MuP LR | Baseline macro | Depth-MuP macro | Baseline tok/s | Depth-MuP tok/s | Speedup |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | d512 | 1x | 3.8104 | 3.8188 | 405,630 | 406,669 | 0.963 |
    | d768 | 1x | 3.4339 | 3.4279 | 273,532 | 274,617 | 1.040 |

  - d1024 `0.5x` caught up and posted Paloma macro 3.6554 at the ~3k-step
    eval, which is slightly better than the previous best observed `0.707x`
    checkpoint at 3.6647.
- Interpretation: depth MuP does not pass gate 1 as a promotable change at the
  current small-scale optima because d512 is slower than baseline on effective
  speedup. The LR-optimum stability signal is still useful: both completed
  small scales prefer `1x`, and the d1024 partial curve is currently centered
  in the `0.5x`-`0.707x`-`1x` basin rather than moving to an extreme LR.
- Next action: continue d1024 through `2.83x`/`4x`, then continue to d1280 for
  the full MoE procedure before making a final recommendation.

### 2026-04-25 20:22 - d1024 ~4k checkpoint

- Hypothesis: after the d512 and d768 optima landed at `1x`, the next d1024
  checkpoint tests whether depth MuP keeps the best LR in the same central
  basin or shifts the optimum with scale.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 480 --max-lines 1400 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows the parent plus 8 running d1024 children and 1 succeeded d768
    child; all active children have `failure_count=0` and no `pending_reason`.
  - d1024 `0.5x` posted its ~4k eval, completing the central-basin checkpoint.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 4199 | 3.6384 | 3.2433 | ~4k eval |
    | 0.354x | running | 4373 | 3.5845 | 3.2604 | ~4k eval |
    | 0.5x | running | 4006 | 3.5592 | 3.2101 | ~4k eval, current best |
    | 0.707x | running | 4152 | 3.5724 | 3.1670 | ~4k eval |
    | 1x | running | 4156 | 3.6264 | 3.2520 | ~4k eval |
    | 1.41x | running | 4155 | 3.7062 | 3.3162 | ~4k eval |
    | 2x | running | 3379 | 3.9430 | 3.5878 | latest eval around 3k |
    | 2.83x | running | 572 | 11.7908 | 4.0274 | startup eval only |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: d1024 currently prefers `0.5x`, with `0.707x` close behind.
  This is one LR step lower than the completed d512/d768 `1x` optima, so depth
  MuP has not made the LR optimum perfectly scale-invariant. It has not moved
  to an extreme LR either: the useful basin remains central (`0.354x`-`1x`),
  and high LR points are clearly worse at the available checkpoints.
- Next action: continue d1024 until `2.83x` and `4x` have comparable evals and
  the d1024 runs complete; then proceed through d1280 before final scaling-law
  interpretation.

### 2026-04-25 21:16 - d1024 ~5k checkpoint

- Hypothesis: if the apparent d1024 optimum shift at ~4k was transient, the
  next central-basin checkpoint should move back toward the completed d512/d768
  `1x` optimum; if it persists, the best point should remain below `1x`.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 180 --max-lines 500 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris still shows the parent plus 8 running d1024 children and 1 succeeded
    d768 child; active children have no failure counts or pending reasons.
  - All sampled started W&B configs still have
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 5181 | 3.5508 | 3.1868 | ~5k eval |
    | 0.354x | running | 5380 | 3.5036 | 3.1477 | ~5k eval |
    | 0.5x | running | 5014 | 3.4841 | 3.1850 | ~5k eval, current best |
    | 0.707x | running | 5165 | 3.5002 | 3.1170 | ~5k eval |
    | 1x | running | 5141 | 3.5450 | 3.2167 | ~5k eval |
    | 1.41x | running | 5135 | 3.6300 | 3.2511 | ~5k eval |
    | 2x | running | 4358 | 3.8529 | 3.4772 | latest eval around 4k |
    | 2.83x | running | 1555 | 4.3148 | 3.9647 | latest eval around 1k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: the d1024 best point remains `0.5x`, with `0.707x` and
  `0.354x` close behind. This is still a central LR basin, but it is not
  scale-invariant in the strong sense: the completed d512/d768 sweeps prefer
  `1x`, while d1024 is currently best one LR step lower. The high-LR side is
  clearly worse at comparable progress (`2x` is much worse by ~4k, and `2.83x`
  is poor by ~1k).
- Next action: continue d1024 to completion and let the executor launch d1024
  `4x` and then d1280 before making the gate-2 scaling-law call.

### 2026-04-25 22:13 - d1024 ~6k checkpoint

- Hypothesis: the d1024 central-basin optimum should become clearer once the
  `0.5x`, `0.707x`, `1x`, and `1.41x` runs all publish comparable ~6k evals.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 1800 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows the parent plus 8 running d1024 children and 1 succeeded d768
    child; there are still no active child failures or pending reasons.
  - All sampled started W&B configs still report
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 6283 | 3.4918 | 3.1274 | ~6k eval |
    | 0.354x | running | 6442 | 3.4441 | 3.1142 | ~6k eval |
    | 0.5x | running | 6082 | 3.4266 | 3.0811 | ~6k eval, current best |
    | 0.707x | running | 6233 | 3.4394 | 3.1220 | ~6k eval |
    | 1x | running | 6230 | 3.4777 | 3.1569 | ~6k eval |
    | 1.41x | running | 6219 | 3.5561 | 3.1399 | ~6k eval |
    | 2x | running | 5450 | 3.7702 | 3.4310 | latest eval around 5k |
    | 2.83x | running | 2639 | 4.2020 | 3.7672 | latest eval around 2k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: the d1024 best point is still `0.5x`, now with `0.707x`
  and `0.354x` both close enough to define a central basin. This remains a
  useful LR-sensitivity signal, but not a clean scale-invariance result:
  d512/d768 completed best at `1x`, while d1024 is currently best one LR step
  lower. The high-LR side is consistently poor; `2x` remains much worse at
  around 5k, and `2.83x` remains poor by around 2k.
- Next action: continue d1024 until `4x` launches and the d1024 sweep
  finishes, then continue through d1280 for the full gate-2 procedure.

### 2026-04-25 23:04 - d1024 ~7k checkpoint

- Hypothesis: if the d1024 optimum is settling, the next comparable central
  checkpoint should identify whether the best point remains at `0.5x` or
  moves toward the completed d512/d768 `1x` optimum.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 600 --max-lines 1600 --tail /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris shows 9 running jobs and 1 succeeded child; there are no active
    failures or pending reasons.
  - All sampled started W&B configs still report
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 7213 | 3.4436 | 3.0721 | ~7k eval |
    | 0.354x | running | 7372 | 3.3989 | 3.1928 | ~7k eval |
    | 0.5x | running | 7010 | 3.3799 | 3.0645 | ~7k eval, current best |
    | 0.707x | running | 7164 | 3.3868 | 3.1016 | ~7k eval |
    | 1x | running | 7156 | 3.4217 | 3.1752 | ~7k eval |
    | 1.41x | running | 7155 | 3.5006 | 3.1962 | ~7k eval |
    | 2x | running | 6378 | 3.6968 | 3.3081 | latest eval around 6k |
    | 2.83x | running | 3564 | 4.1101 | 3.7340 | latest eval around 3k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: the d1024 best point remains `0.5x`, but the `0.707x`
  point is very close. The central basin is strengthening as training
  progresses, while `1x` remains behind and the high-LR side remains clearly
  worse. This still does not match strong depth-wise LR invariance, because
  the completed d512/d768 sweeps prefer `1x` while d1024 continues to prefer
  a lower multiplier.
- Next action: continue d1024 until the current 8 active children complete and
  the executor launches d1024 `4x`, then continue into d1280.

### 2026-04-26 00:01 - d1024 ~8k checkpoint

- Hypothesis: after the partial ~8k evals, `0.5x` needed one more checkpoint
  to determine whether d1024's best point had moved to `0.707x` or remained at
  the lower central multiplier.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris still shows 9 running jobs and 1 succeeded child; there are no active
    failures or pending reasons.
  - All sampled started W&B configs still report
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 8282 | 3.4091 | 3.1121 | ~8k eval |
    | 0.354x | running | 8431 | 3.3613 | 3.1196 | ~8k eval |
    | 0.5x | running | 8068 | 3.3354 | 3.0344 | ~8k eval, current best |
    | 0.707x | running | 8236 | 3.3393 | 3.0529 | ~8k eval |
    | 1x | running | 8216 | 3.3648 | 3.0430 | ~8k eval |
    | 1.41x | running | 8212 | 3.4289 | 3.2039 | ~8k eval |
    | 2x | running | 7439 | 3.6178 | 3.2974 | latest eval around 7k |
    | 2.83x | running | 4627 | 4.0216 | 3.6519 | latest eval around 4k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: the d1024 best point remains `0.5x`, but `0.707x` is only
  about 0.004 Paloma macro behind at comparable progress. The useful LR basin
  is now clearly central (`0.354x` through `1x`), while `2x` and `2.83x` are
  still much worse at their latest evals. This does not show strong LR
  scale-invariance, because completed d512/d768 prefer `1x` and d1024 still
  prefers one LR step lower.
- Next action: continue d1024 until `4x` launches and the d1024 sweep finishes,
  then continue through the d1280 sweep before fitting the gate-2 scaling law.

### 2026-04-26 00:54 - d1024 ~9k checkpoint

- Hypothesis: if the d1024 optimum is drifting within the central basin, the
  next comparable checkpoint should show whether `0.5x` remains best or whether
  `0.707x`/`1x` catch up as training progresses.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris still shows 9 running jobs and 1 succeeded child; there are no active
    failures or pending reasons.
  - All sampled started W&B configs still report
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 9246 | 3.3779 | 3.0685 | ~9k eval |
    | 0.354x | running | 9392 | 3.3301 | 3.0579 | ~9k eval |
    | 0.5x | running | 9028 | 3.3026 | 3.0315 | ~9k eval |
    | 0.707x | running | 9201 | 3.2986 | 2.9236 | ~9k eval, current best |
    | 1x | running | 9184 | 3.3198 | 3.0911 | ~9k eval |
    | 1.41x | running | 9173 | 3.3655 | 3.0964 | ~9k eval |
    | 2x | running | 8396 | 3.5443 | 3.2410 | latest eval around 8k |
    | 2.83x | running | 5586 | 3.9453 | 3.5868 | latest eval around 5k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: the d1024 best point moved from `0.5x` at ~8k to `0.707x`
  at ~9k, with `0.5x` only about 0.004 Paloma macro behind. This is a stronger
  central-basin result than the earlier checkpoints, but it is still not strong
  LR scale-invariance: completed d512/d768 prefer `1x`, while d1024 is best
  one LR step lower at the current checkpoint. The high-LR side remains much
  worse.
- Next action: continue d1024 until the current children finish and the
  executor launches d1024 `4x`, then continue through d1280 for the full gate-2
  procedure.

### 2026-04-26 01:48 - d1024 ~10k checkpoint

- Hypothesis: if depth MuP is reducing LR sensitivity at d1024, the later
  checkpoints should keep the useful region broad and central rather than
  collapsing to a single low-LR point.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /kaiyue/iris-run-job-20260426-020352`
  - `uv run python - <<'PY' ... wandb.Api().runs('understanding-sam/marin_moe', filters={'display_name': name}) ... PY`
- Result:
  - Iris still shows 9 running jobs and 1 succeeded child; there are no active
    failures or pending reasons.
  - All sampled started W&B configs still report
    `model.depth_mup_residual_scaling=True`.
  - d1024 latest W&B summaries:

    | LR multiplier | State | Step | Paloma macro | Train loss | Notes |
    | --- | --- | ---: | ---: | ---: | --- |
    | 0.25x | running | 10239 | 3.3537 | 2.9764 | ~10k eval |
    | 0.354x | running | 10384 | 3.3016 | 3.0810 | ~10k eval |
    | 0.5x | running | 10016 | 3.2694 | 2.9634 | ~10k eval |
    | 0.707x | running | 10191 | 3.2568 | 2.9816 | ~10k eval, current best |
    | 1x | running | 10171 | 3.2648 | 2.9873 | ~10k eval |
    | 1.41x | running | 10163 | 3.2983 | 2.9167 | ~10k eval |
    | 2x | running | 9387 | 3.4593 | 3.1498 | latest eval around 9k |
    | 2.83x | running | 6570 | 3.8586 | 3.4723 | latest eval around 6k |
    | 4x | missing | n/a | n/a | n/a | queued |

- Interpretation: d1024 remains best at `0.707x`, but `1x` is only about
  0.008 Paloma macro behind and `0.5x` is about 0.013 behind. This is the
  strongest central-basin result so far: the useful region now spans `0.5x`
  through `1.41x`, with `0.707x`/`1x` clearly ahead. It still does not prove
  strong LR scale-invariance because d512/d768 completed at `1x`, while d1024
  is currently best one step lower.
- Next action: continue d1024 to completion so `4x` can launch, then continue
  through d1280 for the full gate-2 procedure.
