# Grug MoE EP Ring: Research Logbook

## Scope
- Goal: make [`lib/levanter/src/levanter/grug/grug_moe.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/src/levanter/grug/grug_moe.py) the fastest functional Grug MoE path for the target Qwen3 32B a4b MoE shape, especially at higher expert-parallel (`EP`) ranks.
- Primary metric(s): end-to-end `forward_backward` wall time for the focused harness on `v5p-16`, comparing `legacy` vs `current` kernels at `EP=1,2,4,8`.
- Constraints: do not open a new GitHub issue; preserve numerical equivalence with the pre-change EP ring path.

## Baseline
- Date: 2026-03-06
- Code refs:
  - [`lib/levanter/src/levanter/grug/grug_moe.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - [`lib/levanter/scripts/bench/bench_moe_hillclimb.py`](/Users/dlwh/.codex/worktrees/21ea/marin/lib/levanter/scripts/bench/bench_moe_hillclimb.py)
- Baseline numbers: pending TPU measurement.
- Reference shape:
  - W&B run: `grug-moe-qwen3-32b-a4b-v5p64-perf`
  - v5p-64 training batch size: `32`
  - v5p-16 benchmark tokens: `8 * 4096 = 32768`
  - hidden=`2048`, mlp_dim=`768`, experts=`128`, topk=`8`, shared_expert_dim=`2048`

## Experiment Log
### 2026-03-06 18:00 - Current kernel compaction change
- Hypothesis: high-EP slowdown is dominated by the global `argsort` plus fused full-length `take`s in `_moe_mlp_ep_ring_local`, and replacing that with direct local-assignment compaction should materially reduce time at `EP>=2`.
- Command: local code edit plus focused correctness harness.
- Config:
  - Replaced global sort/filter path with `local_mask` + ordered `top_k` selection over only local assignments.
  - Added a focused benchmark harness that compares `legacy` and `current` kernels and can check forward/grad equivalence.
- Result:
  - CPU smoke equivalence passed for tiny shapes.
  - Existing Grug MoE tests passed locally.
- Interpretation: the new EP ring path is functionally plausible and ready for TPU timing.
- Next action: finish multihost harness startup and run `v5p-16` timings for `EP=1,2,4,8`.

### 2026-03-06 23:45 - Multihost TPU harness validation
- Hypothesis: the focused harness can run directly on a 2-host `v5p-16` slice once it initializes `jax.distributed`, and `legacy` vs `current` remain numerically identical in EP mode.
- Command:
  - Bootstrap two-worker slice `ray-marin-us-central1-worker-e565cfbd-tpu`
  - Run:
    ```bash
    gcloud compute tpus tpu-vm ssh ray-marin-us-central1-worker-e565cfbd-tpu \
      --zone us-central1-a --worker=all --command '
      set -e
      source "$HOME/.local/bin/env"
      cd "$HOME/marin"
      PROC_ID=$(hostname | sed -E "s/.*-w-([0-9]+)$/\1/")
      export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"
      uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
        --coordinator-address 10.128.0.5:12355 \
        --num-processes 2 \
        --process-id "$PROC_ID" \
        --tokens 512 \
        --hidden 256 \
        --mlp-dim 128 \
        --experts 16 \
        --topk 4 \
        --shared-expert-dim 128 \
        --ep-list 1,2,4,8 \
        --iters 1 \
        --warmup 0 \
        --check-equivalence'
    ```
- Config:
  - 8 TPU devices total across 2 hosts
  - `bench_pass=forward_backward`
  - small TPU-valid GMM shape (`mlp_dim=128`)
- Result:
  - `CHECK ep=1 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=2 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=4 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
  - `CHECK ep=8 out_max_abs=0.000000e+00 grad_max_abs=0.000000e+00`
- Interpretation: the current EP-ring compaction path matches the legacy implementation for both forward outputs and backward grads on TPU across the full `EP=1,2,4,8` sweep.
- Next action: run full-shape timings without equivalence mode.

### 2026-03-07 00:00 - Full-shape `v5p-16` benchmark
- Hypothesis: at the Qwen3 32B a4b shape, removing the global sort/full-length take sequence from the EP ring path will be neutral at `EP=1` and increasingly beneficial as `EP` rises.
- Command:
  ```bash
  gcloud compute tpus tpu-vm ssh ray-marin-us-central1-worker-e565cfbd-tpu \
    --zone us-central1-a --worker=all --command '
    set -e
    source "$HOME/.local/bin/env"
    cd "$HOME/marin"
    PROC_ID=$(hostname | sed -E "s/.*-w-([0-9]+)$/\1/")
    export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"
    uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --coordinator-address 10.128.0.5:12355 \
      --num-processes 2 \
      --process-id "$PROC_ID" \
      --tokens 32768 \
      --hidden 2048 \
      --mlp-dim 768 \
      --experts 128 \
      --topk 8 \
      --shared-expert-dim 2048 \
      --ep-list 1,2,4,8 \
      --iters 3 \
      --warmup 1'
  ```
- Config:
  - Shape derived from W&B run `grug-moe-qwen3-32b-a4b-v5p64-perf`
  - v5p-64 training batch size `32`, reduced by `4x` to v5p-16 benchmark tokens `32768`
  - `bench_pass=forward_backward`
  - routing distributions tested: `random`, `runs`
- Result:
  - `random`
    - `EP=1`: legacy `24.330 ms`, current `24.275 ms`
    - `EP=2`: legacy `25.229 ms`, current `23.473 ms`
    - `EP=4`: legacy `22.986 ms`, current `20.121 ms`
    - `EP=8`: legacy `25.939 ms`, current `17.882 ms`
  - `runs`
    - `EP=1`: legacy `24.381 ms`, current `24.302 ms`
    - `EP=2`: legacy `25.209 ms`, current `23.406 ms`
    - `EP=4`: legacy `23.066 ms`, current `20.126 ms`
    - `EP=8`: legacy `26.609 ms`, current `17.811 ms`
- Interpretation:
  - The current kernel is effectively neutral at `EP=1` (`~0.3%` faster on average).
  - It is clearly better at `EP=2` (`~7.6%`), `EP=4` (`~14.4%`), and especially `EP=8` (`~47.2%`).
  - Under the current kernel on this `v5p-16`, `EP=8` is the fastest setting; average current times were:
    - `EP=1`: `24.289 ms`
    - `EP=2`: `23.440 ms`
    - `EP=4`: `20.124 ms`
    - `EP=8`: `17.847 ms`
  - Relative to current `EP=1`, current `EP=8` is `~1.36x` faster for total `forward_backward` time.
- Next action: keep the compact local-selection path in `grug_moe.py`; recommend `EP=8` for this shape on `v5p-16`, with `EP=4` the next-best fallback.
