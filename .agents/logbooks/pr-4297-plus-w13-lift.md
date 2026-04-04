# PR 4297 + 3821-Style W13 Follow-Up: Research Logbook

## Scope
- Goal: measure the additional H100x8 performance lift from layering the #3821-style expert-padded `w13` FC1 lowering on top of the #4297 Triton `ragged_dot` baseline.
- Primary metric(s): `throughput/examples_per_second`, `throughput/tokens_per_second`, `throughput/mfu`, `throughput/duration`, and loss parity on the fixed Grug MoE H100x8 repro harness.
- Constraints:
  - start from `research/pr-4297-followup` / commit `e55747ba2f1c41e1a341b54263079a858a153a30`
  - use the always-on CoreWeave `iris-ci` cluster (`cpu-erapids` + `h100-8x`)
  - keep the comparison apples-to-apples on the overlapping optimization surface
  - the `#4297` branch does not carry the full DeepEP module tree used by `#3821`, so benchmark the shared exact-cap-style `current` EP=8 kernel path rather than replaying the full DeepEP transport ladder
  - answer the incremental question, so the key pair is `4297 triton current` vs `4297 triton current + 3821-style w13 expert padding`
  - GitHub issue: `https://github.com/marin-community/marin/issues/4406`
- Relevant references:
  - #4297: Triton `ragged_dot` kernel for GPU MoE grouped matmul
  - #3821: expert-padded `w13` FC1 lowering on the exact-cap H100x8 path
- Stop criteria:
  - produce at least one clean paired measurement for `4297 triton` vs `4297 triton + w13 expert padded`
  - if the first pair moves materially, add replication until the incremental delta is directionally stable
  - if the integrated path is broken or numerically suspect, stop and report the blocker instead of forcing more runs

## Baseline
- Date: 2026-04-04
- Code refs:
  - `experiments/grug/moe/launch_h100_pr4297.py`
  - `experiments/grug/moe/summarize_h100_pr4297.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `experiments/grug/moe/model.py`
- Fixed baseline case:
  - hardware: CoreWeave H100x8 on `iris-ci`
  - benchmark: exact-cap-style `current` EP ring kernel (`ep_size=8`)
  - `tokens=262144`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `topk=2`
  - `shared_expert_dim=0`
  - `distribution=random`
  - `warmup=5`
  - `iters=20`
- Prior PR 4297 training-harness result carried forward only as context:
  - `triton` beat `xla` by `+8.8%` tokens/s on the fixed ~256M Grug H100x8 training repro
- Working comparison baseline for this thread:
  - `RAGGED_DOT_IMPL=triton`
  - `kernel=current`
  - no `w13` expert-padded lowering enabled

## Experiment Log
### 2026-04-04 01:45 UTC - Kickoff
- Hypothesis:
  - the #3821 expert-padded `w13` FC1 lowering still yields a measurable incremental gain on top of the #4297 Triton `ragged_dot` baseline, but the delta will likely be smaller than the original #3821 isolated win because part of the local expert compute path is already faster under Triton.
- Command:
  - scaffolding only
- Config:
  - worktree: `/home/ubuntu/dev/marin-wt/pr-4297-plus-w13-lift`
  - branch: `research/pr-4297-plus-w13-lift`
  - base branch: `research/pr-4297-followup`
  - base commit: `e55747ba2f1c41e1a341b54263079a858a153a30`
- Result:
  - created a dedicated follow-up research branch/worktree rather than extending the old PR 4297 validation thread
  - confirmed the local repo already contains the PR 4297 Grug H100x8 repro harness and summarizer
  - confirmed the #3821 winning idea was the expert-padded `w13` FC1 lowering, not the earlier `w13-out-first` layout candidate
  - confirmed `lib/iris/examples/coreweave-ci.yaml` provides the always-on `iris-ci` CoreWeave lane the user requested
  - attempted to reuse the historical #3821 bench stack directly, but the PR 4297 branch does not include `levanter.kernels.deepep`, so the full DeepEP transport benchmark path does not import cleanly on this base
- Interpretation:
  - the right low-confound measurement on top of the PR 4297 base is the overlapping exact-cap-style `current` EP=8 kernel path, not the full historical DeepEP transport ladder
  - that still answers the user’s interaction question because both optimizations directly act on the same routed FC1 grouped-matmul surface in the `current` path
- Next action:
  - create a minimal exact-cap benchmark harness for `current` EP=8 on top of the PR 4297 base, then run the four-way matrix `xla`, `xla+padded`, `triton`, `triton+padded` on `iris-ci`

### 2026-04-04 01:46 UTC - First H100 submission failed on default Iris env
- Hypothesis:
  - the new exact-cap benchmark harness should run directly as an Iris H100x8 job from the worktree.
- Command:
  - `KUBECONFIG=~/.kube/coreweave-iris uv run iris --config=lib/iris/examples/coreweave-ci.yaml job run --job-name pr4297-plus-w13-20260404-014558 --gpu H100x8 --cpu 32 --memory 256GB --disk 128GB --timeout 7200 --extra gpu -- bash -lc '<matrix driver using plain python>'`
- Config:
  - Iris job id: `/ubuntu/pr4297-plus-w13-20260404-014558`
  - same benchmark cell planned for the final measurement
- Result:
  - the worker reached the user command, but the benchmark failed immediately with `ModuleNotFoundError: No module named 'numpy'`
  - the default Iris task environment did not install the `levanter` dependency stack needed by the benchmark script
- Interpretation:
  - the benchmark must run inside an explicit `uv run --python 3.11 --package levanter --extra gpu ...` environment inside the task, rather than assuming the default task venv is sufficient
- Next action:
  - resubmit the same H100x8 matrix with an explicit `levanter[gpu]` uv environment inside the container

### 2026-04-04 01:47 UTC - Replicated H100x8 exact-cap matrix on top of #4297
- Hypothesis:
  - if the #3821-style padded `w13` lowering still targets a bottleneck that Triton does not remove, `triton + w13 padded` should remain measurably faster than plain `triton`.
- Command:
  - `KUBECONFIG=~/.kube/coreweave-iris uv run iris --config=lib/iris/examples/coreweave-ci.yaml job run --job-name pr4297-plus-w13-uv-20260404-014704 --gpu H100x8 --cpu 32 --memory 256GB --disk 128GB --timeout 7200 -- bash -lc 'uv run --python 3.11 --package levanter --extra gpu python - <<PY ... seeds=(0,1,2), variants=(xla,xla+padded,triton,triton+padded) ... PY'`
- Config:
  - Iris job id: `/ubuntu/pr4297-plus-w13-uv-20260404-014704`
  - hardware: CoreWeave `iris-ci` / `h100-8x`
  - benchmark script: `lib/levanter/scripts/bench/bench_pr4297_w13_current.py`
  - benchmark cell:
    - `tokens=262144`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `topk=2`
    - `shared_expert_dim=0`
    - `distribution=random`
    - `ep_size=8`
    - `warmup=5`
    - `iters=20`
    - seeds: `0,1,2`
- Result:
  - per-seed paired deltas for the key question (`triton+padded` vs `triton`):
    - seed `0`: `25224053.59` vs `26037983.42` tokens/s, delta `-3.13%`
    - seed `1`: `25219442.80` vs `26093597.99` tokens/s, delta `-3.35%`
    - seed `2`: `25185414.51` vs `26063969.19` tokens/s, delta `-3.37%`
  - aggregate means across the three seeds:

    | Variant | mean time_s | mean tokens/s | delta |
    | --- | ---: | ---: | --- |
    | `xla` | `0.026271` | `9982858.40` | baseline |
    | `xla + w13 padded` | `0.015337` | `17092122.30` | `+71.21%` vs `xla` |
    | `triton` | `0.010057` | `26065183.53` | `+161.10%` vs `xla` |
    | `triton + w13 padded` | `0.010399` | `25209636.97` | `-3.28%` vs `triton`, `+152.53%` vs `xla` |
- Interpretation:
  - on this shared exact-cap-style `current` EP=8 surface, the #3821-style padded `w13` lowering is still a large win for the `xla` backend
  - on top of the #4297 Triton `ragged_dot` path, the same change is not additive; it is a small but very consistent regression across all three seeds
  - the answer to the user’s question on this surface is therefore: assuming #4297 lands, replaying the #3821-style `w13` change does not buy additional throughput here, and likely costs about `3.3%`
  - this should be treated as `replicated` for the exact-cap microbenchmark surface, but not generalized to the full historical #3821 DeepEP transport stack, which is absent on the #4297 base
- Next action:
  - update the experiment issue with the replicated conclusion, seal this snapshot with a tag, and report the measured incremental delta back to the user
