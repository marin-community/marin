# Null-Routed MoE Performance Sweep

## Description
This issue tracks an experiment extending the existing MoE hillclimb harness from #2704 and #2710 to support null routing (tokens can route to "none").

The implementation starts from:
- MoE hillclimb harness and EP path from #2704/#2710
- null-expert Mixtral routing block from `origin/will/null-moe`

We added null-routing controls to `lib/levanter/scripts/bench/bench_moe_hillclimb.py`:
- `--null-route-frac`
- `--null-route-frac-list`
- `--null-route-sweep` (`0.1..0.9`)
- `RESULT ...` summary lines for easier parsing.

## Hypothesis or Goal
At fixed shape, increasing null-routing fraction should reduce routed expert compute and improve throughput, with stronger gains for larger `topk` and larger expert pools.

## Planned Matrix
- Null-routing fraction: `0.1, 0.2, ..., 0.9`
- Shapes:
  - `tokens=32768 hidden=2048 mlp_dim=1408 experts=60`
  - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64`
  - `tokens=65536 hidden=3072 mlp_dim=3072 experts=128`
- Granularity axis: `topk in {2, 4, 8}`
- Primary pass: `forward_backward`

### Links
- Prior issue: https://github.com/marin-community/marin/issues/2704
- Prior issue: https://github.com/marin-community/marin/issues/2710
- Logbook: `.agents/logbooks/null_routed_moe_benchmark.md`

## Status (2026-02-17)
- TPU `gmm` sweep completed for all planned points:
  - `3` shapes x `3` top-k settings x `9` null fractions = `81` measured points
- Added baseline `null=0.0` matrix for the same shape/top-k grid:
  - `3` shapes x `3` top-k settings = `9` baseline points
- Added variable experts-per-token null semantics:
  - New routing mode `take_until_null` in both
    - `experiments/speedrun/custom_mixtral.py` (`MixtralConfig.take_until_null`)
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py` (`--null-routing-mode`)
- Raw run artifact:
  - `.agents/logbooks/null_routed_moe_results_20260217_213956.log`
- Parsed artifact:
  - `.agents/logbooks/null_routed_moe_results_20260217_213956.csv`
- Baseline artifacts:
  - `.agents/logbooks/null_routed_moe_baseline0_20260217_223128.log`
  - `.agents/logbooks/null_routed_moe_baseline0_20260217_223128.csv`
  - `.agents/logbooks/null_routed_moe_results_with_null0_20260217_223128.csv`
- Summary: tokens/s increases with higher null fraction across all combinations; effect size is largest at `topk=8`.
