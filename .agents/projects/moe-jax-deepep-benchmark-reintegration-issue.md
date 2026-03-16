## Description

This experiment continues the sealed JAX DeepEP root-cause thread in `#3677`.

The previous session established two important facts:

1. the earlier negative JAX result in `#3665` was not measuring the real winning DeepEP transport path, because it only replaced layout metadata production and then still used JAX `ragged_all_to_all`;
2. a pure-JAX DeepEP transport path now runs end to end on H100x8, and under corrected world-size-8 defaults it lands within about `1.10x` to `1.54x` of the matched Torch transport baseline on the same fixed-shape workload.

That means the main unresolved question is now narrower:

> if the working pure-JAX DeepEP transport path is reinserted into the original fixed-shape JAX benchmark, does the old negative benchmark story materially change?

This thread is specifically about:

1. localizing the remaining small JAX/Torch transport delta on one authoritative cell;
2. reinserting the working pure-JAX DeepEP transport path into the original fixed-shape JAX benchmark;
3. determining whether the earlier negative benchmark conclusion was mostly a layout-only artifact or whether a broader benchmark bottleneck still dominates.

## Hypothesis or Goal

- Hypothesis: most of the original negative JAX benchmark result came from testing the wrong transport path rather than from an intrinsic JAX-vs-Torch DeepEP deficit.
- Goal: produce a small, decisive benchmark-level control that shows whether the corrected pure-JAX transport materially changes the original fixed-shape JAX ranking.
- Goal: attribute the remaining JAX/Torch transport delta before paying for broad sweeps or deeper tuning.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior layout-only JAX issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior root-cause issue (`#3677`): https://github.com/marin-community/marin/issues/3677
* Research branch: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-benchmark-reintegration
* Research logbook: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-benchmark-reintegration/.agents/logbooks/moe-jax-deepep-benchmark-reintegration.md

## Results

Current state as of 2026-03-16:
- new thread created from sealed `#3677` commit `6baa08edbd8ae9a782d0070a3b7cf0e1f38ba005`
- working branch head: `63862a86b50b3b3b571b19eab033c08a5032dbff`
- experiment issue: https://github.com/marin-community/marin/issues/3711
- baseline from `#3677`:
  - same-shape pure-JAX DeepEP transport now sits within `1.10x` to `1.54x` of Torch transport on H100x8
  - the benchmark-level reintegration question is still unanswered
- one-cell transport attribution on the authoritative `random, topk=2` full-shape cell is complete:
  - JAX: `layout_s=0.000205`, `dispatch_combine_cached_s=0.000700`, `step_s=0.000725`, `45.17M tokens/s`
  - Torch: `layout_s=0.000025`, `dispatch_combine_cached_s=0.000450`, `dispatch_combine_full_s=0.000491`, `66.77M tokens/s`
  - this means layout is not the dominant remaining JAX/Torch delta on the authoritative cell
- benchmark reintegration work landed locally:
  - new `deepep_transport` kernel added to `bench_moe_hillclimb.py`
  - current benchmark-lane fixes already made:
    - launcher switched to the newer working DeepEP ref
    - launcher installs `nodejs` / `npm` so Iris can regenerate protobufs during `uv sync`
    - launcher now uses the stronger DeepEP Python-module load mode instead of the weaker extension-only load mode
- the stronger benchmark-lane load path reached the real benchmark body, but full `deepep_transport` still failed at runtime with:
  - `jax.errors.JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- narrowing controls already completed inside the same hillclimb harness:
  - `shared_expert_dim=0` still fails with the same error
  - `deepep_transport_identity` succeeds
  - `deepep_transport_assignments_identity` succeeds
  - a first trustworthy consumed-intermediate probe, `deepep_transport_first_ragged_dot_probe`, fails again with the same error
- current best factual narrowing:
  - raw DeepEP dispatch/combine transport works in the hillclimb process
  - local assignment pack/collapse also works
  - the remaining frontier is now the first live local `ragged_dot(...)` above DeepEP dispatch inside the same compiled hillclimb graph
