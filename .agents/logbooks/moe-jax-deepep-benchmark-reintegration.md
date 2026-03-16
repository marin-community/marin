# JAX DeepEP Benchmark Reintegration: Research Logbook

## Scope
- Goal: determine whether the working pure-JAX DeepEP transport path from `#3677` materially changes the original fixed-shape JAX benchmark outcome, and localize any remaining JAX/Torch transport delta before broadening the benchmark matrix.
- Primary metric(s):
  - fixed-shape H100x8 forward wall time on the original `#3633`-style benchmark
  - phase-level timing split for layout vs transport vs surrounding bookkeeping
  - matched Torch-vs-JAX transport delta on the authoritative worst-gap cell
- Constraints:
  - Use CoreWeave Iris H100x8 via `~/llms/cw_ops_guide.md`.
  - Reuse the isolated CoreWeave namespace/prefix lane when practical.
  - Keep the next round gated: one-cell attribution first, benchmark reintegration second, broader sweeps only if warranted.
  - Update the GitHub issue only for major milestones.
- Experiment issue: https://github.com/marin-community/marin/issues/3711

## Baseline
- Date: 2026-03-16
- Code refs:
  - sealed prior thread: `research/moe-jax-megatron-root-cause` @ `6baa08edbd8ae9a782d0070a3b7cf0e1f38ba005`
  - key JAX transport files carried forward:
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
    - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
    - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
- Baseline numbers:
  - corrected same-shape transport matrix from `#3677`:
    - `random, topk=2`: Torch `67.44M`, JAX `43.79M`, Torch/JAX `1.54x`
    - `runs, topk=2`: Torch `35.90M`, JAX `29.15M`, Torch/JAX `1.23x`
    - `random, topk=8`: Torch `30.85M`, JAX `25.25M`, Torch/JAX `1.22x`
    - `runs, topk=8`: Torch `25.25M`, JAX `22.94M`, Torch/JAX `1.10x`
  - original fixed-shape negative JAX result (`#3665`) remained layout-only and did not exercise DeepEP transport.

## Initial Hypotheses
- The remaining JAX/Torch transport gap is now small enough that it may mostly be accounted for by non-transport work in the timed region: layout, reductions, wrapper/bookkeeping, or custom-call boundary overhead.
- If the corrected pure-JAX transport is reinserted into the original fixed-shape benchmark, the old negative JAX result should materially improve relative to `deepep_layout_ragged_a2a`.
- `topk=8` drift is likely secondary to the main benchmark-level question and should not gate the first reintegration controls.

## Stop Criteria
- Produce a benchmark-level control on the original fixed-shape JAX path that compares:
  - `current`
  - `ragged_a2a`
  - `deepep_layout_ragged_a2a`
  - full pure-JAX DeepEP transport
- Produce a one-cell attribution split that localizes the remaining JAX/Torch transport gap well enough to decide whether the next step should be transport tuning or broader benchmark reintegration.
- Update the issue body with a concise decision-quality conclusion and a next-step ordering.

## Experiment Log
