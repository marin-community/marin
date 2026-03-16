# JAX DeepEP Autodiff And Scaling: Research Logbook

## Scope
- Goal: add JAX autodiff support to the DeepEP transport path used by the exact-cap reintegrated MoE benchmark, then rerun the decisive fixed-shape `forward_backward` comparison and the requested token sweep.
- Primary metric(s): `step_s` and `tokens/s` on H100x8 for the reintegrated JAX full-layer MoE benchmark, with matched Torch full-layer DeepEP numbers where applicable.
- Constraints:
  - zero Torch in the JAX step path
  - preserve the sealed `#3711` forward exact-cap behavior as the baseline
  - use the isolated CoreWeave lane for H100x8 runs
- GitHub issue: https://github.com/marin-community/marin/issues/3717

## Baseline
- Date: 2026-03-16
- Prior sealed issue: `#3711`
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
  - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
- Baseline numbers:
  - fixed-shape JAX exact-cap forward (`tokens=32768`, `EP=8`, H100x8):
    - `random, topk=2`: `current=0.003876`, `deepep_transport_capped_prewarmed=0.003325`, `1.17x`
    - `random, topk=8`: `current=0.011067`, `deepep_transport_capped_prewarmed=0.009854`, `1.12x`
    - `runs, topk=2`: `current=0.003867`, `deepep_transport_capped_prewarmed=0.003785`, `1.02x`
    - `runs, topk=8`: `current=0.011085`, `deepep_transport_capped_prewarmed=0.010062`, `1.10x`
  - larger-token JAX exact-cap forward wins from `#3711`:
    - `tokens=262144`, `topk=2`: `1.23x` (`random`), `1.18x` (`runs`)
    - `tokens=524288`, `topk=2`: `1.24x` (`random`), `1.21x` (`runs`)
    - `tokens=1048576`, `topk=2`: `1.40x` (`random`), `1.26x` (`runs`)
    - `tokens=262144`, `topk=8`: `1.20x` (`random`), `1.24x` (`runs`)
    - `tokens=524288`, `topk=8`: `1.23x` (`random`), `1.21x` (`runs`)
  - fixed-shape Torch full-layer DeepEP baseline (`tokens=32768`, H100x8):
    - `topk=2`: `alltoall=34.160 ms`, `deepep=11.586 ms`, `2.95x`
    - `topk=8`: `alltoall=34.838 ms`, `deepep=11.025 ms`, `3.16x`
  - current blocker for JAX `forward_backward`:
    - `ValueError: The FFI call to levanter_deepep_dispatch_intranode cannot be differentiated.`

## Experiment Log
### 2026-03-16 18:00 - Kickoff
- Hypothesis:
  - the remaining missing JAX `forward_backward` benchmark column is blocked by missing AD definitions around the DeepEP dispatch/combine custom calls, not by another transport or benchmark integration bug.
- Command:
  - research thread kickoff; no benchmark command yet
- Config:
  - new branch/worktree from sealed `#3711` snapshot `8a1b629e0dc8ab904fa0eca47b4369d59d34c77d`
  - branch: `research/moe-jax-deepep-autodiff-scaling`
  - worktree: `/Users/romain/marin-wt/moe-jax-deepep-autodiff-scaling`
- Result:
  - initialized the new research logbook for the AD/scaling follow-up thread
- Interpretation:
  - the new thread starts from a narrow, concrete blocker rather than a broad benchmark mystery
- Next action:
  - create the experiment issue and then inspect the current transport FFI path to decide where to add JAX AD support
