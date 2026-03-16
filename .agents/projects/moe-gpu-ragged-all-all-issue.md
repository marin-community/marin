## Description

This experiment revisits the MoE Expert Parallelism benchmark thread from #2710, but runs the focused Grug MoE harness on GPU and explicitly evaluates the existing `ragged_all_to_all` path in the harness as the closest in-repo analog to MaxText `ragged_all_all`.

The goal is not to change production defaults immediately. The goal is to produce an apples-to-apples GPU comparison on the same fixed shapes, routing distributions, and EP sizes so we can answer whether the current ring EP path still dominates on GPU or whether `ragged_all_to_all` changes the tradeoff.

## Hypothesis or Goal

- Hypothesis: GPU topology and collective lowering may make the harness `ragged_a2a` kernel materially more competitive with the current ring EP path than it was on TPU.
- Goal: produce reproducible GPU comparison tables for `current` vs `ragged_a2a`, starting from the sealed `grug-moe-ep-ring-20260307` harness snapshot and fixed-shape baseline.

### Links

* Prior experiment issue: https://github.com/marin-community/marin/issues/2710
* Prior sealed tag: https://github.com/marin-community/marin/tree/grug-moe-ep-ring-20260307
* Research logbook: `.agents/logbooks/moe-gpu-ragged-all-all.md`

## Results

Confidence: `replicated` for this fixed-shape H100x8 matrix.

Operational status:
- The CoreWeave Iris cluster is up.
- `iris cluster status` reports the controller healthy and one healthy `h100-8x` worker.
- A raw H100 smoke task from the sealed worktree successfully initialized JAX and saw 8 CUDA devices.
- The benchmark results below were collected from the sealed `grug-moe-ep-ring-20260307` worktree on one H100x8 host.

Repro note:
- A normal `iris job run` from this tagged worktree hit a task setup-path issue (`No pyproject.toml found in current directory or any parent directory`).
- The benchmark was therefore launched through an equivalent raw Iris `LaunchJobRequest` that stages the worktree under `/app`, runs `uv sync --quiet --link-mode symlink --python 3.11 --all-packages --no-group dev --extra gpu`, and then executes the benchmark loop there.

Completed GPU measurements (`forward_backward`, fixed shape `tokens=32768 hidden=2048 mlp_dim=768 experts=128 shared_expert_dim=2048`, EP sweep `1,2,4,8`):

- `topk=2`, `distribution=random`
  - `current`: `52.802 / 35.494 / 18.644 / 11.044 ms`
  - `ragged_a2a`: `52.828 / 60.547 / 30.888 / 16.272 ms`
- `topk=2`, `distribution=runs`
  - `current`: `52.992 / 36.509 / 19.263 / 10.986 ms`
  - `ragged_a2a`: `53.594 / 62.138 / 33.226 / 18.999 ms`
- `topk=8`, `distribution=random`
  - `current`: `215.920 / 139.003 / 66.969 / 34.307 ms`
  - `ragged_a2a`: `213.717 / 234.108 / 116.771 / 59.646 ms`
- `topk=8`, `distribution=runs`
  - `current`: `210.857 / 134.295 / 70.494 / 34.212 ms`
  - `ragged_a2a`: `219.390 / 242.572 / 117.599 / 61.277 ms`

## Decision Log

- 2026-03-13: Treat the existing harness `ragged_a2a` path as the closest in-repo analog to MaxText `ragged_all_all` for the first GPU comparison.
- 2026-03-13: Keep `current` as the recommended path for this workload on GPU. On this fixed H100x8 matrix, `ragged_a2a` did not produce a single win at `EP > 1`.

## Negative Results

- CPU is not a viable backend for this path: `jax.lax.ragged_all_to_all` failed on forced host devices with `UNIMPLEMENTED`.
- On this GPU matrix, `ragged_a2a` lost on every non-control EP point and was about `1.47x` to `1.81x` slower than `current`.

## Conclusion

For this fixed-shape H100x8 comparison, the current ring EP path remains clearly better than `ragged_a2a` on GPU.

- `EP=1` behaves as the expected control and is effectively equal between the two paths.
- For every measured point with `EP > 1`, `current` is faster.
- The slowdown direction is consistent across both tested routing distributions and both tested `topk` values.

This does not rule out a different result for a more MaxText-faithful implementation or a materially different shape/hardware regime, but it does not support changing the default path for this workload.
