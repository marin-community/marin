## Description

This experiment follows the sealed JAX custom-call DeepEP negative result in `#3665`, but it intentionally changes methodology. The new question is not whether a layout-only custom call helps Marin's existing benchmark; it is whether Megatron-LM's own MoE-layer benchmarking approach changes where `alltoall`, `deepep`, and `hybridep` win once the workload is scaled in a Qwen3-like way.

The benchmark target is Megatron-LM's `tests/functional_tests/test_cases/common/moe_perf` path:
- full `MoELayer` forward/backward timing
- fixed input tensor reused across iterations
- force-balanced random router logits for benchmark stability
- dummy GEMM before the timed MoE step to hide router-launch overhead
- manual-GC / repeated timed iterations to reduce jitter

This thread is deliberately anchored to the largest official Qwen3 MoE models rather than arbitrary proxy shapes. The two primary reference configs are:
- `Qwen3-30B-A3B`
- `Qwen3-235B-A22B`

The goal is to see how dispatcher ranking changes with scale in:
1. batch tokens
2. hidden dimension
3. expert hidden size
4. number of experts
5. top-k

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior JAX custom-call issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior JAX custom-call seal tag: https://github.com/marin-community/marin/tree/moe-deepep-jax-layout-ffi-h100-matrix-20260314
* Megatron-LM MoE README: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
* Megatron-LM `moe_perf` benchmark: https://github.com/NVIDIA/Megatron-LM/blob/main/tests/functional_tests/test_cases/common/moe_perf/__main__.py
* Megatron-LM `moe_perf` cases: https://github.com/NVIDIA/Megatron-LM/blob/main/tests/functional_tests/test_cases/common/moe_perf/test_cases.py
* Qwen3-30B-A3B config: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
* Qwen3-235B-A22B config: https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
* Research logbook: `.agents/logbooks/moe-megatron-qwen-scale.md`

## Hypothesis or Goal

- Hypothesis: the fixed-shape DeepEP results so far do not fully capture the scale regime Megatron optimizes for; once the benchmark matches Megatron's full MoE-layer methodology and Qwen3-like shapes, the dispatcher ranking may change.
- Goal: run a Megatron-style H100x8 MoE-layer benchmark that compares `alltoall`, `deepep`, and `hybridep` on Qwen3-shaped proxies.
- Goal: measure where the ranking changes as we scale batch tokens, hidden size, expert hidden size, expert count, and top-k.
- Goal: keep the matrix interpretable by varying one scale axis at a time around official Qwen3 anchors.

## Results

In progress as of 2026-03-14.

Current state:
- The prior JAX custom-call experiment is sealed and closed.
- CoreWeave Iris H100x8 ops guidance is reloaded from `~/llms/cw_ops_guide.md`.
- The methodology reference is identified: Megatron-LM `moe_perf`, not the lower-level DeepEP transport tests.
- The repo-local Megatron-style mirror is now runnable on CoreWeave H100x8.
- The launcher required four pod-side bring-up fixes before the Megatron path ran:
  - explicit cuDNN include/lib exports for Transformer Engine
  - explicit `NVSHMEM_DIR` export for DeepEP
  - RDMA dev packages (`libibverbs-dev`, `rdma-core`) for the NVSHMEM-enabled DeepEP build
  - an unversioned `libnvshmem_host.so` symlink for the NVSHMEM wheel layout DeepEP expects
- The official Qwen3 MoE anchor configs are identified:
  - `Qwen3-30B-A3B`: `hidden=2048`, `moe_ffn_hidden=768`, `num_experts=128`, `topk=8`, `layers=48`
  - `Qwen3-235B-A22B`: `hidden=4096`, `moe_ffn_hidden=1536`, `num_experts=128`, `topk=8`, `layers=94`

Latest exploratory smoke (`qwen3_30b_a3b_anchor`, `warmup=1`, `measure=2`):
- `alltoall`: `forward=23.48 ms`, `backward=40.26 ms`
- `deepep`: `forward=48.13 ms`, `backward=49.11 ms`
- `hybridep`: `forward=25.23 ms`, `backward=37.12 ms`

Interpretation of the smoke:
- The Megatron-style benchmark is operational on H100x8.
- `deepep` is clearly behind at this smoke-level evidence.
- `alltoall` and `hybridep` are close enough that the short smoke is not strong evidence by itself.
- The smoke still emitted the expected HybridEP warning about float32 router probs, so the full sweep should use `moe_router_dtype=\"fp32\"` to match upstream `moe_perf` defaults.

Planned first matrix:
- Dispatchers: `alltoall`, `deepep`, `hybridep`
- Base sequence length: `4096`
- Anchor cases:
  - `qwen3_30b_a3b`
  - `qwen3_235b_a22b`
- One-axis sweeps:
  - batch tokens: `micro_batch in {1, 2, 4}`
  - hidden size: `hidden in {2048, 3072, 4096}`
  - expert hidden size: `moe_ffn_hidden in {768, 1152, 1536}`
  - num experts: `num_experts in {32, 64, 128}`
  - top-k: `topk in {2, 4, 8}`

Implementation decision:
- Use a faithful repo-local mirror of Megatron `moe_perf` semantics, launched directly on CoreWeave H100x8 via KubernetesRuntime.
- Do not regress to the earlier dispatch-only benchmark methodology.

## Decision Log

- 2026-03-14: seal and close `#3665` before starting the next thread.
- 2026-03-14: use Megatron-LM `moe_perf` as the methodology reference for the new experiment.
- 2026-03-14: use official Qwen3 MoE configs as the scale anchors for the sweep.
- 2026-03-14: vary one scale axis at a time for the first milestone to keep the interpretation clean.
- 2026-03-14: use a repo-local Megatron-style harness rather than trying to run upstream pytest machinery directly in the pod.
- 2026-03-14: enable DeepEP NVSHMEM support explicitly in the pod and add the minimal RDMA/NVSHMEM wheel workarounds needed to make Megatron `fused_a2a` runnable on H100x8.

## Negative Results

- The sealed JAX layout-only custom-call experiment (`#3665`) showed that replacing only the layout metadata producer does not move the distributed H100x8 benchmark. That negative result is the reason this new thread pivots to Megatron's full MoE-layer benchmark methodology instead of another metadata-only variant.

## Conclusion

Not complete yet. The Megatron-style benchmark now runs on H100x8; the next milestone is the longer fp32-router Qwen-patterned scaling sweep that turns the current smoke result into a usable comparison table.
