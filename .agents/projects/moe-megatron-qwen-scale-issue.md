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

Complete as of 2026-03-14.

Final state:
- The prior JAX custom-call experiment is sealed and closed.
- CoreWeave Iris H100x8 ops guidance was reloaded from `~/llms/cw_ops_guide.md`.
- The methodology reference stayed faithful to Megatron-LM `moe_perf`: full `MoELayer`, fixed input tensor, force-balanced routing, dummy GEMM, manual GC, and `moe_router_dtype="fp32"`.
- The repo-local Megatron-style mirror ran successfully on CoreWeave H100x8 after four pod-side bring-up fixes:
  - explicit cuDNN include/lib exports for Transformer Engine
  - explicit `NVSHMEM_DIR` export for DeepEP
  - RDMA dev packages (`libibverbs-dev`, `rdma-core`) for the NVSHMEM-enabled DeepEP build
  - an unversioned `libnvshmem_host.so` symlink for the NVSHMEM wheel layout DeepEP expects
- The official Qwen3 MoE anchors remained:
  - `Qwen3-30B-A3B`: `hidden=2048`, `moe_ffn_hidden=768`, `num_experts=128`, `topk=8`
  - `Qwen3-235B-A22B`: `hidden=4096`, `moe_ffn_hidden=1536`, `num_experts=128`, `topk=8`

Authoritative totals (`forward + backward`, milliseconds):

| Case | alltoall | deepep | hybridep | Winner |
| --- | ---: | ---: | ---: | --- |
| `qwen3_30b_a3b_anchor` | 34.95 | 12.41 | 18.05 | `deepep` |
| `qwen3_235b_a22b_anchor` | 25.62 | 11.75 | 13.08 | `deepep` |
| `qwen3_batch_mb2` | 33.77 | 12.42 | 16.61 | `deepep` |
| `qwen3_batch_mb4` | 35.27 | 18.09 | 21.09 | `deepep` |
| `qwen3_hidden_3072` | 45.10 | 10.55 | 19.46 | `deepep` |
| `qwen3_hidden_4096` | 27.22 | 9.71 | 14.59 | `deepep` |
| `qwen3_expert_1152` | 30.56 | 10.13 | 12.50 | `deepep` |
| `qwen3_expert_1536` | 24.62 | 9.59 | 11.38 | `deepep` |
| `qwen3_topk_2` | 31.90 | 9.04 | 14.41 | `deepep` |
| `qwen3_topk_4` | 33.48 | 9.40 | 15.73 | `deepep` |
| `qwen3_experts_32` | 16.12 | 22.82 | 11.91 | `hybridep` |
| `qwen3_experts_64` | 25.38 | 15.52 | 14.55 | `hybridep` |

Important interpretation notes:
- The expert-count axis above comes from a focused rerun that supersedes the earlier rough live scrape. The original full-matrix pod auto-deleted before the raw `32/64 experts` rows could be re-scraped, so those two cases were rerun with local `tee` logging.
- On that authoritative rerun, `deepep` showed large variance at smaller expert counts:
  - `qwen3_experts_32`: `forward_std=29.98 ms`, `backward_std=15.24 ms`
  - `qwen3_experts_64`: `forward_std=11.44 ms`, `backward_std=15.45 ms`
- So the best reading is not "DeepEP always wins." The stronger statement is:
  - `deepep` wins the `128`-expert Qwen-like regime and the batch/hidden/expert-size/top-k sweeps.
  - `hybridep` wins the smaller-expert `32` and `64` slices on mean wall time and is materially more stable there.

Comparison against the sealed fixed-shape GPU benchmark in `#3633`:
- `#3633` used a narrower dispatch-only benchmark and concluded the current path beat ragged dispatch on H100x8.
- This Megatron-style full `MoELayer` benchmark changes the answer materially:
  - for the Qwen-like `128`-expert path, `deepep` is strongly better than `alltoall`
  - for smaller expert counts, `hybridep` becomes the best measured choice
- So the `#3633` result should not be generalized to full Megatron-style MoE-layer behavior.

## Decision Log

- 2026-03-14: seal and close `#3665` before starting the next thread.
- 2026-03-14: use Megatron-LM `moe_perf` as the methodology reference for the new experiment.
- 2026-03-14: use official Qwen3 MoE configs as the scale anchors for the sweep.
- 2026-03-14: vary one scale axis at a time for the first milestone to keep the interpretation clean.
- 2026-03-14: use a repo-local Megatron-style harness rather than trying to run upstream pytest machinery directly in the pod.
- 2026-03-14: enable DeepEP NVSHMEM support explicitly in the pod and add the minimal RDMA/NVSHMEM wheel workarounds needed to make Megatron `fused_a2a` runnable on H100x8.
- 2026-03-14: rerun the expert-count slice (`32/64` experts) with local logging and treat that rerun as authoritative because the earlier live scrape was incomplete.

## Negative Results

- The sealed JAX layout-only custom-call experiment (`#3665`) showed that replacing only the layout metadata producer does not move the distributed H100x8 benchmark. That negative result is the reason this new thread pivots to Megatron's full MoE-layer benchmark methodology instead of another metadata-only variant.
- `deepep` is not uniformly dominant. On the small-expert rerun (`32` and `64` experts), it shows large mean-time variance and loses to `hybridep`.

## Conclusion

Complete. The Megatron-style H100x8 MoE-layer benchmark shows a different dispatcher ranking than `#3633`.

Bottom line:
- For the Qwen-like `128`-expert regime, use `deepep` as the best-performing baseline.
- For smaller expert counts (`32` and `64` in this sweep), `hybridep` is the best measured path and `deepep` becomes variance-sensitive.
- `alltoall` is not the preferred path for this Megatron-style workload once the benchmark is scaled and measured the way Megatron itself does.
