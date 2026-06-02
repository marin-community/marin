# Levanter TPU Inference Parity: Research Notes

## Scope

Goal: design the path for Levanter inference to reach practical performance parity with `vllm-tpu` on Qwen3 8B by reusing or matching `tpu-inference` TPU kernels while preserving Levanter's native JAX model and checkpoint path.

Parity means an apples-to-apples comparison against the pinned Marin `vllm` extra on the same TPU generation, model, dtype, context length, batch mix, and decoding policy. The first target is Qwen3 8B bf16 decode-heavy RL rollout traffic on v5p.

## In-repo Findings

- Marin currently carries `vllm-tpu==0.19.0` and `tpu-inference==0.19.0` in the `vllm` optional dependency group: [lib/marin/pyproject.toml](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/pyproject.toml#L190-L194).
- RL jobs already choose between Levanter and vLLM inference with `inference_type: Literal["levanter", "vllm"]`: [lib/marin/src/marin/rl/job_config.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/job_config.py#L110-L135). The Levanter default path constructs `InferenceServerConfig` with `InferenceEngineConfig`, page size 128, and `hbm_utilization=0.5`: [lib/marin/src/marin/rl/job_config.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/job_config.py#L231-L253).
- The vLLM RL backend imports `vllm.LLM`, disables V1 multiprocessing for direct weight access, and has TPU-specific `tpu_inference` registry patching: [lib/marin/src/marin/rl/environments/inference_ctx/vllm.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py#L30-L65), [lib/marin/src/marin/rl/environments/inference_ctx/vllm.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py#L212-L255).
- Native vLLM server setup already centralizes TPU/JAX cache environment defaults, including `MODEL_IMPL_TYPE=vllm`, `JAX_ENABLE_COMPILATION_CACHE`, `JAX_COMPILATION_CACHE_DIR`, and `VLLM_XLA_CACHE_PATH`: [lib/marin/src/marin/inference/vllm_server.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/inference/vllm_server.py#L270-L299).
- Levanter-to-vLLM weight transfer pads attention projection head dimensions to multiples of 128 for Pallas kernels: [lib/marin/src/marin/rl/weight_utils.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/weight_utils.py#L46-L69). vLLM utilities already map `Qwen/Qwen3-8B` through the Qwen mapping: [lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py#L89-L97).
- Levanter inference already has the right high-level shape: an `InferenceEngineConfig` with page size, max seqs, prefill size, decode rounds, and queue sizing: [lib/levanter/src/levanter/inference/engine.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/engine.py#L43-L120).
- Prefill and decode both call `model.decode(...)`, then sample and enqueue the next tokens. Prefill hot path: [lib/levanter/src/levanter/inference/engine.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/engine.py#L396-L457). Decode loop hot path: [lib/levanter/src/levanter/inference/engine.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/engine.py#L690-L715).
- The engine can already dump generation and prefill jaxprs/HLO, which should become part of the parity benchmark artifact bundle: [lib/levanter/src/levanter/inference/engine.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/engine.py#L1258-L1312).
- Scheduler shape is load-bearing. `SequenceTable.allocate_for_seq` computes dense per-sequence segments, allocates zero-ref pages, and builds `PageBatchInfo`: [lib/levanter/src/levanter/inference/jit_scheduler.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/jit_scheduler.py#L212-L340). `TokenQueue.pack_next_sequence` dequeues up to `max_tokens`, rolls the queue, and stable-sorts by slot ID before decode: [lib/levanter/src/levanter/inference/jit_scheduler.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/jit_scheduler.py#L1062-L1115).
- The KV cache is an interleaved `[page, slot, 2 * kv_heads, head_size]` buffer, with scalar page/slot updates in `kv_update_unified_prefix`: [lib/levanter/src/levanter/layers/kv_cache.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/kv_cache.py#L33-L100), [lib/levanter/src/levanter/layers/kv_cache.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/kv_cache.py#L139-L155).
- `PageBatchInfo` already carries exactly the metadata a ragged paged kernel needs: slot IDs, page indices, sequence lengths, cumulative query lengths, number of sequences, and new token destinations: [lib/levanter/src/levanter/inference/page_table.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/page_table.py#L62-L94).
- Qwen3 support exists in Levanter and reuses the Llama-style LM-head model path: [lib/levanter/src/levanter/models/qwen.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/models/qwen.py#L297-L391). Qwen3 roundtrip tests cover the HF conversion path: [lib/levanter/tests/test_qwen3.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/tests/test_qwen3.py#L39-L87).
- `Attention.paged_decode` computes q/k/v, updates the KV cache, and calls `ragged_paged_attention`: [lib/levanter/src/levanter/layers/attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/attention.py#L1767-L1809).
- `ragged_paged_attention` tries `jax.experimental.pallas.ops.tpu.ragged_paged_attention` on TPU and falls back to a reference implementation: [lib/levanter/src/levanter/layers/attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/attention.py#L21-L26), [lib/levanter/src/levanter/layers/attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/attention.py#L1994-L2050). The TPU wrapper pads head size to 128, masks invalid page metadata, applies scale outside the kernel, and calls through `shard_map`: [lib/levanter/src/levanter/layers/attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/attention.py#L2070-L2167).
- Existing paged attention tests cover single-sequence, multi-sequence, and incremental cases against a naive reference: [lib/levanter/tests/inference/test_paged_attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/tests/inference/test_paged_attention.py#L37-L181).
- Marin has a TPU vLLM smoke test that runs a small GCS Llama checkpoint with `enforce_eager` and `max_model_len`: [tests/vllm/test_llm_inference.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/tests/vllm/test_llm_inference.py#L33-L42).
- The served eval design notes that TPU vLLM generation worked through `/v1/chat/completions` and `/v1/completions`, while prompt-logprob scoring did not pass at that time: [.agents/projects/served_lm_eval.md](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/.agents/projects/served_lm_eval.md#L161-L170).

## Prior Art

- The `vllm-project/tpu-inference` repository says vLLM TPU is now powered by `tpu-inference`, a hardware plugin unifying JAX and PyTorch under a single lowering path, and its support matrix lists `meta-llama/Llama-3.1-8B-Instruct` as passing unit, correctness, and performance tests: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference).
- vLLM's TPU profiling docs split profiling into prefill and decode token shapes, with warmups and TPU profile capture. This is the right comparison shape for Levanter's harness: [vLLM TPU Profiling](https://docs.vllm.ai/en/v0.11.1/examples/offline_inference/profiling_tpu/).
- The Ragged Paged Attention paper describes the TPU kernel target: fine-grained tiling, ragged paged memory, fused KV-cache update plus attention, and specialized decode, prefill, and mixed kernels. It reports Llama 3 8B results on TPU7x: [arXiv:2604.15464](https://arxiv.org/abs/2604.15464).
- The original PagedAttention paper frames why this matters for serving: high throughput depends on batching, while KV cache memory grows dynamically; paging reduces fragmentation and supports sharing: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- JAX Pallas is the supported path for custom TPU kernels lowering through Mosaic: [JAX Pallas TPU docs](https://docs.jax.dev/en/latest/pallas/tpu/).

## Design Pressure

- Levanter already has paged metadata and native Qwen3 decode via the Llama-style path. The likely performance gap is not from lacking pages, but from kernel selection, fusion boundaries, compile bucket discipline, and scheduler shape choices.
- The current KV update is outside the ragged attention kernel. RPA prior art explicitly fuses KV update with attention, so parity likely requires either importing `tpu-inference` kernels that own the update or adding a Levanter wrapper that can present the same input contract.
- The benchmark must compare serving-shaped prefill and decode, not only the attention kernel. Scheduler/queue choices can hide or erase kernel wins.

## Phase 1 Smoke Results

2026-05-31: ran the Phase 1 harness on Iris as `/dlwh/qwen3-parity-vllm-v5e-v6e4-patched-smoke-20260531-0122`.

Command shape:

```bash
uv run iris --config=lib/iris/config/marin.yaml job run \
  --tpu v5litepod-4,v6e-4 \
  --enable-extra-resources \
  --memory 80GB \
  --disk 5GB \
  -e HF_HOME /dev/shm/hf-qwen3-parity \
  -e VLLM_XLA_CACHE_PATH /dev/shm/vllm-xla-qwen3-parity \
  -- \
  bash -lc 'uv run --package marin-core --extra tpu --extra vllm python lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py ...'
```

Observed result for `Qwen/Qwen3-8B`, backend `vllm`, case `decode_b8_i1_o128_n1`, `tensor_parallel_size=4`, one warmup and one measured round:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `vllm-tpu` | 8 | 1 | 1858.05 | 1872.57 | 0.551 | 543.8 | 546.9 |

Environment captured by the harness: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `vllm-tpu==0.19.0`, `tpu-inference==0.19.0`.

Two harness fixes came from this smoke:

- `--backend both` must run backends sequentially. Keeping vLLM and Levanter servers alive at the same time on one TPU is not a valid apples-to-apples setup.
- The vLLM path must not call `jax.devices()` from the harness process while the vLLM server owns libtpu. Version metadata is collected via package metadata; device probing is skipped for vLLM and left available for the Levanter in-process backend.

2026-05-31: ran the matching Levanter smoke on Iris as `/dlwh/qwen3-parity-levanter-v5e-v6e4-smoke-20260531-0132` with `--tpu v5litepod-4,v6e-4`, `--max-seqs 8`, and the same `decode_b8_i1_o128_n1` case.

The job got through dependency setup and server startup, then failed on the first warmup completion:

```text
RESOURCE_EXHAUSTED: E0100: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 556.00M.
That was not possible. There are 289.98M free.; (0x1x0_HBM0)
```

The last successful Levanter server log before the failure reported `Max tokens per seq: 4096, per batch: 142336`. That points at the automatic KV cache sizing leaving too little HBM margin on a 4-chip v6e shape, not at the HTTP harness itself.

Immediate follow-up:

- `/dlwh/qwen3-parity-levanter-v5e-v6e4-maxpages-smoke-20260531-0140` retried the same case with `--max-pages 256`, but failed before Python started because `UV_CACHE_DIR=/dev/shm/uv-cache` ran out of space extracting `tokenizers==0.22.2`.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-maxpages-diskuv-20260531-0143` retried the same capped run with `UV_CACHE_DIR=/tmp/uv-cache`, but failed in the outer Iris container build with `RuntimeError: uv-build was not properly installed`.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-maxpages-inneruv-20260531-0145` retried the same capped run without outer Iris `--extra` flags. The entrypoint still requested TPU resources, and dependencies were resolved by the inner `uv run --package marin-levanter --extra tpu --extra serve`. This got to the prefill compile, reduced the reported batch token capacity to `32768`, and then failed with a scoped VMEM OOM in `ragged_paged_attention_kernel`:

  ```text
  RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom
  bf16[4096,32,128] ... ragged_paged_attention_kernel
  Scoped allocation with size 33.50M and limit 32.00M exceeded scoped vmem limit by 1.50M.
  ```

  For the decode-heavy matrix the prompts are one token, so compiling prefill for `max_prefill_size=4096` is not representative of the target workload.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-prefill128-20260531-0148` retries with `--max-pages 256 --max-prefill-size 128`.
  It failed with the same scoped VMEM OOM, but the error body is now captured by the harness:

  ```text
  response body: {"detail":"RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom:
  ... bf16[128,32,128] ... ragged_paged_attention_kernel ...
  Scoped allocation with size 33.50M and limit 32.00M exceeded scoped vmem limit by 1.50M.
  ```

  Since the scoped allocation stayed at 33.50 MiB after shrinking the prefill token axis, the immediate blocker is the current Levanter/JAX ragged paged attention kernel shape on this TPU target, not just prompt length bucketing.
- A combined `v5p-8,v6e-8` retry was rejected by Iris because the variants have different per-VM chip counts. `/dlwh/qwen3-parity-levanter-v5p8-prefill128-20260531-0153` retries the same capped prefill run on `v5p-8` only, which is the primary target TPU generation for this parity effort.
- The JAX RPA kernel exposes `num_kv_pages_per_block`, `num_queries_per_block`, and `vmem_limit_bytes`; Levanter now threads these through `AttentionConfig` and the benchmark harness. `/dlwh/qwen3-parity-levanter-v5e-v6e4-rpa64-20260531-0158` retried the 4-chip smoke with `--rpa-num-kv-pages-per-block 64`, but JAX rejected it because the value must be no larger than `pages_per_seq=32`.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-rpa16-20260531-0201` retried with `--rpa-num-kv-pages-per-block 16`. It still failed with the same scoped VMEM OOM:

  ```text
  response body: {"detail":"RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom:
  ... bf16[128,32,128] ... ragged_paged_attention_kernel ...
  Scoped allocation with size 33.50M and limit 32.00M exceeded scoped vmem limit by 1.50M.
  ```

  The failing allocation did not move after shrinking the KV page block, so the next smallest test is the direct JAX RPA `vmem_limit_bytes` compiler parameter.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-vmem64-20260531-0906` retried the 4-chip smoke with `--rpa-num-kv-pages-per-block 16 --rpa-vmem-limit-bytes 67108864`. Iris placed it on a `v6e-4` worker and the job succeeded. This proves the immediate compile blocker is the scoped VMEM limit, not a fundamental shape rejection for Qwen3 8B decode-heavy serving.

  The run wrote artifacts only to worker-local `/dev/shm`, so exact `summary.md` and `env.json` were not recoverable from logs. The server logs still show the measured round:

  ```text
  Initial prefill and extraction took 36.333s
  Decode iter: total 0.930s ... 275.18 tok/s, 256 new
  Decode iter: total 0.930s ... 275.30 tok/s, 256 new
  Decode iter: total 0.930s ... 275.37 tok/s, 256 new
  Decode iter: total 0.902s ... 274.83 tok/s, 248 new
  Batch completed in 40.14s, generated 1024 tokens
  ```

  That is roughly 25.5 generated tokens/s end-to-end for `decode_b8_i1_o128_n1`, versus the vLLM smoke's 1858 generated tokens/s on the same benchmark case. The next performance blocker is therefore no longer TPU RPA compilation; it is Levanter serving-path overhead around initial prefill/extraction and host-side per-round extraction/submit cadence. The harness now logs `summary.md` and `env.json` after writing them so future `/dev/shm` runs leave durable benchmark artifacts in Iris logs.
- `/dlwh/qwen3-parity-levanter-v5p8-prefill128-20260531-0153` retried the capped Levanter smoke on `v5p-8` without the VMEM override. It was preempted once, then failed with the same RPA scoped VMEM OOM, with v5p's default scoped limit at 16 MiB:

  ```text
  Scoped allocation with size 33.54M and limit 16.00M exceeded scoped vmem limit by 17.54M.
  ```

  `/dlwh/qwen3-parity-levanter-v5p8-vmem64-20260531-0915` is the direct v5p follow-up with the 64 MiB VMEM override and logged artifacts. It was preempted twice, but eventually succeeded on `TPU v5` with 4 visible devices:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter` | 8 | 1 | 319.38 | 321.88 | 3.206 | 3199.0 | 3203.0 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, device kind `TPU v5`, 4 visible devices. The measured run had `Initial prefill and extraction took 0.054s`, then four decode calls around 333 generated tokens/s. This is the first target-family Levanter number, and it makes a target-family vLLM baseline the next required apples-to-apples comparison.
- `/dlwh/qwen3-parity-vllm-v5p8-logged-20260531-0937` is the matching target-family vLLM baseline on `v5p-8` with `tensor_parallel_size=4`. It succeeded and produced the durable harness table:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `vllm-tpu` | 8 | 1 | 1514.17 | 1525.99 | 0.676 | 670.1 | 672.4 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`. Device probing was intentionally skipped in the harness because the vLLM server process owns libtpu. Against this target-family baseline, the current Levanter v5p run is at 21.1% of vLLM decode throughput and 21.1% of vLLM total throughput for `decode_b8_i1_o128_n1`.
- The next v5p check moved from `b8` to the more rollout-relevant `b32` regime. `/dlwh/qwen3-parity-vllm-v5p8-b32-20260531-0948` intentionally tried both `decode_b32_i1_o128_n1` and `decode_b32_i1_o128_n4`, but failed on the `n=4` warmup because vLLM rejects `n > 1` with greedy sampling:

  ```text
  n must be 1 when using greedy sampling, got 4.
  ```

  The harness now exposes `--temperature` and `--top-p` and fails fast before starting a vLLM server when a greedy vLLM run contains `n > 1`. The cloned-prefix `n=4` case will need a sampled apples-to-apples run.
- `/dlwh/qwen3-parity-vllm-v5p8-b32n1-20260531-0957` is the successful vLLM target-family baseline for `decode_b32_i1_o128_n1`:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b32_i1_o128_n1` | `vllm-tpu` | 32 | 1 | 4682.70 | 4719.28 | 0.875 | 850.6 | 854.7 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`; device probing skipped because vLLM may own libtpu.
- `/dlwh/qwen3-parity-levanter-v5p8-b32-20260531-0948` is the matching Levanter v5p run for `decode_b32_i1_o128_n1` plus the greedy Levanter clone-path case `decode_b32_i1_o128_n4`. It was preempted once and then succeeded:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b32_i1_o128_n1` | `levanter` | 32 | 1 | 187.42 | 188.88 | 21.855 | 11292.4 | 21820.6 |
  | `decode_b32_i1_o128_n4` | `levanter` | 32 | 4 | 319.92 | 320.54 | 12.803 | 12796.3 | 12798.9 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices. The comparable `b32 n=1` Levanter result is 4.0% of vLLM decode throughput and 4.0% of vLLM total throughput. This is worse than the `b8 n=1` ratio, so the current Levanter/JAX RPA serving path is not scaling with more active rollout sequences the way vLLM TPU does. The run also shows request waves in the measured `n=1` case: one batch generated 2432 tokens in 11.21s, then a second generated 1664 tokens in 10.52s, which explains the wide p50/p90 latency split and points at scheduler/loop packing in addition to raw kernel speed.
- `/dlwh/qwen3-parity-levanter-v5p8-b32n1-batch1s-20260531-1012` tested synchronized request launch plus a 1s server batch timeout on the same v5p `decode_b32_i1_o128_n1` case. It was preempted twice and then succeeded:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b32_i1_o128_n1` | `levanter` | 32 | 1 | 104.47 | 105.28 | 39.208 | 39175.8 | 39193.2 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices. The measured run now completed as one batch (`Batch completed in 38.67s, generated 4096 tokens`) rather than the two request waves seen in the earlier b32 run, but throughput fell to 2.2% of the vLLM b32 baseline. The log still shows `Initial prefill and extraction took 25.987s`, so the immediate bottleneck after fixing request arrival skew is the Levanter prefill/extraction path, not only decode loop batching.
- `/dlwh/qwen3-parity-levanter-v5e-v6e4-vmem64-logged-20260531-0922` is the logged-artifact rerun on `v5litepod-4,v6e-4`. It succeeded on `TPU v5 lite` with 4 devices and produced the durable harness table:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter` | 8 | 1 | 31.30 | 31.55 | 32.714 | 32707.8 | 32711.4 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, device kind `TPU v5 lite`, 4 devices, no `vllm-tpu` or `tpu-inference` dependency in the Levanter-only environment. The measured run still shows `Initial prefill and extraction took 27.793s`, followed by four decode calls around 215 generated tokens/s within each call. This makes Levanter about 1.7% of the earlier vLLM smoke's generated-token throughput for this tiny decode-heavy case, with the obvious current bottleneck in host-call cadence plus prefill/extraction.
- The harness now exposes `--max-rounds` and `--max-tokens-per-round` so Phase 1 can deliberately bias Levanter toward decode-heavy rollout throughput. `/dlwh/qwen3-parity-levanter-v5e-v6e4-vmem64-rounds128-20260531-0928` is the direct follow-up with `--max-rounds 128`. It succeeded on `TPU v5 lite` after one preemption, but did not improve throughput:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter` | 8 | 1 | 31.30 | 31.55 | 32.715 | 32707.1 | 32711.3 |

  The measured run still had `Initial prefill and extraction took 27.769s`, then one decode call completed the remaining 1016 tokens. This says the obvious four-host-call decode cadence was not the dominant v5-lite bottleneck.
- For the one-token prompt regime, the baseline still used `--max-prefill-size 128`, so the prefill JIT executes a 128-token prefill buffer even though the actual request batch contains only 8 prompt tokens. `/dlwh/qwen3-parity-levanter-v5e-v6e4-vmem64-prefill8-rounds128-20260531-0933` tested `--max-prefill-size 8 --max-rounds 128` to cover both obvious serving-path overheads from the logged baseline: oversized prefill and four separate decode host calls. It succeeded on `TPU v5 lite` with only a small end-to-end gain:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter` | 8 | 1 | 34.44 | 34.71 | 29.735 | 29727.3 | 29732.1 |

  The measured run still had `Initial prefill and extraction took 24.820s`, followed by one decode call that produced 1016 tokens in 4.720s. This rules out the simple "prefill buffer too large" hypothesis for the v5-lite path and suggests the main gap is in the Levanter/JAX prefill/decode kernels or execution path rather than host-call count alone.

## Phase 2 Backend Boundary Notes

2026-05-31: added the first explicit paged-attention backend boundary under Levanter inference:

- `InferenceEngineConfig` now carries `tpu_paged_attention: TpuPagedAttentionConfig`, and the engine threads it through prefill and decode into Llama/Qwen/Apertus paged attention.
- `levanter.inference.tpu_kernels` owns backend selection for `AUTO`, `TPU_INFERENCE`, `JAX_RPA`, and `REFERENCE`. On TPU, `AUTO` expands to `TPU_INFERENCE`; on CPU/GPU, it expands to `REFERENCE`. Explicit TPU backends fail fast off TPU, and reference fallback on TPU raises unless `fail_on_reference_fallback=False`.
- `levanter.inference.tpu_inference_adapter` wraps the pinned `tpu-inference==0.19.0` v3 ragged paged attention kernel. It adapts Levanter's `[page, slot, 2 * kv_head, head_size]` cache into the TPU-friendly packed cache shape, calls the fused KV-update plus RPA kernel, and converts the returned cache and attention output back to Levanter named axes.
- `tpu-inference==0.19.0` is now part of the `marin-levanter[tpu]` and `marin-core[tpu]` extras, matching the design decision that TPU serving may require it.
- CPU-safe contract tests now exercise real `KvPageCache.update` plus reference paged attention through the new backend dispatcher. The tests cover AUTO-on-CPU, explicit TPU backend failure on CPU, backend-sequence warning/fallback, non-silent TPU reference fallback, duplicate token destination validation, and backend availability reporting.

Validation so far:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/layers/attention.py lib/levanter/src/levanter/models/llama.py lib/levanter/src/levanter/models/apertus.py lib/levanter/src/levanter/inference/engine.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py

uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_paged_attention.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 39 passed

./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/layers/attention.py lib/levanter/src/levanter/models/llama.py lib/levanter/src/levanter/models/apertus.py lib/levanter/pyproject.toml lib/marin/pyproject.toml lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/inference/test_engine.py --fix
# OK, including Pyrefly
```

Open risk: the `tpu-inference` adapter has not yet been exercised on a TPU Levanter server. The likely first TPU integration risks are sharding expectations around Levanter's Haliax mesh and whether the all-mixed request distribution is too conservative for decode-only throughput. The next on-TPU check should run the smallest `decode_b8_i1_o128_n1` case with `backend=AUTO` and then compare against an explicit `JAX_RPA` diagnostic run if the adapter compiles but underperforms.

2026-05-31: first on-TPU smoke for the new `AUTO -> TPU_INFERENCE` backend,
`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-auto-20260531-1036`, received a v5p-8 worker but failed before server startup. The inner Levanter-only TPU solve pulled the default PyPI CUDA PyTorch wheel, then `transformers` imported `torch` while importing `HFCheckpointConverter`:

```text
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
ValueError: libcublasLt.so.*[0-9] not found in the system path [...]
```

This was a packaging/import failure, not a kernel result. `marin-core[tpu]` already routed torch through the CPU PyTorch index, but the benchmark runs as `uv run --package marin-levanter --extra tpu --extra serve`, so `marin-levanter[tpu]` needs the same CPU torch routing. The Levanter TPU extra now pins `torch==2.10.0`, CPU `torchvision`, and a `pytorch-cpu` source for the TPU extra. Follow-up jobs also set `USE_TORCH=0` so Transformers does not import torch during config/tokenizer setup.

Validation:

```text
uv lock
./infra/pre-commit.py --files lib/levanter/pyproject.toml uv.lock --fix
# OK
```

Follow-up TPU smokes:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-auto-cputorch-20260531-1048`: fixed v5p-8 rerun, currently pending capacity.
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-auto-cputorch-20260531-1050`: alternate `v6e-4,v5litepod-4` rerun, currently pending quota-tiering/capacity.

The v5p cputorch rerun received a worker and cleared the CUDA torch import error, but then failed during the first Levanter completion because `tpu-inference==0.19.0` imports `vllm.logger._VllmLogger` while importing its RPA kernel:

```text
File ".../tpu_inference/logger.py", line 3, in <module>
  from vllm.logger import _VllmLogger
ModuleNotFoundError: No module named 'vllm'
```

That means `tpu-inference` is not actually self-contained for this backend path. The TPU extras now install `vllm-tpu==0.19.0` alongside `tpu-inference==0.19.0` in both `marin-levanter[tpu]` and `marin-core[tpu]`, while keeping CPU torch routing for Levanter's TPU extra.

Validation:

```text
uv lock
uv run python -m py_compile lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_paged_attention.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 41 passed
./infra/pre-commit.py --files .agents/projects/levanter_tpu_inference_parity/research.md lib/levanter/pyproject.toml lib/marin/pyproject.toml uv.lock lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/layers/attention.py lib/levanter/src/levanter/models/llama.py lib/levanter/src/levanter/models/apertus.py lib/levanter/src/levanter/models/qwen.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_paged_attention.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Fresh TPU smokes after adding `vllm-tpu` to the TPU extras:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-auto-vllmtpu-20260531-1047`: v5p-8 rerun. It received a worker, installed `vllm-tpu`, and got past the previous `No module named 'vllm'` error. It was preempted once while loading Qwen3 and restarted cleanly.
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-auto-vllmtpu-20260531-1048`: alternate `v6e-4,v5litepod-4` rerun, pending quota-tiering/capacity at launch time.

Both fresh Levanter runs eventually reached the first completion request and failed with the same `tpu-inference` Mosaic partitioning error:

```text
Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.
```

The adapter now wraps the `tpu-inference` v3 `ragged_paged_attention` call in an explicit Haliax `shard_map` with replicated input and output specs. This is intentionally conservative for the first TPU integration proof: it should satisfy Mosaic's explicit partitioning requirement without taking a dependency on `tpu-inference`'s `data`/`model` mesh names before we know the correct Levanter mesh mapping for the serving path. Once the backend compiles end to end, the next optimization is to replace the replicated specs with a head/data sharding spec that matches Levanter's mesh and avoids redundant per-device work.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 26 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK, including Pyrefly
```

Follow-up jobs using the shard-map adapter and default Iris disk:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-shmap-20260531-0645`: primary v5p-8 Levanter `AUTO -> TPU_INFERENCE` smoke.
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-shmap-20260531-0645`: alternate `v6e-4,v5litepod-4` Levanter smoke.
- `/dlwh/qwen3-parity-vllm-v6e-v5lite4-b8-envfix-20260531-0645`: fixed alternate-hardware vLLM rerun. The previous `/dlwh/qwen3-parity-vllm-v5e-v6e-inneruv-smoke-20260531-0104` job measured the first vLLM case, then failed while finalizing because it called `jax.devices()` in the parent process after the vLLM subprocess owned libtpu. The current harness skips JAX device probing for vLLM finalization and has a `main()`-level regression test for that path.

The first v5p shard-map job failed before serving traffic because `write_kernel_jaxprs()` still traced `_run_generation_loop` and `_run_prefill` with their old signatures, missing the new static `tpu_paged_attention` argument. This was a trace-artifact bug, not a serving/kernel result. `write_kernel_jaxprs()` now passes `self.config.tpu_paged_attention` to both traces.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/engine.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 26 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK, including Pyrefly
```

Fresh trace-fixed Levanter jobs:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-shmaptrace-20260531-0650`
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-shmaptrace-20260531-0650`

The fixed alternate-hardware vLLM rerun `/dlwh/qwen3-parity-vllm-v6e-v5lite4-b8-envfix-20260531-0645` succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `vllm-tpu` | 8 | 1 | 1871.53 | 1886.15 | 0.547 | 544.0 | 546.0 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`. Device probing was intentionally skipped because vLLM may own libtpu.

The alternate `shmaptrace` Levanter run reached artifact tracing and failed before serving traffic:

```text
ValueError: shard_map requires a non-empty mesh. Got AbstractMesh((), axis_types=())
```

This was another harness artifact lifecycle bug. `start_levanter_server()` created the server under `trainer.use_device_mesh()`, but wrote kernel JAXPR/HLO artifacts after leaving that context. The harness now writes Levanter kernel artifacts under `trainer.use_device_mesh()` and `hax.axis_mapping(trainer.compute_axis_mapping)`.

Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 20 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Focused Qwen-level correctness evidence for the current decode path:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3.py::test_qwen3_paged_decode_matches_full_logits_for_available_backends -q
# 2 passed
```

This regression compares chunked Qwen3 paged decode logits for the available backends against the full causal logits on a
small synthetic Qwen3 config. It does not replace a full 8B fixed-prompt/reference-logit integration run, but it covers
the model-level paged decode plumbing for `AUTO` and `REFERENCE` on the local platform.

The next sampler optimization replaces `jax.random.categorical` with explicit Gumbel-max sampling in
`levanter.layers.sampler`. For finite logits, `argmax(logits + gumbel(key))` samples exactly from `softmax(logits)`, but
it avoids the candidate-space sorting that `jax.random.categorical` lowered into on TPU. This targets the profile/HLO
evidence above: candidate top-k had already reduced sampling to `128x4096`, but the HLO still contained top-k argsorts
after `chlo.top_k`.

Local validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 16 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK, including Pyrefly
```

The sampler HLO guard now asserts that the top-k path still has `chlo.top_k`, keeps the sampled exponential on the
candidate axis rather than the full vocabulary axis, and has no `call @argsort` in the candidate path.

Submitted Levanter-only TPU measurements for the Gumbel-max sampler:

- `/dlwh/qwen3-parity-levanter-gumbel-v5p8-n4-topk4096-20260531-2340`
- `/dlwh/qwen3-parity-levanter-gumbel-v6e-v5lite4-n4-topk4096-20260531-2340`

Both run `decode_b32_i1_o128_n4` and `decode_b32_i1_o512_n4` with `--top-k 4096`, `--temperature 0.7`,
`--top-p 1.0`, `--return-logprobs`, `--max-tokens-per-round 32`, `--max-rounds 512`, `--max-seqs 128`,
`--max-pages 1024`, `--max-prefill-size 128`, `--warmup-rounds 2`, `--measure-rounds 3`, and
`--dump-levanter-kernels`. At submission, the v5p job was pending quota-pool tiering:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 2 matching group(s) blocked by quota-pool tier monotonicity
```

The secondary job was pending v6e scale-up:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'tpu_v6e-preemptible_4-europe-west4-a'
```

Heartbeat `poll-qwen3-gumbel-sampler-tpu-runs` owns follow-up monitoring and should compare the results against the
prior `mtpr32` candidate-topk baselines.

Fresh artifact-mesh Levanter jobs:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-artmesh-20260531-0654`
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-artmesh-20260531-0654`

The alternate-hardware artifact-mesh run `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-artmesh-20260531-0654` succeeded on `TPU v6 lite` with 4 visible devices. This is the first end-to-end Levanter `AUTO -> TPU_INFERENCE` proof for Qwen3 8B:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 28.74 | 28.97 | 35.629 | 35625.7 | 35626.8 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v6 lite`.

This proves integration but not parity. The comparable v6e/v5lite vLLM baseline is `1871.53` decode tok/s, so the current Levanter AUTO path is 1.5% of vLLM on `decode_b8_i1_o128_n1`. Logs show the same macro bottleneck as the JAX RPA path: measured prefill/extraction was `32.148s`, then the decode kernel call produced `1016` tokens in `3.118s` (`325.90 tok/s`). The conservative replicated shard-map wrapper likely also leaves kernel throughput on the table, but end-to-end parity first needs the prefill/extraction path and artifact-backed v5p result.

The primary v5p artifact-mesh run `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-artmesh-20260531-0654` also succeeded after one preemption. This is the first target-family Levanter `AUTO -> TPU_INFERENCE` proof for Qwen3 8B:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 41.25 | 41.58 | 24.822 | 24819.1 | 24820.8 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices.

Against the target-family vLLM baseline of `1514.17` decode tok/s, the Levanter `tpu-inference` path is still only 2.7% end-to-end on `decode_b8_i1_o128_n1`. The detailed logs are more useful than the summary ratio: measured prefill/extraction took `22.058s`, while the measured decode iteration produced `1016` tokens in `2.512s` (`404.51 tok/s`). That raw decode-loop signal is already above the earlier JAX RPA v5p end-to-end result (`319.38` decode tok/s), but it is still only 26.7% of the vLLM end-to-end baseline and is hidden by the prefill/extraction path.

The harness and adapter now move the Levanter serving mesh from the default `data=4, model=1` shape to `data=1, replica=1, model=tensor_parallel_size`, and the `tpu-inference` shard-map specs shard q/k/v/cache over the Levanter `model` axis while keeping ragged metadata replicated. The backend support contract now treats `tensor_parallel_size=4` as the initial Qwen3 8B target, matching the vLLM comparison shape on four visible TPU devices. Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 28 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK, including Pyrefly
```

Fresh model-axis TPU smokes:

- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-modeltp4-20260531-1407`: alternate `v6e-4,v5litepod-4` run with `--tensor-parallel-size 4`; succeeded on `TPU v6 lite`.

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 29.38 | 29.61 | 34.850 | 34847.9 | 34848.7 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v6 lite`, 4 visible devices. This was only a small change from the earlier replicated artifact-mesh alternate-hardware result (`28.74` decode tok/s). The measured run still had `31.542s` of prefill/extraction and a raw decode iteration of `1016` tokens in `2.988s` (`340.03` tok/s).
- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-modeltp4-20260531-1407`: primary `v5p-8` run with `--tensor-parallel-size 4`; succeeded after one preemption.

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 40.71 | 41.02 | 25.156 | 25152.5 | 25154.0 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices.

  This did not improve the end-to-end metric relative to the replicated artifact-mesh proof (`41.25` decode tok/s). The raw measured decode iteration was slightly higher, `1016` tokens in `2.362s` (`430.16` tok/s), but measured prefill/extraction was still `22.535s`. That points at prefill work shape or the prefill kernel path rather than decode-loop head sharding alone.

The next smallest prefill-path fix is to stop sending a `[max_slots, max_seq_len]` prompt-token table through `_run_prefill` for a one-token decode-heavy prompt batch. Runtime `PrefillWork.prompt_tokens` now uses `[max_slots, max_prefill_size]`, matching the trace-artifact path and the already-bounded prefill queue. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 10 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py .agents/projects/levanter_tpu_inference_parity/design.md .agents/projects/levanter_tpu_inference_parity/spec.md .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK, including Pyrefly
```

Follow-up v5p smoke:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-prefillshape-20260531-1416`: primary `v5p-8` rerun with the bounded prefill-work shape, `--tensor-parallel-size 4`, default Iris disk, and `/dev/shm` caches/output. It succeeded, but did not improve throughput:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 40.98 | 41.30 | 24.986 | 24983.0 | 24984.3 |

  Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices. The measured run still had `22.406s` of prefill/extraction, then a raw decode iteration of `1016` tokens in `2.362s` (`430.23` tok/s). The bounded prompt-token table is a correctness/shape cleanup, not the prefill bottleneck.

The engine now logs prefill submission, device wait, and output-ingestion timing separately:

```text
Initial prefill: submit <seconds>s, device <seconds>s, extraction <seconds>s, total <seconds>s
```

Follow-up diagnostic:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-prefilltiming-20260531-1421`: primary `v5p-8` rerun with the prefill timing split.

The same explicit `jax.block_until_ready` timing is now applied to decode iterations so the `Decode iter` log separates
JIT submission, TPU execution wait, and host-side result drain. This is diagnostic-only; it should not change generated
tokens or scheduler behavior. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py -q
# 1 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py --fix
# OK, including Pyrefly
```

The primary v5p diagnostic has been preempted twice before producing the timing line. A non-preemptible v5p retry was
rejected at submission time because this Iris config has no non-preemptible `v5p-8` scale group. The alternate-hardware
diagnostic `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-blocktiming-20260531-1441` was submitted with the same
block-until-ready timing patch to get the prefill/decode attribution while the v5p job continues.

The v5p diagnostic eventually succeeded after two preemptions:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 39.99 | 40.30 | 25.606 | 25602.4 | 25604.1 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices.

The split timing shows the previous "prefill/extraction" diagnosis was incomplete:

```text
Warmup:  Initial prefill: submit 22.657s, extraction 0.082s, total 22.739s
Warmup:  Decode iter: total 22.069s ... submit 19.683s ... extract 2.385s
Measure: Initial prefill: submit 22.572s, extraction 0.078s, total 22.650s
Measure: Decode iter: total 2.361s ... submit 0.003s ... extract 2.358s
```

So the measured prefill gap is in the synchronous JIT call/compile path, not host output ingestion. A local CPU
reproduction with the dummy engine shows `_run_prefill` compiles on the first `generate()`, compiles again on the second,
then stabilizes, while `_run_generation_loop` compiles once. The one-warmup benchmark was therefore mostly measuring the
second prefill compile. `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-warmup2-20260531-1448` was submitted with two warmup
rounds to measure actual steady state after that second compile.

The alternate-hardware block-timing diagnostic
`/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b8-blocktiming-20260531-1441` succeeded on `TPU v6 lite` and confirmed
the same pattern:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 29.28 | 29.51 | 34.974 | 34969.7 | 34972.3 |

The measured round had `Initial prefill: submit 31.330s, device 0.070s, extraction 0.001s, total 31.401s`, followed by
`Decode iter: total 2.990s (device 2.984s, host 0.006s, submit 0.002s), 339.83 tok/s, 1016 new`. This reinforces that
one warmup measures the second prefill compile on both v5p and v6e/v5lite.

The primary v5p two-warmup diagnostic
`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b8-warmup2-20260531-1448` succeeded after one preemption:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 355.31 | 358.09 | 2.882 | 2878.8 | 2880.4 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind
`TPU v5`, 4 visible devices. The measured round finally removed the prefill compile artifact:

```text
Warmup 0: Initial prefill: submit 22.445s, device 0.077s, extraction 0.001s, total 22.523s
Warmup 0: Decode iter: total 21.866s (device 2.381s, host 19.485s, submit 19.481s), 46.47 tok/s, 1016 new
Warmup 1: Initial prefill: submit 22.011s, device 0.080s, extraction 0.001s, total 22.093s
Warmup 1: Decode iter: total 2.363s (device 2.355s, host 0.007s, submit 0.003s), 430.00 tok/s, 1016 new
Measure:  Initial prefill: submit 0.009s, device 0.034s, extraction 0.001s, total 0.044s
Measure:  Decode iter: total 2.361s (device 2.356s, host 0.006s, submit 0.002s), 430.30 tok/s, 1016 new
```

Against the v5p vLLM b8 baseline of `1514.17` decode tok/s, the correct warmed Levanter number is now 23.5% of vLLM
end-to-end and 28.4% if comparing the raw measured decode-device call (`430.30` tok/s) to vLLM's end-to-end number.
That is a much better diagnosis than the one-warmup runs, but still far outside the 10-15% parity criterion. The harness
now defaults to two warmup rounds, and reports `compile_including_seconds` as the total configured warmup cost, so future
steady-state results do not accidentally include this Levanter second-request compile while still preserving the warmup
tax in `summary.json`. Inspection of the pinned `tpu-inference==0.19.0` wheel confirmed that Levanter's all-decode
distribution convention `[num_decode, num_decode, num_reqs]` matches `PersistentBatchManager._reorder_batch`; the
remaining b8 gap is therefore not explained by a simple request-distribution mismatch. The next rollout-relevant check is
the same warmed protocol at `decode_b32_i1_o128_n1`:
`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b32-warmup2-20260531-1508`.

Follow-up inspection found a tensor-parallel mapping issue in the benchmark harness. The Levanter mesh default maps
`heads` and `mlp` to `model`, but Qwen GQA q/k/v projections use `kv_head` plus `q_heads_per_group` rather than the
flattened `heads` axis. The harness now explicitly maps `kv_head` to `model`, matching the TPU-inference adapter's
packed KV sharding. This should shard q/k/v projections and resident page cache over the tensor-parallel axis instead
of relying on resharding at the paged-attention call boundary. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 23 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

The warmed v5p b32 run before the `kv_head`/sampler follow-up completed:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 419.28 | 422.55 | 9.769 | 9757.0 | 9761.8 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices. Measured timing after two warmups:

```text
Warmup 0: Initial prefill: submit 24.537s, device 0.186s, extraction 0.002s, total 24.725s
Warmup 0: Decode iter: total 30.434s (device 9.181s, host 21.253s, submit 21.239s), 133.53 tok/s, 4064 new
Warmup 1: Initial prefill: submit 28.319s, device 0.626s, extraction 0.001s, total 28.946s
Warmup 1: Decode iter: total 9.231s (device 9.215s, host 0.016s, submit 0.004s), 440.24 tok/s, 4064 new
Measure:  Initial prefill: submit 0.019s, device 0.145s, extraction 0.001s, total 0.165s
Measure:  Decode iter: total 9.163s (device 9.148s, host 0.014s, submit 0.002s), 443.54 tok/s, 4064 new
```

This is only `8.95%` of the vLLM v5p b32 baseline (`4682.70` decode tok/s), so increasing active sequences without further model/sampler fixes does not close the gap.

The `kv_head -> model` mapping also made the b32 run fit on `v6e-4`/`v5litepod-4`, where the previous b32 attempt failed with HBM allocation during reset. The v6e/v5lite `kvheadtp` run succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 306.49 | 308.88 | 13.364 | 13349.8 | 13357.5 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v6 lite`, 4 visible devices. The measured decode iteration was `12.828s` device time for `4064` new tokens (`316.80` tok/s raw).

A second concrete decode-loop hotspot is deterministic sampling. `Sampler.__call__` used to compute top-p sorting and `hax.random.categorical` even when every temperature was zero, then discard the sampled tokens with a `where`. The sampler now has a JAX `lax.cond` greedy fast path that skips top-p/categorical work when all temperatures are zero while still returning the selected token logprob under the model distribution. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 5 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK, including Pyrefly
```

Follow-up TPU runs with both `kv_head -> model` and greedy sampler fast path:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b32-greedykv-20260531-1526`: submitted on `v5p-8`. It reached warmup on a v5p worker, was preempted once, restarted, and succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 391.91 | 394.97 | 10.451 | 10435.8 | 10443.3 |

The measured round had `Initial prefill: submit 0.018s, device 0.145s, extraction 0.001s, total 0.163s`, followed by
`Decode iter: total 9.170s (device 9.156s, host 0.014s, submit 0.002s), 443.17 tok/s, 4064 new`. This is not an
improvement over the earlier v5p b32 raw decode (`443.54` tok/s) and is worse in the summary table (`391.91` versus
`419.28` decode tok/s), which indicates the greedy sampler fast path is not the large missing piece.
- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b32-greedykv-20260531-1528`: submitted on `v6e-4,v5litepod-4` as the faster secondary read. It was placed on a `v6e-4` worker and succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 299.32 | 301.65 | 13.685 | 13668.8 | 13676.6 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, device kind `TPU v6 lite`, 4 visible devices. The measured round had `Initial prefill: submit 0.016s, device 0.207s, extraction 0.001s, total 0.224s`, followed by `Decode iter: total 12.833s (device 12.821s, host 0.012s, submit 0.002s), 316.69 tok/s, 4064 new`. This is slightly slower than the prior `kvheadtp` v6e/v5lite result (`306.49` summary decode tok/s, `316.80` raw decode tok/s), so the greedy sampler fast path is not the dominant bottleneck on this hardware/configuration.

The OpenAI serving path now only asks the sampler for generated-token logprobs when the request needs them. This matters
for RL rollout generation because standard `/v1/completions` and chat generation do not request generated logprobs, but
the old Levanter path still computed the selected-token logprob every decode round. The change threads
`return_logprobs` from OpenAI request parsing through `InferenceContext.submit_request`, `Request`, prefill, and the
generation loop, and tells `Sampler` to return zero placeholders without the per-token logsumexp when logprobs are not
part of the API contract. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_inference_server.py -q
# 20 passed, 1 skipped
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/inference/openai.py lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_inference_server.py --fix
# OK, including Pyrefly
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b32-nologprob-20260531-1539` was submitted on `v5p-8` to measure
`kv_head -> model` + greedy sampler + request-aware no-logprobs together. It succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 408.42 | 411.61 | 10.029 | 10012.3 | 10019.8 |

The measured round had `Initial prefill: submit 0.018s, device 0.144s, extraction 0.001s, total 0.162s`, followed by
`Decode iter: total 9.152s (device 9.141s, host 0.011s, submit 0.002s), 444.04 tok/s, 4064 new`. Suppressing generated
logprobs removes a little host/summary overhead but leaves the raw decode device time essentially unchanged.

The next likely full-model bottleneck is the LM-head/vocab path. The default mesh already maps `heads` and `mlp` to
`model`, and the harness now maps `kv_head` to `model`; however Qwen3 decode still projects to a large vocabulary every
round. Levanter has precedent for `vocab -> model` sharding in `lib/levanter/config/gpt2_20b.yaml`, and Qwen3 8B's
vocabulary size is divisible by the target `tensor_parallel_size=4`. The benchmark trainer config now maps
`vocab -> model` as well, so the LM head/logits axis can shard over the TPU model axis instead of producing a full
replicated vocabulary on each device. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 23 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b32-vocabtp-20260531-1541` was submitted on `v5p-8` to measure the combined
`kv_head`, greedy, no-logprobs, and `vocab -> model` configuration. It succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 986.08 | 993.78 | 4.154 | 4141.2 | 4147.4 |

The measured round had `Initial prefill: submit 0.015s, device 0.061s, extraction 0.000s, total 0.077s`, followed by
`Decode iter: total 3.837s (device 3.827s, host 0.010s, submit 0.002s), 1059.11 tok/s, 4064 new`. This is the first
large throughput gain in the b32 path: roughly `2.4x` over the request-aware no-logprob summary result (`408.42`
tok/s), and raw decode device time dropped from about `9.14s` to `3.83s`. It is still only `21.1%` of the vLLM v5p b32
baseline (`4682.70` decode tok/s), so parity still requires another substantial model/kernel attribution pass.

Because b32 remains far from parity even after vocab sharding, the next matrix point is b128 to test whether Levanter
keeps scaling with more active rollout sequences:

- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-vocabtp-20260531-1552`: Levanter with `kv_head -> model`,
  request-aware no-logprobs, and `vocab -> model`, on `v5p-8`. It succeeded:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 1089.40 | 1097.91 | 15.039 | 14954.6 | 15008.6 |

  The measured round had `Initial prefill: submit 0.042s, device 0.214s, extraction 0.001s, total 0.257s`, followed by
  `Decode iter: total 13.721s (device 13.689s, host 0.032s, submit 0.002s), 1184.77 tok/s, 16256 new`.
- `/dlwh/qwen3-parity-vllm-v5p8-b128n1-20260531-1552`: matching vLLM baseline on `v5p-8`. It succeeded:

  | case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `decode_b128_i1_o128_n1` | `vllm-tpu` | 128 | 1 | 7250.50 | 7307.15 | 2.260 | 2026.5 | 2228.6 |

Levanter improves only modestly going from b32 to b128 (`986.08` to `1089.40` summary decode tok/s), while vLLM improves
from `4682.70` to `7250.50`. The gap is still about `6.7x`, so active-sequence scaling is not enough. The next useful
artifact should split the compiled decode step into transformer/paged-attention versus LM-head/sampling costs or otherwise
identify the remaining fused-kernel/scheduler limitation.

The `vocab -> model` improvement also exposed a stack integration bug: Marin's RL Levanter rollout path replaced the
inference trainer mesh with `MeshConfig(axes={"data": 1, "model": -1})`, which dropped the benchmark's `kv_head` and
`vocab` model-axis mappings. `create_inference_context` now builds a TPU-serving inference mesh that preserves valid
train-time mappings, drops mappings to unavailable axes such as `position -> context`, and explicitly adds
`heads`, `kv_head`, `mlp`, and `vocab` to the `model` axis. It also updates the `LevanterInferenceContextConfig` mesh
and axis mapping so reload/start-server use the same mapping as `InferenceServerConfig.trainer`. Validation:

```text
uv run --with pytest --with pytest-timeout pytest tests/rl/test_rollout_worker.py -q
# 25 passed
./infra/pre-commit.py --files lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK, including Pyrefly
```

The benchmark harness now has a no-LM-head attribution mode for Levanter. It uses normal prefill to seed the decode
queue, then runs decode rounds through the transformer and paged KV-cache update while enqueuing dummy token IDs instead
of projecting hidden states through the LM head and sampling. This is not a user-facing decoding mode; it exists to split
the remaining b128 gap into transformer/paged-attention/cache-update cost versus LM-head/sampling cost. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 26 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/models/llama.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK, including Pyrefly
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-nolmhead-20260531-1616` was submitted on `v5p-8` with default Iris disk,
`/dev/shm` HF/JAX/output scratch, two warmups, one measured round, and `--levanter-diagnose-without-lm-head`. It initially
queued on capacity:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'tpu_v5p-preemptible_8-us-east5-a'
```

Because the primary v5p result is capacity-bound, the same diagnostic was also submitted as a secondary read on
`v6e-4,v5litepod-4`: `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-nolmhead-20260531-1618`. It was assigned a
worker immediately and started running. The v5p run also received a worker shortly afterward, so both no-LM-head
diagnostic jobs are now running.

The initial primary v5p run also failed before reaching the no-LM-head diagnostic, for the same root cause with a larger
inferred cache allocation:

```text
RESOURCE_EXHAUSTED: E0100: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 7.18G.
That was not possible. There are 5.79G free.; (0x1x0_HBM0)
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-nolmhead-pages512-20260531-1626` is the replacement primary v5p run with
`--max-pages 512`. It has been assigned a worker and is running.

The first secondary v6e run failed before reaching the no-LM-head diagnostic because the inferred page-cache size did
not fit on the assigned `v6e-4` worker:

```text
RESOURCE_EXHAUSTED: E0100: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 1.13G.
That was not possible. There are 233.51M free.; (1x0x0_HBM0)
```

The smallest concrete retry is to bound the page cache to the serving shape. `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-nolmhead-pages512-20260531-1621`
was submitted with `--max-pages 512`, which is enough for the b128/o128 no-clone case while leaving headroom on
4-chip v6e/v5lite. It is currently pending quota-tiering:

```text
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 5 matching group(s) blocked by quota-pool tier monotonicity
```

The pages512 secondary run later started and proved the explicit page cap clears the HBM failure. It reached the
diagnostic phase after the normal measured run:

```text
Measure levanter:auto decode_b128_i1_o128_n1 round 0
Initial prefill: submit 0.054s, device 0.374s, extraction 0.001s, total 0.429s
Decode iter: total 23.820s (device 23.793s, host 0.027s, submit 0.002s), 682.45 tok/s, 16256 new
Warmup levanter:auto:no_lm_head decode_b128_i1_o128_n1 round 0
```

It then failed on a harness bug before any no-LM-head measurement:

```text
UnboundLocalError: cannot access local variable 'warmup' where it is not associated with a value
```

The bug was caused by deleting the `warmup` argument inside the diagnostic helper and later using it to set
`compile_including_seconds`. The helper now preserves `warmup`, and a direct regression test exercises the actual
no-LM-head helper for both warmup and measured calls. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 25 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Fixed-bundle warmfix reruns:

- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-nolmhead-warmfix-20260531-1637`
- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-nolmhead-warmfix-20260531-1637`

The secondary v6e/v5lite warmfix succeeded on a `TPU v6 lite` worker with 4 visible devices:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 652.69 | 657.79 | 25.102 | 24543.6 | 25073.4 |
| `decode_b128_i1_o128_n1` | `levanter:auto:no_lm_head` | 128 | 1 | 6608.79 | 6660.42 | 2.479 | 2479.1 | 2479.1 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`,
device kind `TPU v6 lite`, 4 visible devices, `--max-pages 512`, `--max-prefill-size 128`, `--max-rounds 128`,
and `--levanter-diagnose-without-lm-head`.

Detailed measured timing shows the normal Levanter path spent `23.793s` device time to decode `16256` new tokens
(`682.45` raw tok/s), while the no-LM-head diagnostic completed the same decode schedule in `2.479s`. Removing the LM
head plus sampling path therefore makes the decode loop roughly `10.1x` faster on this 4-chip secondary target. This is
the strongest attribution signal so far: after `kv_head` and `vocab` sharding, the remaining huge gap is dominated by
the logits/LM-head/sampling boundary rather than paged-attention, cache update, host cadence, or scheduler packing.

The primary v5p warmfix run is still the target-family result to collect. It was preempted once, restarted, received a
new worker, and is running again.

The primary v5p warmfix run also succeeded after the preemption and restart:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 599.78 | 604.47 | 27.317 | 27231.7 | 27283.2 |
| `decode_b128_i1_o128_n1` | `levanter:auto:no_lm_head` | 128 | 1 | 5951.58 | 5998.08 | 2.753 | 2752.9 | 2752.9 |

Environment: device kind `TPU v5`, 4 visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--max-pages 512`, `--max-prefill-size 128`, and `--max-rounds 128`.
The run used `LIBTPU_INIT_ARGS=--xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_scoped_vmem_limit_kib=50000 --xla_tpu_use_enhanced_launch_barrier=true`.

The normal measured pass split into a first tiny 127-token decode at `11.960s` device time followed by the main 16129-token
decode at `13.850s` device time (`1161.91` raw tok/s for the main call). The no-LM-head measurement finished in `2.753s`
end-to-end. This confirms on the primary target that the transformer plus `tpu-inference` paged-attention/cache-update
path is not the main bottleneck: removing LM-head/sampling is again about `9.9x` faster (`5951.58 / 599.78`).

The harness now adds a second attribution mode, `--levanter-diagnose-lm-head-no-sampling`. It keeps the normal prefill,
runs the transformer and LM-head projection during decode, keeps the logits live with a checksum, then enqueues dummy
token IDs without argmax/top-p/logprob work. The next TPU run should enable both diagnostics on the b128 v5p case:
normal, `lm_head_no_sampling`, and `no_lm_head`. That will split the remaining logits boundary into projection/materialized
vocab cost versus greedy argmax/sampler reduction cost.

Validation for the new diagnostic mode:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 29 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK, including Pyrefly
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-lmheadsplit-20260531-1655` is the v5p follow-up with both diagnostics
enabled. It uses default Iris disk, `/dev/shm` for HF/JAX/output scratch, `--max-pages 512`, `--max-prefill-size 128`,
`--max-rounds 128`, and the same v5p `LIBTPU_INIT_ARGS` as the warmfix run. At submission time it is pending v5p
capacity/quota-tiering:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 2 matching group(s) blocked by quota-pool tier monotonicity
```

The normal serving path now also has an opt-in candidate optimization, `--levanter-streaming-greedy-lm-head`. For greedy
decode requests that do not need generated logprobs, `_run_generation_loop` can run `decode_hidden`, select the sampled
positions, and use the existing fused cross-entropy XLA streaming argmax over `model.get_lm_head()` instead of
materializing full `[position, vocab]` logits and then reducing over vocab. The path is deliberately opt-in until TPU
data proves it helps Qwen3 rollout serving.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 30 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_engine.py --fix
# OK, including Pyrefly
```

`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-streamgreedy-20260531-1701` is the v5p b128 run for the opt-in streaming
greedy path. It uses the same serving shape and cache settings as the warmfix run, plus `--levanter-streaming-greedy-lm-head`.
At submission time it is pending v5p capacity/quota-tiering:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 2 matching group(s) blocked by quota-pool tier monotonicity
```

The LM-head split follow-up `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-lmheadsplit-20260531-1655` later succeeded
on the primary v5p target:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 1122.32 | 1131.09 | 14.598 | 14516.9 | 14563.2 |
| `decode_b128_i1_o128_n1` | `levanter:auto:no_lm_head` | 128 | 1 | 5952.95 | 5999.46 | 2.752 | 2752.2 | 2752.2 |
| `decode_b128_i1_o128_n1` | `levanter:auto:lm_head_no_sampling` | 128 | 1 | 5890.30 | 5936.32 | 2.782 | 2781.5 | 2781.5 |

Environment: device kind `TPU v5`, 4 visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--max-pages 512`, `--max-prefill-size 128`, and `--max-rounds 128`.
The run used `LIBTPU_INIT_ARGS=--xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_scoped_vmem_limit_kib=50000 --xla_tpu_use_enhanced_launch_barrier=true`.

This cleanly splits the remaining b128 gap. Computing transformer hidden states plus the LM-head projection and keeping
the logits live costs essentially the same as skipping the LM head entirely: `5890.30` versus `5952.95` decode tok/s.
The normal greedy path falls back to `1122.32` decode tok/s. The next blocker is therefore not the projection itself;
it is the greedy argmax/sampling reduction over the materialized vocabulary logits. The active streaming-greedy run is
the direct candidate fix for that boundary.

The secondary streaming-greedy run `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-streamgreedy-20260531-1708`
succeeded on a `TPU v6 lite` worker:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 5675.53 | 5719.87 | 2.887 | 2822.6 | 2855.2 |

Detailed timing for the measured round was stronger than the summary latency aggregation: after a `0.304s` prefill, the
decode call completed `16256` generated tokens in `2.173s` total (`2.146s` device, `0.027s` host), or `7481.61` raw
decode tok/s. Against the earlier secondary-target normal run (`652.69` summary decode tok/s) this is an `8.7x` summary
gain. It is also in the same range as the no-LM-head diagnostic (`6608.79` summary decode tok/s), which confirms that
the streaming argmax path removes the dominant materialized-vocab reduction overhead for greedy no-logprob serving.

Because Marin's current RL Levanter context requests generated-token logprobs during rollout collection, the streaming
path now also supports logprobs. It reuses the streaming CE pass to obtain greedy IDs and logsumexp, then computes the
selected-token and dummy-token logits with narrow per-token LM-head gathers. This avoids materializing full
`[position, vocab]` logits while preserving the OpenAI logprob contract for greedy decoding.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 32 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK, including Pyrefly
```

Follow-up logprob-contract TPU jobs:

- `/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-streamlogprobs-20260531-1719`: secondary target with
  `--levanter-streaming-greedy-lm-head --return-logprobs`.
- `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-streamlogprobs-20260531-1719`: primary v5p target with the same
  logprob-capable streaming path.

The benchmark harness now also exposes `--return-logprobs`, which requests generated-token logprobs through the normal
OpenAI-compatible endpoint for both vLLM and Levanter. Marin's current RL Levanter context asks for logprobs during
rollout collection, so the flag is needed for an apples-to-apples RL-serving matrix. The harness rejects
`--return-logprobs` with the no-sampling diagnostics, but the streaming-greedy path now supports generated-token logprobs
using a streaming logsumexp plus narrow selected-token LM-head gathers.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 27 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

2026-05-31: added an explicit top-k sampling path to Levanter's sampler and OpenAI inference surface so the benchmark
can match Marin RL's default `train_decoding_top_k=4096` policy instead of using full-vocabulary categorical sampling.
The implementation threads `top_k` through `SeqDecodingParams`, cloned-prefix prefill, the generation loop, the OpenAI
completion/chat request models, and the RL Levanter inference context. The auto RL `InferenceEngineConfig` now sets
`max_top_k` to the largest top-k requested by any train/eval lesson, which gives the sampler a static candidate axis.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/inference/jit_scheduler.py lib/levanter/src/levanter/inference/openai.py lib/levanter/src/levanter/inference/openai_protocol.py lib/marin/src/marin/rl/environments/inference_ctx/levanter.py lib/marin/src/marin/rl/job_config.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 50 passed
uv run --with pytest --with pytest-timeout pytest tests/rl/test_rollout_worker.py -q
# 25 passed
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_inference_server.py -q
# 14 passed, 1 skipped
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/src/levanter/inference/jit_scheduler.py lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/inference/openai.py lib/levanter/src/levanter/inference/openai_protocol.py lib/levanter/src/levanter/main/sample_lm.py lib/levanter/src/levanter/eval_harness.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/inference/test_inference_server.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/marin/src/marin/rl/environments/inference_ctx/levanter.py lib/marin/src/marin/rl/job_config.py tests/rl/integration/config.py --fix
# OK
```

Submitted paired top-k runs:

- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-topk4096-mtpr32-20260531-2109`
- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-topk4096-mtpr32-20260531-2109`

The secondary job first ran on v6e and got through the vLLM side and part of the Levanter side before preemption. The
useful Levanter signal before preemption was strong for the short rollout case:

```text
decode_b32_i1_o128_n4 measure: Decode iter total 1.175s, 3458.05 tok/s, 4064 new tokens
```

That is the first evidence that the full-vocab categorical HLO hotspot was the dominant `o128 n=4` bottleneck. The
same job had just entered `decode_b32_i1_o512_n4` when it was preempted, so no final long-output Levanter summary was
available. The v5p job finished the vLLM measurements and then was preempted before Levanter startup. Because the
benchmark previously logged only the final summary, those v5p vLLM numbers were not durable in logs.

To make the benchmark robust to exactly this failure mode, the harness now logs every measured `CaseResult`
immediately and writes/logs partial `summary.md`, `env.json`, and `artifacts.json` after each backend completes. Future
preemptions after the vLLM half of a `--backend both` run should still leave the completed backend's table in Iris logs.

Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
```

After reviewing the top-k implementation, I found one semantic footgun: when `InferenceEngineConfig.max_top_k` was set
for RL rollout service shapes, a sampled request with no explicit `top_k` would silently sample from `max_top_k`
candidates instead of the full vocabulary. That is fine for the intended Marin RL train path, where
`train_decoding_top_k=4096` is explicit, but it is wrong for the general OpenAI surface and could make benchmark results
look like a top-k policy when the request did not ask for one. The engine now fails fast for sampled requests that omit
`top_k` when `max_top_k` is configured; greedy requests are still allowed without `top_k`.

The active paired top-k jobs were checked again after this local fix. They are still running under Iris retries rather
than terminal:

- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-topk4096-mtpr32-20260531-2109`: `JOB_STATE_RUNNING`, task building after
  one preemption.
- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-topk4096-mtpr32-20260531-2109`: `JOB_STATE_RUNNING`, task running
  after two preemptions. No new Levanter summary had appeared in the latest 10-minute log window.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/engine.py lib/levanter/tests/inference/test_engine.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_sampler.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 52 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/tests/inference/test_engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK, including Pyrefly
```

The secondary top-k job later succeeded on `TPU v6 lite` after two preemptions:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets | decode/vllm | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 4049.48 | 4057.39 | 1.011 | 2.623 | 1006.9 | 1008.8 | | | 1.000 | |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 4107.03 | 4109.06 | 3.945 | 7.912 | 3932.1 | 3939.6 | | | 1.000 | |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3106.32 | 3112.39 | 1.319 | 96.122 | 1315.7 | 1316.2 | 4831838208 | 2 | 0.767 | fail |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3338.26 | 3339.89 | 4.908 | 12.970 | 4902.6 | 4905.9 | 4831838208 | 2 | 0.813 | fail |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`,
`vllm-tpu==0.19.0`, device kind `TPU v6 lite`, 4 visible devices, `--top-k 4096`,
`--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`, `--max-tokens-per-round 32`,
`--max-rounds 512`, `--max-seqs 128`, `--max-pages 1024`, and `--max-prefill-size 128`.

This was the strongest sampled rollout result until the primary v5p job completed. Explicit top-k closed the previous
full-vocabulary categorical gap enough that Levanter was within 19-23% of vLLM on the secondary target for the
cloned-prefix `n=4` sampled regime.

The primary v5p top-k job also succeeded after one preemption:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets | decode/vllm | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 3580.42 | 3587.42 | 1.144 | 2.750 | 1139.2 | 1142.0 | | | 1.000 | |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 3639.97 | 3641.75 | 4.501 | 8.959 | 4488.1 | 4495.8 | | | 1.000 | |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2671.63 | 2676.84 | 1.533 | 82.886 | 1529.4 | 1530.9 | 4831838208 | 2 | 0.746 | fail |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2926.44 | 2927.87 | 5.599 | 11.159 | 5592.4 | 5595.5 | 4831838208 | 2 | 0.804 | fail |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`,
`vllm-tpu==0.19.0`, device kind `TPU v5`, 4 visible devices, `--top-k 4096`,
`--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`, `--max-tokens-per-round 32`,
`--max-rounds 512`, `--max-seqs 128`, `--max-pages 1024`, and `--max-prefill-size 128`.

The primary-target result is slightly worse than the secondary run, at 74.6% of vLLM for `o128 n=4` and 80.4% for
`o512 n=4`. It still misses the 10-15% parity criterion, and the short-output case is worse than the long-output case,
so the next throughput-biased test is a Levanter-only run with a larger `--max-tokens-per-round 128`. That should
show whether remaining scheduler/loop packing overhead is limiting the sampled RL rollout regime once top-k is fixed.

The throughput-biased secondary follow-up
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-topk4096-mtpr128-20260531-2142` was submitted as a Levanter-only
run with `--max-tokens-per-round 128` and the same sampled `n=4`, `top_k=4096`, generated-logprob contract. At
submission it is pending v6e scale-up:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'tpu_v6e-preemptible_4-us-east1-d'
```

It quickly received a `v6e-4` worker and succeeded. Larger `max_tokens_per_round` regressed the secondary result rather
than closing the remaining gap:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 1884.71 | 1888.39 | 2.173 | 102.294 | 2169.5 | 2170.6 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 1949.56 | 1950.52 | 8.404 | 16.768 | 8397.6 | 8401.4 | 4831838208 | 2 |

Environment: `TPU v6 lite`, 4 visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--max-tokens-per-round 128`, `--max-rounds 512`,
`--top-k 4096`, `--temperature 0.7`, and `--return-logprobs`.

Against the secondary `mtpr32` run, this is a 39-42% throughput regression (`3106.32 -> 1884.71` and
`3338.26 -> 1949.56`). The `max_tokens_per_round` knob is therefore not monotonic; `128` should not be carried to v5p.
The next scheduler-bracketing run should test `64` on secondary hardware before spending primary v5p capacity.

Submitted `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-topk4096-mtpr64-20260531-2148` with the same secondary
shape and `--max-tokens-per-round 64` to bracket the scheduler knob between the good `32` result and the bad `128`
result.

The `mtpr64` run succeeded, also on `TPU v6 lite`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2499.59 | 2504.47 | 1.639 | 102.885 | 1635.6 | 1636.4 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2728.83 | 2730.17 | 6.004 | 12.022 | 5996.8 | 6001.4 | 4831838208 | 2 |

This is better than `128` but still worse than `32` by 18-20%. The observed scheduler curve on secondary hardware is
`mtpr32 > mtpr64 > mtpr128`. Before declaring `32` the local optimum, submit one `mtpr16` run because smaller chunks may
win if sampled top-k/logprob kernel occupancy, not host launch count, dominates.

Local code inspection found a more direct candidate for the remaining sampled top-k gap. The sampler still unsharded the
full Qwen vocabulary before applying `top_k`, then sampled from the 4096 candidate set. For `top_k=4096`, this means the
expensive full-vocab gather happens before the reduction even though the sampler only needs a small candidate set. The
sampler now applies `jax.lax.top_k` to the logits in their existing placement first, then unshards only the candidate
logits and candidate token IDs before categorical sampling/logprob normalization. Full-vocabulary sampling without
`top_k` keeps the old unshard-before-categorical behavior.

The first local regression pass missed two axis details that only showed up when this was exercised through the
full Qwen3 sampler shape. First, `top_ks` may be a per-row `NamedArray`, so the rank mask must explicitly broadcast the
candidate axis and request axes instead of relying on implicit scalar broadcasting. Second, once `top_k` replaces the
full vocabulary axis with a smaller candidate axis, helpers that operate on the sampler vocabulary axis must resolve by
axis name rather than by the original full-size `Axis` object.

Validation after both fixes:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 53 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK, including Pyrefly
```

Submitted `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-topk4096-shardedtopk-mtpr32-20260531-2200` to measure
the sharded-top-k candidate against the known-best secondary `mtpr32` baseline.

That first cluster run failed during warmup on `TPU v6 lite` before producing benchmark results:

```text
RuntimeError: Found axis with same name but different size: vocab(151936) vs vocab(4096)
```

The traceback points to `_with_unsharded_vocab(candidate_logits)`, confirming the second local fix above. The failed
run is therefore not a performance signal.

Submitted replacement `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-topk4096-shardedtopk-fix-mtpr32-20260531-2209`
with the same Levanter-only secondary shape: `--tpu v6e-4,v5litepod-4`, `--case decode_b32_i1_o128_n4`,
`--case decode_b32_i1_o512_n4`, `--top-k 4096`, `--temperature 0.7`, `--return-logprobs`,
`--max-tokens-per-round 32`, `--max-rounds 512`, `--max-seqs 128`, `--max-pages 1024`, `--max-prefill-size 128`,
default Iris disk, and `/dev/shm` for HF/JAX/output scratch. Initial poll showed `JOB_STATE_RUNNING` with one assigned
task.

The v5p `max_pages=1024` n=1 tuning job
`/dlwh/qwen3-parity-levanter-v5p8-n1-streamlogprobs-maxpages1024-20260531-1052` succeeded after one preemption:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 932.92 | 940.20 | 1.098 | 1094.3 | 1095.6 |
| `decode_b8_i1_o512_n1` | `levanter:auto` | 8 | 1 | 1137.48 | 1139.70 | 3.601 | 3597.5 | 3598.8 |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 2585.90 | 2606.10 | 1.584 | 1570.5 | 1577.5 |
| `decode_b32_i1_o512_n1` | `levanter:auto` | 32 | 1 | 3147.91 | 3154.05 | 5.205 | 5185.4 | 5196.7 |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 4765.07 | 4802.29 | 3.438 | 3354.5 | 3396.6 |
| `decode_b128_i1_o512_n1` | `levanter:auto` | 128 | 1 | 5655.58 | 5666.62 | 11.588 | 11462.6 | 11529.3 |

Environment: device kind `TPU v5`, four visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--max-pages 1024`, `--max-prefill-size 128`, `--max-rounds 512`,
`--levanter-streaming-greedy-lm-head`, and `--return-logprobs`. The run produced `summary.json` (4394 bytes),
`summary.md` (809 bytes), and `env.json` (1814 bytes). Relative to the v5p vLLM logprob baseline, this improves the
weak `b32/o128` case from 33.1% to 55.3%, `b128/o128` from 70.6% to 77.8%, and `b128/o512` from 67.9% to 88.8%.
The remaining n=1 gaps are `b8/o128`, `b32/o128`, and `b32/o512`.

The corrected `marin-core` n=4 sampled jobs both failed before measuring Levanter because vLLM's JAX backend rejects
per-request seeds:

```text
requests.exceptions.HTTPError: 400 Client Error: Bad Request ... "JAX does not support per-request seed."
```

The harness now treats request seeding as a backend capability. `start_vllm_server` marks vLLM as not supporting seeds,
`run_case` passes `seed=None` for those handles, and `_send_completion` omits the `seed` field when it is `None`.
Levanter still receives per-request seeds. Replacement jobs were submitted with the no-seed fix:

- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-noseed-20260531-1857`
- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-noseed-20260531-1857`

At submission, both replacements were pending TPU capacity. The v5p job was waiting for scale-up in
`tpu_v5p-preemptible_8-us-east5-a`; the secondary job was waiting for v6e workers in
`tpu_v6e-preemptible_4-us-east1-d`.

Validation after the no-seed fix:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 32 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

The n=4 sampled rollout cases use `temperature=0.7` and `top_p=1.0`. That is an important throughput case because it
does not qualify for the streaming greedy LM-head path, but it also should not pay the full nucleus-sampling sort when
`top_p=1.0` keeps the complete vocabulary distribution. `Sampler` now skips `_apply_top_p` when every requested `top_p`
is at least `1.0`; lower top-p values still use the existing cutoff logic.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 11 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK, including Pyrefly
```

Replacement n=4 jobs with the top-p fast path in the submitted bundle:

- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-toppfast-20260531-1902`
- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-toppfast-20260531-1902`

At submission, the secondary `v6e-4,v5litepod-4` job was pending quota-tiering and the v5p job was pending v5p
scale-up. The older no-seed secondary job is still running and had reached tokenizer setup; keep it as a baseline if it
finishes, but the `toppfast` jobs are the current candidate for sampled rollout parity.

The primary v5p `toppfast` job
`/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-toppfast-20260531-1902` succeeded after one worker preemption:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets | decode/vLLM | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 3578.73 | 3585.72 | 1.145 | 2.753 | 1139.5 | 1142.1 |  |  | 1.000 |  |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 3663.30 | 3665.09 | 4.472 | 8.943 | 4458.5 | 4466.9 |  |  | 1.000 |  |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 321.38 | 322.00 | 12.745 | 119.098 | 12741.5 | 12742.7 | 4831838208 | 2 | 0.090 | fail |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 324.88 | 325.03 | 50.432 | 100.867 | 50425.6 | 50428.3 | 4831838208 | 2 | 0.089 | fail |

Environment: `TPU v5`, four visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`,
`--max-pages 1024`, `--max-prefill-size 128`, and `--max-rounds 512`. The artifact manifest recorded `summary.json`
(3470 bytes), `summary.md` (846 bytes), `env.json` (2250 bytes), and `vllm_profiles/` (2 files, 63086 bytes).

The top-p fast path fixed the obvious wasted sort for `top_p=1.0`, but sampled logprob decode is still roughly `11.2x`
behind vLLM on the cloned-prefix rollout cases. The measured Levanter decode calls are steady after warmup:
`326.73 tok/s` for `o128` and `326.79 tok/s` for `o512`, so the remaining gap is not prefill, cache sizing, or compile
warmup. The next attribution is a Levanter-only n=4 sampled run with the existing no-LM-head and LM-head-no-sampling
diagnostics:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-attrib-20260531-1925`

This run omits `--return-logprobs` so it can enable diagnostics, and should split the sampled path into transformer/RPA,
LM-head projection, and categorical/logprob sampling cost for the same `n=4` cloned-prefix shapes.

The secondary top-p-fast job
`/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-toppfast-20260531-1902` succeeded on `TPU v6 lite`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets | decode/vLLM | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 3197.72 | 3203.97 | 1.281 | 2.952 | 1277.1 | 1279.0 |  |  | 1.000 |  |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 3257.82 | 3259.41 | 5.029 | 10.230 | 5018.7 | 5024.1 |  |  | 1.000 |  |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 176.65 | 177.00 | 23.187 | 156.818 | 23182.5 | 23183.8 | 4831838208 | 2 | 0.055 | fail |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 178.76 | 178.84 | 91.656 | 183.706 | 91648.5 | 91653.0 | 4831838208 | 2 | 0.055 | fail |

Environment: `TPU v6 lite`, four visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`,
`--max-pages 1024`, `--max-prefill-size 128`, and `--max-rounds 512`. The artifact manifest recorded `summary.json`
(3464 bytes), `summary.md` (847 bytes), `env.json` (2255 bytes), and `vllm_profiles/` (2 files, 64568 bytes).

The no-seed secondary baseline
`/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-noseed-20260531-1857` also succeeded and matches the
top-p-fast result within run noise: vLLM was `2817.06`/`3262.56` tok/s for `o128`/`o512`, while Levanter was
`176.86`/`178.86` tok/s. This confirms the secondary sampled gap is not caused by the vLLM seed workaround.

The first attribution jobs were both preempted before producing final `summary.md`, but their completed measured rows
are enough to identify the bottleneck:

- Secondary `v6e-4`: sampled no-logprob decode was `179.68` tok/s for `o128` and `179.71` tok/s for the first `o512`
  warmup after compilation, matching the sampled-logprob run. The `no_lm_head` and `lm_head_no_sampling` diagnostics
  for `o128` measured in roughly 1.3s per 4096-token batch.
- Primary `v5p-8`: sampled no-logprob decode was `327.19` tok/s for measured `o128` and `327.24` tok/s for the first
  `o512` warmup after compilation, matching the sampled-logprob run. The `no_lm_head` and `lm_head_no_sampling`
  diagnostics for `o128` also measured in roughly 1.3s per 4096-token batch.

So returning generated-token logprobs is not the culprit for sampled rollout traffic. The remaining cloned-prefix gap is
inside the categorical sampling/vocab-boundary path after the LM head, not in paged attention, cache sizing, prefill, or
the LM-head matmul itself. `Sampler._sample` now avoids a duplicate full-vocab greedy argmax when all active
temperatures are sampled, preserving the mixed-temperature greedy fallback for slots whose temperature is zero.

Validation for that local sampled-path cleanup:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 12 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK, including Pyrefly
```

The secondary logprob-contract job
`/dlwh/qwen3-parity-levanter-tpuinf-v6e-v5lite4-b128-streamlogprobs-20260531-1719` succeeded on `TPU v6 lite`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 5442.10 | 5484.62 | 3.011 | 2943.8 | 2976.8 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`,
device kind `TPU v6 lite`, four visible devices, and `--levanter-streaming-greedy-lm-head --return-logprobs`.
The measured decode call completed `16256` generated tokens in `2.265s` total (`2.229s` device, `0.036s` host),
or `7176.49` raw decode tok/s. This is close to the no-logprob streaming run (`5675.53` summary, `7481.61` raw),
so returning generated-token logprobs does not reintroduce the materialized-vocab bottleneck on the secondary target.

The primary v5p logprob-contract job
`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-streamlogprobs-20260531-1719` reached warmup, then was preempted during
warmup round 1. Iris restarted it and it is running again. The last completed warmup before preemption had the expected
second-compile shape: prefill `43.305s`, then decode `39.740s` total with only `2.687s` device time and `37.010s`
submit time. This is a preemption/compile-warmup interruption, not a code or kernel failure.

The milestone matrix in the harness now matches the design-doc gate for n=1 decode-heavy rollouts: `batch={8,32,128}`
and `output_len={128,512}`. It also carries cloned-prefix diagnostics at b32 for both `output_len=128` and `512`.
The current cases are:

```text
decode_b8_i1_o128_n1
decode_b8_i1_o512_n1
decode_b32_i1_o128_n1
decode_b32_i1_o512_n1
decode_b128_i1_o128_n1
decode_b128_i1_o512_n1
decode_b32_i1_o128_n4
decode_b32_i1_o512_n4
```

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 27 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Fresh v5p jobs submitted for the expanded n=1 logprob matrix:

- `/dlwh/qwen3-parity-vllm-v5p8-n1-logprobs-20260531-1728`: vLLM baseline with `--return-logprobs` for all six n=1
  cases. It received a v5p worker and is running.
- `/dlwh/qwen3-parity-levanter-v5p8-n1-streamlogprobs-20260531-1728`: Levanter `AUTO -> TPU_INFERENCE` with
  `--levanter-streaming-greedy-lm-head --return-logprobs` for the same six n=1 cases. It received a v5p worker and is
  running.

The matching secondary-hardware n=1 logprob jobs also received workers and are running:

- `/dlwh/qwen3-parity-vllm-v6e-v5lite4-n1-logprobs-20260531-1730`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n1-streamlogprobs-20260531-1730`

The Qwen-level correctness regression now compares a tiny synthetic Qwen3 model's full causal logits against the paged
decode path over chunked prefill/decode calls. The test runs through `AUTO` plus every paged-attention backend reported
as available in the current runtime, which covers `AUTO -> REFERENCE` and explicit `REFERENCE` on CPU and will expand on
TPU runtimes as backends become available.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3.py -q
# 2 passed, 1 skipped
./infra/pre-commit.py --files lib/levanter/tests/test_qwen3.py --fix
# OK
```

The expanded v5p vLLM logprob baseline
`/dlwh/qwen3-parity-vllm-v5p8-n1-logprobs-20260531-1728` succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `vllm-tpu` | 8 | 1 | 1384.33 | 1395.14 | 0.740 | 736.8 | 738.0 |
| `decode_b8_i1_o512_n1` | `vllm-tpu` | 8 | 1 | 1411.09 | 1413.84 | 2.903 | 2898.4 | 2900.8 |
| `decode_b32_i1_o128_n1` | `vllm-tpu` | 32 | 1 | 4675.72 | 4712.24 | 0.876 | 862.6 | 871.0 |
| `decode_b32_i1_o512_n1` | `vllm-tpu` | 32 | 1 | 4897.89 | 4907.46 | 3.345 | 3327.8 | 3338.2 |
| `decode_b128_i1_o128_n1` | `vllm-tpu` | 128 | 1 | 6126.53 | 6174.40 | 2.674 | 2583.2 | 2642.2 |
| `decode_b128_i1_o512_n1` | `vllm-tpu` | 128 | 1 | 6372.31 | 6384.76 | 10.284 | 10165.8 | 10238.1 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`,
`vllm-tpu==0.19.0`, `--return-logprobs`, and the v5p `LIBTPU_INIT_ARGS` scoped-VMEM settings. Device probing was
skipped because vLLM owns libtpu in its server process. The lower b8/b128 numbers relative to earlier no-logprob vLLM
runs quantify the cost of collecting generated-token logprobs in the rollout-like API path.

The secondary-hardware vLLM logprob baseline
`/dlwh/qwen3-parity-vllm-v6e-v5lite4-n1-logprobs-20260531-1730` also succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `vllm-tpu` | 8 | 1 | 1768.55 | 1782.36 | 0.579 | 575.3 | 577.2 |
| `decode_b8_i1_o512_n1` | `vllm-tpu` | 8 | 1 | 1824.87 | 1828.43 | 2.245 | 2240.9 | 2242.6 |
| `decode_b32_i1_o128_n1` | `vllm-tpu` | 32 | 1 | 5118.37 | 5158.36 | 0.800 | 784.8 | 794.5 |
| `decode_b32_i1_o512_n1` | `vllm-tpu` | 32 | 1 | 5403.27 | 5413.83 | 3.032 | 3012.4 | 3023.0 |
| `decode_b128_i1_o128_n1` | `vllm-tpu` | 128 | 1 | 8100.95 | 8164.23 | 2.022 | 1959.7 | 1986.4 |
| `decode_b128_i1_o512_n1` | `vllm-tpu` | 128 | 1 | 8503.62 | 8520.22 | 7.707 | 7620.5 | 7672.4 |

Environment matches the v5p vLLM run except for hardware placement.

The secondary-hardware Levanter logprob matrix
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n1-streamlogprobs-20260531-1730` succeeded on `TPU v6 lite`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 875.94 | 882.78 | 1.169 | 1165.4 | 1167.0 |
| `decode_b8_i1_o512_n1` | `levanter:auto` | 8 | 1 | 1136.97 | 1139.19 | 3.603 | 3599.3 | 3600.6 |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 2743.62 | 2765.05 | 1.493 | 1476.6 | 1486.8 |
| `decode_b32_i1_o512_n1` | `levanter:auto` | 32 | 1 | 2064.53 | 2068.56 | 7.936 | 7914.4 | 7928.8 |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 5477.02 | 5519.81 | 2.991 | 2922.1 | 2955.2 |
| `decode_b128_i1_o512_n1` | `levanter:auto` | 128 | 1 | 6636.61 | 6649.57 | 9.875 | 9776.8 | 9824.6 |

Environment: `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`,
`vllm-tpu==0.19.0`, device kind `TPU v6 lite`, four visible devices, `--levanter-streaming-greedy-lm-head`, and
`--return-logprobs`. Against the secondary vLLM baseline, Levanter reaches about 50% to 78% of vLLM depending on the
case. The strongest case is `b128/o512` at `6636.61 / 8503.62 = 78.0%`; the weakest is `b32/o512` at
`2064.53 / 5403.27 = 38.2%`. This is not the milestone target, but it shows the streaming logprob path scales much
better than the earlier materialized-vocab path.

The Levanter v5p matrix was preempted during its first warmup and restarted. The earlier single-case v5p logprob run
`/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-streamlogprobs-20260531-1719` is also still running after two
preemptions. Neither currently has a code failure.

Both primary v5p Levanter logprob jobs later succeeded. The expanded n=1 matrix
`/dlwh/qwen3-parity-levanter-v5p8-n1-streamlogprobs-20260531-1728` produced:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms | decode/vLLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b8_i1_o128_n1` | `levanter:auto` | 8 | 1 | 668.01 | 673.22 | 1.533 | 1528.5 | 1530.0 | 48.3% |
| `decode_b8_i1_o512_n1` | `levanter:auto` | 8 | 1 | 1116.99 | 1119.17 | 3.667 | 3662.6 | 3664.4 | 79.2% |
| `decode_b32_i1_o128_n1` | `levanter:auto` | 32 | 1 | 1549.51 | 1561.61 | 2.643 | 1359.4 | 2636.4 | 33.1% |
| `decode_b32_i1_o512_n1` | `levanter:auto` | 32 | 1 | 3050.28 | 3056.23 | 5.371 | 5349.4 | 5356.6 | 62.3% |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 4326.17 | 4359.97 | 3.787 | 3676.5 | 3731.1 | 70.6% |
| `decode_b128_i1_o512_n1` | `levanter:auto` | 128 | 1 | 4328.77 | 4337.23 | 15.140 | 14928.8 | 15022.5 | 67.9% |

Environment: device kind `TPU v5`, 4 visible devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--levanter-streaming-greedy-lm-head`, `--return-logprobs`,
`--max-pages 2048`, `--max-prefill-size 128`, and `--max-rounds 512`. The ratios above compare to the v5p vLLM
logprob baseline from `/dlwh/qwen3-parity-vllm-v5p8-n1-logprobs-20260531-1728`.

The targeted b128 rerun `/dlwh/qwen3-parity-levanter-tpuinf-v5p8-b128-streamlogprobs-20260531-1719` also succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms | decode/vLLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b128_i1_o128_n1` | `levanter:auto` | 128 | 1 | 4763.54 | 4800.76 | 3.439 | 3357.9 | 3398.0 | 77.8% |

This targeted run used `--max-pages 512` and `--max-rounds 128`, so it is not the full-matrix configuration, but it
shows the primary v5p path can reach the same rough band as the secondary v6-lite path when cache capacity is tighter.
The remaining milestone gap is therefore not a single global kernel failure. The weakest shape is `b32/o128`, where
latency splits (`p50=1359.4ms`, `p90=2636.4ms`) suggest scheduler/request-wave behavior. The next useful tuning run
should keep the full n=1 v5p logprob matrix but lower Levanter `--max-pages` from 2048 to 1024, which is still enough
for the active rollout matrix and tests whether oversized cache capacity is suppressing the b32/b128 cases.

Submitted that tuning run as
`/dlwh/qwen3-parity-levanter-v5p8-n1-streamlogprobs-maxpages1024-20260531-1052` with default Iris disk,
`--max-pages 1024`, the same six n=1 logprob cases, `--max-rounds 512`, `--levanter-streaming-greedy-lm-head`, and
`--return-logprobs`. At submission it is pending v5p capacity:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 2 matching group(s) blocked by quota-pool tier monotonicity
```

The cloned-prefix `n > 1` cases need sampled decoding for an apples-to-apples vLLM comparison because vLLM rejects
`n > 1` under greedy sampling. Submitted a small v5p job for the two cloned-prefix diagnostics as
`/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-20260531-1100`. It runs vLLM and Levanter sequentially in the same
job with `--backend both`, `--case decode_b32_i1_o128_n4`, `--case decode_b32_i1_o512_n4`, `--temperature 0.7`,
`--return-logprobs`, `--max-pages 1024`, and the same v5p scoped-VMEM settings. This should fill the remaining
decode-heavy matrix evidence for cloned-prefix rollout traffic.

The v5p n=4 sampled job is pending capacity, so a secondary-hardware clone-prefix job was also submitted as
`/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-20260531-1104` with `--tpu v6e-4,v5litepod-4`, the same two
n=4 cases, `--backend both`, `--temperature 0.7`, and `--return-logprobs`. This is not the primary milestone target,
but should give earlier evidence for whether sampled cloned-prefix behavior has the same bottleneck as the greedy n=1
matrix.

The first secondary n=4 job failed immediately due to a submission-command mistake:

```text
error: Extra `vllm` is not defined in the project's `optional-dependencies` table
```

The issue is that `vllm` is a `marin-core` extra, while that job used `--package marin-levanter --extra vllm`. Corrected
replacement n=4 jobs were submitted with `uv run --package marin-core --extra tpu --extra vllm`:

- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-core-20260531-1108`
- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-core-20260531-1108`

The `max_pages=1024` v5p n=1 tuning job reached `Warmup levanter:auto decode_b8_i1_o128_n1 round 0`, was preempted
once, and is pending restart. There is still no code or kernel failure from that run.

The benchmark artifact path is now less fragile for `/dev/shm` runs. `summary.json`, `summary.md`, `env.json`, optional
`levanter_hlo/`, and vLLM server logs under `vllm_profiles/` are summarized in a generated `artifacts.json` manifest.
The harness logs `artifacts.json` next to `summary.md` and `env.json`, so Iris logs contain a durable record of which
evidence files were produced even when the output directory is worker-local scratch. The `vllm_profiles/` directory is
currently a log/profile artifact staging directory rather than a TPU trace capture; a future profiling pass should add
TPU profiler traces there when we need deeper vLLM attribution.

The human-readable `summary.md` table now also includes the non-throughput milestone fields that were previously only
present in `summary.json` or omitted from the table: compile-including warmup seconds, TTFT p50, HBM bytes, and compiled
shape bucket count. TTFT is still blank for the current non-streaming OpenAI completions path because neither Levanter's
handler nor the harness observes the first streamed token yet. This matters because the Iris log copy of `summary.md` is
the artifact most likely to survive scratch-output runs.

For Levanter results, `compiled_shape_count` is now populated as `2`: the fixed serving configuration has one prefill
program and one decode-loop program per backend variant. Diagnostic variants use their own decode program plus the same
prefill program, so their count is also `2`. vLLM remains blank until its TPU profile/log surface exposes an equivalent
shape-bucket count to the harness.

For mixed `--backend both` runs, `summary.json` now includes a machine-readable `comparisons` section in addition to the
raw `results`. Each comparison records the backend, vLLM baseline backend, decode ratio, total ratio, and whether it
meets the current milestone decode target of `0.85` (within 15% of vLLM). This keeps the parity gate auditable without
parsing markdown tables.

The `summary.md` table now also includes a `target` column that marks Levanter rows as `pass` or `fail` against the
same decode-ratio target whenever a vLLM row for the same case exists in the run. This keeps the Iris log summary
actionable for mixed backend runs.

Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 29 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

## Sampled Decode Follow-Up

2026-05-31: the n=4 sampled/logprob attribution runs showed that returning generated-token logprobs is not the main
cause of the cloned-prefix gap. On both v5p and secondary hardware, normal sampled decode without returned logprobs was
still in the same throughput band as sampled decode with logprobs, while the `no_lm_head` and `lm_head_no_sampling`
diagnostics ran much faster. The current bottleneck is therefore the sampled token-selection path over the full Qwen3
vocabulary after the LM-head projection, not RPA, cache update, or the LM-head matmul itself.

The first local optimization replaced Haliax/JAX categorical sampling with inverse-CDF sampling for the full-distribution
`top_p=1.0` case. This reduces random-number generation from one random value per vocab element to one random value per
sample row, while preserving exact sampling from the softmax distribution. Two mixed backend jobs are running that carry
this change:

- `/dlwh/qwen3-parity-both-v6e-v5lite4-n4-sampled-logprobs-cdf-20260531-1946`
- `/dlwh/qwen3-parity-both-v5p8-n4-sampled-logprobs-cdf-20260531-1946`

As of the latest poll, both jobs are running. The secondary job has completed vLLM measurement and reached Levanter
warmup for `decode_b32_i1_o128_n4`; the v5p job has a worker but has not yet emitted benchmark progress lines in the
recent log filter.

Inspecting the pinned `tpu-inference==0.19.0` wheel without installing it on macOS showed that its JAX sampler does not
provide a custom TPU sampling kernel: it uses `jax.random.categorical`, then computes logprobs separately. However, it
does explicitly constrain logits to shard as `(batch, unsharded_vocab)` before sampling, with a comment that unsharding
the logits avoids latency increases. The Levanter benchmark mesh maps `vocab` to the `model` axis, so Levanter was still
sampling over a model-sharded vocab axis.

Levanter's sampler now mirrors that `tpu-inference` sampler placement rule for non-greedy sampled decoding: when a
Haliax axis mapping maps the sampler vocab axis, the sampler applies a sharding constraint with the vocab axis removed
before temperature scaling, optional top-p, sampling, and selected-token logprob computation. This is inactive for CPU
tests and any context without an axis mapping.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 13 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK
```

Follow-up jobs with the vocab-unshard sampler were submitted as Levanter-only checks so they can compare against the CDF
mixed runs without re-running vLLM:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-logprobs-unshard-20260531-1256`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-logprobs-unshard-20260531-1256`

At the latest poll, both unshard follow-ups were running. Early steady decode lines show this change did not materially
move the cloned-prefix sampled bottleneck: the secondary run remains around `178.7 tok/s`, and the primary v5p run
reached about `320.3 tok/s` on measured `decode_b32_i1_o128_n4`, essentially matching the earlier top-p-fast band.

The CDF change also appears unhelpful for this shape. The pinned `tpu-inference==0.19.0` sampler uses
`jax.random.categorical`, not a CDF scan, after constraining sampled logits to `(batch, unsharded_vocab)`. Levanter now
keeps the `tpu-inference`-style unsharded vocab placement but switches the full-distribution sampled path back to
batched `jax.random.categorical`, removing the full-vocab `cumsum` from sampled decode. This is a better apples-to-apples
check against the vLLM TPU sampler than the CDF experiment.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 14 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py --fix
# OK
```

Levanter-only categorical follow-up jobs were submitted with the same two n=4 sampled/logprob cases, `--top-p 1.0`,
`--temperature 0.7`, `--max-pages 1024`, `--max-prefill-size 128`, `--max-rounds 512`, default Iris disk, and `/dev/shm`
for HF/JAX/output scratch:

- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-logprobs-categorical-20260531-2009`: running/building on
  secondary hardware at submission poll.
- `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-logprobs-categorical-20260531-2009`: pending v5p scale-up
  (`tpu_v5p-preemptible_8-us-east5-a`).

The heartbeat watcher now tracks these alongside the CDF and vocab-unshard jobs. None of the Iris jobs should be stopped
or restarted.

The categorical follow-ups both eventually succeeded after one preemption and wrote final `summary.md`/`env.json`
artifacts. The v5p result:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 319.84 | 320.47 | 12.806 | 12802.6 | 12803.7 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 325.45 | 325.61 | 50.343 | 50337.0 | 50339.9 |

Environment: device kind `TPU v5`, 4 visible TPU devices, `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`.

The secondary result:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 268.40 | 268.92 | 15.261 | 15257.6 | 15259.2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 270.75 | 270.89 | 60.512 | 60506.2 | 60509.5 |

Environment: device kind `TPU v6 lite`, 4 visible TPU devices, same package versions. Compared with the earlier
sampled/logprob vLLM baselines, Levanter is still only about `8.9%` of vLLM on v5p and `8.3-8.4%` on secondary
hardware for the cloned-prefix sampled/logprob cases. The sampler micro-variants have therefore not closed the
milestone gap.

The next attribution step is profile/HLO evidence for the sampled/logprob decode loop. I fixed the benchmark artifact
path so `--dump-levanter-kernels` traces the actual requested logprob mode instead of hard-coding
`return_logprobs=False`, and so it passes the current `use_streaming_greedy_lm_head` static argument to the generation
loop. Without that fix, a sampled/logprob profiling run would either fail during artifact tracing or dump the wrong
decode contract.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 46 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

Submitted `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-hlodump-20260531-1321` as a Levanter-only v5p HLO artifact run:
`decode_b32_i1_o128_n4`, `--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`, `--dump-levanter-kernels`,
`--max-pages 1024`, `--max-prefill-size 128`, `--max-rounds 512`, default Iris disk, and `/dev/shm` output/cache paths.
At submission poll it was pending v5p scale-up:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'tpu_v5p-preemptible_8-us-central1-a'
```

Submitted the same artifact run on secondary hardware as
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-hlodump-20260531-1326` with `--tpu v6e-4,v5litepod-4`, so the
fixed HLO dump path can be exercised while the v5p queue is tier-blocked.

Both HLO artifact runs succeeded and proved that the fixed dump path can trace sampled/logprob generation:

| job | device | decode tok/s | artifact directory |
| --- | --- | --- | --- |
| `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-hlodump-20260531-1321` | `TPU v5` | 93.23 | `levanter_hlo`, 4 files, 771491 bytes |
| `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-hlodump-20260531-1326` | `TPU v6 lite` | 80.53 | `levanter_hlo`, 4 files, 773657 bytes |

Those one-round HLO dump runs are slower than the warmed two-case categorical runs because they include trace/dump and
fresh-compile effects, so they should be treated as artifact probes rather than throughput baselines. The important miss
is that the HLO files were written under worker-local `/dev/shm`; after the jobs exited, `iris task exec` could not read
them because completed tasks are no longer running. The artifact manifest survived in logs, but the HLO contents did not.

The benchmark now writes durable `hlo_summary.json` and `hlo_summary.md` whenever `levanter_hlo/` exists. The summary
keeps `/dev/shm` as the bulky artifact location but logs stable counts and representative HLO lines for collectives,
sampling RNG/categorical paths, sort/top-k, logprob operations, and RPA/custom calls. That should make the next HLO dump
run actionable from Iris logs even if the scratch directory disappears with the worker.

Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
```

Submitted replacement HLO-summary probes:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-sampled-hlosummary-20260531-2036`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-hlosummary-20260531-2036`

The secondary run succeeded and confirms the suspected sampled-decode hotspot from durable logs. Its measured table:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 81.34 | 81.49 | 50.360 | 86.545 | 50355.7 | 50357.4 |

`artifacts.json` includes `hlo_summary.json`, `hlo_summary.md`, and `levanter_hlo/` as intended. The HLO summary table
for the secondary run reports:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 150782 | 10646 | 0 | 11 | 91 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 194500 | 16424 | 0 | 15 | 224 | 0 | 39 |

The first summary pass over-counted sort/top-k because it matched the `indices_are_sorted` attribute on gathers/scatters,
but the representative lines still show the real issue: full-vocabulary sampling lowers to `stablehlo.exponential` over
`tensor<128x151936xf32>` and `argsort` over `tensor<128x151936...>`. That is exactly the shape expected from
`jax.random.categorical`'s Gumbel-max implementation and explains why the sampled/logprob path remains far behind vLLM
even after removing top-p sorting and unsharding the vocab axis. The local HLO-summary matcher now counts only
`stablehlo.sort`, `argsort`, and explicit top-k spellings to avoid this noise in future runs.

The v5p replacement was still running at the latest poll and had progressed through Qwen3 weight loading. A heartbeat is
watching both replacement jobs every 10 minutes; it should capture the v5p `hlo_summary.md` when the job finishes.

Validation after tightening the HLO matcher:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

Later checks showed that both HLO-summary jobs succeeded. They were useful for full-vocabulary attribution, but not for
the explicit top-k path: their argv lacked `--top-k`, and both v5p and secondary HLO summaries still showed
`stablehlo.exponential` and `argsort` over `tensor<128x151936...>` in `gen_loop.hlo.txt.gz`. That explains the old
full-vocab categorical bottleneck but does not prove whether the new top-k sampler narrows the HLO to 4096 candidates.

The sharded top-k replacement run
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-sampled-topk4096-shardedtopk-fix-mtpr32-20260531-2209` succeeded on
`TPU v6 lite` with the same installed `tpu-inference==0.19.0`/`vllm-tpu==0.19.0` stack. This confirms the axis-size fix,
but not a throughput win:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3158.80 | 3164.97 | 1.297 | 95.774 | 1293.6 | 1294.5 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3157.13 | 3163.29 | 1.297 | 95.774 | 1293.1 | 1295.5 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3149.69 | 3155.84 | 1.300 | 95.774 | 1296.9 | 1298.0 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3342.85 | 3344.48 | 4.901 | 9.773 | 4895.5 | 4898.8 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3326.70 | 3328.32 | 4.925 | 9.773 | 4920.2 | 4922.3 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3344.76 | 3346.40 | 4.898 | 9.773 | 4892.2 | 4896.0 | 4831838208 | 2 |

Compared with the known-best secondary top-k baseline (`3106.32`/`3338.26` tok/s), this is a small `o128` gain and
essentially flat `o512`. Compared with the secondary vLLM baseline (`4049.48`/`4107.03` tok/s), Levanter remains about
78% and 81% of vLLM on the two sampled rollout cases, still below the 85% milestone line.

I tightened the HLO summary matcher again so `sort_or_topk` is counted from actual `stablehlo.sort`, `argsort`, and
top-k spellings, not incidental `indices_are_sorted` metadata on gather/scatter lines. Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
```

Submitted top-k HLO-summary probes to produce the missing durable attribution evidence:

- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-hlosummary-20260531-2218`
- `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-hlosummary-20260531-2218`

Both run `decode_b32_i1_o128_n4` with `--top-k 4096`, `--temperature 0.7`, `--return-logprobs`,
`--max-tokens-per-round 32`, `--warmup-rounds 1`, `--measure-rounds 1`, and `--dump-levanter-kernels`.

The first poll found the secondary probe still pending on capacity:

```text
Scheduler: Insufficient TPUs (need 4, available 0)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 5 matching group(s) blocked by quota-pool tier monotonicity
```

The secondary probe later started running.

The v5p probe
`/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-hlosummary-20260531-2218` succeeded and produced the missing explicit
top-k HLO attribution:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 141.87 | 142.14 | 28.872 | 48.360 | 28867.5 | 28869.6 | 4831838208 | 2 |

The slow token rate is expected because the run dumps HLO with one warmup and one measure round; use the HLO, not the
throughput, as the signal. The env confirms `TPU v5`, 4 visible devices, `jax==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, `vllm-tpu==0.19.0`, `--top-k 4096`, `--temperature 0.7`, `--return-logprobs`, and the v5p
`LIBTPU_INIT_ARGS`.

The v5p HLO summary:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 152343 | 10798 | 0 | 12 | 13 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 196928 | 16657 | 0 | 16 | 13 | 0 | 39 |

Representative lines show the current top-k candidate path is present and narrowed before the main categorical path:

```text
%values, %indices = chlo.top_k(%6481, k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6502 = call @argsort_662(%6500) : (tensor<128x4096xf32>) -> tensor<128x4096xi32>
%6511 = stablehlo.exponential %6510 : tensor<128x4096xf32>
```

There is still an unexpected full-vocabulary random path in the same graph:

```text
%6636 = stablehlo.exponential %6635 : tensor<128x151936xf32>
```

That means the top-k fix did narrow at least one categorical path to 4096 candidates, but the HLO still contains a
full-vocabulary categorical-like operation. The next inspection should identify whether that full-vocab branch comes
from the mixed-temperature fallback in `Sampler._sample`, the `lax.cond` branches being retained in HLO, or another
token/logprob path. If it is only a dead conditional branch, the runtime gap is more likely `chlo.top_k`/scheduler. If it
executes, removing that branch is a concrete next optimization.

The secondary HLO probe
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-hlosummary-20260531-2218` also succeeded on `TPU v6 lite`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 130.45 | 130.71 | 31.398 | 59.256 | 31395.2 | 31396.0 | 4831838208 | 2 |

The env matches the same Qwen3 sampled top-k contract, with `TPU v6 lite`, `jax==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`. Its HLO summary is effectively the same shape as v5p:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 152320 | 10798 | 0 | 13 | 12 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 196550 | 16657 | 0 | 15 | 13 | 0 | 39 |

Representative HLO again shows `chlo.top_k` over `tensor<128x151936xf32>` producing `tensor<128x4096xf32>` and
`tensor<128x4096xi32>`, so the secondary run confirms the top-k candidate lowering is not v5p-specific.

I also inspected `tpu-inference==0.19.0` locally from the wheel. Its TPU sampler does not reduce logits to a 4096-wide
candidate distribution before categorical sampling. It scales logits, applies `topk_mask` and `topp_mask`, and then
calls `jax.random.categorical` over the masked full-vocabulary logits. `topk_mask` is a binary-search threshold mask
from `tpu_inference.layers.common.binary_search`: it does 32 reductions over the vocab axis, masks logits below the
threshold, and relies on the vocab axis being unsharded. This is a different kernel shape from Levanter's current
`jax.lax.top_k(..., 4096)` candidate extraction plus 4096-way categorical path. Since the sharded-candidate attempt was
flat, the next concrete performance experiment should be a Levanter sampler mode that matches `tpu-inference`'s
threshold-mask top-k implementation and measures it against the existing top-k candidate path on the same v5p/v6e smoke
cases.

Local follow-up on the unexpected full-vocabulary exponential found a simpler immediate fix: the sampler's outer
all-greedy `lax.cond` traced the full-vocabulary greedy/logprob branch even for explicit top-k sampled requests. The
top-k path now enters `_sample` directly when `top_ks` is provided, so the regular greedy fast path is preserved for
non-top-k serving while the RL top-k path no longer keeps the full-vocab greedy branch in HLO.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 13 passed
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 41 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/test_sampler.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

The new sampler regression lowers a small top-k sample and checks the HLO exponential path is on the candidate axis
(`tensor<2xf32>` in the test) and not the original vocabulary axis (`tensor<8xf32>`). This is local CPU evidence only;
the TPU confirmation jobs below are the authoritative serving-shaped evidence.

Submitted confirmation HLO probes:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-skipgreedy-hlosummary-20260531-2234`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-skipgreedy-hlosummary-20260531-2234`

Both use the same `decode_b32_i1_o128_n4`, `n=4`, `--top-k 4096`, `--temperature 0.7`, `--return-logprobs`,
`--max-tokens-per-round 32`, `--warmup-rounds 1`, `--measure-rounds 1`, and `--dump-levanter-kernels` contract as the
previous top-k HLO probes, with default Iris disk and `/dev/shm` model/cache/output paths. The initial status check
found v5p pending on capacity/tier monotonicity and the secondary v6e/v5lite job running. A heartbeat
`poll-qwen3-skip-greedy-hlo-probes` is watching both jobs every 10 minutes.

The secondary skip-greedy confirmation job
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-skipgreedy-hlosummary-20260531-2234` succeeded on
`TPU v6 lite` with the same `tpu-inference==0.19.0`/`vllm-tpu==0.19.0` stack:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 131.69 | 131.95 | 31.104 | 58.893 | 31098.9 | 31101.0 | 4831838208 | 2 |

The low throughput is expected for a one-round HLO-dump run; the durable HLO is the useful signal. The HLO summary
confirmed the sampler fix removed the earlier full-vocabulary exponential examples from the serving-shaped top-k path:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 151596 | 10696 | 0 | 12 | 12 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 195436 | 16509 | 0 | 13 | 13 | 0 | 39 |

Representative lines now show `chlo.top_k` narrowing from the full vocabulary to 4096 candidates, followed by sampled
exponentials over the candidate axis:

```text
%values, %indices = chlo.top_k(%6468, k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6498 = stablehlo.exponential %6497 : tensor<128x4096xf32>
%6600 = stablehlo.exponential %6599 : tensor<128x4096xf32>
```

The latest status check found the matching v5p confirmation job
`/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-skipgreedy-hlosummary-20260531-2234` running, past Qwen3 weight loading
and server startup. It still needs its v5p HLO summary captured once it reaches a terminal state.

The earlier vLLM smoke failure was not capacity-related. It failed after vLLM finished a warmup/measure case because
the benchmark tried to call `jax.devices()` for env output while the vLLM server process still owned libtpu:

```text
RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid 2001.
```

The benchmark now treats env capture as backend-aware: Levanter env snapshots include JAX devices, while vLLM snapshots
record version fields and a `devices_skipped` reason instead of initializing JAX. Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 13 passed
```

The matching v5p skip-greedy confirmation job
`/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-skipgreedy-hlosummary-20260531-2234` also succeeded:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 137.48 | 137.75 | 29.793 | 49.894 | 29788.1 | 29790.4 | 4831838208 | 2 |

The env confirms `TPU v5`, `jax==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`. The HLO
summary is now aligned with the secondary run: no representative full-vocabulary sampled exponential remains, and the
serving-shaped top-k path narrows from full vocab to 4096 candidates before categorical sampling:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 151626 | 10696 | 0 | 11 | 13 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 195811 | 16509 | 0 | 14 | 13 | 0 | 39 |

Representative lines:

```text
%values, %indices = chlo.top_k(%6468, k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6498 = stablehlo.exponential %6497 : tensor<128x4096xf32>
%6600 = stablehlo.exponential %6599 : tensor<128x4096xf32>
```

The fixed-env vLLM replacement job
`/dlwh/qwen3-parity-vllm-v6e-v5lite4-n4-topk4096-fixedenv-20260531-1544` succeeded on the same secondary TPU contract.
The env capture no longer initializes JAX devices while vLLM owns libtpu; it records version fields plus
`devices_skipped`.

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 4100.50 | 4108.51 | 0.999 | 1.408 | 994.3 | 997.1 |
| `decode_b32_i1_o128_n4` | `vllm-tpu` | 32 | 4 | 4101.65 | 4109.66 | 0.999 | 1.408 | 995.3 | 996.7 |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 4197.61 | 4199.66 | 3.903 | 3.905 | 3892.2 | 3897.4 |
| `decode_b32_i1_o512_n4` | `vllm-tpu` | 32 | 4 | 4163.97 | 4166.02 | 3.891 | 3.905 | 3878.4 | 3885.0 |

Against the latest secondary Levanter top-k candidate run (`3158.8` / `3342.9` tok/s), Levanter is roughly 77% on
`o128` and 80% on `o512`, still below the 85% milestone line. The remaining top-k candidate HLO now looks clean, so the
next experiment is a true sampler-shape A/B rather than another full-vocab cleanup.

I added an explicit Levanter sampler top-k mode for that A/B:

- `candidate` keeps the current `jax.lax.top_k(..., 4096)` candidate distribution path and remains the default.
- `threshold_mask` keeps the full vocabulary axis, masks logits below a top-k threshold via 32 reduction-only binary
  search iterations, then samples from the masked full-vocabulary distribution. This matches the shape of
  `tpu-inference`/vLLM TPU sampling more closely, but is experimental until TPU throughput is measured.

Validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 15 passed
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 33 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/layers/sampler.py lib/levanter/src/levanter/inference/engine.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_sampler.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py .agents/projects/levanter_tpu_inference_parity/research.md --fix
# OK
```

Submitted threshold-mask HLO probes:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-thresholdmask-hlosummary-20260531-1551`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-thresholdmask-hlosummary-20260531-1553`

Both use `decode_b32_i1_o128_n4`, `n=4`, `--top-k 4096`, `--levanter-sampler-top-k-mode threshold_mask`,
`--temperature 0.7`, `--return-logprobs`, `--max-tokens-per-round 32`, `--warmup-rounds 1`,
`--measure-rounds 1`, and `--dump-levanter-kernels`. Initial status:

```text
v5p: pending; waiting for worker scale-up in scale group 'tpu_v5p-preemptible_8-us-east5-a'
v6e/v5lite: pending; waiting for worker scale-up in scale group 'tpu_v6e-preemptible_4-us-east5-b'
```

The active heartbeat `poll-qwen3-skip-greedy-hlo-probes` now watches both threshold-mask jobs and should capture the
summary/env/HLO evidence or the smallest concrete failure.

Local threshold-mask follow-up:

```text
uv run python -m py_compile lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py -q
# 16 passed
```

The added sampler test covers per-row `top_k` values (`[1, 3]`) in threshold-mask mode, matching the mixed-request shape
used by serving.

The secondary threshold-mask HLO probe
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-thresholdmask-hlosummary-20260531-1553` succeeded, but it is a
clear negative result for throughput:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 100.34 | 100.54 | 40.821 | 72.946 | 40818.0 | 40819.0 | 4831838208 | 2 |

The env confirms `levanter_sampler_top_k_mode=threshold_mask`, TPU v6 lite, `jax==0.9.2`, and `libtpu==0.0.39`.
The HLO confirms why it is slow: the masked full-vocabulary categorical path avoids `chlo.top_k`, but it introduces
full-vocabulary argsorts and full-vocabulary sampled exponentials:

```text
%6784 = func.call @argsort_653(%6782) : (tensor<32x151936xf32>) -> tensor<32x151936xi32>
%6793 = stablehlo.exponential %6792 : tensor<32x151936xf32>
%6491 = call @argsort_661(%6489) : (tensor<128x151936xf32>) -> tensor<128x151936xi32>
%6500 = stablehlo.exponential %6499 : tensor<128x151936xf32>
```

This means the Levanter-side threshold-mask experiment does not reproduce the fast `tpu-inference` sampling kernel
shape; it mostly trades `chlo.top_k` over the vocabulary for worse full-vocabulary sampling/sort work. The current
candidate-topk path is still the better Levanter baseline.

The primary v5p threshold-mask HLO probe
`/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-thresholdmask-hlosummary-20260531-1551` also succeeded and confirmed the
same negative result:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 114.35 | 114.57 | 35.821 | 58.255 | 35817.0 | 35819.0 | 4831838208 | 2 |

The env confirms `levanter_sampler_top_k_mode=threshold_mask`, TPU v5, `jax==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`. The HLO again contains full-vocabulary sampled work:

```text
%6784 = func.call @argsort_653(%6782) : (tensor<32x151936xf32>) -> tensor<32x151936xi32>
%6793 = stablehlo.exponential %6792 : tensor<32x151936xf32>
%6491 = call @argsort_661(%6489) : (tensor<128x151936xf32>) -> tensor<128x151936xi32>
%6500 = stablehlo.exponential %6499 : tensor<128x151936xf32>
```

The threshold-mask branch should therefore remain experimental/disabled-by-default, and the next throughput work should
continue from the candidate-topk path. Since `max_tokens_per_round=32` is still the best measured scheduler point and
`64`/`128` regress on secondary hardware, I submitted the missing smaller scheduler bracket:

- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-mtpr16-20260531-1604`
- `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-mtpr16-20260531-1604`

Both are Levanter-only candidate-topk runs for `decode_b32_i1_o128_n4` and `decode_b32_i1_o512_n4` with
`--top-k 4096`, `--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`, `--max-tokens-per-round 16`,
`--max-rounds 512`, `--max-pages 1024`, `--max-prefill-size 128`, default Iris disk, and `/dev/shm` cache/output
scratch.

Both `mtpr16` jobs succeeded, and the smaller decode chunk regresses on both target families:

| job | TPU | case | measured decode tok/s |
| --- | --- | --- | --- |
| `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-mtpr16-20260531-1604` | TPU v6 lite | `decode_b32_i1_o128_n4` | 127.83, 1955.39, 1949.06 |
| `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-mtpr16-20260531-1604` | TPU v6 lite | `decode_b32_i1_o512_n4` | 2100.72, 2096.77, 2100.14 |
| `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-mtpr16-20260531-1604` | TPU v5 | `decode_b32_i1_o128_n4` | 139.66, 2255.61, 2142.32 |
| `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-mtpr16-20260531-1604` | TPU v5 | `decode_b32_i1_o512_n4` | 2337.93, 2342.26, 2337.21 |

The first `o128` measured row in each run is polluted by a second shape compile and should not be treated as steady
state, but the remaining rows are still well below the `mtpr32` candidate-topk baselines:

| TPU family | `mtpr16` steady `o128` | `mtpr32` steady `o128` | `mtpr16` steady `o512` | `mtpr32` steady `o512` |
| --- | --- | --- | --- | --- |
| TPU v6 lite / v5lite secondary | ~1950 tok/s | ~3158 tok/s | ~2100 tok/s | ~3343 tok/s |
| TPU v5p primary | ~2200 tok/s | ~2672 tok/s | ~2340 tok/s | ~2926 tok/s |

The envs match the intended comparison: `levanter_sampler_top_k_mode=candidate`, `max_tokens_per_round=16`,
`jax==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`. The scheduler bracket now shows
`mtpr32` as the best measured point among `16`, `32`, `64`, and `128`; further work should move back to sampler/logprob
kernel cost attribution rather than chunk-size tuning.

To make the remaining-gap attribution satisfy the design doc's "profile artifacts" requirement rather than relying only
on HLO summaries, the benchmark harness now has a `--profile-levanter` flag. It wraps measured Levanter rounds in
`jax.profiler.start_trace(..., create_perfetto_trace=True)`, labels the round with `StepTraceAnnotation`, writes traces
under `levanter_profiles/<backend>/<case>/measure_<round>/`, and includes `levanter_profiles/` in `artifacts.json`.

Validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 34 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Submitted Levanter-only top-k profile probes for the current best candidate path:

- `/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-profile-20260531-2317`
- `/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-profile-20260531-2317`

Both run `decode_b32_i1_o128_n4` with `--top-k 4096`, `--temperature 0.7`, `--top-p 1.0`, `--return-logprobs`,
`--max-tokens-per-round 32`, `--warmup-rounds 2`, `--measure-rounds 1`, `--dump-levanter-kernels`, and
`--profile-levanter`. The v5p job is pending worker scale-up; the secondary `v6e-4,v5litepod-4` job has a worker and is
building. The heartbeat `poll-qwen3-top-k-profile-probes` watches both jobs and should capture `summary.md`,
`env.json`, `hlo_summary.md`, and the `artifacts.json` entry proving `levanter_profiles/` was produced.

The secondary profile probe
`/dlwh/qwen3-parity-levanter-v6e-v5lite4-n4-topk4096-profile-20260531-2317` succeeded and produced the expected
profile artifacts. The first warmup included compile overhead, the second warmup reached a steady raw decode line of
`3478.74 tok/s`, and the profiled measured round completed at `2856.45 decode tok/s`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2856.45 | 2862.03 | 1.434 | 90.546 | 1427.2 | 1431.3 | 4831838208 | 2 |

The env confirms `profile_levanter=true`, `levanter_sampler_top_k_mode=candidate`, `top_k=4096`, `top_p=1.0`,
`temperature=0.7`, TPU v6 lite, `jax==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`.
The HLO still has the expected candidate-topk shape rather than the threshold-mask regression:

```text
%values, %indices = chlo.top_k(%6468, k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6489 = call @argsort_654(%6487) : (tensor<128x4096xf32>) -> tensor<128x4096xi32>
%6498 = stablehlo.exponential %6497 : tensor<128x4096xf32>
```

The artifact manifest proves the profiling payload was written:

| artifact | kind | bytes | file count |
| --- | --- | --- | --- |
| `summary.json` | file | 799 | - |
| `summary.md` | file | 423 | - |
| `env.json` | file | 1620 | - |
| `hlo_summary.json` | file | 40438 | - |
| `hlo_summary.md` | file | 27434 | - |
| `levanter_hlo` | directory | 776338 | 4 |
| `levanter_profiles` | directory | 222611680 | 3 |

This is enough to show the harness can emit profile artifacts for the best current candidate-topk Levanter path. The
profiled measured throughput is lower than the earlier unprofiled secondary `mtpr32` rows (`~3158 tok/s`), so use the
trace for attribution and the unprofiled rows for throughput comparison.

The primary v5p profile probe
`/dlwh/qwen3-parity-levanter-v5p8-n4-topk4096-profile-20260531-2317` reached a v5p-8 worker, loaded Qwen3, wrote HLO,
started the Levanter server, and began warmup round 0. It was then preempted before any measured/profiled round
completed:

```text
preemption_count=1
task_state_counts={"pending": 1}
task error: Worker marin-tpu-v5p-preemptible-8-us-east5-a-20260531-2315-b55bb0cd-worker-0 failed: worker ping threshold exceeded
```

Iris still reports the job as `JOB_STATE_RUNNING`, so this is a capacity/preemption wait rather than a benchmark failure
so far. The heartbeat should keep watching for replacement capacity and capture the same `summary.md`, `env.json`,
`hlo_summary.md`, and `artifacts.json` evidence if the retry reaches the profiled measured round.

The same v5p profile probe then retried on replacement capacity and succeeded after one preemption. It ran on `TPU v5`
with a `v5p-8` worker, `jax==0.9.2`, `libtpu==0.0.39`, `tpu-inference==0.19.0`, `vllm-tpu==0.19.0`,
`levanter_sampler_top_k_mode=candidate`, `top_k=4096`, `top_p=1.0`, `temperature=0.7`, `profile_levanter=true`, and
`LIBTPU_INIT_ARGS=--xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_scoped_vmem_limit_kib=50000 --xla_tpu_use_enhanced_launch_barrier=true`.

The retry's second warmup reached a steady raw decode line of `3073.73 tok/s`; the profiled measured round's decode body
logged `3012.09 tok/s`, then the wall-clock measured row completed at `2534.91 decode tok/s` after profiler overhead:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2534.91 | 2539.86 | 1.616 | 77.093 | 1608.7 | 1611.2 | 4831838208 | 2 |

The HLO summary again shows the candidate-topk shape and no collectives in the serving loop:

| file | collective | rng_sampling | sort_or_topk | custom_call |
| --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 0 | 11 | 13 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 0 | 14 | 13 | 39 |

Representative HLO lines:

```text
%values, %indices = chlo.top_k(%6468, k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6489 = call @argsort_654(%6487) : (tensor<128x4096xf32>) -> tensor<128x4096xi32>
%6498 = stablehlo.exponential %6497 : tensor<128x4096xf32>
```

The v5p artifact manifest proves profiler output was produced:

| artifact | kind | bytes | file count |
| --- | --- | --- | --- |
| `summary.json` | file | 799 | - |
| `summary.md` | file | 423 | - |
| `env.json` | file | 1693 | - |
| `hlo_summary.json` | file | 40466 | - |
| `hlo_summary.md` | file | 27447 | - |
| `levanter_hlo` | directory | 773895 | 4 |
| `levanter_profiles` | directory | 292965264 | 3 |

With v5p and secondary profile artifacts in hand, the remaining vLLM gap can now be attributed more concretely: the
attention path is using the `tpu-inference` RPA custom calls and no collectives, while the sampled top-k path still
contains full-vocabulary `chlo.top_k` plus top-k argsort and Gumbel/exponential sampling over `128x4096`. That is the
specific kernel/sampling limitation to attack next if we want to move from the current best unprofiled v5p ratio
(`2671.63 / 3580.42 ~= 74.6%` on `o128`, `2926.44 / 3639.97 ~= 80.4%` on `o512`) to the 85-90% parity band.

The benchmark harness now writes `prompt_corpus.json` before any backend starts, recording the exact prompt text, token
IDs, target input/output lengths, active sequence count, `n`, and request count for each case. This makes the rollout
matrix prompts durable and reproducible rather than implicit in `_prompt_for_token_count`. Local validation for that
change:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 36 passed
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

2026-05-31: changed sampled top-k sampling from `jax.random.categorical` to explicit Gumbel-max and ran a new
secondary unprofiled probe,
`/dlwh/qwen3-parity-levanter-gumbel-v6e-v5lite4-n4-topk4096-20260531-2340`. It succeeded on `TPU v6 lite` with
`tpu-inference==0.19.0` and `vllm-tpu==0.19.0` installed:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3109.17 | 3115.24 | 1.317 | 90.635 | 1312.9 | 1315.4 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3156.01 | 3162.18 | 1.298 | 90.635 | 1294.1 | 1295.8 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3146.64 | 3152.79 | 1.302 | 90.635 | 1297.6 | 1299.9 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3337.09 | 3338.72 | 4.910 | 9.761 | 4903.5 | 4907.1 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3340.91 | 3342.54 | 4.904 | 9.761 | 4898.4 | 4901.0 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3336.01 | 3337.64 | 4.911 | 9.761 | 4901.4 | 4903.8 | 4831838208 | 2 |

This did not materially improve throughput over the previous secondary `mtpr32` candidate-topk baseline. The HLO still
showed candidate-space `argsort` over `128x4096`; inspection showed it came from the dynamic `top_p=1.0` path, not from
the Gumbel-max replacement. `_apply_optional_top_p` used `lax.cond`, which still traces the nucleus-sampling sort branch
even when every request's runtime `top_p` is one.

The engine now computes whether any admitted request actually needs nucleus filtering at the Python request boundary.
When all requests have `top_p >= 1.0`, prefill, clone sampling, and decode pass `top_ps=None` into the JIT, making
top-p filtering a static no-op. The sampler also short-circuits concrete scalar `top_ps=1.0` before building the dynamic
branch. Local validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/layers/sampler.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_sampler.py lib/levanter/tests/inference/test_engine.py -q
# 25 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/src/levanter/layers/sampler.py lib/levanter/tests/inference/test_engine.py lib/levanter/tests/test_sampler.py --fix
# OK
```

Follow-up top-p-static TPU probes were submitted and are watched by heartbeat `poll-qwen3-gumbel-sampler-tpu-runs`:

- `/dlwh/qwen3-parity-levanter-toppstatic-v6e-v5lite4-n4-topk4096-20260531-2350`
- `/dlwh/qwen3-parity-levanter-toppstatic-v5p8-n4-topk4096-20260531-2350`

The first secondary top-p-static submission requested `--memory 120GB`, which excludes available v5lite workers with
about 116.7GB. A replacement secondary probe was submitted at `--memory 110GB` without stopping the original pending
job:

- `/dlwh/qwen3-parity-levanter-toppstatic-v6e-v5lite4-n4-topk4096-mem110-20260531-2352`

The remaining old v5p Gumbel probe
`/dlwh/qwen3-parity-levanter-gumbel-v5p8-n4-topk4096-20260531-2340` succeeded after one preemption and served as the
pre-top-p-static comparison point. It ran on `TPU v5` with 4 visible devices, `jax==0.9.2`, `libtpu==0.0.39`,
`tpu-inference==0.19.0`, and `vllm-tpu==0.19.0`:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2811.30 | 2816.79 | 1.457 | 76.186 | 1452.9 | 1454.5 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2737.54 | 2742.88 | 1.496 | 76.186 | 1493.1 | 1494.3 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 2733.60 | 2738.94 | 1.498 | 76.186 | 1494.8 | 1496.3 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2972.88 | 2974.33 | 5.511 | 11.025 | 5505.1 | 5508.2 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2975.44 | 2976.89 | 5.506 | 11.025 | 5500.1 | 5503.7 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2972.64 | 2974.09 | 5.512 | 11.025 | 5505.5 | 5508.8 | 4831838208 | 2 |

Its HLO confirmed that explicit Gumbel-max alone did not remove the candidate-space top-p sort:

```text
%values, %indices = chlo.top_k(..., k = 4096) : tensor<128x151936xf32> -> (tensor<128x4096xf32>, tensor<128x4096xi32>)
%6489 = call @argsort_654(%6487) : (tensor<128x4096xf32>) -> tensor<128x4096xi32>
%6498 = stablehlo.exponential %6497 : tensor<128x4096xf32>
%6515 = call @argsort_692(%6490) : (tensor<128x4096xi32>) -> tensor<128x4096xi32>
```

The top-p-static v5p follow-up
`/dlwh/qwen3-parity-levanter-toppstatic-v5p8-n4-topk4096-20260531-2350` then succeeded on `TPU v5` with the same
dependency set. It materially improved the primary v5p target:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3194.45 | 3200.68 | 1.282 | 74.595 | 1278.4 | 1279.9 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3167.76 | 3173.95 | 1.293 | 74.595 | 1289.5 | 1291.3 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3151.51 | 3157.66 | 1.300 | 74.595 | 1296.0 | 1297.7 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3558.67 | 3560.41 | 4.604 | 9.236 | 4598.4 | 4601.0 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3559.75 | 3561.49 | 4.603 | 9.236 | 4596.7 | 4599.6 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3555.95 | 3557.69 | 4.607 | 9.236 | 4601.6 | 4605.1 | 4831838208 | 2 |

The HLO summary shrank from `sort_or_topk=13` to `sort_or_topk=9` in `gen_loop.hlo` and from `13` to `7` in
`run_prefill.hlo`. The candidate-space `argsort(... tensor<128x4096x...>)` lines disappeared; the remaining
sorts are the full-vocabulary `chlo.top_k`, scheduler/page-table integer sorts such as `tensor<32xi32>` and
`tensor<129xi32>`, and stable dimension-0 sorts. Against the v5p vLLM sampled baseline (`3580.42` tok/s for `o128`
and `3639.97` tok/s for `o512`), the best top-p-static Levanter rows are `89.2%` and `97.8%` of vLLM decode
throughput, respectively. This satisfies the 10-15% throughput band for the primary v5p decode-heavy `n=4` regime.

Both secondary top-p-static probes also reached terminal success on `TPU v6 lite`. The original 120 GB request,
`/dlwh/qwen3-parity-levanter-toppstatic-v6e-v5lite4-n4-topk4096-20260531-2350`, produced:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3191.85 | 3198.08 | 1.283 | 88.663 | 1279.2 | 1281.3 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3141.55 | 3147.68 | 1.304 | 88.663 | 1300.9 | 1302.3 | 4831838208 | 2 |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 3144.99 | 3151.13 | 1.302 | 88.663 | 1298.3 | 1301.1 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3556.26 | 3558.00 | 4.607 | 9.202 | 4602.0 | 4604.4 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3559.84 | 3561.58 | 4.602 | 9.202 | 4596.4 | 4600.1 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3558.36 | 3560.10 | 4.604 | 9.202 | 4598.4 | 4602.2 | 4831838208 | 2 |

The 110 GB replacement,
`/dlwh/qwen3-parity-levanter-toppstatic-v6e-v5lite4-n4-topk4096-mem110-20260531-2352`, produced nearly the same
throughput envelope with a best `o128` row of `3351.11` tok/s and a best `o512` row of `3560.45` tok/s. Relative to
the secondary vLLM sampled baseline (`4100.50` tok/s for `o128`, `4197.61` tok/s for `o512`), that is still only
`81.7%` and `84.8%` of vLLM, so the primary v5p target is now the clear success case while secondary TPU generations
need more work.

The benchmark harness now has an explicit fixed-prompt Levanter correctness gate for the design-doc milestone:
`--check-levanter-reference-logits`. When enabled for the Levanter backend, the harness deduplicates the exact
`prompt_corpus.json` prompts, runs full causal Levanter logits as the reference, runs the same tokens through paged
decode with the configured `TpuPagedAttentionConfig`, and writes `levanter_reference_logits.json` plus
`levanter_reference_logits.md` with max absolute/relative logit error and pass/fail under TPU tolerances. These
artifacts are included in `artifacts.json` and logged with the rest of the benchmark outputs, so a TPU smoke can now
prove the fixed-corpus correctness requirement directly instead of relying on unit tests alone.

Local validation for the correctness-gate plumbing:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 39 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Submitted the first v5p TPU run that enables the new fixed-prompt correctness artifact:

- `/dlwh/qwen3-parity-levanter-refcheck-v5p8-n4-topk4096-20260601-0008`

It uses the same top-p-static sampled `n=4` shape that reached the v5p parity band, with
`--check-levanter-reference-logits --reference-logit-max-prompts 1`, writes outputs to
`/dev/shm/qwen3-parity-refcheck`, and keeps UV/JAX caches on default disk. Initial Iris state is pending on v5p
capacity:

```text
JOB_STATE_PENDING
pending_reason: Waiting for worker scale-up in scale group 'tpu_v5p-preemptible_8-us-central1-a'
```

Heartbeat `poll-qwen3-refcheck-tpu-run` is watching this job and should update these notes with
`levanter_reference_logits.md`, `summary.md`, `env.json`, and HLO/artifact evidence when it reaches a terminal state.

Backend correctness tests were broadened to cover the remaining design-doc shapes that were not represented by the
original single-sequence dispatcher smoke. The new CPU/reference dispatcher case exercises two active sequences,
multi-token same-sequence updates that cross a page boundary, a second sequence on a separate page, `q_heads_per_group=2`,
and both soft-cap disabled and enabled. This complements the Qwen-level paged-decode-vs-full-logit regression.

Local validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 13 passed
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3.py::test_qwen3_paged_decode_matches_full_logits_for_available_backends -q
# 2 passed
./infra/pre-commit.py --files lib/levanter/tests/inference/test_tpu_paged_attention_backends.py --fix
# OK
```

The reference-logit TPU job later acquired a v5p worker and started running. The first logs show dependency setup,
tokenizer mirror copy, and Qwen3 8B HF checkpoint load; no benchmark or correctness result has been emitted yet.

That first reference-logit TPU job failed before serving because the harness's standalone reference-logit decode cache
used `trainer.mp.compute_dtype`, which was float32 in this run, while the `tpu-inference` kernel requires
`kv_cache.dtype == k.dtype == v.dtype`:

```text
ValueError: Expected kv_cache.dtype=dtype('float32') to be equal to k.dtype=dtype(bfloat16) and v.dtype=dtype(bfloat16).
```

The correctness gate now chooses a bf16 KV cache whenever the configured paged-attention policy is `AUTO` or
`TPU_INFERENCE`, while preserving the caller's default dtype for explicit reference/JAX paths. Local validation:

```text
uv run python -m py_compile lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 40 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Submitted replacement run:

- `/dlwh/qwen3-parity-levanter-refcheck-v5p8-n4-topk4096-bf16cache-20260601-0016`

It immediately acquired a v5p worker and started building. Heartbeat `poll-qwen3-refcheck-tpu-run` now watches this
replacement job.

The bf16-cache replacement passed the previous dtype failure, wrote HLO, started the server, and began the first
`decode_b32_i1_o128_n4` warmup. It then failed with HBM exhaustion from auto KV-cache sizing:

```text
Auto-computed KV cache budget: base=0 bytes, per_page=4.72 MB, budget=69.36 GB, used=69.36 GB, next=69.37 GB -> max_pages=14700
RESOURCE_EXHAUSTED: Attempting to allocate 7.18G. That was not possible. There are 5.79G free.
```

This is not a reference-logit correctness failure. It is the same cache-budget issue seen earlier when Levanter was run
without an explicit page cap. The successful parity-band top-p-static runs used an explicit cache around 4.83 GB, so the
next replacement pins `--max-pages 1024`:

- `/dlwh/qwen3-parity-levanter-refcheck-v5p8-n4-topk4096-pages1024-20260601-0022`

Heartbeat `poll-qwen3-refcheck-tpu-run` now watches this page-capped replacement job.

Initial page-capped status: the job acquired a v5p worker, loaded Qwen3 8B, started the inference context, and reached
the TPU kernel initialization point without repeating the dtype failure or auto-cache HBM failure. No reference-logit or
benchmark artifact has been emitted yet.

The page-capped reference-logit job succeeded:

- `/dlwh/qwen3-parity-levanter-refcheck-v5p8-n4-topk4096-pages1024-20260601-0022`
- Iris state: `JOB_STATE_SUCCEEDED`, `exit_code=0`, `failure_count=0`, `preemption_count=0`
- TPU: `v5p-8`, `tensor_parallel_size=4`
- Pages/cache: `--max-pages 1024`, reported HBM bytes `4,831,838,208`
- Sampling shape: `temperature=0.7`, `top_p=1.0`, `top_k=4096`, `n=4`, `return_logprobs=true`
- Dependencies: `tpu_inference_version=0.19.0`, `vllm_tpu_version=0.19.0`

Reference-logit correctness output:

| prompt | cases | positions | vocab | max abs error | max rel error | passed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `decode_b32_i1_o128_n4,decode_b32_i1_o512_n4` | 1 | 151936 | 0.28125 | 1711.3 | true |

Levanter-only benchmark output from the same run:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 145.97 | 146.25 | 28.061 | 45.761 | 28056.9 | 28059.2 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 3562.18 | 3563.92 | 4.599 | 4.621 | 4593.2 | 4596.2 | 4831838208 | 2 |

The low `o128` measured decode throughput in this refcheck run is from the one-round measurement including a long
initial prefill/submit path after the warmup shape. The prior matched vLLM comparison run remains the throughput gate:
`o128 n4` at 3194.45 decode tok/s vs vLLM 3580.42 (89.2%) and `o512 n4` at 3559.75 vs vLLM 3639.97 (97.8%).

HLO summary for this run shows the TPU kernel in the decode loop and the expected remaining sort/top-k sites:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 150405 | 10545 | 0 | 10 | 9 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 194342 | 16293 | 0 | 12 | 7 | 0 | 39 |

Artifacts logged by the job include `summary.json`, `summary.md`, `env.json`, `prompt_corpus.json`,
`levanter_reference_logits.{json,md}`, `hlo_summary.{json,md}`, and the `levanter_hlo/` directory.

After reviewing the `0.28125` max-logit delta, the reference-logit artifact was expanded to report whether that error
lands in tokens that matter for rollout logprobs. The JSON/Markdown now includes mean/RMS/p50/p90/p99/p99.9 absolute
logit error, a fixed absolute-error histogram, top-1 agreement, reference-top-k overlap for `k in {1,10,100,1000,4096}`,
and logprob absolute error on those reference-top-k sets. Local validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 41 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Submitted a v5p rerun to capture the richer diagnostics:

- `/dlwh/qwen3-parity-levanter-refdiag-v5p8-n4-topk4096-pages1024-20260601-0122`

Initial Iris state is pending on v5p capacity/quota-tiering:

```text
JOB_STATE_PENDING
pending_reason: Scheduler: Insufficient TPUs (need 4, available 0) - 24 worker(s)
Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 2 matching group(s) blocked by quota-pool tier monotonicity
```

Heartbeat `poll-qwen3-refdiag-tpu-run` and subagent `Plato` are watching for terminal state. The run should update these
notes with the expanded `levanter_reference_logits.md`, `summary.md`, `env.json`, and HLO/artifact evidence when it
succeeds.

The expanded refdiag run succeeded:

- `/dlwh/qwen3-parity-levanter-refdiag-v5p8-n4-topk4096-pages1024-20260601-0122`
- Iris state: `JOB_STATE_SUCCEEDED`, `exit_code=0`, `failure_count=0`, `preemption_count=0`
- TPU: `v5p-8`, `tensor_parallel_size=4`
- Pages/cache: `--max-pages 1024`, reported HBM bytes `4,831,838,208`

Expanded reference-logit diagnostics:

| prompt | cases | positions | vocab | max abs error | mean abs error | rms abs error | p99 abs error | p99.9 abs error | max rel error | top1 agreement | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `decode_b32_i1_o128_n4,decode_b32_i1_o512_n4` | 1 | 151936 | 0.25 | 0.109653 | 0.115925 | 0.1875 | 0.25 | 1537 | 1 | true |

Reference-top-k diagnostics:

| k | max logit abs error | mean logit abs error | max logprob abs error | mean logprob abs error | overlap fraction |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.25 | 0.25 | 0.0754366 | 0.0754366 | 1 |
| 10 | 0.25 | 0.225 | 0.0754366 | 0.0504366 | 0.9 |
| 100 | 0.25 | 0.214375 | 0.0754366 | 0.0447679 | 0.99 |
| 1000 | 0.25 | 0.19875 | 0.0754366 | 0.0338019 | 0.981 |
| 4096 | 0.25 | 0.183586 | 0.112063 | 0.0247864 | 0.986328 |

Absolute-error histogram:

| lower | upper | count | fraction |
| --- | --- | --- | --- |
| 0 | 0.0001 | 113 | 0.000743734 |
| 0.0001 | 0.0003 | 0 | 0 |
| 0.0003 | 0.001 | 6 | 3.94903e-05 |
| 0.001 | 0.003 | 23 | 0.00015138 |
| 0.003 | 0.01 | 264 | 0.00173757 |
| 0.01 | 0.03 | 1551 | 0.0102082 |
| 0.03 | 0.1 | 62543 | 0.41164 |
| 0.1 | 0.3 | 87436 | 0.575479 |
| 0.3 | 1 | 0 | 0 |
| 1 | inf | 0 | 0 |

The top-1 token agrees, but the top-token logprob differs by `0.0754`, and the reference top-4096 contains broad
bf16-scale logit drift (`mean_abs_error=0.183586`, `mean_logprob_abs_error=0.0247864`). This is a much stronger signal
than the original max-only table: the mismatch is not just a near-zero tail-token relative-error artifact. The next
correctness follow-up should compare against a same-dtype full-forward baseline and, if this remains, isolate whether
the drift comes from `tpu-inference` RPA accumulation, cache layout/position handling, or bf16 full-forward vs decode
numerics.

Levanter-only benchmark output from the same run:

| case | backend | active seqs | n | decode tok/s | total tok/s | steady s | compile incl s | p50 ms | p90 ms | hbm bytes | shape buckets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_b32_i1_o128_n4` | `levanter:auto` | 32 | 4 | 167.99 | 168.31 | 24.383 | 45.205 | 24378.8 | 24380.6 | 4831838208 | 2 |
| `decode_b32_i1_o512_n4` | `levanter:auto` | 32 | 4 | 2525.12 | 2526.35 | 6.488 | 6.511 | 6482.0 | 6485.5 | 4831838208 | 2 |

HLO summary:

| file | bytes | lines | collective | rng_sampling | sort_or_topk | logprob | custom_call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `levanter_hlo/gen_loop.hlo.txt.gz` | 150250 | 10525 | 0 | 10 | 8 | 0 | 39 |
| `levanter_hlo/run_prefill.hlo.txt.gz` | 194342 | 16293 | 0 | 12 | 7 | 0 | 39 |

2026-06-01: added the backend/cache matrix needed to isolate the reference-logit drift:

- decode backends: `tpu_inference` and slow Levanter `reference`
- cache dtype policies: `auto` and `bfloat16`
- `--reference-logit-only` so correctness artifacts are logged before running serving benchmarks
- diagnostic controls for `tpu-inference` RPA precision: `--levanter-tpu-inference-out-dtype` and
  `--levanter-preserve-attention-output-dtype`

Two matrix attempts failed before producing correctness artifacts:

- `/dlwh/qwen3-parity-levanter-refmatrix-v5p8-n4-topk4096-pages1024-20260601-0713` failed after one preemption with
  a slow-reference loop carry dtype mismatch: input carry `bfloat16[1,8,4,128]`, output carry `float32[1,8,4,128]`.
- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-20260601-0725` failed with
  `NameError: name 'config' is not defined` in `_run_tpu_inference_backend`.

Both are harness/backend-dispatch issues, not numerical results. Local fixes:

- `_run_tpu_inference_backend` now accepts `TpuPagedAttentionConfig`, and `_run_backend` passes it through.
- `default_ragged_paged_attention` initializes its output accumulator as f32 so the loop carry dtype is stable.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 54 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/layers/attention.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py docs/debug-log-qwen3-tpu-logprob-numerics.md --fix
# OK
```

Submitted the fixed ref-only matrix as:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-fixed-20260601-0733`

Initial Iris state is pending on v5p scale-up:

```text
JOB_STATE_PENDING
pending_reason: Scheduler: Insufficient TPUs (need 4, available 0) - 1 worker(s)
Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'tpu_v5p-preemptible_8-us-east5-a'
```

Because the fixed v5p job was still waiting on capacity, submitted the same ref-only matrix against the secondary
4-chip TPU pool:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v6e-v5lite4-fixed-20260601-0736`
- TPU alternatives: `v5litepod-4,v6e-4`
- Same backend/cache matrix and `--reference-logit-only` contract as the v5p fixed run.

That secondary run acquired a `v6e-4` worker and reached the reference-logit check, but failed before logging the
already-written artifacts because the default diagnostic threshold was too low:

```text
AssertionError: Levanter reference-logit check failed; see
/dev/shm/qwen3-parity-refonly-matrix-v6e-v5lite4-fixed/levanter_reference_logits.json for details
```

Submitted a capture rerun with `--reference-logit-atol 999` so it logs the matrix artifact:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v6e-v5lite4-capture-20260601-0743`

The v5p fixed job later started with the same low diagnostic threshold, so submitted a v5p capture twin:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-capture-20260601-0745`

The secondary capture acquired `v6e-4` but failed during f32 model loading before any reference-logit result:

```text
RESOURCE_EXHAUSTED: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 288.00M. That was not possible. There are 71.37M free.;
(1x0x0_HBM0)
```

This is not a matrix result. The v5p capture remains the primary path for the f32-reference diagnostic.

The restarted v5p fixed run completed the reference-logit check, then failed in `log_output_artifacts` because
reference-only mode does not write `summary.md`:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/dev/shm/qwen3-parity-refonly-matrix-fixed/summary.md'
```

Local fix: `log_output_artifacts` now logs whichever artifacts exist instead of requiring `summary.md`, `env.json`, and
`artifacts.json`.

The pending v5p capture was submitted before this fix, so submitted a logfix v5p capture:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-logfix-20260601-0757`

That logfix job was also preempted shortly after TPU init. Submitted the same logfix matrix without `--preemptible`:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-logfix-nonpreempt-20260601-0807`

2026-06-01 follow-up: the expanded refdiag prompt has `positions=1`, so the error is visible even when attention
should reduce to reading back the current token's V. That makes KV-cache dtype a likely source. Added f32-KV diagnostic
support:

- `--levanter-compute-dtype` controls the Levanter inference engine KV-cache dtype.
- `--levanter-trainer-mp` controls the model mixed-precision policy for smaller-TPU same-dtype checks.
- The harness can diagnose f32 KV cache, but `tpu_inference` shape support remains bf16-only after TPU evidence showed
  that f32 output/cache paths fail for this Qwen3 decode shape.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 57 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py docs/debug-log-qwen3-tpu-logprob-numerics.md .agents/projects/levanter_tpu_inference_parity/research.md scratch/20260601-0816_f32kv_v5p_monitoring_state.json scratch/20260601-0816_bf16mp_f32kv_v6e_v5lite_monitoring_state.json --fix
# OK
```

Submitted the authoritative f32-reference/f32-KV v5p diagnostic:

- `/dlwh/qwen3-parity-levanter-refonly-f32kv-v5p8-20260601-0816`
- It acquired `v5p-8`, loaded the f32 model, and failed when `tpu_inference` hit the explicit f32 KV-cache case:

```text
TypeError: cannot reshape array of shape (2, 4, 128) (size 1024) into shape (4, 128) (size 512)
```

The traceback points inside `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` while storing the decode output.
This rules out "just turn on f32 KV cache" as an immediate fix for `tpu-inference==0.19.0` decode.

Submitted narrower v5p diagnostics:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32out-v5p8-20260601-0823`: `tpu_inference` with bf16 cache and
  f32 RPA output/accumulator.
- `/dlwh/qwen3-parity-levanter-refonly-reference-cachematrix-v5p8-20260601-0823`: slow Levanter `reference` backend
  with bf16-vs-f32 cache policies.

The `tpu_inference` f32-output job retried once after preemption and then failed with the same reshape error even with
bf16 KV cache:

```text
TypeError: cannot reshape array of shape (2, 4, 128) (size 1024) into shape (4, 128) (size 512)
```

So current `tpu-inference==0.19.0` decode does not support f32 output/accumulator for this shape. Tried to submit the
slow-reference cache matrix with `--no-preemptible`, but Iris rejected it because no non-preemptible `v5p-8` group
exists. The preemptible slow-reference job is retrying after one preemption.

Submitted a smaller-TPU same-dtype attempt:

- `/dlwh/qwen3-parity-levanter-refonly-bf16mp-f32kv-v6e-v5lite4-20260601-0816`
- It acquired `v6e-4`, but failed before reference-logit artifacts during model load:

```text
RESOURCE_EXHAUSTED: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 288.00M. That was not possible. There are 71.37M free.;
(0x0x0_HBM0)
```

So `v6e-4`/`v5litepod-4` still are not useful for Qwen3 8B reference-logit diagnostics without changing the load path.
The next decisive result is the v5p f32-KV job.

2026-06-01 follow-up: the reduced v5p slow-reference/f32-cache diagnostic succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-v5p8-20260601-0839`
- backend: slow Levanter `reference`
- cache dtype: `float32`
- service compute dtype in env: `bfloat16`
- `--levanter-preserve-attention-output-dtype` was not set

Result:

| backend | cache dtype | positions | max logit abs | mean logit abs | rms logit abs | p99 logit abs | top1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reference` | `float32` | 1 | 0.25 | 0.0444493 | 0.055767 | 0.15625 | 1.0 |

Reference-top-k logprob errors:

| k | max logprob abs | mean logprob abs | overlap |
| --- | --- | --- | --- |
| 1 | 0.0341177 | 0.0341177 | 1.0 |
| 10 | 0.159118 | 0.0716177 | 1.0 |
| 100 | 0.159118 | 0.0645588 | 0.96 |
| 1000 | 0.159118 | 0.0461274 | 0.966 |
| 4096 | 0.159118 | 0.0391007 | 0.978027 |

Absolute logit-error histogram:

| lower | upper | count | fraction |
| --- | --- | --- | --- |
| 0 | 0.0001 | 14954 | 0.098423 |
| 0.0001 | 0.0003 | 1 | 0.00000658 |
| 0.0003 | 0.001 | 57 | 0.000375 |
| 0.001 | 0.003 | 268 | 0.001764 |
| 0.003 | 0.01 | 5862 | 0.038582 |
| 0.01 | 0.03 | 25867 | 0.170249 |
| 0.03 | 0.1 | 97062 | 0.638835 |
| 0.1 | 0.3 | 7865 | 0.051765 |
| 0.3 | 1 | 0 | 0 |
| 1 | inf | 0 | 0 |

Interpretation: f32 cache plus the slow reference paged-attention loop does not fix the mismatch. Since the diagnostic
still has `positions=1`, this rules out page traversal, multi-token attention accumulation, and bf16 KV cache as the
sole cause. The quantized-looking 0.125/0.25 error scale now points at a dtype transition outside the RPA kernel,
most likely decode hidden/logit output being rounded relative to the full-forward path.

Added artifact fields for the next diagnostic:

- `reference_logits_dtype`
- `decode_logits_dtype`
- `max_abs_error_if_reference_rounded_to_decode_dtype`
- `residual_max_abs_error_after_reference_rounding`

These identify whether a run's error is simply explained by rounding the full-forward logits to the decode-logit dtype.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 56 passed
```

Submitted the next narrow v5p diagnostic:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-preserve-v5p8-20260601-0848`
- same slow-reference/f32-cache check, but with `--levanter-compute-dtype float32` and
  `--levanter-preserve-attention-output-dtype`
- state after placement: running/building on v5p, loading checkpoint shards

2026-06-01 follow-up: inspected the `tpu-inference==0.19.0` RPA f32-output failure. The v3 kernel wrapper allocates
its output double buffer and output HBM shape from the prepared query dtype/packing, not from `out_dtype`:

```text
bo_double_buf = bq_double_buf
out_shape = q.shape, q.dtype
```

So passing `out_dtype=float32` while leaving Q as bf16 creates exactly the observed packing mismatch when the kernel
tries to store f32 output through a bf16-shaped buffer:

```text
TypeError: cannot reshape array of shape (2, 4, 128) (size 1024)
into shape (4, 128) (size 512)
```

Local fix: when Levanter requests `tpu_inference_out_dtype`, the adapter casts Q to that dtype before calling the RPA
kernel. K/V and the KV cache stay bf16 for the initial Qwen3 target.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 57 passed
```

Submitted the fixed f32-output `tpu_inference` diagnostic:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32qout-v5p8-20260601-0854`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- `--levanter-tpu-inference-out-dtype float32`
- `--levanter-preserve-attention-output-dtype`

2026-06-01 follow-up: the slow-reference/f32-cache preserve run completed and made the mismatch worse:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-preserve-v5p8-20260601-0848`
- backend: slow Levanter `reference`
- cache dtype: `float32`
- `--levanter-compute-dtype float32`
- `--levanter-preserve-attention-output-dtype`

Result:

| backend | cache dtype | preserve attention output | positions | max logit abs | mean logit abs | rms logit abs | p99 logit abs | top1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reference` | `float32` | true | 1 | 1.15741 | 0.391914 | 0.41717 | 0.748282 | 1.0 |

Reference-top-k logprob errors:

| k | max logprob abs | mean logprob abs | overlap |
| --- | --- | --- | --- |
| 1 | 0.458656 | 0.458656 | 1.0 |
| 4096 | 0.458656 | 0.0555125 | 0.977295 |

Interpretation: preserving f32 attention output is not the parity path for the slow reference backend. The full-forward
baseline evidently includes a cast back to the residual/model dtype before the output projection; forcing f32 through
the residual path changes the model numerics rather than matching full forward.

The fixed f32-output `tpu_inference` diagnostic then failed before artifacts:

```text
ValueError: Expected kv_cache.dtype=dtype(bfloat16) to be equal to k.dtype=dtype('float32')
and v.dtype=dtype('float32').
```

This was a real adapter-boundary bug. Levanter's normal `KvPageCache.update` casts new K/V into the cache dtype, but
the fused `tpu-inference` path passed raw K/V directly to the external kernel. Local fix: cast new K/V to
`kv_cache.kv_pages.dtype` before padding/packing for `tpu-inference`, while still casting Q to the requested f32 output
dtype.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 16 passed
```

2026-06-01 follow-up: after the adapter K/V cast fix, the f32-output `tpu_inference` diagnostic succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32qout-v5p8-20260601-0902`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: true
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`, `levanter_compute_dtype=bfloat16`

Reference-logit diagnostics:

| backend | cache dtype | rpa out dtype | preserve attention output | positions | ref dtype | decode dtype | max logit abs | mean logit abs | p99 logit abs | top1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tpu_inference` | `bfloat16` | `float32` | true | 1 | `bfloat16` | `float32` | 1.15741 | 0.391914 | 0.748282 | 1.0 |

Reference-top-k logprob errors:

| k | max logprob abs | mean logprob abs | overlap |
| --- | --- | --- | --- |
| 1 | 0.458656 | 0.458656 | 1.0 |
| 10 | 0.458656 | 0.345714 | 0.9 |
| 100 | 0.458656 | 0.21619 | 0.99 |
| 1000 | 0.458656 | 0.108289 | 0.97 |
| 4096 | 0.458656 | 0.0555125 | 0.977295 |

Absolute logit-error histogram:

| lower | upper | count | fraction |
| --- | --- | --- | --- |
| 0 | 0.0001 | 3 | 0.0000197 |
| 0.0001 | 0.0003 | 2 | 0.0000132 |
| 0.0003 | 0.001 | 8 | 0.0000527 |
| 0.001 | 0.003 | 46 | 0.000303 |
| 0.003 | 0.01 | 113 | 0.000744 |
| 0.01 | 0.03 | 341 | 0.002244 |
| 0.03 | 0.1 | 2234 | 0.014704 |
| 0.1 | 0.3 | 36141 | 0.23787 |
| 0.3 | 1 | 113035 | 0.743965 |
| 1 | inf | 13 | 0.0000856 |

Interpretation: the runtime integration is now fixed, but preserving f32 attention output through the output projection
is the wrong numerics. It reproduces the slow-reference preserve result exactly (`max_logit_abs=1.15741`,
`top1_logprob_abs=0.458656`). The next targeted run should keep the f32 RPA accumulator/output inside
`tpu-inference` but allow Levanter to cast attention output back to the residual/model dtype before `o_proj`.

Added hidden-state attribution to the reference-logit artifact for the next diagnostic run:

- reference/decode hidden dtype
- hidden max absolute error
- hidden mean absolute error
- hidden RMS absolute error

This will separate transformer/paged-decode drift from LM-head projection/logprob drift if the cast-back run still
misses the target.

Validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 42 passed

./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

2026-06-01 follow-up: the f32-output/cast-back `tpu_inference` diagnostic succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32out-castback-v5p8-20260601-0910`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`, `levanter_compute_dtype=bfloat16`

Reference-logit diagnostics:

| backend | cache dtype | rpa out dtype | preserve attention output | positions | ref dtype | decode dtype | max logit abs | mean logit abs | p99 logit abs | top1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tpu_inference` | `bfloat16` | `float32` | false | 1 | `bfloat16` | `bfloat16` | 0.25 | 0.0444493 | 0.15625 | 1.0 |

Reference-top-k logprob errors:

| k | max logprob abs | mean logprob abs | overlap |
| --- | --- | --- | --- |
| 1 | 0.0341177 | 0.0341177 | 1.0 |
| 10 | 0.159118 | 0.0716177 | 1.0 |
| 100 | 0.159118 | 0.0645588 | 0.96 |
| 1000 | 0.159118 | 0.0461274 | 0.966 |
| 4096 | 0.159118 | 0.0391007 | 0.978027 |

Interpretation: f32 RPA output with cast-back fixes the top-1 logprob gap below `0.05`, but the max reference-top-4096
logprob gap remains `0.159118`. This exactly matches the slow-reference/f32-cache result, which means the remaining
error is no longer specific to `tpu-inference` RPA. The next useful run is the same cast-back configuration with the
new hidden-state attribution artifact to determine whether the shared residual error is before or after `lm_head_logits`.

2026-06-01 follow-up: the hidden-attribution cast-back run succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-castback-hidden-v5p8-20260601-0922`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false
- positions: 1
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`

Reference-logit diagnostics:

| ref hidden dtype | decode hidden dtype | hidden max abs | hidden mean abs | hidden RMS abs | max logit abs | top-1 logprob abs | top-4096 max logprob abs | top-4096 mean logprob abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bfloat16` | `bfloat16` | 1.3125 | 0.0160123 | 0.0325309 | 0.25 | 0.0341177 | 0.159118 | 0.0391007 |

Interpretation: the remaining top-k error is already visible before the LM head. This rules out a pure logprob
normalization or LM-head projection bug. Since the prompt has one token and the slow-reference/f32-cache path matches
the same top-k result, the next attribution target is the full-forward-vs-paged-decode transformer path shared across
RPA backends. The harness now has a pending one-token direct-attention diagnostic: it replaces each self-attention kernel
with the exact `V -> o_proj` path and records whether that direct path matches full forward or paged decode.

Local follow-up before the TPU direct run placed: a tiny bf16 Qwen3 reproduction showed direct/unrolled attention matches
paged decode exactly, while `model.activations()` still differs:

```text
direct-full 0.01953125
direct-decode 0.0
full-decode 0.01953125
```

The first-layer attention output itself matched exactly after adding a safe singleton-attention fast path, so the
remaining local mismatch is the layer container: full forward uses `Stacked.fold`, while paged decode manually unrolls
layers. The reference-logit checker now computes its full-sequence reference by explicitly unrolling
`model.transformer.layers.unstacked()` before applying the final norm. This leaves the training loss path alone but
aligns the diagnostic reference with the serving/decode layer order.

Local validation after this change:

```text
unrolled-decode hidden max 0.0
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 61 passed, 1 skipped
./infra/pre-commit.py --files ... --fix
# OK, including Pyrefly
```

TPU proof run:

- `/dlwh/qwen3-parity-levanter-refonly-unrolledref-v5p8-20260601-0952`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false
- `--reference-logit-atol 0.05`
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`

Reference-logit diagnostics:

| hidden max abs | hidden mean abs | hidden RMS abs | max logit abs | top-1 logprob abs | top-4096 max logprob abs | top-4096 overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 | 1 |

Conclusion: the earlier `0.159118` top-4096 max logprob gap was a reference-side artifact from comparing serving decode
against the scanned `Stacked.fold` full-forward path. In the one-token rollout regime, the serving/decode layer order
matches an explicitly unrolled full-reference path exactly after the adapter fixes (`Q` cast to f32 for f32 RPA output,
new K/V cast to bf16 cache dtype, and attention output cast back before `o_proj`).

The earlier direct-attention attribution run also completed:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-directv-v5p8-20260601-0941`

Its row before switching the checker to unrolled full reference was:

| hidden max abs | direct-ref hidden max abs | direct-decode hidden max abs | max logit abs | direct-ref logits max abs | direct-decode logits max abs | top-4096 max logprob abs |
| --- | --- | --- | --- | --- | --- | --- |
| 1.3125 | 1.3125 | 0 | 0.25 | 0.25 | 0 | 0.159118 |

That confirms the exact one-token attention path matched paged decode, while the scanned full-forward reference was the
side that drifted.
