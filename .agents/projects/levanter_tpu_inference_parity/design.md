# Levanter TPU Inference Parity

We want Levanter inference to be a credible default for Qwen3 8B RL rollouts on TPU, not a convenience path that users switch away from when decode throughput matters. Marin already ships both Levanter and `vllm-tpu`; the practical goal is for Levanter to match the decode-heavy RL rollout regime on the same 8B checkpoint, TPU generation, context length, batch mix, and decoding policy.

The proposal is to keep Levanter's native JAX model, checkpoint, scheduler, and OpenAI server surfaces, but introduce an explicit TPU inference-kernel backend underneath paged decode. On TPU serving jobs, `tpu-inference` is a required dependency and the default path should use its RPA kernels. The current JAX Pallas ragged paged attention wrapper remains a diagnostic fallback when explicitly configured, and the reference path is for tests only.

## Background

The source notes for this design are in [research.md](research.md). The short version is that Marin already pins `vllm-tpu==0.19.0` and `tpu-inference==0.19.0`, while Levanter already has a paged inference engine, Qwen3 support via the Llama-style paged decode stack, and a TPU RPA dispatch path. We do not have a reliable baseline gap yet; phase 1 of this design exists to establish that gap before committing to invasive kernel work. The missing implementation layer is an explicit backend contract that makes `tpu-inference` the TPU serving backend without spreading its private shapes through the model code.

## Challenges

Levanter already has the right broad shape: `InferenceEngineConfig` controls page size, max sequences, prefill size, queueing, and decode rounds in [engine.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/engine.py#L43-L120); prefill and decode both call the same `model.decode` hot path; Qwen3 reuses the Llama LM-head decode path in [qwen.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/models/qwen.py#L379-L391). The gap is that the implementation boundary is too implicit. `Attention.paged_decode` updates the KV cache in `KvPageCache.update`, then calls `ragged_paged_attention` in [attention.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/layers/attention.py#L1767-L1809). On TPU, that function opportunistically imports JAX's experimental ragged paged attention kernel and otherwise falls back to a slow reference path.

`vllm-tpu` is moving in the opposite direction: the TPU backend is organized around a validated `tpu-inference` kernel package, serving-shaped prefill/decode profiling, and Qwen support already exists in Marin's vLLM mapping. The RPA paper also points at a stronger kernel contract than Levanter currently exposes: ragged paged metadata plus fused KV update and attention, with specialized decode, prefill, and mixed kernels. If we only swap one function call, we risk improving a microbench while missing end-to-end serving parity.

## Costs / Risks

- This makes `tpu-inference` a required dependency for TPU serving, and its kernel APIs may churn faster than Levanter's internal attention API.
- Keeping the native Levanter scheduler means we do not get all vLLM scheduling behavior "for free"; we are explicitly choosing throughput over scheduler similarity when those trade off.
- Fusing KV update into a kernel backend changes an ownership boundary that is currently simple and testable in `KvPageCache.update`.
- A real parity claim needs TPU time across at least one primary generation, plus repeated runs for compile-including and steady-state numbers.

## Design

Phase 1 is benchmark-only. Add a harness that can run the same Qwen3 8B checkpoint through Levanter and Marin's `vllm` extra, using the same tokenizer, prompts, `max_model_len`, tensor parallel size, batch shape, dtype, temperature, stop behavior, and compile-cache settings. The primary matrix is decode-heavy RL rollout traffic: many active sequences, short per-step decode calls, shared/cloned prefixes when `n > 1`, and output lengths long enough for steady-state decode to dominate. The harness should report time-to-first-token, steady-state decode tokens/sec, total tokens/sec, p50/p90 request latency, HBM used by KV cache, compile-including latency, and number of compiled shape buckets. It should also save Levanter's existing generation and prefill jaxpr/HLO artifacts. This phase can use the current JAX RPA path; it should not wait for a new kernel adapter.

Phase 2 adds a small, explicit backend layer under `levanter.layers.attention.ragged_paged_attention`. The public Levanter model path should continue to call `Attention.paged_decode`, but the attention module should delegate to `levanter.inference.tpu_kernels` for TPU-specific paged attention and KV-update behavior. The new layer should consume Levanter's existing `PageBatchInfo`, which already contains slot IDs, page indices, sequence lengths, cumulative query lengths, and token destinations in [page_table.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/page_table.py#L62-L94).

Then introduce backend selection:

```python
class TpuPagedAttentionBackend(StrEnum):
    AUTO = "auto"
    TPU_INFERENCE = "tpu_inference"
    JAX_RPA = "jax_rpa"
    REFERENCE = "reference"
```

`AUTO` means: on TPU, require `tpu_inference` and raise if it cannot run. On CPU/GPU, use the reference implementation. JAX RPA and reference are never silent TPU serving fallbacks; they can run on TPU only when explicitly requested in a test/diagnostic config. An explicit backend must fail fast if unsupported. This mirrors the existing Pallas-kernel skill rule: explicit backends raise, backend sequences warn and fall back.

The first supported kernel target is intentionally narrow: Qwen3 8B bf16, page size 128, head size 128 after any required padding, 32 query heads, 8 KV heads, tensor parallel size 4 on the Levanter `model` axis, max model length 4096, on TPU v5p first. v6e can be used as a secondary check, but it is not the tuning target because Marin usually has more v5p capacity available. Other models and TPU generations may run if the backend reports support, but they are not part of the initial parity claim.

The backend contract should accept Levanter's existing `PageBatchInfo` metadata and `KvPageCache` layout at the boundary. Internally, the `tpu_inference` backend may repack metadata or call a fused update-attention kernel, but the rest of Levanter should not learn `tpu-inference`'s private shapes. If the fastest kernel needs KV update fused with attention, the backend returns both `(attn_tokens, updated_cache)`. Backends may use JAX donation/aliasing internally, but callers must treat the returned cache as the only valid cache after the call. The returned cache must preserve Levanter's `KvPageCache` axes, dtype, and sharding. Duplicate token destinations in one call are invalid input and should raise before lowering. Token ordering follows the existing packed-position order from `TokenQueue.pack_next_sequence`.

Scheduler changes should be evidence-driven. Start with Levanter's current `max_tokens_per_round`, `max_rounds`, `max_seqs_in_prefill`, and `max_prefill_size` knobs. Use the benchmark harness to find where Levanter loses to vLLM in the decode-heavy RL matrix first; prefill-heavy and mixed serving cases are secondary diagnostics. The scheduler hot spots to inspect first are page allocation in [jit_scheduler.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/jit_scheduler.py#L212-L340) and queue packing/sorting in [jit_scheduler.py](https://github.com/marin-community/marin/blob/7caf24e88fd5cadedcfa0173f0c01c35ef69f12c/lib/levanter/src/levanter/inference/jit_scheduler.py#L1062-L1115). We should accept scheduler divergence from vLLM if it improves rollout throughput. Latency and exact vLLM scheduling semantics are secondary to high steady-state decode tokens/sec and high TPU utilization.

Success criteria for the first parity milestone:

- Qwen3 8B bf16 runs through Levanter on v5p with the same installed `vllm-tpu`/`tpu-inference` versions used by Marin.
- Correctness matches the current Levanter paged-attention tests and a fixed prompt corpus against Levanter reference logits within TPU tolerances.
- On the fixed decode-heavy RL rollout matrix, Levanter reaches within 10-15% of `vllm-tpu` on steady-state decode tokens/sec after warmup, or the remaining gap is attributed to a specific kernel/scheduler limitation with profile artifacts.
- The default `AUTO` path does not silently fall back to the reference implementation on TPU serving shapes.

## Testing

Extend `lib/levanter/tests/inference/test_paged_attention.py` so every backend is checked against the existing naive attention reference on single-sequence, multi-sequence, and incremental cases. Add CPU/reference tests that never import TPU-only packages, plus TPU-marked tests that require `tpu-inference` or JAX RPA and fail fast when an explicit backend is requested. The test matrix should include ragged mixed prefill/decode batches, page-boundary writes, multi-token same-sequence updates, soft-cap on/off, nontrivial `q_heads_per_group`, and explicit fallback/error cases.

Add a Qwen-level regression that compares logits for a small synthetic Qwen3 config through the current Levanter decode path and each available backend. Levanter already has Qwen3 roundtrip tests for the HF conversion path. For full 8B, use the benchmark/integration script rather than pytest: it should run a fixed prompt corpus through Levanter and `vllm-tpu`, save the exact command lines, environment variables, HLO/profile directories, and a JSON summary table. Marin's existing TPU vLLM smoke test and served-eval notes are good starting points for the vLLM side.

The minimum performance matrix is: decode-heavy RL rollout cases with `batch={8,32,128}`, `input_len=1`, `output_len={128,512}`, plus cloned-prefix `n > 1` cases that exercise Levanter's clone path. Prefill-only `input_len={256,1024,4096}, output_len=1` and mixed serving with varied prompt lengths remain diagnostic, not the milestone gate. Report both compile-including and steady-state numbers.

## Open Questions

- Should v6e be a required secondary validation before announcing parity, or is v5p enough for the first milestone?
- How much additional rollout latency is acceptable if it buys materially higher steady-state decode throughput?
