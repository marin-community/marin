# Draft Design Document for Inference

> **Note**: This document contains the technical design and architecture for
> the inference server.

## References

* [JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/blob/b282713880943cebe7183815918fb7dd60922b14/llama3/llama3_jax/model.py) <-- pretty basic kv cache, but easy to follow
* [Fancier JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/pull/22/files)

### MaxText

* [Page Manager](https://github.com/AI-Hypercomputer/maxtext/blob/eac885edb371e6141a2bb784f9060f816ce17b23/MaxText/inference/page_manager.py)
* [Paged Attention](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/inference/paged_attention.py#L298)
* [Jetstream](https://github.com/AI-Hypercomputer/JetStream) <-- session management and such
* [MaxEngine](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/maxengine.py) <-- maxtext impl of jetstream protocol

### EasyDel

* [VSurge](https://github.com/erfanzar/EasyDeL/tree/main/easydel/inference/vsurge)

### JAX Repo

- [Ragged Paged Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py)

### Nano-VLLM

torch, not much interesting here given jetstream. focused on batch inference I think?

* https://github.com/GeeeekExplorer/nano-vllm/

### SGLang

* [SGLang](https://github.com/sgl-project/sglang)

## Architecture Overview

**Goal**: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

**Core Architecture**:
- Reuse `run_generation_loop`, `_one_round`, and `GenState` patterns from `src/levanter/main/sample_lm.py` but extract into a reusable service.
- Maintain decode state in a long-lived service object that owns: `JitScheduler`, `PageTable`, `KvPageCache`, `DecodeState`, `Sampler`, `tokenizer`, and the `LlamaLMHeadModel` (or other `LmHeadModel`).
- Requests enqueue prompt tokens into the service; a background loop performs prefill+decode rounds and streams tokens back to callers.
- Keep JIT-safety: no Python control flow in jitted sections; use existing named-jit and Equinox modules.

## Technical Design

### Service Architecture
The inference server is built around a `GenerationService` that encapsulates:
- Model loading and initialization via `HFCheckpointConverter`
- Tokenizer management
- JAX device mesh configuration via `TrainerConfig`
- KV cache management via `PageTable` and `KvPageCache`
- Generation loop orchestration via `JitScheduler` and `DecodeState`
- Sampling via `Sampler`

### API Design
- **OpenAI Compatibility**: Follows OpenAI API v1 specification for `/v1/completions` and `/v1/chat/completions`
- **Request Schema**: Supports `prompt`, `max_tokens`, `temperature`, `stop`, `seed` parameters
- **Response Schema**: Returns structured responses with `choices`, `usage`, and metadata
- **Error Handling**: Proper HTTP status codes and error messages

### Performance Considerations
- **JIT Compilation**: Warmup generation on startup to trigger JIT compilation
- **Memory Management**: Efficient KV cache page allocation and deallocation
- **Batching**: Support for concurrent requests with `JitScheduler`
- **Streaming**: Server-Sent Events (SSE) for real-time token streaming

### Streaming and cancellation

`stream=true` sends text deltas and requested token IDs and logprobs after prefill and
at each host decode boundary. `max_rounds` controls how many device decode rounds
run between those boundaries; larger values trade response latency for throughput.
Incomplete Unicode characters are held until their bytes can be decoded. Completion
requests with `echo=true` retain buffered responses because echo logprobs rescore
the complete sequence.

Set `x-request-id` to name an HTTP request group. `InferenceServer.abort([id])`
cancels that group at the next host boundary and returns its exact partial tokens
with an `abort` finish reason. Other groups in the batch continue; the cancelled
response does not wait for them to finish. Disconnecting a client cancels its group
and releases the request ID. No separate remote cancellation endpoint is exposed.

### Remote weight publication

Set `InferenceServerConfig.weight_transfer` to `WeightTransferConfig(backend="gloo",
max_staging_bytes=...)` to enable SkyRL weight control routes. Install PyTorch in the
serving environment (`torch_test` supplies it for local validation). Use `nccl` for
a single GPU; CPU/Gloo has numerical integration coverage, while GPU/NCCL still
requires hardware validation. Multi-device and multi-process serving are rejected.
The receiver uses Torch broadcast and DLPack on the serving device. It does not
materialize a complete checkpoint on the host.

The trainer initializes `/init_weight_update_communicator`, then brackets its
ordered `/update_weights` broadcasts with `/begin_weight_reload` and
`/finish_weight_reload`. Every tensor and finish request must echo the begin
response's `publication_id` and `model_version`. A new begin invalidates an unfinished
publication. Received names, shapes, and dtypes must match the model's HF state dict;
expert banks may also arrive as individual expert projections. Missing, duplicate,
mismatched, and stale publications fail without replacing the current model.

The current model remains available while tensors arrive. Finish validates the
complete candidate, rechecks its expected model version, aborts active requests at
the pause barrier, resets KV and model-specific cache state, and installs the whole
model with one version increment. An already paused server remains paused.
`/reset_prefix_cache` also aborts active requests and clears cached state, without
changing weights or version. It preserves an existing pause. `/destroy_weights_update_group`
invalidates staged weights and closes the transport.

`max_staging_bytes` bounds received parameter bytes. Provision additional memory
for the current model, inference cache, and layout conversion temporaries. This
single-device adapter does not implement distributed JAX shard placement or a
production-sized memory budget. The paired SkyRL remote client must forward the
reload bracket and publication receipt; the repository's SkyRL pin is unchanged.

## File/Code References
- `src/levanter/main/sample_lm.py`: `SampleLmConfig`, `_load_model`, `GenState`, `run_generation_loop`, `_one_round`, `extract_outputs`.
- `src/levanter/inference/jit_scheduler.py`: `JitScheduler`, `DecodeState`, `SeqDecodingParams`.
- `levanter.inference.page_table.PageTable`, `levanter.layers.kv_cache.KvPageCache`.
- `levanter.layers.sampler.Sampler`.
- `levanter.models.llama.LlamaLMHeadModel` and `levanter.models.lm_model.LmHeadModel`.
- `levanter.compat.hf_checkpoints.HFCheckpointConverter`, `load_tokenizer`.
- `levanter.trainer.TrainerConfig` and `levanter.utils.jax_utils.use_cpu_device`.

## Implementation Notes

### Current Status
- Basic FastAPI server with `/v1/completions` endpoint implemented
- `GenerationService` with single-sequence generation working
- Warmup JIT compilation on startup implemented
- Health check endpoint (`/healthz`) functional
- Accurate token counting using actual tokenizer instead of rough estimates
- End-to-end testing completed successfully with tiny HF model on CPU

### Key Design Decisions
- **Optional Dependencies**: FastAPI and Uvicorn are optional `serve` dependencies
- **Configuration**: Uses `draccus` for configuration management, consistent with other Levanter components
- **Error Handling**: Comprehensive error reporting with proper HTTP status codes
- **Logging**: Structured logging with configurable verbosity levels

## Paged short-convolution history

`levanter.layers.paged_short_conv` provides the causal history needed by Hero's
key-projection, attention-output, and MoE-output convolutions. It uses the
attention page allocation and keeps `min(page_size, kernel_size - 1)` input rows
per physical page. A page-local ring updates in packed token order; each token
reads its own request's preceding positions before overwriting a ring entry.
This supports chunked prefill, incremental decode, and kernels wider than a page.

`ShortConvPageCache` implements `PageCache.copy_page` and `reset`, so a cloned
partial page receives independent convolution history. Padding does not update
the rings. The implementation uses a JAX scan and matches the existing
short-convolution reference's lag-ordered arithmetic. CPU FP32/BF16 parity covers
mixed request order, page crossings, clone divergence, and reset. The native Hero model uses this history for incremental decode; accelerator
validation remains pending.

### Native Hero schema-v2 model

`HeroConfig` and `HeroLMHeadModel` load schema-v2 `grug_moe` exports through
`HFCheckpointConverter`. The recipe preserves latent routed experts, independent
shared experts, local/global KV-head counts, all three short-convolution sites,
and the checkpoint's fused or unfused RoPE convention. Snowball schema-v1 and
Hero schema-v2 configurations are resolved separately. Packed expert banks and
per-expert export weights map to the same native model.

Paged prefill and decode use `HeroLayerCache`, combining KV pages with the
short-convolution histories described above. Supply absolute token positions and
a `compact_grug_mesh` with context size one. Packed token buffer lengths must
be divisible by the data and expert mesh axes. Convolution currently gathers
packed activations for a causal scan; this is a correctness baseline and still
needs accelerator profiling. The stored KV layout duplicates global heads to
the maximum local/global count so all layers share one scan shape.

CPU tests compare the native full forward path with the experiment using
nonidentity convolution taps, both RoPE conventions, packed documents, and both
checkpoint layouts. Mixed-request incremental tests compare against full forward
across page and sliding-window boundaries. FP32 parity checks pin highest matmul
precision, as the default GPU precision can round full-sequence and incremental
matrix shapes differently. One H100 default-precision case exceeded the 1e-4
comparison tolerance; default-precision parity remains a separate validation target.
Full-checkpoint accelerator throughput and matched vLLM performance remain unmeasured.
