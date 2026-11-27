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

### Short-convolution state

Each request slot keeps the last `kernel_size - 1` convolution inputs.
Packed prefill and decode convolve each contiguous chunk in parallel. A tap
that reaches before the start of a chunk reads from the request's slot. The
end of each chunk supplies the history for the next step.
Position-zero inputs clear history on slot reuse. Cloning copies history into
the new slot, so the two continuations can update it independently. Resetting
the cache clears every slot.

The current history describes the processed prefix at one position. Reusing KV
pages for an earlier prefix requires restoring the convolution history at that
position, or recomputing it.

### Native Hero serving

Levanter loads Hero `grug_moe` schema-v2 and schema-v3 checkpoints with banked
or individually stored expert weights. Schema-v3 exports must have global
query log scaling disabled. The inference engine uses absolute token
positions, paged KV caches, and request-slot convolution history for prefill
and decode. Serving requires an explicit Grug mesh with context parallelism
set to one.

The CPU parity tests cover packed documents, both rotary layouts, and
multi-choice generation with nonidentity convolution weights.
