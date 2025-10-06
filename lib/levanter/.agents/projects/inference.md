# OAI-compatible Inference Server (Levanter)

> **Note**: For somewhat more detailed technical design and architecture, see [`docs/design/inference.md`](/docs/design/inference.md).

**Goal**: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

## Current Milestone: Minimal single-turn completions (FastAPI)

**Scope**: Implement only `POST /v1/completions` (no chat, no streaming, batch size = 1). Single request -> single output text. Use tokenizer + model via a simplified wrapper of `sample_lm` logic (prefill + decode loop), with `max_tokens`, `temperature`, and optional `stop` (string or list of strings). Return OpenAI completion response shape.

**Out of scope**: SSE streaming, batching >1, usage tokens accounting beyond basic counts, top_p/top_k penalties, function/tool calls.

### Progress Tracking

- [ ] **Server scaffold**
  - [ ] Add optional dependency group `serve` (fastapi + uvicorn).
  - [ ] Create `src/levanter/serve/min_server.py` exposing `app` and a `main()` to run uvicorn.
  - [ ] Endpoint: `POST /v1/completions` with minimal OpenAI-compatible schema.
  - [ ] Health: `GET /healthz` returns 200 when model is ready.

- [x] **Minimal model wrapper**
  - [x] `GenerationService` (single-seq) that loads tokenizer + model via `_load_model` and sets up `PageTable`, `KvPageCache`, `JitScheduler`, `DecodeState`, `Sampler`.
  - [x] `generate_once(prompt, max_tokens, temperature, stop) -> text` using the same enqueue + `run_generation_loop` + `extract_outputs` flow as `sample_lm`, but limited to a single sequence and a blocking loop until finish or `max_tokens`.
  - [x] Convert `stop` strings to token sequences with tokenizer and pass via `SeqDecodingParams`.

- [x] **Request/response models** (subset of OpenAI)
  - [x] Request: `{ model: str, prompt: str, max_tokens?: int, temperature?: float, stop?: str | string[], seed?: int }`
  - [x] Response: `{ id, object: "text_completion", created, model, choices: [{ index, text, finish_reason }], usage: { prompt_tokens, completion_tokens, total_tokens } }`

- [] **CLI**
  - [ ] `python -m levanter.serve.min_server --host 0.0.0.0 --port 8000 --hf_checkpoint <org/model> --tokenizer <org/model>`
  - [ ] Warmup one-token decode on startup to trigger JIT.

- [ ] **Acceptance criteria**
  - [ ] `curl -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"<id>","prompt":"Hello","max_tokens":8}'` returns a valid response with non-empty `choices[0].text`.
  - [ ] End-to-end runs on CPU with a tiny HF model; no crashes; initial JIT compile acknowledged in logs.

### Next Steps

- [ ] **Fix usage token counting** - Replace `len(text.split())` with actual tokenizer counts
- [ ] **Improve API design** - Replace tuple returns with GenerationResult dataclass for better extensibility
- [ ] **Test end-to-end** - Verify everything works with a tiny HF model on CPU
- [ ] **Add `/v1/models` endpoint** - Return model information
- [ ] **Implement queue-based concurrency strategy** - Replaced single lock with request queue and scheduler thread for true concurrency

## Implementation Tasks

### 1. Extract reusable generation backend
- [x] Create `src/levanter/inference/service.py` with `GenerationService`:
  - [x] Initialization: device mesh (`TrainerConfig`), tokenizer loading, `_load_model`, `PageTable.init`, `model.initial_cache`, `JitScheduler.init`, `DecodeState.init`, `Sampler`.
  - [x] Public APIs:
    - [x] `generate_once(prompt, options) -> text` (single sequence)
    - [ ] `generate(prompt: str | list[str], options) -> GeneratedResult` (batch)
    - [ ] `stream_generate(...)-> AsyncIterator[StreamChunk]` (streaming)
  - [x] Internals mirror `sample_lm` flow: enqueue prompts, `run_generation_loop`, `extract_outputs`, stop-seq handling, temperature, seed.
  - [x] Accept per-request options: `max_tokens`, `temperature`, `stop` (list[str]).
  - [ ] Accept additional options: `top_p`
  - [ ] Additional option: `n`
  - [ ] Additional option: `logprobs`
- [x] Refactor `run_generation_loop` logic into a method usable from both sync and async paths; keep jitted parts unchanged.
- [ ] Unit tests targeting service with a tiny config (CPU) to decode 1–2 tokens.

### 2. HTTP server scaffold
- [ ] Add optional dependency group `serve`: FastAPI + Uvicorn (or Starlette). Optional: Pydantic v2.
- [ ] New module `src/levanter/serve/min_server.py` exporting `app`.
  - [ ] Endpoints:
    - [ ] `GET /healthz` returns 200 once model is loaded and warm.
    - [ ] `GET /v1/models` returns loaded model id(s).
    - [ ] `POST /v1/completions` for text completion (prompt-based).
    - [ ] `POST /v1/chat/completions` for chat.
- [ ] Authentication: accept `Authorization: Bearer <token>` if `LEVANTER_API_KEY` set; otherwise open.
- [ ] Streaming: if `stream=true`, return SSE with `data: {delta...}` chunks and `data: [DONE]` per OpenAI semantics.

### 3. Request/response schema mapping
- [ ] Define Pydantic models for requests/responses (subset of OpenAI spec):
  - [ ] Completions: `prompt`, `model`, `max_tokens`, `temperature`, `stop`, `seed`.
  - [ ] Chat: `messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `user`.
  - [ ] Response objects include `id`, `object`, `created`, `model`, `choices`, `usage`.
  - [ ] Translate chat `messages` -> prompt via a simple, configurable chat template (default: Llama 3 style). Support `system` and `user` roles; ignore tools/functions initially.
  - [ ] fix chat template / get from tokenizer by default.
  - [ ] Default to using <eot> as turn end for chats
  - [ ] Compute `usage` based on tokenized prefix and generated tokens.

### 4. Scheduling, batching, and backpressure
- [ ] Async request queue that batches requests and feeds `JitScheduler`.
- [ ] Per-request state in `DecodeState` rows; enforce limits: `max_seq_len`, `pages_per_seq`, `page_size`, `max_new_tokens`.
- [ ] Backpressure: bound queue size and timeout; return 429/503 on saturation.
- [ ] Cancellation: detach sequence and `purge_queue_of_seq` on client disconnect.

### 5. Model loading, configuration, and CLI
- [x] Support either `checkpoint_path` or `hf_checkpoint` (mutually exclusive), same as `SampleLmConfig`.
- [x] Reuse `TrainerConfig` for device mesh and `mp` dtype; allow YAML config file.
- [ ] CLI `python -m levanter.serve.min_server`:
  - [ ] Args: `--hf-checkpoint`, `--checkpoint-path`, `--tokenizer`, `--port`, `--host`, `--seed`, `--log-level`, `--access-log`.
  - [ ] Warmup: perform one tiny decode to compile.

### 6. Observability
- [ ] Structured logging for lifecycle and request events (start, end, errors, latency, tps).
- [ ] Optional Prometheus `/metrics` exporter with request counts, latencies, tokens/sec, compile time.
- [ ] Error surfaces include truncated stack traces and 4xx/5xx mapping.

### 7. Testing
- [ ] Unit: schema validation, prompt templating, stop sequence application.
- [x] Integration: spin up server with tiny CPU config; `POST /v1/completions` generating <= 4 tokens.
- [ ] Streaming: verify incremental deltas and `[DONE]` terminator.
- [ ] Load smoke: concurrent requests (N=4–8) exercise batching path.

### 9. Documentation & examples
- [ ] `README` snippet with curl examples for both endpoints.
- [ ] OpenAI SDK example:
  - Python: set `OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")`.
  - Node: `new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "sk-local" })`.
- [ ] Note unsupported options initially (function calling, tools, `response_format`, images, vision).

### 10. Stretch/Follow-ups
- [x] `n>1` parallel sampling and multi-choice responses.
- [ ] `logprobs` surface and token-level details.
- [ ] Presence/frequency penalties; repetition penalty.
- [ ] `/v1/embeddings` if/when model supports.
- [ ] Tracing (OpenTelemetry) and richer metrics.

## Core Infrastructure Tasks

### Attention & Caching
- [x] Attention: do prefill for a batch of inputs
- [x] Attention: do decode for a batch of inputs
- [x] Attention: unit test that decode gives same output as prefill
- [x] simple sampler
- [x] initialize cache for a model (add a method to the lmheadmodel class?)
- [x] make a main to test the above
- [x] use paged attention for simple kvcache decoding (should be doable)
- [x] add support for ragged paged attention
- [x] stop sequences

### Offline Milestone
- [x] make an offline batch inference that does continuous decoding
- [x] Update JIT scheduler to recognize sequences properly using `hax.where` with size/fill
- [x] Remove tokens from the queue based on sequence ID using a JIT-safe mask
- [x] Automatically slide tokens forward when removing entries
- [x] Integrate stop sequence into decode state
- [x] Implement max tokens per sequence limit
- [x] Add option to record logprobs
- [ ] add swapping in new sequences into jit scheduler
- [x] per-seq PRNG states inside the JIT scheduler
- [x] Rename jit scheduler to something more appropriate (DecodeState?)
- [ ] Finish offline batch inference support

### LM Eval Harness milestone
- [ ] Support LM Eval Harness generation tasks
- [ ] Stop sequences
- [ ] max num tokens
- [ ] temperature

### Prefix Cache Integration
- [ ] Use radix tree draft to actually drive scheduling
- [ ] Enable automatic prefix caching
- [ ] When returning from JIT scheduler, insert allocated pages into the radix cache. careful to handle the case where there were the sampled tokens already happen to exist.
- [ ] Radix cache must return freed pages when a sequence is released
- [ ] Manage list of free pages outside the radix cache
- [ ] Double check radix LRU evict

### Optimization Ideas
- [ ] microoptimize jit dispatch
- [ ] Investigate why tensor parallelism doesn't work for logits / vocab mat mul
- [ ] Avoid computing full logit matrix during prefill unless `logprobs` are requested
- [ ] figure out why the profiler isn't giving me anything useful
- [ ] software pipeline / double buffer decode and post-processing similar to
- [ ] push prefill into main jit decode loop
-
#### Mixed Prefill + Decode
- [ ] possibly add decode tokens to prefill (should be pretty easy with prefill in decode loop)

### Quality of Life

- [ ] automatically calculate cache size based on model config and what the core inference loop needs
- [ ]

### Sample LM Integration
- [ ] expose a free list of pages in `PageTable`
- [ ] allocate pages from the free list inside the generation loop
- [x] store partial sequences in `DecodeState` instead of `JitScheduler`
- [x] check `DecodeState.is_finished` during generation
- [ ] integrate any remaining pieces needed for `sample_lm`

## Future Milestones

### Milestone: Chat Completions
- [ ] Implement `POST /v1/chat/completions` endpoint
- [ ] Add chat message templating (Llama 3 style)
- [ ] Support streaming responses with SSE

### Milestone: Batching & Concurrency
- [ ] Implement request queuing and batching
- [ ] Add concurrent request handling
- [ ] Implement graceful backpressure

### Milestone: Production Features
- [ ] Add authentication (optional API key)
- [ ] Implement proper metrics and observability
- [ ] Add graceful shutdown handling
- [ ] Create Docker container and deployment examples

## Acceptance criteria
- Can run: `python -m levanter.serve.min_server --hf-checkpoint <org/model> --tokenizer <tok> --port 8000` and:
  - [x] `GET /healthz` returns ready status.
  - [x] `POST /v1/completions` returns a valid OpenAI response.
  - [ ] `GET /v1/models` returns model id.
  - [ ] `POST /v1/chat/completions` returns a valid OpenAI response; streaming works with SSE; `usage` is populated.
  - [ ] Concurrency: 4 parallel requests; batching occurs; no crashes; graceful shutdown.

## Notes

This milestone deliberately avoids concurrency and streaming to minimize moving parts while validating the integration of tokenizer, model, scheduler, and response schema.

For detailed technical design and architecture, see [`docs/design/inference.md`](/docs/design/inference.md).
