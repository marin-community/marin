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

### Teacher scoring

Completion requests with `max_tokens=0` return no generated tokens. With `echo=true`
and positive `logprobs`, they score the exact prompt sequence, including a sequence
that fills `max_seq_len`. Scoring uses the installed model under the same lock as
weight publication and generation; it does not allocate or reset paged decode state.
Paused requests still return `abort` and are not scored.

This supports SkyRL's external `OpenAICompatibleTeacherOracle` chosen-token and
top-K distribution evidence. Set `return_tokens_as_token_ids=true` to retain exact
candidate identities. The existing first-token sentinel is zero because that token
has no preceding context. Arbitrary student-selected token scoring and rollout
candidate capture are separate contracts and are not provided by this endpoint.

### Remote weight-sync pause

`POST /pause_generation` takes `{"mode": "abort", "clear_cache": true}`. It closes
admission, returns each active request's exact partial token IDs and logprobs with
finish reason `abort`, then clears KV and model-specific cache state before
acknowledging. Requests submitted while paused return an empty `abort` result.
`POST /resume_generation` reopens admission. SkyRL's single-request retry loop
waits for resume, appends the partial IDs to the prompt, and requests the remaining
tokens. Deterministic greedy continuation matches uninterrupted generation when
weights stay unchanged; cache state is rebuilt.

Configure the paired remote SkyRL client with `generator.weight_sync_pause.mode=abort`
and `clear_cache=true`. Native serving rejects `keep`, `wait`, and cache retention
before changing state. SkyRL's default `keep` policy is not silently converted.
The current Marin SkyRL launcher still requires local engines; remote transport
support alone does not enable a remote-engine training topology.

The paired retry regression is
`test_remote_skyrl_pause_retries_exact_tokens_after_resume` in
`tests/inference/test_inference_server.py`. Add the paired SkyRL repository root and
its `skyrl-train` directory to `PYTHONPATH`, install its CPU client dependencies
(Ray, OmegaConf, and loguru), and run that test explicitly with `-m integration`.
It uses a local HTTP server and synthetic weights; it does not launch training.

### Remote weight publication

Set `InferenceServerConfig.weight_transfer` to `WeightTransferConfig(backend="gloo",
max_staging_bytes=...)` to enable SkyRL weight control routes. Install PyTorch in the
serving environment (`torch_test` supplies it for local validation). Use `nccl` for
a single GPU. CPU/Gloo and a tiny single-receiver H100/NCCL gate pass in FP32 and
BF16. Multi-device and multi-process serving are rejected.
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

### Weight-transfer validation

`experiments.benchmarks.remote_weight_transfer` runs the real SkyRL remote client,
a native HTTP server, and a separate Torch sender with a bounded synthetic Llama.
It checks incomplete and stale publications, exact generation IDs/logprobs, cache
reset, and one successful model-version transition. The script requires SkyRL
commit `b3297eddfe67358004ee925a42c9ffca81ac8f6e` on `PYTHONPATH` and records both
repository revisions and runtime versions in its JSON result. Pass the native
source revision explicitly; a local checkout verifies it against Git, while an
Iris source bundle records the revision supplied by its launcher. It does not launch
training or change the repository's fork pin.

In a prepared CPU environment with serving and Torch dependencies:

```bash
PYTHONPATH=/path/to/MarinSkyRL/skyrl-train JAX_PLATFORMS=cpu \
  uv run --no-sync python -m experiments.benchmarks.remote_weight_transfer \
  --marin-revision "$(git rev-parse HEAD)" --backend gloo --dtype bfloat16 --output /tmp/weight-transfer-gloo.json
```

For the NCCL gate, use a node with two GPUs and a prepared JAX/Torch CUDA
environment. The sender owns physical GPU 0 and the receiver owns physical GPU 1;
each process sees its GPU as `cuda:0`. The script checks their UUIDs differ, that the
receiver sees one JAX device, and that installed arrays stay on that device.

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/path/to/MarinSkyRL/skyrl-train \
  uv run --no-sync python -m experiments.benchmarks.remote_weight_transfer \
  --marin-revision "$(git rev-parse HEAD)" --backend nccl --sender-device 0 --receiver-device 1 --dtype bfloat16 \
  --output /tmp/weight-transfer-nccl.json
```

Run both dtypes (`float32` and `bfloat16`). The sender uses host-generated toy
weights; the native receiver uses device-resident DLPack. This gate does not cover
sharded production models, production-sized staging memory, or trainer execution.
The CPU regression in `tests/inference/test_weight_reload.py` also covers Snowball
and Hero with individual expert tensors and populated short-convolution history.

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

## Reproducible batch benchmarks

`levanter.main.inference_benchmark` measures the native batch engine with fixed
prompt token IDs, greedy sampling, and a fixed number of generated tokens. The
checked-in small Snowball workload exercises both short and long attention
layers with random weights:

```bash
RAGGED_DOT_IMPL=xla uv run --package marin-levanter python -m levanter.main.inference_benchmark \
  --model-config lib/levanter/config/inference/snowball_tiny.json \
  --workload lib/levanter/config/inference/tiny_workload.json \
  --dtype bfloat16 --hardware-label v5p-8 --output /tmp/snowball-v5p.json
```

Use the hardware label for the actual provisioned slice. The driver supports a
single host and records every visible device, mesh dimensions, engine settings,
model configuration, seed, dtype, code revision, and selected XLA environment
flags. Source bundles inherit Iris launch provenance through `MARIN_PROVENANCE`;
`--source-revision` and `--source-dirty` can supply it explicitly. Without source
metadata the revision and dirty state are null. `--model-axis-size` defaults to 1; the remaining local devices occupy the
Grug data axis. Before changing tensor parallelism, check head divisibility:
production Snowball has 20 query heads and 5 KV heads, so an 8-way head partition
is invalid. The native driver requires Snowball's paged `decode` implementation.

The output retains raw samples and hashes the complete token workload. Model
initialization and cache setup are timed separately. The first batch includes
compilation and execution; subsequent warmup batches are separate from measured
batches. `compile_only` is null because the harness does not isolate compiler
time. Persistent compiler caches can change first-batch latency.

All prompts must fit in one prefill. Native first-token time is observed after
prefill and host extraction, immediately before the first decode callback. The
JSON records batch output throughput and mean time from first token to batch
completion per remaining output token. This amortized measure includes host
scheduling, synchronization, and extraction; it is not a distribution of device
kernel or per-token latencies. Incomplete generations fail instead of producing
a throughput number. The OpenAI server currently renders streaming events from
a finished response, so its HTTP first event cannot measure native prefill time.

### Checkpoint-backed comparison

Replace `--model-config` with `--checkpoint` to load the same Snowball HF export
used by vLLM. Supply `--revision` for a Hub revision and `--checkpoint-identity`
with the immutable export identity or weight digest from the baseline manifest.
The existing HF converter reads the checkpoint's model config and tokenizer;
Schema-v1 Snowball and schema-v2 Hero exports select their respective native model. The result
records the checkpoint location, identity, requested revision, full HF config,
and tokenizer vocabulary hash. Export identities for object-storage paths are
operator-supplied; the driver does not rehash multi-gigabyte weight files.

```bash
python -m levanter.main.inference_benchmark \
  --checkpoint /regional/snowball-hf-export \
  --checkpoint-identity exported-training-step-and-weight-digest \
  --workload workload.json --dtype bfloat16 \
  --hardware-label H100-8 --output /tmp/levanter-checkpoint.json
```

Keep checkpoints in the accelerator's region. Model loading remains outside the
measured generation samples. Compare checkpoint-backed results only when the
weight/config identities, token workload hash, dtype, accelerator count,
topology, and parallelism match.

### vLLM baseline

Snowball and Hero require the Marin vLLM fork that registers `GrugMoeForCausalLM`.
Use the promoted fork wheel and CUDA/PyTorch pins from Marin serving
(`IsolatedCudaVllm` with `VllmType.MARIN_FORK`), not an upstream vLLM install.
Run `levanter.main.vllm_inference_benchmark` in that serving environment with
three JSON files:

- `--workload`: the identical prompt token IDs and output count used by Levanter.
- `--engine-args`: keyword arguments for vLLM `EngineArgs`, including `model`, an
  immutable `revision` when loading from the Hub, `dtype`, tensor parallelism,
  context and batch capacity. Prefix caching must be disabled.
- `--provenance`: `checkpoint` (immutable revision or weight digest), the complete
  `model_config`, `dtype`, and `hardware_label` including accelerator count and
  topology. These fields are supplied by the operator and retained alongside
  actual runtime versions, visible CUDA devices, and engine arguments.

```bash
python -m levanter.main.vllm_inference_benchmark \
  --engine-args vllm-engine.json --provenance checkpoint-manifest.json \
  --workload lib/levanter/config/inference/tiny_workload.json \
  --output /tmp/vllm-result.json
```

This uses the public [LLMEngine step interface](https://docs.vllm.ai/en/v0.10.2/api/vllm/engine/llm_engine.html)
and cumulative output token IDs. First-token observations include admission and
scheduling. If vLLM admits requests across multiple prefills, its first-token
and decode overlap differs from Levanter's single-prefill measurement. Compare
end-to-end throughput with that scheduling difference recorded. The adapter
requires validation against the deployed vLLM version, including the Marin
GrugMoE model registration. It has not been validated on a TPU vLLM runtime.

### Coverage and remaining model work

No accelerator measurements are checked in with this harness. Random-weight
Snowball measurements characterize execution only. They cannot establish a
speedup over the checkpoint-backed vLLM baseline. A matched model comparison
also requires identical weight/config
identities, tokenizer provenance for the token workload, dtype, prompt/output
lengths, concurrency, accelerator count, topology, and parallelism. Compare
output hashes or token arrays and investigate differences before reporting a
speed ratio. Production checkpoint loading and real serving latency remain separate
validation steps.

| Target | Native Snowball benchmark | Native Hero benchmark | vLLM comparison |
| --- | --- | --- | --- |
| H100 | Driver available; unmeasured | Driver available; unmeasured | Baseline adapter; unmeasured |
| GB200 | Driver available; unmeasured | Driver available; unmeasured | Baseline adapter; unmeasured |
| TPU v4 | Driver available; unmeasured | Driver available; unmeasured | Backend validation required |
| TPU v5p | Driver available; unmeasured | Driver available; unmeasured | Backend validation required |
| TPU v6e | Driver available; unmeasured | Driver available; unmeasured | Backend validation required |

Hero's `experiments/grug/moe_hero_ep/heuristic.py:HERO_MODEL` uses 48 layers,
width 6144, 384 experts with top-8 routing, two shared experts, latent dimension
3072, short convolutions, and 12 local / 6 global KV heads. Snowball pins the
June recipe with 26 layers, width 2560, 256 experts with top-4 routing, one shared
expert, and 5 KV heads. Both export `model_type=grug_moe`; that name does not
establish architectural equivalence. The native adapters preserve their distinct schema versions and architectures.
For a random-weight Hero smoke benchmark, use `config/inference/hero_tiny.json`
with the same token workload and driver arguments as Snowball. Synthetic results
measure the tiny configuration, not the production Hero model.
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

For checkpoint-scale comparisons, set `--model-axis-size` and
`--expert-axis-size` to match the vLLM tensor and expert parallel configuration.
The remaining local devices partition the data axis; the result records the
complete effective mesh and device list. Both drivers must use the same
checkpoint identity, tokens, dtype, and device allocation.

### Matched synthetic checkpoint smoke comparison

`experiments.benchmarks.matched_inference` writes a small BF16 checkpoint with
head dimension 128, a tokenizer, an immutable weight digest, and a token workload.
It verifies every exported tensor through native HF loading. Both backends then
load this identical checkpoint from local disk; this measures a complete small
model and does not represent a production Snowball or Hero checkpoint.

Run the following on one CUDA accelerator, with the same single device visible
to both processes. Keep the fixture directory on that worker. The vLLM command
uses Marin serving's promoted fork, PyTorch pin, and CUDA toolchain through an
isolated environment. It does not install vLLM into the JAX environment.

```bash
export CUDA_VISIBLE_DEVICES=0
uv run python -m experiments.benchmarks.matched_inference export \
  --recipe hero --output /tmp/hero-comparison
uv run python -m experiments.benchmarks.matched_inference native \
  --fixture /tmp/hero-comparison --hardware-label H100x1
uv run python -m experiments.benchmarks.matched_inference vllm \
  --fixture /tmp/hero-comparison --hardware-label H100x1
```

Use `--recipe snowball` and a separate fixture directory for Snowball. Both
measurement commands print their result JSON for remote log retention. Compare
workload and generated-token hashes before comparing throughput. vLLM uses eager
execution for this startup smoke test. Pass `--execution-mode compiled` to
allow vLLM compilation and supported CUDA graph capture; effective compilation
and graph modes are recorded separately from requested engine arguments. The
promoted fork disables CUDA graphs for Hero short convolutions. Tiny fixtures
reserve 64 MiB of KV cache per rank; override `--kv-cache-memory-bytes` for larger
workloads. Both modes use a fresh per-invocation Triton
cache so inherited JAX cache settings cannot disable vLLM kernel compilation.
FlashInfer builds use two Ninja workers by default (`--compile-workers`) and one
nvcc thread per command. These limits and the effective execution modes are
recorded in the report; eager inference can still compile CUDA kernels during startup.
For the GB200 SM100 startup gate, `--flashinfer-jit-cache-wheel URL#sha256=DIGEST`
installs an explicit compatible precompiled cache in the isolated runtime. It
loads and hashes `fused_moe_100.so` before engine initialization and records its
path, package versions, and successful load. JIT compilation is disabled for
this gate: missing or incompatible precompiled modules fail instead of starting
a long build. FlashInfer 0.6.18.post1's
[installer](https://github.com/flashinfer-ai/flashinfer/blob/8bc3b578027791336c6ae87db5c9d76f82cef8bc/flashinfer/__main__.py#L108)
selects its CUDA 13.0 cache for CUDA 13.2, and the
[official ARM64 wheel](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/)
contains this MoE library. This cache
selection preserves the backend and autotuning configuration; accelerator
validation is still required for a given wheel/runtime pair.

The async vLLM client distributes requests across data-parallel ranks when
configured, while preserving per-request first-token timestamps. Larger
workloads require a separate benchmark configuration.

After both runtime commands finish, write the paired report:

```bash
uv run python -m experiments.benchmarks.matched_inference compare --fixture /tmp/hero-comparison
```

`comparison.json` checks checkpoint identity, loaded HF configuration, architecture,
dtype, exact token workload, accelerator allocation label and kinds, and the
supported parallelism pairing: native EP=N/TP1/data1 versus local vLLM DP=N with
EP enabled for N>1. Backend attention selectors remain visible execution choices.
Other configuration differences reject the comparison.

Each runtime saves one additional validation batch after the timed samples. Its
tokens and hash remain outside the throughput summary. The comparison records
hash agreement across every batch, the first differing request and output-token
position, and effective execution modes. It withholds the throughput ratio if
outputs differ across backends or batches. Reports from before this validation
capture must be rerun; timing hashes alone cannot identify a divergent token.

For a divergent Snowball token, write a prefixes JSON with `sequences` containing
both original prompts followed by their common generated tokens, and
`prefill_length` equal to the original prompt length. The bounded diagnostics
support up to eight equal-length sequences and 128 tokens:

```bash
uv run python -m experiments.benchmarks.diagnose_native_prefix \
  --fixture /tmp/snowball-comparison --prefixes prefixes.json --expert-axis-size 2
```

Run `python -m levanter.main.vllm_prefix_diagnostic` in the same isolated vLLM
environment, passing the fixture's `--engine-args`, `--provenance`, the same
`--prefixes`, and an `--output` path. It records next-token top-20 logprobs from a
single prefill. The native diagnostic records full-vocabulary logits from both
a single prefill and the original prefill followed by forced one-token steps.
It independently varies BF16/FP32 router weights and baseline/highest matmul
precision, recording all four combinations for each input mode. These diagnostic
interventions do not change the model's training or serving defaults.
These passes are numerical diagnostics and do not produce throughput claims.

The diagnostic also records the residual before final RMSNorm, the normalized
and gated hidden states, and a host FP64 projection using the exact BF16 head
weights. Native capture substitutes an identity output head in separate untimed
passes; vLLM installs read-only model hooks after startup through a named worker
extension and removes them after the probe. The RPC carries method names and
plain results; it does not require callable serialization or unsafe-deserialization
settings. The vLLM capture requires eager execution and TP1,
and records token IDs and positions to align data-parallel rows. Both sides
record a canonical head-weight digest. These tensors distinguish differences
already present in the hidden state from final projection rounding; the FP64
projection is diagnostic evidence, not a new serving precision contract.

### Matched TPU fixture gate

The `vllm-tpu` fixture command provisions the existing `IsolatedTpuVllm` fork
pins, JAX 0.11.0, and libtpu 0.0.44 in a separate environment. It selects
`MODEL_IMPL_TYPE=vllm` explicitly and uses single-process SPMD data parallelism.
The native command uses `expert-axis-size=1`, leaving the remaining devices on
its data axis. Set `data-parallel-size` to the actual local JAX device count;
the driver discovers the devices in a subprocess that exits before vLLM starts.

```bash
uv run python -m experiments.benchmarks.matched_inference export \
  --recipe snowball --output /tmp/snowball-tpu-fixture
uv run python -m experiments.benchmarks.matched_inference native \
  --fixture /tmp/snowball-tpu-fixture --hardware-label v6e-local4 --expert-axis-size 1
uv run python -m experiments.benchmarks.matched_inference vllm-tpu \
  --fixture /tmp/snowball-tpu-fixture --hardware-label v6e-local4 --data-parallel-size 4
uv run python -m experiments.benchmarks.matched_inference compare \
  --fixture /tmp/snowball-tpu-fixture
```

The comparison requires the same TPU allocation, native data=N/EP1/TP1 and
vLLM SPMD data=N/EP1/TP1. Reports retain the runtime pins, discovered devices,
effective sharding, and package versions. `enforce_eager` disables vLLM's Torch
compilation path; TPU execution still uses JAX compilation. This is a prepared
full-model correctness gate, not a measured TPU throughput result. The pinned
Torchax MoE currently bypasses Grug's custom router in its monolithic kernel
path; a custom-routing bridge is required before this gate can establish
architecture parity. The native JAX Grug fallback also lacks the current
combine-weight normalization and is not an equivalent baseline.

Hero remains unsupported by this pinned TPU fixture. The vLLM fork at
`70ea9ae8f2601f06d820ee9d70e3afbdc52683b1` parses schema-v2 Hero, but its
`grug_moe_short_conv` operator has no Torchax bridge in tpu-inference
`29548fbab663b7ea946546ca7efaa473dab55ba5`. The separate native JAX Grug model
accepts only schema v1. A Hero TPU baseline needs a cache-preserving short-conv
bridge and full-model parity before timing; disabling short convolutions would
change the model.
