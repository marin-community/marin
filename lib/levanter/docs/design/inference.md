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

### Snowball paged serving

Snowball serving uses `levanter.grug.attention.ragged_paged_attention` for
mixed prefill and decode batches. Queries use the layout
`[tokens, kv_heads, query_heads_per_group, head_dim]`; KV pages use
`[pages, page_size, 2 * kv_heads, head_dim]`, with interleaved key/value
slots. `implementation="reference"` is the portable CPU/GPU path and
`implementation="tpu"` selects JAX's standard TPU ragged paged-attention
kernel for eligible FP32 inputs. The TPU path falls back to the reference
implementation for lower-precision inputs.

The public implementation set is limited to `reference` and `tpu`. The former
`tpu_fp32_tiles`, `tpu_dma_buffers`, custom TPU decode kernel, compensated
FP64 arithmetic, and related precision-oracle helpers are removed. Callers
should use the reference path for CPU validation and the standard TPU path for
accelerator validation; no compatibility aliases remain.

### Production Snowball engine route replay

`levanter.inference.snowball_engine_replay.generate_snowball_replay` opts an
existing `InferenceEngine` into serving route capture. It runs the same prompt
admission, prefill, sampling and device `while_loop` used by `engine.generate`.
Its `generation` field retains the ordinary generated tokens, sampled
logprobs and finish reasons. `payloads` contains one host
`SnowballReplayPayload` per request in request order.

```python
import haliax as hax
from levanter.inference.snowball_engine_replay import (
    generate_snowball_replay, replay_snowball_logits, replay_snowball_loss,
)
from levanter.inference.snowball_replay_payload import SnowballReplayPayload

# The caller owns the serving mesh/precision context and an immutable identity
# for these exact policy weights. The learner verifies that same identity.
result = generate_snowball_replay(engine, requests, policy_id="checkpoint-sha256:...")
wire_bytes = result.payloads[0].to_bytes()
payload = SnowballReplayPayload.from_bytes(wire_bytes)
logits = replay_snowball_logits(model, payload, policy_id="checkpoint-sha256:...")
with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"),
                       "position": None, "vocab": "model"}):
    loss = replay_snowball_loss(model, payload, policy_id="checkpoint-sha256:...")
```

The supported scope is one choice per request (`n_generations=1`), nonempty
heterogeneous prompts with contiguous absolute positions, a causal mask,
STOP/EOS stop-token sequences and LENGTH completion. Pending requests may
outnumber slots; ownership is resolved to request ID and choice before host
extraction releases or reuses a slot. Clones and packed document segments are
unsupported. The capture API has no callbacks or abort parameter. It validates
its supported model, sampler and request scope before resetting the engine.
The page budget must cover the largest concurrently admitted requests through
their terminal input, including a terminal token on a new page.

Capture uses `SnowballLMHeadModel.decode_with_route_captures` for the actual
prefill and decode inputs. Since the scheduler purges a sampled terminal token
from the normal decode queue, capture performs one additional bounded serving
decode of finished sequences before their pages are released. That call samples
no output. Every input row, including the last sampled token, therefore has its
actual serving routes. Capture buffers are bounded by prefill capacity or
`max_rounds * max_tokens_per_round`, plus one terminal row per slot. They remain
on device until the existing host extraction boundary. Ordinary generation
allocates no route buffers and performs no terminal capture decode.

Each payload contains int32 tokens and absolute positions, the prompt boundary,
request ID/choice, effective sampling parameters and key, sampled float32
logprobs, finish reason, and per-layer `selected_experts`, `token_valid` and
`qb_alpha`. It transports no KV pages, weights or vocabulary logits. Versioned
JSON metadata plus safetensors arrays provides a bytes format without pickle.
A SHA256 binding covers exact tokens, row order, prompt, routes, metadata,
model configuration/vocabulary and the caller's policy ID. This detects altered
or truncated evidence; it does not authenticate an untrusted sender or hash the
model weights. The caller must assign and verify the immutable source-policy identity.
Learner parameters may change under replay; numerical equality comparisons
additionally require unchanged weights and matching arithmetic contexts.
The mask contract is contiguous causal attention, with Snowball's configured
layer/window schedule; arbitrary masks and packed segments cannot be supplied.

Learner `training_inputs` reconstructs `LmExample` and
`SnowballRouteCaptureBatch` directly from those transported arrays. Row `p`
predicts token `p+1`; the generated targets use predictor rows
`prompt_length-1` through `len(tokens)-2`, including the first and terminal
target. The final input row predicts no sampled token. `replay_snowball_loss`
accepts optional per-generated-target weights on the `target` axis. It calls
the public next-token loss, preserving current-parameter combine weights,
weighted reduction and gradients. It creates no teacher route capture or
optimizer state. Replayed experts and `qb_alpha` do not include training
capacity/drop state or router auxiliary losses.

Engine sampled logprobs describe its temperature/top-p sampling logits;
temperature zero selects greedily using unscaled logits. Raw-policy comparison
is initially supported only at temperature one and top-p one.
`payload.raw_policy_logprobs` rejects other settings. Learner logits are untransformed logits with the retained routing policy. CPU reference tests exercise scheduling,
transport and numerical replay at the existing tolerances. They do not
establish TPU bitwise equality, throughput or full production qualification.
External vLLM/SkyRL rollout transport and learner integration remain outside
this API.

### Opt-in fixed-schedule projections

Snowball’s `projection_implementation="fixed_schedule"` selects the fixed
BM=BK=BN=128 TPU contraction for Q/K/V projections, the shared DenseMLP’s
gate/up/down projections, and the LM head in full forward, prefill and cached
decode. Standard next-token training loss uses the same LM-head contraction,
with its existing weighting, reduction, dtype, soft-cap and auxiliary-loss
semantics. This training-loss path materializes full-vocabulary logits, so its
activation memory grows with token count and vocabulary size. The native
projection path remains the default.

The implementation supports a fully addressable single-device TPUv4 path and
a same-process explicit `NamedSharding` TPUv4 path with one row-only
data-parallel mesh axis. The Snowball helper retains flattened row sharding
and replicates weights before the contraction. Genuine contraction/output
feature sharding is outside the supported contract. For FP32 or BF16 input
and weight storage, the helper zero-pads contraction and output widths to
multiples of 128 and trims the output to its public width. Parameter shapes
and checkpoint keys stay the same.
The contraction casts activations to the weight storage dtype and restores outputs
to the activation dtype, which covers the Snowball caller’s FP32-parameter/
BF16-activation pairing. Unsupported backends, device kinds, layouts, device
mesh layouts, and widths fail explicitly; there is no shape-dependent fallback.

The multi-device contract is bounded to four same-process, fully addressable
TPUv4 devices in the actual Grug `compact_grug_mesh`: an Explicit mesh shaped
`replica_dcn=1, data=4, context=1, expert=1, model=1`. Grug activation specs
may name the row dimension as `(replica_dcn, data, expert)`; the fixed-schedule
kernel validates those names and canonicalizes only the known size-one axes to
physical row sharding on `data`. It enters the local Pallas body through a
same-topology all-Manual `shard_map` and restores the caller's Explicit
NamedSharding at the boundary, including an all-singleton Explicit mesh.
Unknown mesh axes, genuine TP/EP/model
sharding, remote or non-addressable topology, and feature/contraction sharding
remain unsupported. A custom-VJP cotangent must use the same mesh and either
the output’s row sharding or replicated rows. Returned primal, input-gradient,
and weight-gradient shardings retain the caller’s named singleton axes.
Row-sharded backward weight gradients remain FP32 through the deliberate
data-axis reduction and cast only afterward.

Callers must keep `jax.set_mesh(mesh)` active while constructing activation
specs and tracing or executing Explicit-mesh projections. The legacy
`with mesh:` context does not establish JAX's ambient Explicit mesh.

The forward uses DEFAULT products and FP32 accumulation with a K schedule
determined by the weight width. Its custom VJP uses separate HIGHEST-precision
FP32 reference GEMMs. A bounded four-device TPUv4 forward run established
raw-bit equality for a repeated shared-MLP token at 4, 512 and 1024 rows and
reduced cached-decode error. Training, prefill and cached decode still differed
in that fixture. Native custom-VJP qualification and throughput measurements
remain outstanding; the forward result does not establish full-model parity
or distribution-level equality. The opt-in does not change attention, RoPE, KV-cache casts,
MoE routing, or Splash selection. The scoped tests cover physical training
projection `[4, 128, 256]` and a true jitted attention decode with four active
sequences, logical sequence length one, physical token destinations
`[0, 8, 16, 24]`, and a 32-slot cache (`4` pages of `8` slots). Four-way
decode is bounded to this supported flat-token serving layout; it does not
claim arbitrary sequence or context sharding. The committed tests include the
dtype and tile-boundary matrix, independent custom-VJP gradient references,
forward and backward lowering checks, and an Explicit-mesh Snowball decode
caller that checks K/V placement, cache dtype and sharding, and untouched slots.

### Opt-in fixed-schedule GatedNorm

Snowball’s `gated_norm_implementation="fixed_schedule"` selects the existing
fixed-schedule contraction for every Snowball GatedNorm: the embedding gate,
each block attention and MLP gate, and the final gate. The default is
`"native"`; the selector is runtime-only and is omitted from HF architecture
configuration and checkpoint state. Gate parameter shapes, names,
initialization order, and checkpoint identity remain unchanged.

### Opt-in shared weighted RMSNorm

Snowball also exposes `rms_norm_implementation="reference"` or
`"pallas_tpu"` as a runtime-only opt-in. The default `"native"` path and all
HF architectural configuration and state-dict keys remain unchanged. The
existing `RMSNorm` objects are shared by embedding, both per-block weighted
norms, and the final norm, so the selected implementation is used by training,
route capture, prefill, and cached decode, with the accepted
`batch_invariant_rms_norm` custom VJP supplying the backward path.
The readable fixed-schedule reference is the independent AD oracle for
gradient checks; the ordinary mean-reduction expression remains the separate
forward and scalar-value comparison.

The accepted kernel validates the public contract: FP32/BF16 inputs and
weights, hidden dimensions 128/256/2560, and a global flattened leading-row
count from 1 through 512. The TPU implementation additionally requires a
same-mesh explicit `NamedSharding` with row-sharded flattened activations and a
replicated rank-1 weight (`P(None)`). Snowball does not add a fallback, gather,
chunking, or reshape around these limits; unsupported layouts fail through the
accepted kernel validation. In particular, a context-sharded mixed-axis
flatten is rejected at the Snowball RMSNorm boundary when its later leading
axis is genuinely partitioned. A length-one context axis is accepted because
it does not partition the rows.

### Opt-in Snowball shared attention forward

Snowball’s `shared_attention_implementation="fixed_schedule"` is a bounded
forward selector for the H128/E2top1/I32/sharedI32/Q2/KV1/D128/V23 fixture:
four 256-token slots with 128-token pages. The public boundary accepts only
unsegmented full-causal masks; external windowed, noncausal, and segmented masks
are rejected. Each layer derives its window from the model schedule.
Cached decode writes use the physical page map supplied by `PageBatchInfo`,
with placement checks at positions 127 and 128. The selector is runtime-only;
checkpoint fields and the native default remain unchanged.

The contiguous four-slot training adapter now retains that exact fixed forward
under a custom VJP. Its reverse mode is a readable grouped-query softmax
reference over BF16-materialized K/V operands, with FP32 QK and output
accumulation, causal positions, and reduction from two query heads to one KV
head. This is a cast-aware mathematical gradient convention: it does not claim
to differentiate discrete BF16 rounding or establish backward bitwise parity.
The reverse rule supports the bounded one- or two-layer, two-expert fixture;
short layers retain half-RoPE and a 128-token sliding window while long layers
remain NoPE and full-causal. Serving decode follows the same per-layer
schedule. The custom reverse rule applies only to the contiguous training
adapter; paged-cache training gradients are not supported or qualified here.
Other layer/expert counts remain unchanged.
Focused CPU reference checks cover
nontrivial cotangents at positions 127, 128, and 255 and all Q/K/V gradients.
Native TPU gradient qualification, including sharding and full-model
value-and-grad checks, remains pending and is owned by the parent validation
stage.

Snowball Q/K projection has a separate runtime-only
`qk_rms_norm_implementation` selector. Its default is the existing native
non-parametric RMSNorm; `reference` and `pallas_tpu` reuse
`batch_invariant_rms_norm` with unit weights and the fixed Q/K epsilon `1e-6`.
Q/K rows are processed in static chunks of at most 512, so larger
teacher-forced batches and cached decode callers retain the same public path.
The reference implementation composes the accepted primitive inside an
explicit row-local `shard_map` for multi-device layouts; size-one head and
context annotations are removed before flattening. Genuine partitioning of a
later leading axis, head axis, or hidden axis is rejected before lowering.
The selector is not serialized into HF configuration or checkpoint state, and
it does not claim full-model zero-KL parity. The TPU option accepts only the
primitive's qualified row-sharded, hidden-replicated layout and reports
unsupported layouts through the existing validation.

This is a bounded RMSNorm forward/backward step toward inference parity, not a
whole-model zero-KL claim. TPU layout and numerical validation remain a
separate runtime gate.

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

## Paged short-convolution history

`levanter.layers.paged_short_conv` provides the causal history needed by Hero's
key-projection, attention-output, and MoE-output convolutions. Each persistent
request slot stores the last `kernel_size - 1` input rows. Packed tokens select
history through `PageBatchInfo.slot_ids`, so request order and KV page allocation
do not change history ownership. New position-zero prompts clear reused slots;
padding does not write history.

`ShortConvSequenceCache.copy_sequence` copies the current processed prefix when
the engine clones a request, including clones at complete KV page boundaries.
KV pages retain their existing copy-on-write behavior. Global reset clears both
KV and convolution state. Earlier-prefix replay must recompute convolution
history; KV page reuse alone does not restore an earlier recurrent state.

The JAX scan preserves the reference convolution's lag-ordered arithmetic.
FP32/BF16 tests cover packed request reordering, kernels wider than a page,
chunked prefill, clone divergence, padding, and slot reuse. Actual Hero engine
tests compare cloned tokens and logprobs with independent generation and reuse
freed slots for queued requests.

For the 48-layer Hero configuration with hidden width 6144, stored KV width
1536, and three four-tap convolutions per layer, BF16 history occupies 3,981,312
bytes (3.796875 MiB) per request slot. A 4096-token request with 128-token pages
previously reserved 121.5 MiB of history across its 32 pages. CPU evaluation of
the actual cache initializer's shapes gives 1,211,940,864 bytes total after this
change, including 1,207,959,552 unchanged KV bytes: a 117.703125 MiB reduction.
These are logical allocation sizes; accelerator memory and latency measurements
for request-owned history remain pending.

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

## Snowball fixed-schedule attention postprocessing

The opt-in `projection_implementation="fixed_schedule"` path uses the bounded
XSA kernel for the per-head projection away from the current value row. It then
uses the same fixed-schedule linear projection for the head gate and `w_o`.
The XSA reference computes one float32 dot and value norm per flattened token
row with `1e-6` epsilon; the TPU implementation requires width 128 and keeps
the row schedule fixed on the bounded four-device data-parallel mesh (context,
expert and model axes each have size one). Forward and reference-backward paths
preserve caller layouts, including singleton mesh annotations. CPU uses the
portable reference. Native behavior remains the default.
