# Native Levanter inference

Levanter serves decoder models with a paged KV cache, continuous request
admission, and JIT-compiled prefill and decode. Snowball uses the same model
parameters and router as the learner, with paged attention and expert dispatch
selected independently for serving.

## Entry points

| Layer | Responsibility |
| --- | --- |
| `marin.inference.levanter_backend.LevanterBackend` | Resolve the HF snapshot, tokenizer, device mesh and compute dtype, then own the server lifecycle. |
| `levanter.inference.runtime.InferenceContext` | Normalize generation inputs, queue requests, fence policy versions and own the engine. |
| `levanter.inference.openai` and `levanter.inference.binary` | Adapt HTTP JSON, SSE and tensor frames to the shared runtime. |
| `levanter.inference.engine.InferenceEngine` | Admit requests, manage KV pages, and run compiled prefill and decode rounds. |
| `levanter.models.snowball.SnowballLMHeadModel` | Implement cache allocation, paged forward, and the learner's full forward. |

For programmatic serving, pass a `ModelSpec` and `LevanterEngineConfig` to
`LevanterBackend.serve`. The context manager performs mesh setup and checkpoint
loading, starts the server and releases it on exit. For standalone entry points
that already own a model and mesh, create an `InferenceServer` with an
`InferenceServerConfig` and an `InferenceEngineConfig`. The engine config sets
the context bound and physical page budget. Use `snowball_layerwise_load` for
large HF checkpoints to bound the temporary weight-conversion memory to one
layer.

## Generation contract

OpenAI text completions accept exact prompt IDs. Chat accepts
`_skyrl_exact_prompt_token_ids` for clients that already own the tokenized
conversation. With `return_token_ids`, the response returns the prompt and
completion IDs. Each request starts at position zero. Its completion positions
start at the prompt length. A request must leave space within the configured
context and its output is capped by the remaining space.

Both protocols pass effective temperature, top-p, seed, stop tokens and choice
count to the same engine. The scheduler emits the token that completes a stop
sequence. Its finish reason and scores are aligned with those emitted IDs.
Responses include the install version and immutable policy identity. Clients
can pin either with `expected_model_version` or `expected_policy_id`. A request
for a stale snapshot gets HTTP 409.

- `behavior_logprobs` gives selected-token scores under temperature and top-p.
  Greedy behavior assigns the selected token log probability zero.
- `policy_logprobs` gives selected-token scores on the untransformed cached
  logits. Enable `InferenceEngineConfig.return_policy_logprobs` on the server
  and request `return_policy_logprobs`. At temperature and top-p equal to one,
  the sampler can reuse the behavior normalization.
- OpenAI `logprobs` retains the API's sampled-token field. With greedy decoding
  its value is the model score after the sampler's logit transform. The explicit
  fields above distinguish raw policy and behavior scores.

Incremental chat SSE emits text during decoding and returns optional policy
scores and route evidence in the final event. Requests for exact token IDs
or OpenAI logprobs use buffered SSE. Aborted requests carry their emitting
version and partial outputs.

### Binary requests

POST `/v1/levanter/generate` with content type
`application/vnd.levanter.generate.v1`. Use `encode_frame` and `decode_frame`
from `levanter.inference.binary`. A frame consists of the versioned magic,
a big-endian 32-bit metadata length, UTF-8 JSON, and a safetensors tensor table.

The request tensor table contains one one-dimensional int32
`prompt_token_ids` tensor. JSON metadata is validated by `GenerateOptions`.
The response echoes the prompt tensor and returns per-choice tensors under
`choice.<index>.`. The choice metadata contains `finish_reason`,
`model_version`, and `policy_id`. Output tensors carry token IDs, behavior
scores, and optional policy scores and route bytes. The OpenAI route field
uses base64 for the same route bytes.

## Policy installation

Call `InferenceServer.reload` with a snapshot loader, the expected current
version, and a unique policy identity for the candidate snapshot. The loader
can stage sharded candidate tensors while the old model serves. The runtime
validates the tree, shapes, dtypes and shard placement and waits for device
readiness before admission changes. At install it pauses admission, aborts or
drains active requests, invalidates queued requests from the previous epoch,
clears KV and prefix state, and commits the new model and version together.
Failed staging preserves the previous serving version.

A serving instance starts at version zero. Its policy identity names the
snapshot across instances. Assign that identity from the publisher's snapshot
manifest when coordinating refreshes across replicas.

## Router replay

Enable `InferenceServerConfig.capture_routes` for Snowball and request
`return_route_captures` with one choice per prompt. Capture records the
per-layer expert IDs, QB threshold and validity at the same forward that
produces the cached logits. Captured prefix rows follow the retained KV
pages and the active policy. A final route-only decode covers the terminal
token before the scheduler releases the slot.

The wire payload binds tokens, positions, sampling parameters, layer order
and policy identity. Version 2 stores expert IDs as one byte per ID when they
fit. The decoder restores the canonical int32 IDs before checking the digest
and can read existing version 1 archives. `align_snowball_policy_routes`
checks those coordinates against padded learner rows. `PolicyRouteCaptures` stores replay arrays on
named batch and position axes. `pack_policy_batch` moves the route rows with
the tokens and masks. `response_logprobs` passes them to Snowball for both
forward scoring and backward recomputation. Current weights still determine
router combine weights and differentiable activations.

## Qualification

Run fixed input sequences and frozen weights through the learner forward,
prefill and one-token cached decode. Compare selected-token logprobs and
full-vocabulary forward KL independently. Route replay controls the discrete
router decisions. It leaves other sources of numerical error to the score
comparison. Zero reference-KL penalty requires both the loss and reward KL
flags off.

The production-engine A/B compares the same requests and seeds with route
capture off and on, including payload serialization. The full AgentX test
uses the same checkpoint, tokenizer, frozen sessions, context and concurrency
as the compiled vLLM results. See the branch [integration
handoff](https://github.com/marin-community/marin/blob/codex/dlwh-levanter-rl/INFERENCE_HANDOFF.md)
and the [Snowball measurement
report](../../../../docs/reports/snowball-inference-status-2026-11-20.md).
