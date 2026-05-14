# Iris vLLM Inference Service

Run Marin evals against vLLM on Iris without making each evaluator start or
manage a model. Evaluators should keep consuming `RunningModel` and
OpenAI-compatible HTTP. Iris should own vLLM startup, readiness, request
routing, worker cleanup, and diagnostics.

The first milestone is a standard Hugging Face model that vLLM already serves
well. That is enough to validate the service boundary, worker lifecycle,
OpenAI proxy, result upload, and one real lm-eval generation path. Harder
model- and scoring-specific checks stay as follow-up compatibility work, not
as blockers for the service shape.

See [research.md](./research.md) for code references and pressure-test
evidence. See [spec.md](./spec.md) for the concrete API contracts.

## Challenges

The HTTP API is not the hard part; vLLM already exposes OpenAI-compatible
endpoints. The hard part is where lifecycle and failure handling live.

- Workers are elastic Iris jobs. They need to start vLLM, wait for readiness,
  pull work, forward requests locally, and report terminal results.
- The eval-side proxy must wait on long-running broker operations without
  losing Iris/Fray actor context.
- Prompt-logprob scoring uses `/v1/completions`, which has different failure
  modes from generation through `/v1/chat/completions`.
- The first implementation should prove batch eval serving, not become a
  multi-tenant online inference gateway.

## Costs / Risks

- vLLM is the only served engine. If vLLM cannot serve a scoring task, that
  task stays on an existing eval path until vLLM is fixed.
- The broker is a single in-memory coordinator. Broker or eval-job restart
  loses in-flight requests.
- A pull-broker design adds latency and one more failure surface between
  lm-eval and vLLM.
- The MVP does not include streaming, cancellation, persistent queues,
  always-on serving, Harbor/Evalchemy migration, or in-training-loop weight
  updates.

## Design

The service has three roles.

- **Eval job:** runs the evaluator and a local OpenAI-compatible proxy. It sees
  `RunningModel(endpoint=OpenAIEndpoint(...))`, not vLLM process details.
- **Broker actor:** owns one eval run's in-memory request table. It accepts
  opaque OpenAI request envelopes, leases pending work to workers, expires old
  leases, and accepts terminal results only from the current lease.
- **vLLM worker group:** runs one Iris job with `N` replicas. Each replica
  starts native vLLM, waits for `/v1/models`, records readiness, leases work
  from the broker, forwards to `http://127.0.0.1:<port>/v1`, and reports the
  vLLM response.

The proxy exposes only:

- `/v1/completions`
- `/v1/chat/completions`

The proxy validates the minimal HTTP shape: JSON object body, supported path,
safe request id, bounded request size, and no streaming. It forwards the body
opaquely to the broker, waits for a terminal broker result, and returns the
worker response. Duplicate submits with the same request id are idempotent only
when the endpoint and payload match exactly.

The worker also treats payloads opaquely. It chooses the local vLLM endpoint
from the envelope path and forwards the JSON body. It does not parse lm-eval,
Harbor, Evalchemy, or OpenAI semantics.

This design intentionally uses a pull broker instead of direct proxy
load-balancing. Replacement workers can pick up expired work without the proxy
tracking worker liveness. The tradeoff is one in-memory coordinator, which is
acceptable for first batch-eval serving.

Implementation should reuse `VllmEnvironment` for native vLLM startup,
readiness, and diagnostics, then wrap it in an Iris/Fray `ModelLauncher`. The
launcher returns a `RunningModel` whose endpoint base URL points at the local
proxy and whose model id matches vLLM's served model name. Engine kwargs remain
vLLM kwargs; they are not generic service fields.

## Testing

Normal CI should not require real Iris or real vLLM. It should cover:

- broker lifecycle: submit, lease, expiry, complete/fail, duplicate submit,
  stale lease result ignored, and first valid terminal result kept;
- proxy safety: request id validation before echoing headers, invalid JSON,
  oversized body, unsupported path, and `stream: true`;
- proxy -> broker -> worker -> deterministic OpenAI-compatible stub for both
  completions and chat completions;
- a tiny lm-eval path when optional eval dependencies are installed.

The manual Iris smoke should run stock lm-eval generation against a standard
vLLM-friendly Hugging Face model. The saved lm-eval artifact is the source of
truth for the actual task and few-shot settings. Follow-up manual runs should
exercise prompt-logprob scoring through `/v1/completions`.

## Open Questions

- Which standard Hugging Face model should be the named MVP smoke target?
- What minimum throughput makes the single-broker design acceptable for near
  term eval runs?
- Should manual pressure-test results upload directly to GCS, or is local
  output plus Iris logs enough for the first implementation PR?
- What vLLM changes are needed before prompt-logprob scoring is reliable on
  the target TPU serving stack?
