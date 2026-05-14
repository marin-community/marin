# Spec — iris_inference_service

This is the contract layer for the vLLM-only Iris inference service. It names
the files, public shapes, routing behavior, and test obligations reviewers are
approving.

## Files

| File | Status | Purpose |
|---|---|---|
| `lib/marin/src/marin/inference/iris_vllm.py` | new | Iris vLLM launcher, proxy, broker, worker actors |
| `tests/evals/test_iris_vllm_inference.py` | new | CI-safe broker/proxy/worker tests with a deterministic OpenAI stub |
| `scripts/iris/run_vllm_eval_pressure_test.py` | new, manual-only | Iris pressure-test runner |

No proto changes. No persistent schema. No new engine abstraction layer.

Reused contracts:

- `OpenAIEndpoint`, `RunningModel`, `ModelDeployment`, `ModelLauncher` from
  `marin.inference.types`
- `LmEvalRun`, `LmEvalAdapter`, `run_lm_eval` from `marin.evaluation.lm_eval`
- `VllmEnvironment` from `marin.inference.vllm_server`

## Scope

This service supports native vLLM only. It must not introduce
`EngineKind.LEVANTER`, a JAX OpenAI HTTP server, or a generic engine adapter.

The first manual pressure-test lane uses a standard vLLM-friendly Hugging Face
model and stock lm-eval generation. Scoring tasks that require prompt logprobs
remain follow-up compatibility checks until the TPU vLLM `/v1/completions`
contract is reliable.

## Constants

```python
MARIN_REQUEST_ID_HEADER = "X-Marin-Inference-Request-Id"
JSON_CONTENT_TYPE = "application/json"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MAX_REQUEST_BODY_BYTES: int = 8 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT: float = 3600.0
DEFAULT_LEASE_TIMEOUT: float = DEFAULT_REQUEST_TIMEOUT + 60.0
DEFAULT_WORKER_LEASE_WAIT_TIMEOUT: float = 1.0
DEFAULT_WORKER_READY_TIMEOUT: float = 900.0
DEFAULT_CLEANUP_TIMEOUT: float = 10.0
```

`lease_timeout` must be greater than `request_timeout`. The proxy may echo
`MARIN_REQUEST_ID_HEADER` only after validation or generation.

## Endpoint Kind

```python
class OpenAIEndpointKind(StrEnum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"

    @property
    def http_path(self) -> str: ...

    @property
    def api_path(self) -> str: ...

    @staticmethod
    def from_http_path(path: str) -> "OpenAIEndpointKind | None": ...
```

`http_path` is `/v1/completions` or `/v1/chat/completions`. `api_path` is the
path appended under a vLLM `/v1` API root.

## Broker Shapes

```python
@dataclass(frozen=True)
class OpenAIRequestEnvelope:
    request_id: str
    endpoint: OpenAIEndpointKind
    payload_json: str


@dataclass(frozen=True)
class OpenAIResponseEnvelope:
    status_code: int
    payload_json: str
    content_type: str = JSON_CONTENT_TYPE


@dataclass(frozen=True)
class InferenceLease:
    lease_id: str
    worker_id: str
    request: OpenAIRequestEnvelope
    expires_at: float


@dataclass(frozen=True)
class WorkerReadyState:
    worker_id: str
    model_id: str


class BrokerRequestStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool: ...


class BrokerWaitOutcome(StrEnum):
    READY = "ready"
    TIMEOUT = "timeout"
    UNKNOWN_REQUEST = "unknown_request"
    BROKER_STOPPED = "broker_stopped"


@dataclass(frozen=True)
class BrokerWaitResult:
    outcome: BrokerWaitOutcome
    response: OpenAIResponseEnvelope | None = None


@dataclass(frozen=True)
class LeaseResult:
    lease: InferenceLease | None
    stopped: bool = False
```

`BrokerWaitResult.response` is set only for `READY`. `LeaseResult.lease=None`
means no work was available before timeout unless `stopped=True`.

## Broker API

```python
class IrisInferenceBroker:
    """In-memory broker scoped to one eval run."""

    def __init__(
        self,
        lease_timeout: float = DEFAULT_LEASE_TIMEOUT,
        now: Callable[[], float] = time.monotonic,
    ) -> None: ...

    def submit(self, request: OpenAIRequestEnvelope) -> bool: ...

    def lease(self, worker_id: str, wait_timeout: float | None = None) -> LeaseResult: ...

    def complete(
        self,
        request_id: str,
        lease_id: str,
        response: OpenAIResponseEnvelope,
    ) -> bool: ...

    def fail(
        self,
        request_id: str,
        lease_id: str,
        response: OpenAIResponseEnvelope,
    ) -> bool: ...

    def poll(self, request_id: str) -> OpenAIResponseEnvelope | None: ...

    def wait(self, request_id: str, timeout: float | None = None) -> BrokerWaitResult: ...

    def status(self, request_id: str) -> BrokerRequestStatus | None: ...

    def record_worker_ready(self, state: WorkerReadyState) -> None: ...

    def wait_for_workers_ready(
        self,
        worker_count: int,
        timeout: float,
    ) -> tuple[WorkerReadyState, ...]: ...

    def stop(self) -> None: ...
```

Broker behavior:

- `submit` returns `True` for a new request and `False` for an identical
  duplicate. Reusing a request id with different endpoint or payload raises
  `ValueError`.
- `lease` expires overdue leases before choosing work. Expired work becomes
  pending again.
- `complete` and `fail` store a terminal response only for the current lease.
  Stale lease ids and already-terminal requests return `False`.
- `wait` distinguishes ready, timeout, unknown request, and stopped broker.
- `record_worker_ready` is idempotent for the same `(worker_id, model_id)` and
  raises on conflicting model ids.
- Mutations are linearizable under one lock. Blocked lease, wait, and readiness
  calls wake on submit, terminal result, readiness, lease expiry, and stop.

The broker has no persistence contract. Restarting it starts with an empty
request table.

## Worker API

```python
@dataclass(frozen=True)
class IrisVllmWorkerConfig:
    model: ModelDeployment
    host: str = "127.0.0.1"
    port: int | None = None
    vllm_timeout: int = 3600
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT
    extra_vllm_args: tuple[str, ...] = ()


class IrisVllmWorker:
    """Actor that starts vLLM and forwards broker leases."""

    def __init__(self, broker: ActorHandle, config: IrisVllmWorkerConfig) -> None: ...

    def run(self, max_requests: int | None = None) -> int: ...
```

Worker behavior:

- Use `VllmEnvironment` in native mode.
- Set `ModelConfig.name = deployment.model_name`,
  `ModelConfig.path = deployment.model_path`, and
  `ModelConfig.engine_kwargs = dict(deployment.engine_kwargs)`.
- Add `--served-model-name <deployment.model_name>` and reject
  `extra_vllm_args` that include another `--served-model-name`.
- Wait for `/v1/models`, record readiness, lease work, forward JSON bodies
  opaquely, and return the vLLM response body/status to the broker.
- Map network failures while forwarding to a `502` response via `broker.fail`.
  Let vLLM startup failures and non-network worker exceptions propagate.

Endpoint mapping:

| Envelope endpoint | Worker target |
|---|---|
| `COMPLETIONS` | `{env.server_url}/completions` |
| `CHAT_COMPLETIONS` | `{env.server_url}/chat/completions` |

## Proxy API

```python
@dataclass(frozen=True)
class RunningIrisInferenceProxy:
    base_url: str
    request_id_header: str = MARIN_REQUEST_ID_HEADER


def serve_iris_inference_proxy(
    broker: ActorHandle,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> AbstractContextManager[RunningIrisInferenceProxy]: ...
```

The yielded `base_url` is an OpenAI API root ending in `/v1`.

Supported paths:

| Method | Path |
|---|---|
| `POST` | `/v1/completions` |
| `POST` | `/v1/chat/completions` |

Proxy errors:

| Status | Condition |
|---|---|
| `400` | unsafe request id, invalid JSON, non-object JSON, or `stream: true` |
| `404` | unsupported path |
| `409` | request id reused with different endpoint or payload |
| `413` | body exceeds `MAX_REQUEST_BODY_BYTES` |
| `415` | explicit non-JSON content type |
| `503` | broker stopped or reports unknown request after submit |
| `504` | broker wait timed out |

The proxy should call the long-running actor operation for `broker.wait`. Its
HTTP serving implementation must either be single-threaded, copy Iris/Fray
context into handler threads, or capture a broker handle that does not depend
on thread-local context lookups.

## Launcher API

```python
@dataclass(frozen=True)
class IrisVllmLauncherConfig:
    worker_count: int
    worker_resources: ResourceConfig
    broker_resources: ResourceConfig
    lease_timeout: float = DEFAULT_LEASE_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    worker_lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT
    worker_ready_timeout: float = DEFAULT_WORKER_READY_TIMEOUT
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 0
    service_name: str | None = None
    extra_vllm_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class IrisVllmLauncher(ModelLauncher):
    client: Client
    config: IrisVllmLauncherConfig

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]: ...
```

Launcher validation:

- `worker_count > 0`
- all timeout values are positive
- `lease_timeout > request_timeout`
- `extra_vllm_args` does not include `--served-model-name`
- `worker_resources` requests the accelerator topology for each worker replica

Startup:

1. Create the broker.
2. Start worker replicas.
3. Wait for `worker_count` readiness records within `worker_ready_timeout`.
4. Require every readiness model id to equal `deployment.model_name`.
5. Start the local proxy.
6. Yield `RunningModel(OpenAIEndpoint(base_url=proxy.base_url, model=deployment.model_name), tokenizer=deployment.tokenizer)`.

Context exit:

1. Stop the proxy.
2. Call `broker.stop()`.
3. Wait up to `DEFAULT_CLEANUP_TIMEOUT` for worker operations to end.
4. Surface partial-start and cleanup failures with broker state, worker ids,
   and vLLM diagnostics when available.

## Manual Pressure-Test Runner

`scripts/iris/run_vllm_eval_pressure_test.py` should accept:

```text
--model-name <served name>
--model-path <gs://... or hf id>
--tokenizer <hf id or local path>
--output-path <local or gs:// path>
--tpu-type <iris tpu type>
--worker-count <int>
--task standard_humaneval_smoke|mmlu_sl_verb_5shot|humaneval_5shot
--limit <int>
--dry-run
```

Runner behavior:

- Default to `standard_humaneval_smoke` with
  `LmEvalAdapter.LOCAL_CHAT_COMPLETIONS` and `apply_chat_template=True`.
- Treat `mmlu_sl_verb_5shot` as a follow-up prompt-logprob check using
  `LmEvalAdapter.LOCAL_COMPLETIONS`.
- Treat `humaneval_5shot` as valid only if saved lm-eval artifacts record the
  requested few-shot setting.
- Record Iris job ids, served model id from `/v1/models`, output directory,
  request counts by endpoint, throughput if available, and failure diagnostics.
- Upload lm-eval outputs when `--output-path` is `gs://`; otherwise write
  locally.

This script is manual-only and must not run in normal CI.

## Tests

CI should cover:

- broker submit/lease/complete/wait lifecycle;
- expired lease requeue;
- stale lease terminal result ignored;
- duplicate terminal result keeps the first valid response;
- conflicting duplicate submit returns `409` through the proxy;
- unsafe request id returns `400` and is not echoed;
- invalid JSON, non-object JSON, `stream: true`, non-JSON content type, and
  oversized body errors;
- proxy/worker routing for completions and chat completions against a
  deterministic OpenAI-compatible stub;
- tiny real lm-eval path gated by optional dependencies.

Manual testing should run the standard-model pressure test on Iris before the
first implementation PR is treated as production-shaped. Prompt-logprob scoring
is a follow-up compatibility gate.

## Out of Scope

- Levanter/JAX behind OpenAI HTTP.
- Engine polymorphism.
- Persistent broker queue or broker restart recovery.
- OpenAI server-sent streaming.
- Cancellation of in-flight vLLM requests.
- Multi-tenant always-on inference service.
- Harbor/Evalchemy migration in the first implementation PR.
- Parameter syncing for RL or in-training-loop evals.
