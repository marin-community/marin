# RFC 1: Served-Model LM Eval

Status: draft.

This RFC supersedes [`01-rfc-evals-over-openai-http-legacy.md`](./01-rfc-evals-over-openai-http-legacy.md) and the closed draft PR [#4841](https://github.com/marin-community/marin/pull/4841). The old RFC is kept as history for the broader eval migration attempt.

This RFC does not supersede:
- [`02-rfc-levanter-http-parity.md`](./02-rfc-levanter-http-parity.md), which tracks Levanter/JAX serving parity against this contract.
- [`03-rfc-iris-inference-service.md`](./03-rfc-iris-inference-service.md), which is the intended production serving path for this contract.

## Raison d'être

Marin eval code should not load or serve models. It should evaluate a served model.

The first concrete target is `lm_evaluation_harness`. The eval runner should consume a `RunningModel` that exposes the Marin eval OpenAI-compatible HTTP subset. The model may be served by direct vLLM, Levanter/JAX HTTP, or the future Iris inference service. That serving choice should not change lm-eval code.

The implementation should validate the whole architecture with one vertical slice:
- define the served-model boundary
- define the lm-eval runner that consumes that boundary
- prove it with direct vLLM first
- leave the API shape compatible with the RFC 3 Iris launcher

## What changed since the legacy RFC

The old RFC tried to move `lm_evaluation_harness`, Harbor, and Evalchemy in one broad migration. That is too much for the first replacement PR.

Several assumptions are stale:
- Ray migration is no longer the main forcing function.
- The eval surface has grown: raw PPL, long-tail PPL, FineWeb2, perplexity-gap, and capability slices should not all be forced through a served-model abstraction.
- PR [#4841](https://github.com/marin-community/marin/pull/4841) is stale and closed; it should be used only as reference.
- Levanter's OpenAI-compatible server exists, but it does not yet satisfy lm-eval prompt-scoring parity. That remains RFC 2 work.
- RFC 3 is now the future golden path for production serving, so the eval contract must be launcher-neutral from the start.

## Goals

- Decouple lm-eval from model loading and serving.
- Introduce a narrow served-model contract that lm-eval can consume.
- Define the Marin eval OpenAI-compatible HTTP subset required by lm-eval.
- Make direct vLLM the first concrete launcher.
- Keep the public boundary compatible with a future `IrisInferenceLauncher` implementing RFC 3.
- Provide a lean test stub plus reusable conformance tests for the HTTP subset.
- Leave existing eval paths in place until the new path has proved itself.

## Non-goals

- Migrating Harbor in the first PR.
- Migrating Evalchemy in the first PR.
- Deleting `EvaluationConfig`, `Evaluator`, `run.py`, or the old dispatcher in the first PR.
- Deleting `levanter_lm_evaluation_harness` before RFC 2 is implemented.
- Implementing the RFC 3 Iris proxy/broker/worker system in the first PR.
- Forcing raw PPL, perplexity-gap, validation-slice, or direct Levanter analysis jobs through served-model eval.
- Implementing full OpenAI API parity.

## Design principle

Keep three concerns separate:

1. **Eval workload**: which benchmark/task to run and where to write results.
2. **Model deployment**: which model artifact should be served and with what engine settings.
3. **Service deployment**: where and how the model server runs.

The lm-eval runner should depend only on the first concern plus a `RunningModel`. It should not know whether the model came from vLLM, Levanter, or Iris.

## Core types

Names are provisional, but the boundary should look like this:

```python
@dataclass(frozen=True)
class OpenAIEndpoint:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True)
class RunningModel:
    endpoint: OpenAIEndpoint
    tokenizer: str | None = None


@dataclass(frozen=True)
class ModelDeployment:
    model_name: str
    model_path: str
    tokenizer: str | None = None
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)


class ModelLauncher(Protocol):
    def launch(self, deployment: ModelDeployment) -> ContextManager[RunningModel]:
        ...
```

`OpenAIEndpoint.base_url` should be the OpenAI-compatible API root, such as `http://host:port/v1`, not the specific `/completions` endpoint. The lm-eval runner derives the concrete endpoint path from `apply_chat_template`.

`RunningModel` is intentionally small. It is a handle to an already running model. It should not expose backend-specific details such as Docker containers, TPU topology, or Iris job ids unless a caller explicitly needs debug metadata outside the eval runner.

`ModelLauncher` owns lifecycle. Direct vLLM, Levanter HTTP, and Iris should be separate launcher implementations.

## RFC 3 compatibility

RFC 3 is the intended production serving path. This RFC must not bake direct vLLM assumptions into the eval runner.

The first PR may implement only:

```python
class VllmLauncher:
    def launch(self, deployment: ModelDeployment) -> ContextManager[RunningModel]:
        ...
```

But the boundary must also allow:

```python
class IrisInferenceLauncher:
    def launch(self, deployment: ModelDeployment) -> ContextManager[RunningModel]:
        ...
```

Under RFC 3, `IrisInferenceLauncher` would start or attach to the EvalJob proxy/broker/worker deployment and return a `RunningModel` pointing at the local proxy. The lm-eval runner should not change.

This means the first PR validates the architecture if:
- direct vLLM can provide a `RunningModel`
- lm-eval can run against that `RunningModel`
- the runner accepts only endpoint/tokenizer/run config, not vLLM-specific lifecycle state

It does not need to validate Iris preemption, broker replay, or worker scheduling.

## Marin eval OpenAI-compatible subset

This is not the OpenAI API. It is the subset Marin evals need.

### `/v1/completions`

Required for lm-eval scoring tasks.

Request fields:
- `model`
- `prompt`
- `max_tokens`
- `temperature`
- `stop`
- `seed`
- `echo`
- `logprobs`

Response fields:
- `choices[*].text`
- `choices[*].index`
- `choices[*].logprobs.tokens`
- `choices[*].logprobs.token_logprobs`
- `choices[*].logprobs.top_logprobs`
- `choices[*].logprobs.text_offset`

Behavioral requirements:
- `echo=True` with `logprobs` must support prompt / continuation scoring.
- `token_logprobs` must align with returned tokens in the way `lm_eval`'s local-completions client expects.
- `top_logprobs` must be present for scored tokens and include enough information to determine whether the chosen token was greedy.
- Batched prompts should either work or fail with a clear unsupported-capability error. Silent partial support is not acceptable.

The exact edge semantics should be pinned by conformance tests, not prose alone.

### `/v1/chat/completions`

Required for chat-template generation tasks.

Request fields:
- `model`
- `messages`
- `max_tokens`
- `temperature`
- `stop`
- `seed`

Response fields:
- `choices[*].message.content`
- `choices[*].index`

Prompt-side scoring on chat completions is not required for the first slice. Scoring tasks should use `/v1/completions`.

### Tokenizer handling

The MVP does not require tokenizer HTTP endpoints.

The lm-eval runner may pass an explicit Hugging Face tokenizer name/path to lm-eval. This is enough for the current Marin flow and avoids requiring Levanter, vLLM, and Iris to expose a tokenizer side API before we need it.

Remote-tokenizer endpoints can be reconsidered later if they remove real complexity.

## LM-eval runner

The new path should be a small function, not a new evaluator framework:

```python
@dataclass(frozen=True)
class LmEvalRun:
    tasks: Sequence[str]
    output_path: str
    apply_chat_template: bool = False
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = None
    extra_model_args: Mapping[str, object] = field(default_factory=dict)


def run_lm_eval(model: RunningModel, run: LmEvalRun) -> None:
    ...
```

`run_lm_eval` should build the correct lm-eval model name and `model_args`:
- `local-completions` when `apply_chat_template=False`
- `local-chat-completions` when `apply_chat_template=True`
- `base_url` by appending `/completions` or `/chat/completions` to `model.endpoint.base_url`
- `model` from `model.endpoint.model`
- `tokenizer` from `model.tokenizer` when present

It should not:
- launch vLLM
- stage model artifacts
- know about Iris
- dispatch by evaluator string
- own backend-specific cleanup

## First PR scope

The first implementation PR should validate one vertical slice.

In scope:
- Add the served-model types.
- Add a direct vLLM launcher that wraps the existing vLLM lifecycle code.
- Add the new lm-eval runner.
- Add a lean OpenAI-compatible stub server for tests.
- Add conformance tests for the Marin eval OpenAI subset.
- Add one real, gated smoke path that launches vLLM and runs a tiny lm-eval task through the new runner.

Out of scope:
- Harbor.
- Evalchemy.
- Levanter parity.
- Iris proxy/broker/worker implementation.
- Migration of every existing eval helper.
- Deletion of legacy eval framework code.

## Test strategy

### Stub server

The stub server is a test fixture. It should implement only the Marin eval subset and return deterministic responses.

It validates:
- URL construction
- request shape
- response parsing
- tokenizer/model arg plumbing
- error handling at the HTTP boundary

It does not validate:
- model quality
- vLLM correctness
- Levanter correctness
- tokenizer parity
- benchmark score equivalence

### Conformance tests

The same HTTP conformance tests should be reusable against:
- the stub server
- direct vLLM
- Levanter/JAX HTTP once RFC 2 lands
- the RFC 3 Iris proxy once it exists

This gives Levanter and Iris a concrete target without coupling them to the lm-eval runner internals.

### Real smoke

The first real smoke should be gated/manual or TPU-CI only:
- launch a tiny model with direct vLLM
- construct a `RunningModel`
- run one small lm-eval task through `run_lm_eval`
- write results to a temporary output path

This proves the architecture end to end without making the first PR depend on Iris.

## Levanter status

Levanter already exposes OpenAI-compatible completion and chat-completion endpoints, including generated-token logprobs.

It is not yet sufficient for normal lm-eval scoring because `/v1/completions` does not implement the echoed prompt-logprob response shape needed by `lm_eval local-completions`. RFC 2 should make Levanter pass the same conformance tests described here.

Expected RFC 2 work:
- add a prompt-scoring path for `/v1/completions`
- compute per-token next-token logprobs with a full forward pass
- return `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset` in lm-eval-compatible shape
- validate with the shared conformance tests

## Migration plan

1. Land this RFC.
2. Implement the direct vLLM + lm-eval vertical slice.
3. Add Levanter/JAX HTTP parity against the conformance tests.
4. Implement the RFC 3 Iris launcher/proxy path.
5. Migrate selected lm-eval call sites to the new runner.
6. Migrate Harbor and Evalchemy only after the lm-eval boundary has held up.
7. Delete legacy eval dispatcher/framework code once call sites no longer need it.

## Open questions

- Should `RunningModel.tokenizer` be required for `local-completions`, or should the runner fail only when lm-eval requires it?
- Should `OpenAIEndpoint.api_key` be part of the core type now, or added when an external endpoint needs it?
- Where should conformance tests live so Levanter and Marin can both use them without creating an awkward dependency direction?
- Should `ModelDeployment.model_path` be `str`, or should we introduce a typed artifact reference before this lands?
