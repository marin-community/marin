# RFC 1: Served-Model LM Eval

Status: draft.

Supersedes [`01-rfc-evals-over-openai-http-legacy.md`](./01-rfc-evals-over-openai-http-legacy.md) and closed draft PR [#4841](https://github.com/marin-community/marin/pull/4841).

Does not supersede:
- [`02-rfc-levanter-http-parity.md`](./02-rfc-levanter-http-parity.md): Levanter/JAX support for this contract.
- [`03-rfc-iris-inference-service.md`](./03-rfc-iris-inference-service.md): the intended production serving path.

## Decision

Marin's lm-eval code should evaluate a served model. It should not load checkpoints, start vLLM, manage Iris jobs, or know which backend is serving the model.

The first implementation should prove this boundary with one vertical slice:

```text
ModelDeployment -> ModelLauncher -> RunningModel -> run_lm_eval(...)
```

For the first PR, `ModelLauncher` can be direct vLLM. Later, RFC 3 should provide an Iris launcher that returns the same `RunningModel`. `run_lm_eval` should not change.

## Why Now

The old RFC tried to migrate lm-eval, Harbor, and Evalchemy together. That made the first PR too broad.

Main has also moved:
- Ray migration is no longer the main driver.
- The eval surface now includes raw PPL, long-tail PPL, FineWeb2, perplexity-gap, and validation slices. Those should not all be forced through a served-model abstraction.
- Levanter has an OpenAI-compatible server, but it still lacks the lm-eval prompt-scoring behavior tracked in RFC 2.
- RFC 3 is now the future golden path for production serving, so the eval contract needs to be launcher-neutral.

## Scope

In the first PR:
- Add the served-model types.
- Add a direct vLLM launcher.
- Add a small lm-eval runner that consumes `RunningModel`.
- Add a lean HTTP stub and reusable conformance tests for the Marin eval OpenAI subset.
- Add one gated real smoke that launches vLLM and runs a tiny lm-eval task.

Not in the first PR:
- Harbor.
- Evalchemy.
- Levanter parity.
- Iris proxy/broker/worker implementation.
- Deleting `EvaluationConfig`, `Evaluator`, `run.py`, or the old dispatcher.
- Deleting `levanter_lm_evaluation_harness`.
- Forcing raw PPL or perplexity-gap evals through this path.

## Core Boundary

Names are provisional, but the shape should stay small:

```python
@dataclass(frozen=True)
class OpenAIEndpoint:
    base_url: str  # OpenAI-compatible API root, e.g. http://host:port/v1
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

The important rule: `ModelLauncher` owns serving lifecycle. `run_lm_eval` only receives a `RunningModel`.

Direct vLLM and future Iris serving should both fit:

```python
with launcher.launch(deployment) as model:
    run_lm_eval(model, run)
```

## LM-Eval Runner

The runner should be a function, not a new eval framework:

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

It should choose:
- `local-completions` when `apply_chat_template=False`
- `local-chat-completions` when `apply_chat_template=True`

It should build lm-eval `model_args` from:
- `model.endpoint.base_url`
- `model.endpoint.model`
- `model.tokenizer`, when present
- `run.extra_model_args`

It should not launch servers, stage model artifacts, dispatch by evaluator string, or clean up backend resources.

## HTTP Contract

This is not the OpenAI API. It is the Marin eval subset needed by lm-eval.

`/v1/completions` is required for scoring tasks. It must support:
- request fields: `model`, `prompt`, `max_tokens`, `temperature`, `stop`, `seed`, `echo`, `logprobs`
- response fields: `choices[*].text`, `choices[*].index`, `choices[*].logprobs.tokens`, `token_logprobs`, `top_logprobs`, `text_offset`
- `echo=True` plus `logprobs` for prompt / continuation scoring

`/v1/chat/completions` is required for chat-template generation tasks. It must support:
- request fields: `model`, `messages`, `max_tokens`, `temperature`, `stop`, `seed`
- response fields: `choices[*].message.content`, `choices[*].index`

Tokenizer HTTP endpoints are not part of the MVP. The runner may pass an explicit Hugging Face tokenizer name/path to lm-eval.

Exact edge semantics should live in conformance tests rather than in prose.

## RFC 3 Compatibility

RFC 3 is the intended production path. This RFC should not assume direct vLLM is the final operational shape.

The first PR can implement:

```python
VllmLauncher.launch(...) -> RunningModel
```

RFC 3 should later implement:

```python
IrisInferenceLauncher.launch(...) -> RunningModel
```

Under RFC 3, the returned `RunningModel` should point at the EvalJob's local proxy. The lm-eval runner should not know about the broker, worker pool, preemption, or replay.

## Testing

Use three test layers:

1. **Stub server tests**
   - Tiny deterministic HTTP server.
   - Validates request shape, URL construction, model args, and response parsing.
   - Does not validate model quality or backend correctness.

2. **Conformance tests**
   - Reusable tests for the Marin eval OpenAI subset.
   - Should eventually run against stub, vLLM, Levanter, and the RFC 3 Iris proxy.

3. **Real smoke**
   - Gated/manual or TPU-CI only.
   - Launch tiny vLLM.
   - Run one small lm-eval task through `run_lm_eval`.

## Levanter Status

Levanter already has `/v1/completions` and `/v1/chat/completions`, including generated-token logprobs.

It is not yet enough for normal lm-eval scoring. `/v1/completions` accepts `echo` and `logprobs`, but it does not implement the echoed prompt-logprob shape that `lm_eval local-completions` needs.

RFC 2 should make Levanter pass the same conformance tests by adding prompt scoring over `/v1/completions`.

## Plan

1. Land this RFC.
2. Implement direct vLLM + lm-eval vertical slice.
3. Implement Levanter HTTP parity against the conformance tests.
4. Implement the RFC 3 Iris launcher/proxy path.
5. Migrate selected lm-eval call sites.
6. Migrate Harbor and Evalchemy after the lm-eval boundary holds up.
7. Delete legacy eval framework code after call sites no longer need it.
