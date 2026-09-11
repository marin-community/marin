# RFC: Served LM Eval

## Goal

This RFC is a small step toward running more Marin evaluations against served
models, instead of requiring every evaluator to work with an in-process model
object. It is not the full inference-service plan. The goal here is to define
the handoff object between model-serving code and eval code.

That boundary should keep eval runners focused on eval frameworks, while
letting serving code deal with vLLM, Iris, Fray, Docker, TPUs, auth, and
cleanup. `lm_eval` is the first consumer, but the useful abstraction is the
generic `RunningModel`: a model that has already been launched or connected to,
and can now be evaluated.

Related context:

- [archived inference-service drafts](https://github.com/marin-community/marin/tree/archive/served-lm-eval-rfcs/.agents/projects/inference-service)
  are background only. They are not part of this review and should not be
  treated as the current spec.

## Proposal

Split served-model evals into two pieces.

Launchers own serving. A launcher can start or connect to vLLM, run through
Iris or Fray, choose native or Docker mode, set up auth, manage package
versions, allocate resources, collect diagnostics, and clean up. Once all of
that is ready, it returns a `RunningModel`.

The eval runner owns `lm_eval`. It receives a `RunningModel`, turns it into the
right `lm_eval` adapter name and model args, calls
`lm_eval.evaluator.simple_evaluate`, and writes results with
`EvaluationTracker`.

In production, the shape should look like this:

```python
deployment = ModelDeployment(
    model_name="llama-200m",
    model_path="gs://...",
    tokenizer="gs://...",
)

with launcher.launch(deployment) as running_model:
    run_lm_eval(
        running_model,
        LmEvalRun(tasks=["arc_easy"], output_path="..."),
    )
```

Concrete launchers, such as a future Iris/vLLM launcher, implement
`ModelLauncher`. Eval code should not instantiate vLLM, Iris, or Fray objects
directly.

Core contract:

```python
@dataclass(frozen=True)
class OpenAIEndpoint:
    base_url: str  # API root, e.g. http://127.0.0.1:8000/v1
    model: str
    api_key: str | None = None

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


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
    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        ...


class LmEvalAdapter(StrEnum):
    LOCAL_COMPLETIONS = "local-completions"
    LOCAL_CHAT_COMPLETIONS = "local-chat-completions"


@dataclass(frozen=True)
class LmEvalRun:
    tasks: Sequence[str]
    output_path: str
    adapter: LmEvalAdapter = LmEvalAdapter.LOCAL_COMPLETIONS
    apply_chat_template: bool = False
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = None
    extra_model_args: Mapping[str, str | int | float | bool] = field(default_factory=dict)


def run_lm_eval(model: RunningModel, run: LmEvalRun) -> None:
    ...
```

`OpenAIEndpoint.base_url` is the API root. The runner derives the endpoint that
`lm_eval` expects:

- `local-completions` gets `base_url={endpoint.base_url}/completions`.
- `local-chat-completions` gets
  `base_url={endpoint.base_url}/chat/completions`.

`OpenAIEndpoint.model` is the server-side served-model id that clients should
send in OpenAI-compatible requests. It is not meant to be a human display name.
Launchers should choose a stable id that the backing server accepts.

`RunningModel.tokenizer` is the tokenizer staging handoff. If the checkpoint
needs a local tokenizer path or a Hugging Face tokenizer id, the launcher is
responsible for resolving that before it yields `RunningModel`; eval runners
only consume the resulting string.

Default model args are `model`, `base_url`, `tokenizer_backend=huggingface`,
and `tokenized_requests=False`. `api_key` and `tokenizer` are included only
when present. `extra_model_args` is applied last so callers can override these
defaults when needed.

`LmEvalRun.output_path` is an output directory. With today's
`EvaluationTracker`, results land under
`<output_path>/<model_name_sanitized>/results_<date>.json`, with samples in
`samples_<task>_<date>.jsonl`.

## lm-eval Adapter Contract

The runner should match the OpenAI adapter shape that `lm_eval` actually uses.
For text scoring, `local-completions` wants the endpoint-specific URL, not the
`/v1` API root. It also needs tokenizer information for context length
calculation, even when the HTTP request itself uses string prompts.

The tested scoring path sends `max_tokens=1`, not `max_tokens=0`, and it does
not require `stop`. Prompt-logprob responses need echoed prompt tokens plus one
generated token, because `lm_eval` scores the `token_logprobs[ctxlen:-1]`
slice.

Observed scoring request:

```json
{
  "model": "gpt2",
  "prompt": "A B",
  "temperature": 0,
  "max_tokens": 1,
  "logprobs": 1,
  "seed": 1234,
  "echo": true
}
```

## vLLM Stance

Real vLLM should remain a manual validation step for this RFC. Validation so
far showed native TPU vLLM generation on Iris through both generation routes:

- `/v1/chat/completions`
- `/v1/completions`

The representative `lm_eval` prompt-logprob path did not pass. vLLM reached
readiness, then the first `/v1/completions` loglikelihood request returned HTTP
500:

```json
{"error":{"message":"14924","type":"Internal Server Error","param":null,"code":500}}
```

So this served lm-eval boundary should not claim that served vLLM supports
MCQ/loglikelihood evals. Generation compatibility is useful, but it is not the
same contract as
`lm_eval` scoring compatibility. This RFC leaves the existing Levanter-backed
path as the supported route for MCQ/loglikelihood evals.

## Tests

Use a focused test with real `lm_eval` and a deterministic fake
OpenAI-compatible server to catch adapter-contract drift. The test should run a
tiny `loglikelihood` task through `local-completions`, then check the request
payload and `EvaluationTracker` output layout.

Manual validation should separately run real vLLM generation smokes through
Iris, with pinned and logged native vLLM TPU packages.

## Non-Goals

- legacy evaluator call-site migration;
- Harbor, Evalchemy, raw perplexity, long-tail perplexity, FineWeb2, or
  perplexity-gap work;
- Levanter parity or Iris inference-service implementation;
- served vLLM MCQ/loglikelihood scoring.

## Follow-Up Work

The next steps should stay separate from this RFC so the first PR remains a
small interface brick.

- Levanter OpenAI HTTP parity: make Levanter's `/v1/completions` response
  satisfy the real `lm_eval local-completions` prompt-logprob contract,
  including echoed prompt-token logprobs.
- Iris served-model launcher: build an Iris-facing launcher that returns
  `RunningModel` and owns tokenizer staging, endpoint discovery, auth setup,
  and context-manager cleanup.
- vLLM scoring: track prompt-logprob support separately. Until that passes,
  served vLLM should be treated as generation-compatible but not as a
  MCQ/loglikelihood backend.
- Additional eval frameworks: add Harbor and Evalchemy adapters as
  framework-specific consumers of `RunningModel`, without moving their config
  concepts into the generic served-model boundary.
- Results: decide whether Marin should expose raw `EvaluationTracker` layout
  or add a normalized result artifact.
- Conformance tests: promote reusable OpenAI subset assertions once Levanter
  and vLLM contract work has stabilized.

## Pressure-Test Result

Follow-up probing exercised this boundary against Levanter's OpenAI request
schemas, a fake Iris launcher, and Harbor's hosted-vLLM config shape. It did
not find a reason to add fields beyond `base_url`, `model`, `api_key`, and
`tokenizer`. In particular, typed request headers and Harbor-style `model_info`
should stay out of `RunningModel` until a real consumer needs them.
