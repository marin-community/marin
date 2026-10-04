# TaskCompendium

## What problem does it solve?

Training and evaluation tasks arrive with different prompt formats, answer rules, tools, and graders. TaskCompendium separates the problem a model must solve from the way a framework runs and grades it. A caller can choose among compatible presentations of a task while keeping its reference answer private. Additional Harbor environment configurations can use the same task definition.

TaskCompendium exports direct-chat Harbor tasks for text, number, and native-action results. The separate [rolloutengine library](../rolloutengine/README.md) executes tasks through Shellbox. See the [task rollout reference](../../docs/references/task-rollouts.md) for the Parquet format, engine interfaces, and SkyRL integration.

## What does it contain?

- **Task specs** describe the source problem, the required capabilities, the kind of result, and how to verify it.
- **Submission conventions** describe how to ask for and extract a result, such as a plain answer, a JSON object, or a final function call.
- **Harbor environment configurations** describe the capabilities and tools exposed during execution. The only configuration in the current implementation is direct chat, which records submission calls but does not execute tools.
- **Lowering tools** find compatible convention and environment configuration pairs, select a pair, and export a runnable Harbor task package.
- **A Harbor adapter** runs the exported task against an OpenAI-compatible chat endpoint and records a grading result. Harbor acts as the harness: it orchestrates the model and environment after lowering.

```mermaid
flowchart LR
    S[TaskSpec] --> C[Find compatible lowerings]
    V[Submission conventions] --> C
    E[Harbor environment configurations] --> C
    C --> P[Select a lowering]
    P --> H[Export Harbor task package]
    H --> T[Harbor trial]
    L[Launch: agent and model] --> T
    T --> G[Private verifier and result]
```

## What is a task spec?

`TaskSpec` is the private definition of one source task. An importer or author creates it before choosing a submission convention or a framework launch.

| Field | Meaning |
| --- | --- |
| `id` | Stable identity for this task. |
| `context` | The ordered model-visible conversation: text messages, historical assistant function calls, and tool results. |
| `environment_requirements` | Required capabilities, pinned initial workspace, and named tool-provider contracts. |
| `final_tools` | An ordered list of functions that terminate a chat. They are not backed by a tool provider. |
| `answer_type` | The semantic result: `text`, `number`, `file`, `state`, `workspace_state`, or `native_action`. |
| `source` | Upstream dataset, revision, row, and importer revision retained as audit provenance. |
| `verifier` | Private grading rule and configuration. See [What is a verifier?](#what-is-a-verifier) |
| `environment` | Executable environment, public files, resource limits, and optional task session. |
| `metadata` | Application metadata, including an optional teacher route. |
| `resources` | Inline files grouped under `all`, `worker`, `oracle`, and `verifier` visibility. |
| `tags` | Ordered descriptive strings. |
| `schema_version` | Version of the serialized spec, checked when the record is loaded. |

`context.events` is the model-visible conversation prefix. A text event retains its role and content. Historical assistant calls and tool results retain their call IDs and order; the adapter sends them as OpenAI-compatible chat messages without executing them again. `answer_type` does not prescribe a wrapper such as JSON.

`ConversationInput` is not a lossless Responses API transcript. It excludes provider reasoning items. The NeMo importer omits an unencrypted historical reasoning summary only when a visible assistant message or function call follows it. It rejects encrypted reasoning and reasoning left at the decision point. Exact provider continuation from reasoning state is outside this contract.

For example, a task asking “What is 7 + 5?” can have `answer_type=number` and a private expected answer of `12`. The plain, JSON, and `answer_call` conventions carry that answer as `12`, `{"answer":"12"}`, and a final `submit_answer({"answer":"12"})` call, respectively. Each extracts a string for the same numeric verifier, which accepts both `12` and `12.0`. A task asking for a function call has `answer_type=native_action`; its context retains the conversation and its tools describe the source functions. The verifier and expected answer are never added to the model-visible input. Importers must make source output instructions neutral to the supported conventions, or reject rows they cannot safely rewrite. A raw-output requirement in a source message would conflict with a JSON or function-call convention; `answer_type=text` alone cannot detect that conflict in prose.

## What can a task represent?

### Text answers

A text task uses `answer_type=text`. Plain text, a JSON object with an `answer` string, and `submit_answer(answer: string)` can carry its answer. The `exact` verifier compares normalized text; `mcq` grades a single option letter. The same verifier grades the extracted answer across these conventions.

### Numeric answers

A numeric task uses `answer_type=number` and can use the same submission conventions as text. The `numeric` verifier parses the extracted string as a number and applies the explicitly configured absolute and relative tolerances. For example, both `12` and `12.0` can satisfy an expected value of `12.0`.

### Final function calls

A task whose result is a function call uses `answer_type=native_action`. Its `final_tools` field declares the available functions. `FinalAction` defines whether a call is required and the maximum call count. The final-action submission convention captures the assistant's calls, and `predicted_action` compares their function names and decoded argument objects with the private expected calls. Direct chat stops after recording the response; it does not execute the calls.

With the `answer_call` convention, the chat agent adds `submit_answer(answer: string)` alongside the task’s final tools and records the assistant's final response. It never invokes the function. The convention extracts the call's `answer` argument and passes it to the task's ordinary verifier. A non-call response or a call to another function receives `extraction_error` with no reward. The same final-action decoder handles native-action tasks; their verifier compares the recorded call's function name and argument dictionary with the private expected call.

### Files and state

`answer_type=file` names a file result. `answer_type=state` names the resulting environment state. The Shellbox engine executes these tasks with the environment and private shell verifier in the task spec. Direct-chat Harbor export does not accept executable environments. `environment_requirements` declares capabilities and tool-provider contracts. `environment` supplies the executable resources.

`resources` keeps inline files in `all`, `worker`, `oracle`, and `verifier` groups. Worker files are model-visible. Oracle and verifier files are private. Each `TaskResource` has a relative destination, base64 file content, and optional mode and modification time. Runtimes must reject unsafe paths and collisions. The Shellbox rollout engine does not implement this legacy mount contract; use `environment.files` and private verifier files for executable tasks.

## What can we import?

### TaskTrove MCQA

The TaskTrove MCQA importer reads archives from a cleaned release. See the [published TaskTrove Clean dataset](https://huggingface.co/datasets/open-athena/task-trove). Its caller passes the archive bytes, upstream subset, archive path, and release provenance to `read_archive`. The reader checks the subset and path against the archive manifest; the release URI and revision are caller-supplied provenance. The importer checks the source answer-line template before replacing it with a one-letter instruction. Its text answer works with plain and JSON submission conventions. The private `mcq` verifier stores the expected letter and option count. Any author can use that verifier; it currently calls the shared `verifyit` MCQ scorer after extracting the submission. This importer supports only MCQ mode. Executable TaskTrove modes still need private resources and an isolated verifier runtime.

### NeMo predicted function calls

`taskcompendium.importers.nemo_predicted_action.import_row` accepts a NeMo predicted-function-call row and a caller-pinned digest of that row. `canonical_sha256(row)` hashes its UTF-8 JSON with sorted keys and compact separators; record the digest with the source revision before importing. The importer returns `(specification, convention)`, with `answer_type=native_action` and a `FinalAction` convention. A hand-authored task can select the same convention with `FinalAction(id="final-call")`. The context carries the source conversation; `final_tools` carries advertised functions. `FinalAction.require_call` and `FinalAction.max_calls` carry the source call constraints. The convention describes how Harbor captures the final action and can be reused across tasks. The expected function calls remain in the private `predicted_action` verifier. There is one stored conversation, with no second flattened prompt to keep in sync.

For a chat launch, the Harbor adapter sends the source turns and function definitions to the model, records its final function call, and stops without dispatching the call. The verifier compares function names and JSON arguments. The importer rejects rows whose expected action is an assistant text message because the source comparator gives any message full credit; it also rejects request settings it cannot carry. The pinned fixture records the NeMo Gym repository revision and blob SHA in `tests/fixtures/nemo/predicted-action.provenance.json`. Numeric tolerance is used only when explicitly set in the private verifier.

## What is a verifier?

Each spec selects a private verifier and stores its configuration in `VerifierSpec`. TaskCompendium uses the submission convention to extract a candidate answer; the selected verifier grades it. `answer_type` controls which submission conventions can carry the result; the verifier determines how to score it.

`VerifierSpec.kind` is an open nonempty string. Its private typed `environment_requirements` defaults to empty and declares the capabilities, image, and workspace needed by the verifier. The harness owns these requirements. `parameters_json` is an opaque private JSON object owned by the shared verifier library (`verifyit`); the harness must not extract structural environment fields such as an image from it.

`verifier` grades one acquired answer. Comparative scoring across several attempts, cohort membership, and grading phase belong to the trainer. The ordinary per-attempt grader can score an already acquired answer regardless of worker workspace requirements.

Schema loading accepts descriptors without a grader implementation. Shared verifier validation, export, and launch validate executable configuration separately; an unimplemented kind raises `NotImplementedError`. Future kinds such as `script`, `test_suite`, or `llm_judge` can carry private entrypoints, paths referencing `TaskSpec.resources`, or rubrics in `parameters_json`. These graders have no implementation in this package. Direct chat rejects nonempty verifier environment requirements before export or launch, including for its implemented pure graders.

Standard verifier contracts and pure candidate scoring live in `verifyit`. Conversion pipelines select a shared spec and store its parameters in the task's private `verifier` slot. TaskCompendium extracts submission evidence and translates shared rewards into Harbor outcomes; it has no verifier registry or separate standard verifier schema.

The implemented canonical kinds are `exact` for normalized text, `numeric` for numbers with explicit absolute and relative tolerances, `mcq` for a single option letter, and `predicted_action` for final function calls. `structured_exact` is a schema descriptor without an implementation in this package. The expected answer and grading settings stay out of the model-visible instruction.

## What is a lowering?

A lowering is one runnable presentation of a spec for a target framework. It combines a compatible submission convention with a Harbor environment configuration, then writes the target's task files. The spec says *what* result is needed; the convention says *how* the model delivers it; the environment configuration says *which capabilities* the environment provides. Agent and model selection happens when the task is launched.

`convention.supports(spec.answer_type)` checks the result kind. `compatible_lowerings` uses that check and the environment requirements; it does not read convention IDs from the spec. The direct-chat environment configuration accepts only tasks with no worker or verifier environment capabilities, images, workspace initialization, tool providers, or resources; a task requiring `shell` has no candidate in the current implementation. File, workspace-state, arbitrary-state, and unimplemented-verifier tasks also have no direct-chat candidate. Export and launch raise `NotImplementedError` for these requirements before producing artifacts or requesting a model response. Chat request construction also rejects unsupported environment and resource requirements. `select_lowerings` can keep all candidates, take the first, or sample one with an explicit RNG key. The order of the caller-supplied convention and environment configuration sequences determines the first candidate and the sample order. A training caller should record those ordered inputs, the selection policy and key, and the TaskCompendium code revision.

Submission conventions preserve the task’s advertised functions. Plain-text and JSON submissions keep those functions; the `answer_call` convention adds `submit_answer` and requires one call with an answer string. An existing function named `submit_answer` conflicts with that convention and is rejected. `FinalAction(require_call=True)` requests a call, and `FinalAction(max_calls=1)` disables parallel calls. Before scoring, the grading boundary rejects missing required calls or calls exceeding `max_calls` as `extraction_error` with no reward. Direct chat captures the final assistant turn without executing advertised functions.

An author can require a particular execution environment without changing the semantic `TaskSpec`. Pass `required_environment="shellsim"` to `select_lowerings`; it keeps only ShellSim candidates and raises if none are compatible. A `shell` capability requests an operation, while ShellSim names a concrete execution choice. The current implementation offers only direct chat, so a ShellSim request fails rather than falling back to chat. The selected environment configuration is recorded in the exported Harbor package.

```python
from pathlib import Path

from taskcompendium.lowering import (
    HarborEnvironmentConfig,
    SelectionPolicy,
    compatible_lowerings,
    lower_to_harbor,
    select_lowerings,
)
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import AnswerFormat, SubmissionConvention

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    context=ConversationInput(events=(TextMessage(role="user", content="What is 7 + 5?"),)),
    environment_requirements=EnvironmentRequirements(),
    answer_type=AnswerType.NUMBER,
    verifier=numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0),
    source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
)
conventions = (
    SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    SubmissionConvention(id="json", answer_format=AnswerFormat.JSON),
    SubmissionConvention(id="answer-call", answer_format=AnswerFormat.ANSWER_CALL),
)
candidates = compatible_lowerings(spec, conventions, (HarborEnvironmentConfig(),))
chosen = select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=1234)[0]
lower_to_harbor(spec, chosen.convention, chosen.environment_config, Path("/tmp/arithmetic-task"))
```

## Dataset conversion

`TaskSpec.model_dump_json()` defines the serialized task contract.
`TaskSpec.model_validate_json()` reads that contract. Applications own dataset
conversion, file formats, and storage. JSON decoding preserves valid unsupported
requirements; export and launch validate runtime support separately.

## How does Harbor run it?

`lower_to_harbor` writes `instruction.md` and `task.toml` for Harbor, plus `specification.json`, `submission_convention.json`, and `environment_config.json` for the launcher and custom verifier. The package also has an empty `environment/` directory. A chat launch sends the structured conversation from the spec, then adds the convention's final answer instruction when needed. The agent has no tool to read the package files. Harbor's custom verifier can read the spec and private reference answer. The convention file tells it how to extract the submitted answer.

`run_trial` takes the exported directory, its environment configuration, and a chat launch. The Harbor harness selects and runs the agent and environment; those choices are absent from `TaskSpec`. Provide the endpoint's base URL and, if needed, the name of an environment variable containing the API key. The agent resolves that variable in its process; the trial configuration retains only its name.

```python
import asyncio

from taskcompendium.harbor.runner import ChatLaunch, run_trial

result = asyncio.run(
    run_trial(
        Path("/tmp/arithmetic-task"),
        chosen.environment_config,
        ChatLaunch(model="model-id", api_base="https://example.com/v1", api_key_env="MODEL_API_KEY"),
        Path("/tmp/arithmetic-trials"),
        "arithmetic-run",
    )
)
```

Each direct-chat Harbor trial runs one `ChatAgent` using the exported submission convention. The lowering prepares the conversation and tool configuration; the agent makes one request, validates the chat protocol, and writes a typed `ConversationTrace` to `submission.json`. This trace contains the complete model-visible conversation, including submission instructions and the final assistant message. Function-call arguments are decoded objects in both source context and grading evidence. The raw provider response is retained separately in `chat-response.json` for diagnostics.

The direct-chat environment exposes no filesystem or shell tools. Harbor's custom verifier reads the typed trace and calls the synchronous `grade_answer` submission adapter. TaskCompendium extracts text or numeric candidates according to the convention and passes final function calls directly to shared scoring. Expected values stay in private verifier configuration. Each harness translates its protocol into the shared conversation types.

A valid but wrong answer receives reward `0.0`. A text, numeric, or final-action answer that violates its submission convention receives `extraction_error` with no reward. A native-action submission that satisfies its convention but differs from the expected function calls receives reward `0.0`. A malformed provider message or tool-call argument fails at the harness boundary with no reward and the raw response retained. Verifier infrastructure failures are recorded as `infra_error` with no reward in `taskcompendium-result.json`. The package requires Harbor's [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155) and does not use `tests/test.sh`. Install the pinned Harbor fork with `uv sync --project lib/taskcompendium --extra harbor`; its revision is declared in `lib/taskcompendium/pyproject.toml`.

The package tests use a test-only `ReplayAgent` in `tests/harbor_replay.py` to write fixed assistant messages and exercise Harbor grading without a model request. They also replay responses at the HTTP boundary through the production launcher. Replay is absent from the installed package and public launcher.

TaskCompendium requires Python 3.12 or 3.13 and uses `marin-rigging` for shared portable path and mount-collision validation. The validator leaf module performs no storage access.

Run the package tests from the repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q

# Type-check the package from its own project directory after installing its dependencies.
cd lib/taskcompendium
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```
