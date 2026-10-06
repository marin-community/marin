# TaskCompendium

The [task curation pipeline](../../docs/references/task-curation.md) downloads
pinned sources, normalizes tasks, and runs grading checks and model review.
It writes filtering decisions to sharded Parquet and retains source locators,
edits, and rejection reasons. Library families define normalization, checks,
and rubrics. Experiments bind pinned inputs, download artifacts, inference clients,
and intended use.

For ingestion work, start with the [pipeline overview](src/taskcompendium/pipeline/README.md)
and the [experiment flow](../../experiments/post_training/task_curation/README.md).

TaskCompendium stores a task's semantic contract, extracts one final submission, and
grades it through the shared verifier library. A caller chooses a submission convention
while expected answers and verifier files stay private. Execution, model requests, tool
dispatch, environment setup, and lifecycle management belong to the caller's runtime; the
separate [rollout engine](../rolloutengine/README.md) executes tasks through Shellbox.

## Task records

A task contains:

- `context`: Public text messages, function calls, and tool results before the first model turn.
- `answer_type`: Text, number, JSON, final function calls, files, or environment state.
- `final_tools`: Advertised functions that terminate the task.
- `interaction_tools`: Executable function declarations for the episode runtime.
- `output_paths`: Absolute paths that the episode runtime captures.
- `verifier`: Private grading parameters, private files, and capability requirements.
- `environment_requirements`: Task capabilities.
- `environment`: Executable machine inputs and an optional task-session selector.
- `source`: Dataset, revision, row, and importer revision.
- `metadata` and `tags`: Application data and labels.

`TaskSpec.model_dump_json()` serializes a task.
`TaskSpec.model_validate_json()` validates it. Applications own dataset file formats and storage.
The serialized task contains private reference answers. Do not send the whole record to the model.
Readers reject other `schema_version` values; the current version is `0.25`.

Conversation events retain tool-call IDs and order. They exclude provider reasoning state.
`answer_type` describes the result, independently of its submission format.
Importers must remove source instructions that conflict with the supported conventions or reject the row.

`environment` describes the task machine. `environment_requirements` declares
task capabilities.
Place public files in `environment.files` and private files in `VerifierSpec.files`.
`TaskExecution` stores attempt and agent deadlines, agent users, and stage
preparation separately from the task definition.
See [task rollouts](../../docs/references/task-rollouts.md) for executable fields, stages, and token contracts.

## Submission conventions

An answer is the task's semantic result, identified by `answer_type`. A submission is the typed
value a convention extracts from a completed attempt for grading. `PlainText`, `JsonAnswer`
(an `{"answer": "..."}` object), and `AnswerCall` (a final `submit_answer(answer: string)` call)
each extract a `TextSubmission` for the same text or numeric verifier. `JsonValueAnswer` parses
the complete final text as one JSON value for `answer_type=json`. `FinalAction` captures native
function calls as an `ActionSubmission`, with optional required-call and maximum-call
constraints. No convention executes final function calls.

`convention.supports(answer_type)` checks the result kind. `submission_compatibility(task,
convention)` also checks that the convention produces an evidence type the private candidate
verifier accepts, that `FinalAction` has final tools, and that no task function is named
`submit_answer`. A custom convention extends `SubmissionConvention`, declares its
`submission_types`, and extracts already acquired evidence from a `GradingAttempt`; it must not
reach a live workspace. `submission_instruction(convention)` and `answer_call_tool()` supply the
instruction and tool definition that a runtime adds to the public conversation.

## Grading

Pure candidate scoring uses `verifyit`: `exact`, `numeric`, `mcq`, `predicted_action`, and
`structured_exact`. `grade_answer(task, convention, GradingAttempt(conversation))` validates the
private verifier and the convention's compatibility, calls `extract` once, and scores the
submission. Numeric references are literal strings with explicit absolute and relative
tolerances. `structured_exact` compares JSON values with numeric value equality by default and
`numeric_types="strict"` for exact scalar types. Verifier parameters reject duplicate keys and
nonfinite numbers at every nesting level.

A valid wrong answer is `graded` with reward `0.0`. Malformed text, JSON, numeric, or
final-action evidence is `submission_failure` with reward `0.0`. Invalid private configuration
raises; a candidate verifier with capability requirements or a private grading environment
raises `NotImplementedError` from the pure boundary. Malformed provider messages fail at the
harness boundary. Verifier failures are `infra_error` with no reward.

Runtime evidence grading lives in `taskcompendium.runtime.task_grading`. Its synchronous
`grade_task` accepts a conversation and already acquired `RuntimeEvidence`, delegates candidate
kinds to `grade_answer`, and materializes `environment.files` and `VerifierSpec.files` for
file and script graders. Script verdicts retain `invalid_task` and `infra_error` status and
details separately from graded rewards.

Shell tasks use `ShellVerifierSpec`. Application sessions use `ExternalVerifierSpec` for
private parameters and `environment.interaction` to select the session. Staged tasks use
`StageVerifierSpec`. `validate_verifier` checks each kind's payload. Group grading belongs to
the training application.

Private verifier factories:

```python
from taskcompendium.grading import exact_answer, numeric_answer, structured_exact, verifier_descriptor
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer
from taskcompendium.verifiers.predicted_action import predicted_action_verifier

text_verifier = exact_answer("expected text")
number_verifier = numeric_answer("12", tolerance_abs=0.0, tolerance_rel=0.0)
mcq_verifier = multiple_choice_answer("C", options=4)
json_verifier = structured_exact({"value": 16})
```

Create a semantic task and grade a final response:

```python
from taskcompendium.grading import grade_answer, numeric_answer
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import PlainText

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    context=ConversationInput(events=(TextMessage(role="user", content="What is 7 + 5?"),)),
    environment_requirements=EnvironmentRequirements(),
    answer_type=AnswerType.NUMBER,
    verifier=numeric_answer("12", tolerance_abs=0.0, tolerance_rel=0.0),
    source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
)
convention = PlainText(id="plain")


def score_final_response(content: str):
    conversation = ConversationTrace(events=(*spec.context.events, TextMessage(role="assistant", content=content)))
    return grade_answer(spec, convention, GradingAttempt(conversation))
```

Built-in conventions read `conversation.events[-1]`, the final assistant message or call
batch. Runtimes decode provider responses into `TextMessage` or `AssistantToolCalls` with
`taskcompendium.chat` before constructing a `ConversationTrace`; tool-call arguments become
decoded JSON objects there.

## Importers

- `taskcompendium.importers.tasktrove.convert.read_archive` reads MCQ archives with caller-supplied release provenance.
  It checks archive identity and answer instructions before producing a private `mcq` verifier.
- `taskcompendium.importers.nemo_predicted_action.import_row` accepts a source row and its pinned
  `canonical_sha256(row)` digest. It returns a task and `FinalAction` convention.
  Unsupported request settings, encrypted reasoning, and reasoning at the decision point cause rejection.
  Assistant-message targets cause rejection because the source comparator does not compare their content.
- Harbor, SWE, and SkyRL importers produce executable tasks for the
  [rollout engine](../rolloutengine/README.md).

## Local checks

From the Marin repository root:

```bash
task_test_prefix=$(mktemp -d -t taskcompendium-tests.XXXXXX)
MARIN_PREFIX="$task_test_prefix" uv run --project lib/taskcompendium --frozen --extra pipeline --group test pytest lib/taskcompendium/tests -q
```

Python 3.12 or 3.13 is required. Package dependencies are in [pyproject.toml](pyproject.toml).
For type checks, run `uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check` from this package directory after dependency installation.
