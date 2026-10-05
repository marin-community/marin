# TaskCompendium

The [task curation pipeline](../../docs/references/task-curation.md) downloads pinned sources, normalizes tasks, runs grading checks and GLM review, then writes final filtering decisions to sharded Parquet. Its audit retains every selected input, source locator, edit and rejection reason. Library families define normalization, checks and rubrics; the experiment binds pinned inputs, intended use, download artifacts and inference clients.

For ingestion work, start with the [pipeline overview](src/taskcompendium/pipeline/README.md)
and the [experiment flow](../../experiments/post_training/task_curation/README.md).

TaskCompendium defines private `TaskSpec` records, submission conventions, importers,
and grading contracts. It exports direct-chat Harbor tasks. The separate
[rollout engine](../rolloutengine/README.md) executes tasks through Shellbox.
Harbor runs model trials and invokes private graders. Shellbox supplies isolated execution machines.

## Task records

A task contains:

- `context`: Public text messages, historical function calls, and tool results.
- `answer_type`: Text, number, final function calls, files, or environment state.
- `final_tools`: Advertised functions that terminate the task.
- `interaction_tools`: Executable function declarations for the episode runtime.
- `output_paths`: Absolute paths that the episode runtime captures.
- `verifier`: Private grading parameters and capability requirements.
- `environment_requirements`: Task capabilities.
- `environment`: Executable machine inputs and an optional task-session selector.
- `source`: Dataset, revision, row, and importer revision.
- `metadata` and `tags`: Application data and labels.

`TaskSpec.model_dump_json()` serializes a task.
`TaskSpec.model_validate_json()` validates it. Applications own dataset file formats and storage.
The serialized task contains private reference answers. Do not send the whole record to the model.

Conversation events retain tool-call IDs and order. They exclude provider reasoning state.
`answer_type` describes the result, independently of its submission format.
Importers must remove source instructions that conflict with the supported conventions or reject the row.

`environment` is the single machine description. Place public files in `environment.files` and private files in `VerifierSpec.files`.
Schema `0.24` removes the legacy machine fields and `resources`. `environment_requirements` declares capabilities only.
`TaskExecution` stores attempt and agent deadlines, agent users, and stage preparation separately from the task definition.
Rebuild earlier task exports with the current importer.
See [task rollouts](../../docs/references/task-rollouts.md) for executable fields, stages, and token contracts.

## Submissions and grading

`SubmissionConvention` supports plain text, an `{"answer": "..."}` JSON object,
or a final `submit_answer(answer: string)` call. These conventions extract a string for the same verifier.
`FinalAction` captures native function calls with optional required-call and maximum-call constraints.
Neither convention executes final function calls.

Pure candidate scoring uses `verifyit`: `exact`, `numeric`, `mcq`, and `predicted_action`.
Numeric tolerances must be explicit. Shell tasks use `ShellVerifierSpec`.
Application sessions use `ExternalVerifierSpec` for private parameters and
`environment.interaction` to select the session. Staged tasks use `StageVerifierSpec`.
Group grading belongs to the training application.

Schema loading accepts verifier descriptors without an implementation.
Export and launch validate runtime support and reject unsupported kinds or requirements.
`structured_exact` has no implementation in this package.

A wrong answer receives a numeric grade. An invalid submission receives `extraction_error` with no reward.
Malformed provider messages fail at the harness boundary. Verifier failures receive `infra_error` with no reward.

## Importers

- `taskcompendium.importers.tasktrove.convert.read_archive` reads MCQ archives with caller-supplied release provenance.
  It checks archive identity and answer instructions before producing a private `mcq` verifier.
- `taskcompendium.importers.nemo_predicted_action.import_row` accepts a source row and its pinned
  `canonical_sha256(row)` digest. It returns a task and `FinalAction` convention.
  Unsupported request settings, encrypted reasoning, and reasoning at the decision point cause rejection.
  Assistant-message targets cause rejection because the source comparator does not compare their content.
- Harbor, SWE, and SkyRL importers produce executable tasks for the
  [rollout engine](../rolloutengine/README.md).

## Direct-chat Harbor export

A lowering pairs a compatible submission convention with a Harbor environment configuration.
The current direct-chat configuration accepts text, numeric, and native-action tasks without machine or resource requirements.
It records final calls without execution. Shell, file, and state tasks require the separate rollout engine.

For multiple presentations, use `compatible_lowerings` and `select_lowerings` with an explicit selection policy and RNG key.

```python
from pathlib import Path

from taskcompendium.grading import numeric_answer
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
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
convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)
environment_config = HarborEnvironmentConfig()
lower_to_harbor(spec, convention, environment_config, Path("/tmp/arithmetic-task"))
```

The exported package contains `instruction.md`, `task.toml`, `specification.json`,
`submission_convention.json`, and `environment_config.json`.
The model receives the public conversation and submission instructions. It cannot read package files.
The custom verifier reads the private task and typed `submission.json` conversation.
`chat-response.json` retains the provider response for diagnostics.
`taskcompendium-result.json` records the grading outcome.

Run the exported task through the pinned Harbor fork:

```python
import asyncio

from taskcompendium.harbor.runner import ChatLaunch, run_trial

result = asyncio.run(
    run_trial(
        Path("/tmp/arithmetic-task"),
        environment_config,
        ChatLaunch(model="model-id", api_base="https://example.com/v1", api_key_env="MODEL_API_KEY"),
        Path("/tmp/arithmetic-trials"),
        "arithmetic-run",
    )
)
```

`api_key_env` stores the environment-variable name. The agent resolves its value in its process.
The fork supplies [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155).

## Local checks

From the Marin repository root:

```bash
task_test_prefix=$(mktemp -d -t taskcompendium-tests.XXXXXX)
MARIN_PREFIX="$task_test_prefix" uv run --project lib/taskcompendium --frozen --extra harbor --extra pipeline --group test pytest lib/taskcompendium/tests -q
```

Python 3.12 or 3.13 is required. Package dependencies and the Harbor revision are in [pyproject.toml](pyproject.toml).
For type checks, run `uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check` from this package directory after dependency installation.
