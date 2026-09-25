# TaskCompendium direct-chat slice

`TaskSpec` holds one fixed answer task, its source provenance, semantic requirements, permitted answer formats, and a private verifier descriptor. The task, rendering, binding, and verifier records are frozen Pydantic models, validated again when read from a Harbor export. A `Rendering` selects either a plain final answer or a JSON object with an `answer` string. Plain text is the conservative default; an importer must explicitly permit JSON when that wrapper preserves the task. `compatible_lowerings` enumerates allowed rendering and environment pairs. `select_lowerings` takes all pairs, the first pair, or one reproducible keyed sample. The library order determines the first pair and the order used for sampling. This slice provides only a direct-chat binding with no tools. `HarborLaunch` selects a replay agent for validation or an OpenAI-compatible chat agent for a model run.

The exporter writes `instruction.md`, `task.toml`, an empty `environment/` directory, `specification.json`, `rendering.json`, and `binding.json`. The specification stays private to the Harbor custom verifier; the launcher reads the rendering to configure public output functions for action tasks. Launch checks the stored binding before starting Harbor. The agent has no filesystem or shell tools. The package requires Harbor at the revision containing [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155); exported tasks contain no `tests/test.sh`.

The custom verifier has Harbor's verifier-side environment available for tasks that need workspace state. The registry dispatches by private verifier kind and passes the environment to each handler. The exact-answer handler reads the saved final response, extracts the answer according to the rendering, and compares it directly with the private reference. It does not create an answer file or use a TaskTrove grading runtime.

For an authenticated model run, pass the environment variable name as `HarborLaunch("chat", model="...", agent_kwargs={"api_base": "...", "api_key_env": "MODEL_API_KEY"})`. The agent reads its value at request time; Harbor trial configuration retains only the variable name. The variable must be set in the trial process environment.

```python
from pathlib import Path

from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborTaskBinding, SelectionPolicy, compatible_lowerings, lower_to_harbor, select_lowerings
from taskcompendium.models import AnswerFormat, Source, TaskRequirements, TaskSpec
from taskcompendium.rendering import Rendering

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    instructions="What is 7 + 5?",
    verifier=exact_answer("12"),
    source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    requirements=TaskRequirements(),
    permitted_answer_formats=(AnswerFormat.PLAIN, AnswerFormat.JSON),
)
renderings = (Rendering(id="plain", answer_format=AnswerFormat.PLAIN), Rendering(id="json", answer_format=AnswerFormat.JSON))
candidates = compatible_lowerings(spec, renderings, (HarborTaskBinding(),))
for candidate in select_lowerings(candidates, SelectionPolicy.ALL):
    lower_to_harbor(spec, candidate.rendering, candidate.binding, Path(f"/tmp/arithmetic-{candidate.rendering.id}"))

# A training caller can choose one variant reproducibly instead:
sampled = select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=1234)
```

The caller records the selection policy, RNG key, and library revision in its dataset or rollout manifest. Each exported Harbor package records the chosen rendering and binding. An arbitrary source prompt does not become format-agnostic automatically: the importer must allow only rewrites that preserve its instructions.

`VerifierSpec(kind, parameters)` is the private serialized form. Each `taskcompendium.verifiers` entry point names a kind and resolves to a zero-argument factory returning a `VerifierHandler(payload_type, grade)`. The payload type is a frozen Pydantic model that validates its parameters. The grade function receives that typed payload plus a `GradingAttempt` containing the response, rendering, transcript, and Harbor's verifier-side environment. Validation loads the factory by kind before export or launch. An unknown kind or invalid parameters fail preflight; `Rendering` determines answer extraction but does not select the verifier. The exact-answer handler compares extracted text directly; other handlers may inspect the verifier-side environment.

Run the Python-only package tests from the repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
```

The Harbor trial tests exercise correct, wrong, malformed, and infrastructure-failure outcomes. A malformed submission has no reward. A failed comparison has reward `0.0`. An infrastructure failure has no reward and is recorded separately in `taskcompendium-result.json`.

## NeMo predicted actions

`import_row` in `taskcompendium.importers.nemo_predicted_action` accepts a NeMo single-step row and its pinned canonical SHA-256 digest. It returns a private `TaskSpec` and a public final-action `Rendering`. The rendering retains source message roles, function definitions, tool choice, and parallel-call setting; the expected function calls stay in the private `nemo_predicted_action` verifier. The importer rejects nonempty request settings it cannot carry. Harbor export checks that the readable task instruction still matches the structured messages. The chat agent sends the source turns separately, records one native final action, and stops. The no-tool environment never dispatches the call.

The pinned fixture records the NeMo Gym repository revision and blob SHA in `tests/fixtures/nemo/predicted-action.provenance.json`. The importer pins the dataset revision in `Source`. Imported targets must be advertised function calls. Message targets are rejected because the source comparator gives any message full credit. TaskCompendium strengthens function-call scoring: call counts, JSON types, object keys, list structure, and string values must match exactly. Numeric tolerance is available only when set explicitly on the private verifier.
