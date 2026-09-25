# TaskCompendium direct-chat slice

`TaskSpec` holds one fixed answer task, its source provenance, semantic requirements, coarse answer type, and private exact-answer verifier. A `SubmissionConvention` decides how a result is requested, delivered, and extracted. This slice supports plain final answers and JSON objects with an `answer` string for text and number results. An importer can restrict conventions when source instructions require a particular submission shape. `compatible_lowerings` enumerates convention and environment pairs that satisfy the task. `select_lowerings` takes all pairs, the first pair, or one reproducible keyed sample. The library order determines the first pair and the order used for sampling. This slice provides only a direct-chat binding with no tools. `HarborLaunch` selects a replay agent for validation or an OpenAI-compatible chat agent for a model run.

The exporter writes `instruction.md`, `task.toml`, an empty `environment/` directory, `specification.json`, `submission_convention.json`, and `binding.json`. The specification and convention files remain private to the Harbor custom verifier. Launch checks the stored binding before starting Harbor. The agent receives the resulting instruction and has no filesystem or shell tools. The package requires Harbor at the revision containing [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155); exported tasks contain no `tests/test.sh`.

The custom verifier has Harbor's verifier-side environment available for tasks that need workspace state. This direct-chat grader reads the saved final response, extracts the answer according to the convention, and compares it directly with the private reference. It does not create an answer file or use a TaskTrove grading runtime.

For an authenticated model run, pass the environment variable name as `HarborLaunch("chat", model="...", agent_kwargs={"api_base": "...", "api_key_env": "MODEL_API_KEY"})`. The agent reads its value at request time; Harbor trial configuration retains only the variable name. The variable must be set in the trial process environment.

```python
from pathlib import Path

from taskcompendium.lowering import HarborTaskBinding, SelectionPolicy, compatible_lowerings, lower_to_harbor, select_lowerings
from taskcompendium.models import AnswerType, ExactAnswer, Source, TaskRequirements, TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    instructions="What is 7 + 5?",
    verifier=ExactAnswer(expected="12"),
    source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    requirements=TaskRequirements(),
    answer_type=AnswerType.NUMBER,
)
conventions = (SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN), SubmissionConvention(id="json", answer_format=AnswerFormat.JSON))
candidates = compatible_lowerings(spec, conventions, (HarborTaskBinding(),))
for candidate in select_lowerings(candidates, SelectionPolicy.ALL):
    lower_to_harbor(spec, candidate.convention, candidate.binding, Path(f"/tmp/arithmetic-{candidate.convention.id}"))

# A training caller can choose one variant reproducibly instead:
sampled = select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=1234)
```

The caller records the selection policy, RNG key, and library revision in its dataset or rollout manifest. Each exported Harbor package records the chosen convention and binding. An arbitrary source prompt does not become format-agnostic automatically: the importer must allow only rewrites that preserve its instructions.

Run the Python-only package tests from the repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
```

The Harbor trial tests exercise correct, wrong, malformed, and infrastructure-failure outcomes. A malformed submission has no reward. A failed comparison has reward `0.0`. An infrastructure failure has no reward and is recorded separately in `taskcompendium-result.json`.
