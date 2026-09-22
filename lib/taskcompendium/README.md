# TaskCompendium direct-chat slice

`TaskSpec` holds one fixed answer task, its source provenance, semantic requirements, permitted answer formats, and a private verifier descriptor with an explicit `kind`. The core package provides an `exact_answer` handler; other packages can provide further kinds without changing the task schema or grading dispatch. A `Rendering` selects either a plain final answer or a JSON object with an `answer` string, independently of the private verifier kind. Plain text is the conservative default; an importer must explicitly permit JSON when that wrapper preserves the task. Lowering rejects renderings outside the specification's allowlist. `HarborTaskBinding` selects the direct-chat environment with no tools and must satisfy the task requirements. `HarborLaunch` selects a replay agent for validation or an OpenAI-compatible chat agent for a model run.

The exporter writes `instruction.md`, `task.toml`, an empty `environment/` directory, `specification.json`, `rendering.json`, and `binding.json`. The specification and rendering files remain private to the Harbor custom verifier. Launch checks the stored binding before starting Harbor. The agent receives the rendered instruction and has no filesystem or shell tools. The package requires Harbor at the revision containing [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155); exported tasks contain no `tests/test.sh`.

For an authenticated model run, pass the environment variable name as `HarborLaunch("chat", model="...", agent_kwargs={"api_base": "...", "api_key_env": "MODEL_API_KEY"})`. The agent reads its value at request time; Harbor trial configuration retains only the variable name. The variable must be set in the trial process environment.

```python
from pathlib import Path

from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborTaskBinding, lower_to_harbor
from taskcompendium.models import AnswerFormat, Source, TaskRequirements, TaskSpec
from taskcompendium.rendering import Rendering

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    instructions="What is 7 + 5?",
    verifier=exact_answer("12"),
    source=Source("hand-authored", "2026-09-16", "arithmetic-7-plus-5", "1"),
    requirements=TaskRequirements(),
    permitted_answer_formats=(AnswerFormat.PLAIN, AnswerFormat.JSON),
)
binding = HarborTaskBinding()
lower_to_harbor(spec, Rendering("plain", AnswerFormat.PLAIN), binding, Path("/tmp/arithmetic-plain"))
lower_to_harbor(spec, Rendering("json", AnswerFormat.JSON), binding, Path("/tmp/arithmetic-json"))
```

`VerifierSpec(kind, parameters)` is the private serialized form. Each `taskcompendium.verifiers` entry point names a kind and resolves to a zero-argument factory returning a frozen `VerifierHandler(payload_type, grade)`. The payload type is a `msgspec.Struct` that validates its own parameters. The grade function receives that typed payload plus a `GradingAttempt`. Validation loads the factory by kind before export or launch. An unknown kind or invalid parameters fail preflight; `Rendering` does not select the verifier. The current attempt carries a rendered answer and transcript. Script and judge handlers will need verifier-side resources and runtime context added to that input before they can run.

Run the Python-only package tests from the repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
```

The Harbor trial tests exercise correct, wrong, malformed, and infrastructure-failure outcomes. A malformed submission has no reward. A failed comparison has reward `0.0`. An infrastructure failure has no reward and is recorded separately in `taskcompendium-result.json`.
