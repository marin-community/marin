# TaskCompendium direct-chat slice

`TaskSpec` holds one fixed answer task, its source provenance, semantic requirements, and a private exact-answer verifier. A `Rendering` selects either a plain final answer or a JSON object with an `answer` string. Both renderings use the same reference answer. `HarborTaskBinding` selects the direct-chat environment with no tools and must satisfy the task requirements. `HarborLaunch` selects a replay agent for validation or an OpenAI-compatible chat agent for a model run.

The exporter writes `instruction.md`, `task.toml`, `specification.json`, `rendering.json`, and `binding.json`. The specification and rendering files remain private to the Harbor custom verifier. Launch checks the stored binding before starting Harbor. The agent receives the rendered instruction and has no filesystem or shell tools. The package requires Harbor at the revision containing [custom-verifier task loading](https://github.com/marin-community/harbor/pull/155); exported tasks contain no `tests/test.sh`.

```python
from pathlib import Path

from taskcompendium.lowering import HarborTaskBinding, lower_to_harbor
from taskcompendium.models import ExactAnswer, Source, TaskRequirements, TaskSpec
from taskcompendium.rendering import AnswerFormat, Rendering

spec = TaskSpec(
    id="arithmetic-7-plus-5",
    instructions="What is 7 + 5?",
    verifier=ExactAnswer("12"),
    source=Source("hand-authored", "2026-09-16", "arithmetic-7-plus-5", "1"),
    requirements=TaskRequirements(),
)
binding = HarborTaskBinding()
lower_to_harbor(spec, Rendering("plain", AnswerFormat.PLAIN), binding, Path("/tmp/arithmetic-plain"))
lower_to_harbor(spec, Rendering("json", AnswerFormat.JSON), binding, Path("/tmp/arithmetic-json"))
```

Run the Python-only package tests from the repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
```

The Harbor trial tests exercise correct, wrong, malformed, and infrastructure-failure outcomes. A malformed submission has no reward. A failed comparison has reward `0.0`. An infrastructure failure has no reward and is recorded separately in `taskcompendium-result.json`.
