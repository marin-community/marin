# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A deliberately basic prompt asking GLM for a task program, and the submission GLM replies with.

``program_messages`` builds a system message and a user message for a catalog capability. The
system message carries the rules for the program, the TaskSpec JSON schema and ``example_program.py``
as a worked example. The user message carries the capability record.

GLM replies with a ``ProgramSubmission`` JSON object, decoded under its schema as the response
format. The messages here leave the reply format out. ``taskforge.llm.structured.complete_structured``
appends a last user turn asking for the object, rendered from ``ProgramSubmission``'s docstring and
the description of its ``source`` field, so both are written as instructions to the model.

The driver in ``generate_tasks`` runs the submitted program as
``python3 program.py <task id> <output path>``. The program writes a TaskSpec to the output path,
for a task exercising the capability. The driver then confirms that the program exited with status 0
and that its output parses as a TaskSpec. The prompt limits its rules to what the driver checks,
plus guidance on grading and reproducibility. From there on, the task's own grader is the only check
on the task.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from taskcompendium.models import TaskSpec
from taskforge.capability import CapabilityIdea, capability_idea_record
from taskforge.content_hash import pretty_json
from taskforge.llm.policy import Message

EXAMPLE_PROGRAM_FILE = "example_program.py"

SYSTEM_PROMPT = """\
Your job is to write one Python program that generates one task for training a language-model agent.

- The program runs as `python3 program.py <task id> <output path>` with Python 3.12 or newer, the \
standard library, and the `taskcompendium` and `verifyit` packages. It has no network and no other \
packages.
- It must exit with status 0 and write to `<output path>` one JSON object that \
`taskcompendium.models.TaskSpec.model_validate_json` accepts, with the given task id as its `id`. \
Build the spec with the `taskcompendium.models` classes and write `spec.model_dump_json()`.
- The task carries its own grader. Compute the reference answer in the program and grade it with a \
verifyit spec (`taskcompendium.grader.verifyit_package`); never put the answer in the instruction or \
in agent-visible files.
- The task should exercise the capability below. Derive any randomness from the task id so a run is \
reproducible.

The TaskSpec JSON schema:

```json
{schema}
```

A complete program for a different capability:

```python
{example}
```
"""

USER_PROMPT = """\
# Capability

```json
{capability}
```

Write the program for this capability.
"""


class ProgramSubmission(BaseModel):
    """The program submission for the capability above."""

    source: str = Field(
        description="The complete Python program text, the whole module and nothing else. "
        "It runs as program.py and writes the TaskSpec JSON to argv[2]."
    )

    @field_validator("source")
    @classmethod
    def compiles(cls, source: str) -> str:
        # For some sources compile raises ValueError, RecursionError or MemoryError instead of SyntaxError,
        # the last two when deep nesting overflows the parser's stack. The validator reports each as an
        # invalid submission.
        try:
            compile(source, "program.py", "exec")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            raise ValueError(f"the program does not compile: {error}") from error
        return source


def example_program_source() -> str:
    """Read the source of the example task program for the prompt."""
    return Path(__file__).with_name(EXAMPLE_PROGRAM_FILE).read_text()


def program_messages(idea: CapabilityIdea) -> list[Message]:
    """Build the system and user messages for ``idea``.

    The system message holds the rules, the TaskSpec schema and the example program. The user message
    holds the capability record.
    """
    system = SYSTEM_PROMPT.format(
        schema=json.dumps(TaskSpec.model_json_schema(), indent=2), example=example_program_source()
    )
    user = USER_PROMPT.format(capability=pretty_json(capability_idea_record(idea)))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
