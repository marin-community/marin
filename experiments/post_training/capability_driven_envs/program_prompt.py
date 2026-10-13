# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deliberately basic prompt that asks GLM for a task program, and the submission its reply must be.

GLM gets one catalog capability and writes one Python program that, run as
``python3 program.py <task id> <output path>``, writes a TaskSpec for a task exercising that
capability. The rules are only the ones the driver checks: the program runs, and its output parses
as a TaskSpec. The task carries its own grader; nothing here checks the task itself. The reply is a
``ProgramSubmission`` JSON object; the server enforces its schema as the response format.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from taskcompendium.models import TaskSpec
from taskforge.content_hash import pretty_json
from taskforge.llm.policy import Message

from experiments.post_training.capability_driven_envs.catalog import CapabilityIdea, capability_idea_record

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

Write the program for this capability. Reply with the JSON object for the program submission.
"""


class ProgramSubmission(BaseModel):
    """Submit the complete task program."""

    source: str = Field(description="The complete Python program: it writes the TaskSpec JSON to argv[2].")

    @field_validator("source")
    @classmethod
    def compiles(cls, source: str) -> str:
        # Besides SyntaxError, compile raises ValueError, RecursionError or MemoryError (the parser's
        # stack overflowing on deep nesting) for some sources; each is an invalid submission, not a crash.
        try:
            compile(source, "program.py", "exec")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            raise ValueError(f"the program does not compile: {error}") from error
        return source


def example_program_source() -> str:
    """The source of the example task program shown to GLM."""
    return Path(__file__).with_name(EXAMPLE_PROGRAM_FILE).read_text()


def program_messages(idea: CapabilityIdea) -> list[Message]:
    """The system rules with the schema and example, then the capability record."""
    system = SYSTEM_PROMPT.format(
        schema=json.dumps(TaskSpec.model_json_schema(), indent=2), example=example_program_source()
    )
    user = USER_PROMPT.format(capability=pretty_json(capability_idea_record(idea)))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
