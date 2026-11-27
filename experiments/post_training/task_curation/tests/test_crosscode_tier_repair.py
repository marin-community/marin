# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The tier repair removes the hack tiers and the coaching; the clean contract scores 1.0/0."""

import json

from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.crosscode_tier_repair import (
    REMOVED_TIERS,
    REPAIR_VERSION,
    REPAIRED_TEST_SCRIPT,
    convert_population,
    repair_task,
    repaired_instruction,
)

TIERED_SCRIPT = b"""#!/bin/bash
# --- laion v4 patch: 4-tier forgiving verifier ---
echo "0" > /logs/verifier/reward.txt
# Tier 4: non-empty completion fallback. Award partial reward 0.25
echo "0.25" > /logs/verifier/reward.txt
"""

COACHING_INSTRUCTION = """# Cross-File Code Completion Task

## The context

```csharp
using System;
```

### Constraints on the fragment itself

- Output ONLY the short continuation. Do NOT re-emit any of the snippet above.
- The verifier byte-compares your fragment to a reference fragment after
  whitespace normalisation. It also accepts:
    * a leading-identifier match (your fragment's first identifier token
      matches the reference's first identifier token), and
    * a non-empty partial credit if your fragment contains at least one
      C# identifier and is under ~400 characters.
  So even if you can't guess the exact reference identifier, writing
  *something* shaped like a C# completion fragment is strictly better than
  writing nothing.

### Worked example

Run write_solution 'Module> Modules => GetModules();'
"""


def _tiered_task() -> TaskSpec:
    dockerfile = b"FROM python:3.10-slim\nWORKDIR /app\n"
    return TaskSpec(
        id="TaskTrove:laion__fixture-v4/task-1",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=(
                        "# Cross-File Code Completion Task\n\nComplete the fragment.\n\n"
                        "- The verifier byte-compares your fragment to a reference fragment after\n"
                        "  whitespace normalisation. It also accepts:\n"
                        "    * a leading-identifier match, and\n"
                        "    * a non-empty partial credit if your fragment contains at least one\n"
                        "      C# identifier and is under ~400 characters.\n"
                        "  So even if you can't guess the exact reference identifier, writing\n"
                        "  *something* shaped like a C# completion fragment is strictly better than\n"
                        "  writing nothing.\n"
                    ),
                ),
            ),
        ),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", dockerfile),)),
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The native Harbor Trial executes this package's declared verifier"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("environment/Dockerfile", dockerfile),
                inline_resource("instruction.md", COACHING_INSTRUCTION.encode()),
                inline_resource("task.toml", b"[verifier]\ntimeout_sec = 720.0\n"),
                inline_resource("tests/test.sh", TIERED_SCRIPT),
                inline_resource("tests/solution.txt", b"gold\n"),
            ),
            oracle=(
                inline_resource("solution/solution.txt", b"gold\n"),
                inline_resource(
                    "solution/solve.sh", b'#!/bin/bash\ncp "$(dirname "$0")/solution.txt" /app/solution.txt\n'
                ),
            ),
        ),
        tags=("harbor",),
    )


def test_repair_removes_the_hack_tiers_and_records_the_lineage():
    task = _tiered_task()
    repaired = repair_task(task)
    script = next(r for r in repaired.resources.verifier if r.path == "tests/test.sh")
    text = resource_bytes(script).decode()
    assert "0.25" not in text
    assert "leading-identifier" not in text
    assert "normalize" in text


def test_repair_also_rewrites_the_public_conversation_message():
    """The coaching lives in the public context message too (the SAMPLE QC gap)."""
    task = _tiered_task()
    public = task.context.events[0].content
    assert "leading-identifier" in public or "partial credit" in public
    repaired = repair_task(task)
    public = repaired.context.events[0].content
    assert "partial credit" not in public
    assert "leading-identifier" not in public
    assert "exact match" in public
    # the task description stays
    assert "Cross-File Code Completion Task" in public
    provenance = next(r for r in repaired.resources.oracle if r.path == "taskcompendium/crosscode-tier-repair.json")
    doc = json.loads(resource_bytes(provenance).decode())
    assert doc["version"] == REPAIR_VERSION
    assert len(doc["removed_tiers"]) == 3
    assert doc["input_task_sha256"] != task_spec_sha256(repaired)


def test_repaired_instruction_drops_the_coaching_and_keeps_the_task():
    repaired = repaired_instruction(COACHING_INSTRUCTION)
    assert "strictly better than writing nothing" not in repaired
    assert "0.25" not in repaired
    assert "partial credit" not in repaired
    assert "Cross-File Code Completion Task" in repaired
    assert "Output ONLY the short continuation" in repaired
    assert "exact match" in repaired
    assert "### Worked example" in repaired


def test_repaired_script_scores_gold_one_and_noop_zero():
    """The clean contract: exact normalized match -> 1.0; anything else -> 0."""
    assert 'echo 0 > "$REWARD"' in REPAIRED_TEST_SCRIPT
    assert 'echo 1 > "$REWARD"' in REPAIRED_TEST_SCRIPT
    assert "0.25" not in REPAIRED_TEST_SCRIPT
    assert "leading-identifier" not in REPAIRED_TEST_SCRIPT


def test_convert_population_repairs_every_row():
    task = _tiered_task()
    rows = [{"task_spec": task.model_dump_json()}]
    population = convert_population(rows)
    assert population.counts["converted_rows"] == 1 and population.counts["excluded_rows"] == 0
    repaired = TaskSpec.model_validate_json(population.tasks[0]["task_spec"])
    script = next(r for r in repaired.resources.verifier if r.path == "tests/test.sh")
    assert "0.25" not in resource_bytes(script).decode()
    assert population.receipts[0]["removed_tiers"] == list(REMOVED_TIERS)


def test_repair_passes_through_an_already_clean_task():
    """A task clean on BOTH surfaces (script + public message) passes through."""
    task = _tiered_task()
    clean_public = (
        "# Cross-File Code Completion Task\n\n"
        "- The verifier compares your fragment after normalisation and accepts only an exact match.\n"
    )
    clean = task.model_copy(
        update={
            "context": task.context.model_copy(
                update={
                    "events": tuple(event.model_copy(update={"content": clean_public}) for event in task.context.events)
                }
            ),
            "resources": task.resources.model_copy(
                update={
                    "verifier": tuple(
                        (
                            inline_resource("tests/test.sh", REPAIRED_TEST_SCRIPT.encode()).model_copy(
                                update={"mode": "0644", "mtime_ns": 0}
                            )
                            if r.path == "tests/test.sh"
                            else r
                        )
                        for r in task.resources.verifier
                    )
                }
            ),
        }
    )
    assert repair_task(clean) is clean
