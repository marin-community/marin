# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic generation archives preserve failures and consumer identities."""

import json

from harbor.verifier.base import BaseVerifier
from tasktrove_verify.spec import Mode

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig, HarborTaskBinding, NoEnvironment
from taskcompendium.harbor.generation import GenerationRequest, generate_attempts, write_attempts
from taskcompendium.importers.sequential import sentence_revision_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    JsonPath,
    Outcome,
    Rendering,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskTroveVerifier,
)


class UnavailableVerifier(BaseVerifier):
    async def verify(self):
        raise RuntimeError("Verifier service unavailable")


async def test_mixed_generation_retains_correct_wrong_malformed_and_launch_failure(tmp_path):
    specification = TaskSpec(
        id="generation/exact",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(Source("generation-fixture", "v1", "0", "v1")),
        steps=(
            StepSpecification(
                instructions="Reply with the word blue.", verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["blue"]})
            ),
        ),
    )
    binding = HarborTaskBinding(NoEnvironment())
    requests = []
    for index, response in enumerate(['{"answer":"blue"}', '{"answer":"red"}', "not JSON"]):
        task = lower_to_harbor(
            specification,
            (Rendering("json", AssistantFinal(JsonPath())),),
            binding,
            tmp_path / f"task-{index}",
            reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
            agent_kwargs={"response": response},
        )
        requests.append(
            GenerationRequest(task, json.loads((task / "reference-execution.json").read_text()), "example", index)
        )
    requests.append(GenerationRequest(tmp_path / "missing-package", {}, "missing", 0))

    attempts = await generate_attempts(requests, tmp_path / "trials", concurrency=2)
    archive = tmp_path / "attempts.jsonl"
    write_attempts(attempts, archive)
    records = [json.loads(line) for line in archive.read_text().splitlines()]

    assert [(attempt.status, attempt.reward) for attempt in attempts] == [
        (Outcome.GRADED, 1.0),
        (Outcome.GRADED, 0.0),
        (Outcome.EXTRACTION_ERROR, None),
        (Outcome.INFRA_ERROR, None),
    ]
    assert [(record["instance_id"], record["repetition_id"]) for record in records] == [
        ("example", 0),
        ("example", 1),
        ("example", 2),
        ("missing", 0),
    ]
    assert records[2]["reward"] is None and records[3]["reward"] is None
    assert records[3]["exception"]["exception_type"] == "FileNotFoundError"
    assert records[0]["source"]["dataset"] == "generation-fixture"
    assert records[0]["messages"][-1]["content"] == '{"answer":"blue"}'
    assert len({record["trial_dir"] for record in records}) == 4


async def test_ordered_verifier_crash_retains_named_missing_results_and_step_exception(tmp_path):
    specification = sentence_revision_task()
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (Rendering("first", AssistantFinal()), Rendering("second", AssistantFinal())),
        binding,
        tmp_path / "ordered",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
        agent_kwargs={
            "steps": [
                {"response": "Mira will meet Leo on Tuesday.", "commands": []},
                {"response": "Mira will meet Leo on Thursday.", "commands": []},
            ]
        },
    )
    execution = json.loads((task / "reference-execution.json").read_text())
    execution["verifier"] = {"import_path": f"{__name__}:UnavailableVerifier"}
    attempts = await generate_attempts(
        [GenerationRequest(task, execution, "ordered", 0)], tmp_path / "trials", concurrency=1
    )
    write_attempts(attempts, tmp_path / "attempts.jsonl")
    record = json.loads((tmp_path / "attempts.jsonl").read_text())

    assert record["status"] == "infra_error" and record["reward"] is None
    assert record["step_names"] == ["step-1", "step-2"]
    assert record["step_results"] == [None, None]
    assert record["step_exceptions"][0]["exception_type"] == "RuntimeError"
    assert record["exception"]["exception_type"] == "RuntimeError"
    assert record["messages"][1]["content"] == "Mira will meet Leo on Tuesday."
