# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small generated shell fixtures and a converter for exported TaskTrove captures."""

import base64
from collections.abc import Iterator
from typing import Any

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentFixture,
    EnvironmentRequirements,
    FunctionDefinition,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    GeneratedSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)
from taskcompendium.verifiers.runtime import CaptureOutputVerifier

OUTPUT_PATH = "/output/command_capture.txt"
CONTROL_PATH = "/controls/reference.sh"
BASH = FunctionDefinition(
    name="Bash",
    description="Run a shell command in /workspace",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
)
RUBRIC = ReviewRubric(
    "shell-capture",
    "1",
    (
        "Check that the public files and setup instructions suffice to run the requested command.",
        "Grade the capture file; an assistant's final text is not the submitted artifact.",
        "The inherited grader compares normalized output records, tolerates non-error extras, and ignores ordering.",
        "Flag tasks that request ordering or exact output when that inherited comparator would ignore it.",
        "Private reference scripts and expected output must not be supplied to the solving actor.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    data = row.data
    required = ("instruction", "reference_script", "expected_output")
    if any(not isinstance(data.get(key), str) for key in required) or not data["instruction"].strip():
        return ImportRejection(
            reason="malformed_capture", detail="Instruction, reference script and expected output are required"
        )
    files = data.get("public_files", {})
    controls = data.get("control_files", {})
    if not isinstance(files, dict) or not isinstance(controls, dict):
        return ImportRejection(reason="malformed_files", detail="File maps must contain base64 bytes by absolute path")
    if any(
        not isinstance(path, str) or not isinstance(encoded, str)
        for mapping in (files, controls)
        for path, encoded in mapping.items()
    ):
        return ImportRejection(reason="malformed_files", detail="File paths and base64 content must be strings")
    resources = [task_resource(CONTROL_PATH, data["reference_script"].encode(), ResourceVisibility.CONTROL)]
    try:
        for visibility, mapping in ((ResourceVisibility.AGENT, files), (ResourceVisibility.CONTROL, controls)):
            resources.extend(
                task_resource(path, base64.b64decode(encoded, validate=True), visibility)
                for path, encoded in mapping.items()
            )
        verifier = CaptureOutputVerifier(output_path=OUTPUT_PATH, expected_output=data["expected_output"])
        return TaskSpec(
            id=row.id,
            context=ConversationInput(events=(TextMessage(role="user", content=data["instruction"]),)),
            environment_requirements=EnvironmentRequirements(
                capabilities=("shell", "filesystem"), action_interfaces=("shell:v1",)
            ),
            interaction_tools=(BASH,),
            fixture=EnvironmentFixture(interface="shell:v1", revision="1", initial_state_json="{}"),
            resources=tuple(resources),
            output_paths=(OUTPUT_PATH,),
            answer_type=AnswerType.FILE,
            verifier=VerifierSpec(kind=VerifierKind.CAPTURE_OUTPUT, parameters_json=verifier.model_dump_json()),
            source=row.source,
        )
    except ValueError as error:
        return ImportRejection(reason="malformed_files", detail=str(error))


def generate_rows(limit: int) -> Iterator[dict[str, Any]]:
    for index in range(limit):
        name = f"team-{index}"
        table = "name,team\n" + "".join(
            f"person-{index}-{entry},{name if entry % 2 == 0 else 'other'}\n" for entry in range(4 + index)
        )
        # Select records rather than requiring ordering, which the inherited grader ignores.
        expected = "".join(f"person-{index}-{entry}\n" for entry in range(4 + index) if entry % 2 == 0)
        command = f"awk -F, '$2 == \"{name}\" {{print $1}}' /workspace/people.csv"
        yield {
            "instruction": (
                f"Read /workspace/people.csv. List the names belonging to {name}, one per line. "
                f"Write command stdout and stderr to {OUTPUT_PATH}."
            ),
            "public_files": {"/workspace/people.csv": base64.b64encode(table.encode()).decode()},
            "reference_script": f"#!/bin/bash\nset -eu\n{command} > {OUTPUT_PATH} 2>&1\n",
            "expected_output": expected,
        }


recipe = DatasetRecipe(
    name="shell-files-mock",
    version="shell-files-v1",
    source=GeneratedSource("mock/shell-files", "1", "default", "train", __name__),
    inputs=RecipeInputs(SourceFiles(("*.jsonl",), SourceFormat.JSONL), ()),
    normalize=normalize,
    rubric=RUBRIC,
    intended_use=IntendedUse.TRAIN,
)
