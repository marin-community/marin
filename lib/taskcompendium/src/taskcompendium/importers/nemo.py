# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import bounded NeMo Gym answer-only source rows without exposing their verifiers."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from tasktrove_verify.spec import Mode

from taskcompendium.models import (
    AnswerRequirements,
    CodeAnswerVerifier,
    ConstraintVerifier,
    ContainerRuntime,
    Embedded,
    InstructionConstraint,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskTroveVerifier,
)

GYM_REVISION = "1e668906d2e69a9e8ee9aaafc60050a4025d9688"
IFEVAL_DATASET = "nvidia/Nemotron-RL-instruction_following"
IFEVAL_REVISION = "3b253899665cb71334bb54c14eb5d91751beaad7"
CODE_DATASET = "nvidia/Nemotron-RL-coding-competitive_coding"
CODE_REVISION = "ae1f446f299823ea3c4c00217942b53787278b31"
IMPORTER_REVISION = "taskcompendium-nemo-v0.2"
_CHECKER_PATH = "check_code_answer.py"
_SOURCE_ROW_PATH = "source-row.json"
_SOURCE_PROVENANCE_PATH = "source-provenance.json"
_CODE_PROMPT_PREFIX = (
    "You are a helpful and harmless assistant. You should think step-by-step before responding to the "
    "instruction below.\n\n"
    "Please use python programming language only.\n\n"
    "You must use ```python for just the final solution code block with the following format:\n"
    "```python\n# Your code here\n```\n\n"
)

# This checker is an intentionally bounded replacement for the source's remote
# LiveCodeBench worker. It supports only the selected standard-input shape.
# The raw source row stays verifier-only; packed cases are decoded only inside
# the isolated verifier after the model response has been materialized.
_CODE_CHECKER = b"""import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib


def cases():
    root = Path(os.environ["TASKTROVE_TESTS_DIR"])
    row = json.loads((root / "source-row.json").read_text())
    unit_tests = row["verifier_metadata"]["unit_tests"]
    if "packed" in unit_tests:
        packed = base64.b64decode(unit_tests["packed"], validate=True)
        unit_tests = json.loads(zlib.decompress(packed))
    inputs = unit_tests["inputs"]
    outputs = unit_tests["outputs"]
    if unit_tests.get("fn_name") is not None or not inputs or len(inputs) != len(outputs):
        raise ValueError("unsupported LiveCodeBench unit-test shape")
    if not all(isinstance(value, str) for value in (*inputs, *outputs)):
        raise ValueError("unit-test inputs and outputs must be strings")
    return tuple(zip(inputs, outputs, strict=True))


def matches(actual, expected):
    actual_tokens = actual.strip().split()
    expected_tokens = expected.strip().split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    for got, want in zip(actual_tokens, expected_tokens, strict=True):
        if got == want:
            continue
        try:
            if abs(float(got) - float(want)) <= 1e-6:
                continue
        except ValueError:
            pass
        return False
    return True


def main():
    workspace = Path(os.environ["TASKTROVE_WORKSPACE"])
    code = (workspace / "answer.txt").read_text()
    if not code.strip():
        print(0)
        return
    candidate_environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    for task_input, expected in cases():
        try:
            run = subprocess.run(
                [sys.executable, "-I", "-c", code],
                input=task_input,
                text=True,
                capture_output=True,
                timeout=10,
                cwd=workspace,
                env=candidate_environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            print(0)
            return
        if run.returncode or not matches(run.stdout, expected):
            print(0)
            return
    print(1)


if __name__ == "__main__":
    main()
"""


def _row(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("source record must be a JSON object")
    return value


def _prompt(row: Mapping[str, Any]) -> str:
    request = row.get("responses_create_params")
    if not isinstance(request, Mapping):
        raise ValueError("source record has no responses request")
    messages = request.get("input")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("bounded importer requires one source input message")
    message = messages[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise ValueError("bounded importer requires one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("source user message is empty")
    if request.get("tools", []) != []:
        raise ValueError("answer-only importer does not accept source tool declarations")
    return content


def _metadata(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    nested = row.get("verifier_metadata")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("verifier_metadata must be an object")
        if field in nested:
            return nested
    return row


def _code_prompt(prompt: str) -> str:
    """Keep source problem semantics while leaving answer representation to rendering."""
    if not prompt.startswith(_CODE_PROMPT_PREFIX):
        raise ValueError("unsupported code source prompt template")
    return "Write a Python solution for the following problem.\n\n" + prompt.removeprefix(_CODE_PROMPT_PREFIX)


def _import_instruction_following(
    data: bytes,
    *,
    aggregation: Literal["binary", "fraction"] | None = None,
    source_row: str | None = None,
    source_provenance: dict[str, str] | None = None,
) -> TaskSpecification | Rejected:
    """Convert one NeMo Gym IFEval record, preserving source aggregation semantics."""
    try:
        row = _row(data)
        identifier = row["id"]
        if not isinstance(identifier, int):
            raise ValueError("IFEval record id must be an integer")
        prompt = _prompt(row)
        verifier_data = _metadata(row, "instruction_id_list")
        names = verifier_data["instruction_id_list"]
        values = verifier_data["kwargs"]
        source_aggregation = verifier_data.get("grading_mode", "binary")
        if source_aggregation not in {"binary", "fraction"}:
            raise ValueError("IFEval grading_mode must be binary or fraction")
        effective_aggregation = aggregation or source_aggregation
        if not isinstance(names, list) or not isinstance(values, list) or not names or len(names) != len(values):
            raise ValueError("IFEval constraints and kwargs must be nonempty parallel lists")
        constraints: list[InstructionConstraint] = []
        for name, params in zip(names, values, strict=True):
            if not isinstance(name, str) or not name:
                raise ValueError("IFEval constraint names must be nonempty strings")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError("IFEval kwargs must be objects or null")
            constraints.append(InstructionConstraint(name, params))
        source = Source(IFEVAL_DATASET, IFEVAL_REVISION, source_row or str(identifier), IMPORTER_REVISION)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        source = Source(IFEVAL_DATASET, IFEVAL_REVISION, "unparseable", IMPORTER_REVISION)
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    resources = [Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data))]
    if source_provenance is not None:
        resources.append(
            Resource(
                _SOURCE_PROVENANCE_PATH,
                (ResourceRole.VERIFIER,),
                Embedded(json.dumps(source_provenance, sort_keys=True, separators=(",", ":")).encode()),
            )
        )
    return TaskSpecification(
        id=f"nemo/ifeval/{identifier}/{effective_aggregation}",
        requirements=TaskRequirements(),
        resources=tuple(resources),
        metadata=TaskMetadata(source, competencies=("instruction-following",), task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=prompt,
                verifier=ConstraintVerifier(tuple(constraints), effective_aggregation),
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


def import_instruction_following(
    data: bytes, *, aggregation: Literal["binary", "fraction"] | None = None
) -> TaskSpecification | Rejected:
    """Convert one NeMo Gym IFEval record, preserving source aggregation semantics."""
    return _import_instruction_following(data, aggregation=aggregation)


def import_hub_instruction_row(
    data: bytes, *, split: str, offset: int, aggregation: Literal["binary", "fraction"] | None = None
) -> TaskSpecification | Rejected:
    """Convert a selected IFEval Hub row while retaining its split and offset privately."""
    return _import_instruction_following(
        data,
        aggregation=aggregation,
        source_row=str(offset),
        source_provenance={
            "dataset": IFEVAL_DATASET,
            "revision": IFEVAL_REVISION,
            "split": split,
            "offset": str(offset),
        },
    )


def _import_code_answer(
    data: bytes,
    *,
    verifier_image: str | None,
    source_row: str | None = None,
    source_provenance: dict[str, str] | None = None,
) -> TaskSpecification | Rejected:
    """Convert one NeMo Gym code response task to an isolated private checker."""
    try:
        row = _row(data)
        identifier = row["hash_id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("code record hash_id must be a nonempty string")
        prompt = _code_prompt(_prompt(row))
        verifier_data = _metadata(row, "unit_tests")
        unit_tests = verifier_data["unit_tests"]
        if not isinstance(unit_tests, dict):
            raise ValueError("unit_tests must be an object")
        if "packed" in unit_tests:
            if not isinstance(unit_tests["packed"], str) or not unit_tests["packed"]:
                raise ValueError("packed unit_tests must be nonempty base64 text")
        else:
            inputs = unit_tests.get("inputs")
            outputs = unit_tests.get("outputs")
            if (
                not isinstance(inputs, list)
                or not isinstance(outputs, list)
                or not inputs
                or len(inputs) != len(outputs)
                or not all(isinstance(value, str) for value in (*inputs, *outputs))
            ):
                raise ValueError("unit_tests must contain parallel nonempty string inputs and outputs")
        if unit_tests.get("fn_name") is not None:
            raise ValueError("bounded code importer supports standard-input unit tests only")
        if verifier_image is None:
            raise LookupError("an immutable isolated verifier image must be supplied by the caller")
        runtime = ContainerRuntime(verifier_image)
        source = Source(CODE_DATASET, CODE_REVISION, source_row or identifier, IMPORTER_REVISION)
    except LookupError as error:
        source = Source(CODE_DATASET, CODE_REVISION, "unparseable", IMPORTER_REVISION)
        return Rejected(source, RejectionReason.UNSUPPORTED_ENVIRONMENT, str(error))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        source = Source(CODE_DATASET, CODE_REVISION, "unparseable", IMPORTER_REVISION)
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    verifier = CodeAnswerVerifier(
        TaskTroveVerifier(Mode.SCRIPT, {"path": _CHECKER_PATH, "timeout": 60.0}, runtime=runtime), "answer.txt"
    )
    resources = [
        Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data)),
        Resource(_CHECKER_PATH, (ResourceRole.VERIFIER,), Embedded(_CODE_CHECKER), executable=True),
    ]
    if source_provenance is not None:
        resources.append(
            Resource(
                _SOURCE_PROVENANCE_PATH,
                (ResourceRole.VERIFIER,),
                Embedded(json.dumps(source_provenance, sort_keys=True, separators=(",", ":")).encode()),
            )
        )
    return TaskSpecification(
        id=f"nemo/code-answer/{identifier}",
        requirements=TaskRequirements(),
        resources=tuple(resources),
        metadata=TaskMetadata(source, competencies=("competitive-programming",), task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=prompt,
                verifier=verifier,
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


def import_code_answer(data: bytes, *, verifier_image: str | None = None) -> TaskSpecification | Rejected:
    """Convert one pinned NeMo Gym code response task to an isolated private checker."""
    return _import_code_answer(data, verifier_image=verifier_image)


def import_hub_code_row(
    data: bytes, *, split: str, offset: int, verifier_image: str | None = None
) -> TaskSpecification | Rejected:
    """Convert a selected Hub row while retaining its split and offset privately."""
    return _import_code_answer(
        data,
        verifier_image=verifier_image,
        source_row=str(offset),
        source_provenance={"dataset": CODE_DATASET, "revision": CODE_REVISION, "split": split, "offset": str(offset)},
    )


def load_instruction_sample(
    path: Path, *, aggregation: Literal["binary", "fraction"] | None = None
) -> TaskSpecification | Rejected:
    """Load one already-pinned instruction-following row for a build or test."""
    return import_instruction_following(path.read_bytes(), aggregation=aggregation)


def load_code_sample(path: Path, *, verifier_image: str | None = None) -> TaskSpecification | Rejected:
    """Load one already-pinned code-answer row for a build or test."""
    return import_code_answer(path.read_bytes(), verifier_image=verifier_image)
