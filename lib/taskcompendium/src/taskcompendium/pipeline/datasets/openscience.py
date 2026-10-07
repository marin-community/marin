# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated science MCQA normalization with private reference reasoning."""

import json
import re

from verifyit.modes.extract import extract_boxed
from verifyit.spec import McqSpec

from taskcompendium.grading_contract import resolve_verifier
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer

RUBRIC = ReviewRubric(
    id="openscience-quality",
    version="1",
    criteria=(
        "The boxed option is a generated response conclusion, not an independent gold label. Assess each "
        "public option and the private derivation before accepting it.",
        "Check missing scientific context, overlapping options and incorrect causal claims. Specialist "
        "difficulty alone is not a defect.",
        "The MCQ cleanup comparator grades one option letter; source reward parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    prompt, reference = row.data.get("input"), row.data.get("output")
    if not isinstance(prompt, str) or not isinstance(reference, str):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="input and generated output strings are required"
        )
    choices = re.findall(r"(?m)^([A-Z]):", prompt)
    expected = extract_boxed(reference)
    if (
        not isinstance(expected, str)
        or not choices
        or choices != [chr(65 + i) for i in range(len(choices))]
        or expected not in choices
    ):
        return ImportRejection(
            reason="unsupported_choice_contract",
            detail="Contiguous labeled choices and a boxed option conclusion are required",
        )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt + "\n\nReturn one option letter."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        resources=ResourceGroups(
            verifier=(inline_resource("reference/generated-response.json", json.dumps({"output": reference}).encode()),)
        ),
        verifier=multiple_choice_answer(expected, options=len(choices)),
    )


def controls(task: TaskSpec) -> VerificationReport:
    verifier = resolve_verifier(task.verifier)
    assert isinstance(verifier, McqSpec)
    wrong = next(chr(65 + index) for index in range(verifier.options) if chr(65 + index) != verifier.expected)
    return VerificationReport(checks=verify_witness(task, verifier.expected, wrong))


def pipeline() -> TaskPipeline:
    return TaskPipeline(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="openscience-controls", revision="1", parameters={"comparator": "cleanup-mcq"}, run=controls
        ),
    )
