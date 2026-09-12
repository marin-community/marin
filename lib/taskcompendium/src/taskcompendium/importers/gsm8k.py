# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned GSM8K rows without recovering answers heuristically."""

import re

from tasktrove_verify.spec import MathType, Mode

from taskcompendium.models import (
    AnswerRequirements,
    Embedded,
    NoEnvironment,
    PythonRuntime,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    Source,
    TaskMetadata,
    TaskSpecification,
    VerifierSpec,
)

DATASET = "openai/gsm8k"
REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
IMPORTER_REVISION = "taskcompendium-gsm8k-v1"
_DELIMITER = re.compile(r"(?m)^####[ \t]+([^\n]+)[ \t]*$")


def _gold(answer: str) -> str | None:
    matches = _DELIMITER.findall(answer)
    if len(matches) != 1 or not matches[0].strip() or _DELIMITER.fullmatch(answer.rstrip().splitlines()[-1]) is None:
        return None
    return matches[0].strip()


def import_row(question: str, answer: str, source: Source) -> TaskSpecification | Rejected:
    """Convert one GSM8K row, retaining the full rationale as an oracle resource."""
    if source.dataset != DATASET or source.revision != REVISION:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, "GSM8K source is not the pinned revision")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        return Rejected(source, RejectionReason.UNDERSPECIFIED, "GSM8K row has empty question or answer")
    expected = _gold(answer)
    if expected is None:
        return Rejected(
            source, RejectionReason.BROKEN_GRADER, "GSM8K answer must contain exactly one #### gold delimiter"
        )
    return TaskSpecification(
        id=f"gsm8k-{source.row}",
        instructions=question.strip(),
        environment=NoEnvironment(),
        resources=(Resource("oracle/reasoning.txt", (ResourceRole.ORACLE,), Embedded(answer.encode())),),
        verifier=VerifierSpec(Mode.MATH, {"expected": expected, "math_type": MathType.SCALAR}),
        verifier_runtime=PythonRuntime(),
        metadata=TaskMetadata(source=source, competencies=("math",), task_shape="answer"),
        answer_requirements=AnswerRequirements("value"),
    )
