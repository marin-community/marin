# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned GSM8K rows without recovering answers heuristically."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from tasktrove_verify.spec import MathType, Mode

from taskcompendium.models import (
    AnswerRequirements,
    Embedded,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskTroveVerifier,
)

DATASET = "openai/gsm8k"
REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
IMPORTER_REVISION = "taskcompendium-gsm8k-v4"
_DELIMITER = re.compile(r"(?m)^####[ \t]+([^\n]+)[ \t]*$")


def _gold(answer: str) -> str | None:
    matches = _DELIMITER.findall(answer)
    if len(matches) != 1 or not matches[0].strip() or _DELIMITER.fullmatch(answer.rstrip().splitlines()[-1]) is None:
        return None
    return matches[0].strip()


def import_row(question: str, answer: str, source: Source) -> TaskSpec | Rejected:
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
    return TaskSpec(
        id=f"gsm8k-{source.row}",
        requirements=TaskRequirements(),
        resources=(Resource("oracle/reasoning.txt", (ResourceRole.ORACLE,), Embedded(answer.encode())),),
        metadata=TaskMetadata(source=source, competencies=("math",), task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=question.strip(),
                verifier=TaskTroveVerifier(Mode.MATH, {"expected": expected, "math_type": MathType.SCALAR}),
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )


@dataclass(frozen=True)
class Gsm8kTaskFamily:
    """Pinned source family; keyed rows are resolved once before any rendering."""

    rows: Mapping[str, tuple[str, str]]

    def instantiate(self, key: str) -> TaskSpec | Rejected:
        question, answer = self.rows[key]
        return import_row(question, answer, Source(DATASET, REVISION, key, IMPORTER_REVISION))
