# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversation tasks whose final reply a packaged grader scores."""

from collections.abc import Mapping, Sequence
from typing import Any

from taskcompendium.convert.answers import evidence_resource
from taskcompendium.grader import GraderPackage
from taskcompendium.models import (
    AnswerType,
    ConversationEvent,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
)
from taskcompendium.pipeline.models import RawRow


def conversation_task(
    row: RawRow,
    *,
    events: Sequence[ConversationEvent],
    package: GraderPackage,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec:
    """A task whose plain-text reply to ``events`` the packaged grader scores.

    ``evidence`` is source material kept beside the grader for review, never shown to the solver.
    """
    verifier = (*package.resources, *((evidence_resource(evidence),) if evidence else ()))
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=tuple(events)),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(verifier=verifier),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
    )
