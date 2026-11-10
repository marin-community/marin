# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert prefix rows mined from pi coding-agent transcripts into one-turn reasoning-markup tasks.

A row holds the chat messages before an assistant reply that pi recorded with a malformed reasoning
span, and the tools pi advertised, both as JSON text so rows with different tool sets share one
Parquet schema. Its other fields are ``benchmark``, ``model`` and ``recorded_violation``. The miner
that produces rows from ``pi.txt`` event logs is archived on marin-logbooks branch
``codex/25-thoughtcrime``.

A task is graded by the ``reasoning_markup`` mode and also carries the ``reject`` markup policy. The
SkyRL worker strips reasoning from the text of a reply without function calls, so the policy's
token-level check covers that runtime while the grader covers runtimes that keep the markers.
The grader scores an empty reply text: a bare function call carries no markup, so it is well formed.
"""

import json

from verifyit.json_objects import unique_object
from verifyit.spec import EmptyOutputPolicy, ReasoningMarkupSpec

from taskcompendium.chat import chat_input
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    ReasoningMarkupPolicy,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow


def prefix_row_task(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """The one-turn task for a mined prefix row, graded on the reasoning markup of its reply."""
    data = row.data
    messages = json.loads(data["messages"], object_pairs_hook=unique_object)
    tools = json.loads(data["tools"], object_pairs_hook=unique_object)
    return TaskSpec(
        id=row.id,
        context=chat_input(messages),
        environment_requirements=EnvironmentRequirements(),
        final_tools=tuple(FunctionDefinition.model_validate(tool) for tool in tools),
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(),
        grader=verifyit_package(ReasoningMarkupSpec(empty_output=EmptyOutputPolicy.GRADE)).grader,
        reasoning_markup=ReasoningMarkupPolicy.REJECT,
        source=row.source,
        tags=(data["benchmark"], data["model"], data["recorded_violation"]),
    )
