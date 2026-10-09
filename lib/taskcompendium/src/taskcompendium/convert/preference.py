# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preference rows as review-only conversation tasks.

A preference label ranks the source's own candidate replies; scoring a new reply would need a
reward model. These tasks therefore carry a ``NoGrader`` whose contract keeps the labeled
candidates, and the public context holds only the conversation before the final candidate.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue, ValidationError

from taskcompendium.chat import chat_conversation
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.grader import GraderPackage
from taskcompendium.models import ConversationEvent, ConversationInput, NoGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.models import ImportRejection, RawRow

NO_REWARD_MODEL = "Preference labels rank source candidates; scoring a new reply needs a reward model"


def _preference_task(
    row: RawRow,
    events: Sequence[ConversationEvent],
    contract: dict[str, JsonValue],
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec:
    package = GraderPackage(NoGrader(reason=NO_REWARD_MODEL, contract=contract))
    return conversation_task(row, events=events, package=package, evidence=evidence)


def pairwise_preference_task(
    row: RawRow, *, chosen: Sequence[TextMessage], rejected: Sequence[TextMessage]
) -> TaskSpec | ImportRejection:
    """Keep the shared history public and both final candidates in the grader contract.

    Both transcripts must end in an assistant candidate after the same history, and that history
    must end in a user request.
    """
    if not chosen or not rejected or chosen[-1].role != "assistant" or rejected[-1].role != "assistant":
        return source_defect("missing_preference_completion", "Both transcripts must end in assistant turns")
    if tuple(chosen[:-1]) != tuple(rejected[:-1]):
        return source_defect("preference_prompt_conflict", "Candidates have different public histories")
    if len(chosen) < 2 or chosen[-2].role != "user":
        return source_defect("invalid_preference_prompt", "Public history must end in a user request")
    contract: dict[str, JsonValue] = {
        "kind": "pairwise",
        "chosen": [chosen[-1].model_dump(mode="json")],
        "rejected": [rejected[-1].model_dump(mode="json")],
    }
    return _preference_task(row, chosen[:-1], contract)


def binary_preference_task(
    row: RawRow,
    *,
    prompt: Any,
    completion: Any,
    label: Any,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec | ImportRejection:
    """Keep an unpaired labeled completion in the grader contract instead of inventing a partner.

    ``prompt`` and ``completion`` are chat message lists and ``label`` is a boolean; anything else
    is a source defect.
    """
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return source_defect("invalid_binary_preference", "Prompt and completion must be chat message lists")
    try:
        transcript = chat_conversation(prompt + completion)
        context = ConversationInput(events=transcript.events[: len(prompt)])
        if not isinstance(label, bool):
            raise ValueError("Preference label must be a boolean")
        contract: dict[str, JsonValue] = {
            "kind": "binary",
            "response": [event.model_dump(mode="json") for event in transcript.events[len(prompt) :]],
            "preferred": label,
        }
    except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as error:
        return source_defect("invalid_binary_preference", str(error))
    if not completion:
        return source_defect("missing_preference_messages", "Public prompt and labeled completion required")
    return _preference_task(row, context.events, contract, evidence)
