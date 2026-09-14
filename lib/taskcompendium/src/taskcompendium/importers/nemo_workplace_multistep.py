# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An explicit derived Workplace workflow anchored to the pinned source row."""

from dataclasses import dataclass
from pathlib import Path

import msgspec

from taskcompendium.execution import ChatWithTools, HarborTaskBinding, ProviderEnvironment, ProviderToolBinding
from taskcompendium.importers.nemo_workplace import (
    _PROVIDER_IMPORT_PATH,
    DATASET,
    DATASET_REVISION,
    FIXTURE_NAME,
    FIXTURE_SHA256,
    INTERFACE,
    ProviderCall,
    import_row,
)
from taskcompendium.models import (
    AnswerRequirements,
    AssistantFinal,
    ContextRequirement,
    Embedded,
    ProviderStateVerifier,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskSuccessPolicy,
)
from taskcompendium.providers.nemo_workplace.provider import ADAPTER, SEED_SHA256

DERIVATION_VERSION = "taskcompendium-nemo-workplace-derived-multistep-v1"
DERIVED_ROW = "0-derived-multistep-v1"
_SOURCE_ROW_PATH = "source-row.json"
_DERIVATION_PATH = "derivation.json"

_REPLY = ProviderCall(
    "email_reply_email",
    '{"email_id":"00000057","body":"Thanks for the update - I will get back to you tomorrow."}',
)
_CREATE_FOLLOW_UP = ProviderCall(
    "project_management_create_task",
    (
        '{"task_name":"Follow up on report-generation prototype",'
        '"assigned_to_email":"carlos.rodriguez@atlas.com","list_name":"Backlog",'
        '"due_date":"2023-12-04","board":"Front end"}'
    ),
)
_MOVE_TO_IN_PROGRESS = ProviderCall(
    "project_management_update_task",
    '{"task_id":"00000300","field":"list_name","new_value":"In Progress"}',
)
_MOVE_TO_IN_REVIEW = ProviderCall(
    "project_management_update_task",
    '{"task_id":"00000300","field":"list_name","new_value":"In Review"}',
)
_NOTIFY_CARLOS = ProviderCall(
    "email_send_email",
    (
        '{"recipient":"carlos.rodriguez@atlas.com","subject":"Prototype follow-up ready for review",'
        '"body":"The report-generation prototype follow-up is ready for review."}'
    ),
)


@dataclass(frozen=True)
class WorkplaceMultistepSample:
    """One derived, source-pinned workflow plus independent scripted attempts."""

    specification: TaskSpecification
    renderings: tuple[Rendering, ...]
    binding: HarborTaskBinding
    all_good: tuple[tuple[ProviderCall, ...], ...]
    later_wrong: tuple[tuple[ProviderCall, ...], ...]
    missing_step: tuple[tuple[ProviderCall, ...], ...]


def _source() -> Source:
    return Source(DATASET, DATASET_REVISION, DERIVED_ROW, DERIVATION_VERSION)


def _verifier(actions: tuple[ProviderCall, ...]) -> ProviderStateVerifier:
    return ProviderStateVerifier(
        INTERFACE,
        ADAPTER,
        {"ground_truth": [{"name": call.name, "arguments": call.arguments} for call in actions]},
    )


def _derivation(data: bytes) -> bytes:
    return msgspec.json.encode(
        {
            "kind": "explicit_derived_workflow",
            "derived_from": {
                "dataset": DATASET,
                "revision": DATASET_REVISION,
                "row": "0",
                "raw_sha256": FIXTURE_SHA256,
            },
            "description": (
                "This is a new ordered workflow over the original pinned Workplace seed. "
                "It is not an upstream multi-step row. Each private step verifier replays the cumulative action prefix."
            ),
        }
    )


def build_multistep_sample(fixture_root: Path) -> WorkplaceMultistepSample:
    """Build a three-request derived workflow whose later calls require prior provider mutations."""
    data = (fixture_root / FIXTURE_NAME).read_bytes()
    source_specification = import_row(data)
    if not isinstance(source_specification, TaskSpecification):
        raise ValueError(source_specification.detail)
    all_good = ((_REPLY, _CREATE_FOLLOW_UP), (_MOVE_TO_IN_PROGRESS,), (_MOVE_TO_IN_REVIEW, _NOTIFY_CARLOS))
    cumulative = (
        (_REPLY, _CREATE_FOLLOW_UP),
        (_REPLY, _CREATE_FOLLOW_UP, _MOVE_TO_IN_PROGRESS),
        (_REPLY, _CREATE_FOLLOW_UP, _MOVE_TO_IN_PROGRESS, _MOVE_TO_IN_REVIEW, _NOTIFY_CARLOS),
    )
    steps = (
        StepSpecification(
            instructions=(
                "Reply to Carlos's last email about 'Task Update on Develop prototype for report generation' "
                "with 'Thanks for the update - I will get back to you tomorrow.' Then create a Front end Backlog "
                "task called 'Follow up on report-generation prototype' for Carlos, due 2023-12-04."
            ),
            verifier=_verifier(cumulative[0]),
            answer_requirements=AnswerRequirements("text"),
        ),
        StepSpecification(
            instructions="The prototype work is underway. Move the follow-up task you just created to In Progress.",
            verifier=_verifier(cumulative[1]),
            answer_requirements=AnswerRequirements("text"),
            context_requirement=ContextRequirement.PRIOR_CONVERSATION,
        ),
        StepSpecification(
            instructions=(
                "The work is ready for review. Move that same follow-up task to In Review and let Carlos know "
                "by email with subject 'Prototype follow-up ready for review' and body "
                "'The report-generation prototype follow-up is ready for review.'"
            ),
            verifier=_verifier(cumulative[2]),
            answer_requirements=AnswerRequirements("text"),
            context_requirement=ContextRequirement.PRIOR_CONVERSATION,
        ),
    )
    specification = TaskSpecification(
        id="nemo/workplace/0-derived-multistep",
        requirements=TaskRequirements(action_interfaces=(INTERFACE,)),
        resources=(
            Resource(_SOURCE_ROW_PATH, (ResourceRole.VERIFIER,), Embedded(data)),
            Resource(_DERIVATION_PATH, (ResourceRole.VERIFIER,), Embedded(_derivation(data))),
        ),
        metadata=TaskMetadata(
            _source(), competencies=("stateful_tool_use", "multi_step_workflow"), task_shape="derived_stateful_domain"
        ),
        success_policy=TaskSuccessPolicy.MEAN,
        steps=steps,
    )
    return WorkplaceMultistepSample(
        specification=specification,
        renderings=(Rendering("nemo-workplace-chat", AssistantFinal()),) * len(steps),
        binding=HarborTaskBinding(
            ProviderEnvironment(
                INTERFACE,
                _PROVIDER_IMPORT_PATH,
                {"interface": msgspec.to_builtins(INTERFACE), "seed_sha256": SEED_SHA256},
            ),
            ChatWithTools((ProviderToolBinding(INTERFACE.name),)),
            context="conversation",
        ),
        all_good=all_good,
        later_wrong=(
            (_REPLY, _CREATE_FOLLOW_UP),
            (
                ProviderCall(
                    "project_management_update_task",
                    '{"task_id":"00000300","field":"list_name","new_value":"Completed"}',
                ),
            ),
            (_MOVE_TO_IN_REVIEW, _NOTIFY_CARLOS),
        ),
        missing_step=((_REPLY, _CREATE_FOLLOW_UP), (), (_MOVE_TO_IN_REVIEW, _NOTIFY_CARLOS)),
    )
