# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove MultiChallenge: the next assistant turn of a conversation, graded by an LLM judge on the source's checklist.

Each archive's ``tests/judge.toml`` lists yes/no requirements that the source's RewardKit suite put
to a Together-hosted judge, passing a response only when every requirement held. The task asks
verifyit's checklist judge about each requirement separately, with the conversation
(``tests/conversation.txt``) as reference context, and scores the fraction met. The source's judge
configuration stays beside the grader for review. Verification cannot reach a judge endpoint, so the
source has no controls and its kept rows are admitted without them.
"""

import tomllib
from dataclasses import replace

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasktrove import archive_file
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import RUBRIC_CHECKLIST, JudgeSpec

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.judged import REWRITE_REASON, response_instruction
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = DataSourceMetadata(id="", name="", origin="Task Trove", recorded_at="2026-10-08")

CONFIG = "laion__nemotron-gym-multichallenge-advanced-v4"
CONTEXT_FILE = "conversation.txt"
REQUIREMENT_MARKER = "\nRequirement:\n"
"""Separates each source criterion's instructions about RewardKit's attached files from the requirement itself."""

RUBRIC = """
Read the full persona and conversation; grade only the requested next response, not a historical turn.

Compare every hidden criterion with the public conversation and final user request, including earlier rules.

Preserve negated criterion polarity. The source passed a response only when every criterion held; this grader scores
the fraction of criteria met.

Reject conflicting required formats, absent context, and invented checklist conditions.

Judge availability is a verification limitation; no canonical response should be fabricated.
"""


def source_requirements(configuration: dict) -> tuple[str, ...]:
    """The requirement each ``[[criterion]]`` of a source ``judge.toml`` asks about."""
    descriptions = (criterion.get("description") for criterion in configuration.get("criterion", []))
    return tuple(
        description.rpartition(REQUIREMENT_MARKER)[2].strip() if isinstance(description, str) else ""
        for description in descriptions
    )


def convert_multichallenge(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    judge_toml, conversation = archive_file(row.data, "tests/judge.toml"), archive_file(
        row.data, "tests/conversation.txt"
    )
    if judge_toml is None or conversation is None:
        return unsupported("missing_judge_config", "The source judge.toml and conversation.txt are required")
    try:
        configuration = tomllib.loads(judge_toml.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return source_defect("malformed_judge_config", f"judge.toml does not parse: {type(error).__name__}")
    criteria = source_requirements(configuration)
    if not criteria or not all(criteria):
        return source_defect("empty_criteria", "The source provides no complete checklist criteria")
    instruction = row.data["instruction"]
    if not instruction.strip():
        return source_defect("missing_instruction", "Public instruction is required")
    package = verifyit_package(
        JudgeSpec(criteria=criteria, context=CONTEXT_FILE, rubric=RUBRIC_CHECKLIST),
        (inline_resource(CONTEXT_FILE, conversation), inline_resource("source/judge.toml", judge_toml)),
        environment=required_grader_environment(context),
    )
    prompt = TextMessage(role="user", content=response_instruction(instruction))
    task = conversation_task(row, events=(prompt,), package=package)
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-multichallenge-advanced-v4",
                name="laion__nemotron-gym-multichallenge-advanced-v4",
                display_name="laion/nemotron-gym-multichallenge-advanced-v4",
                url="https://huggingface.co/datasets/open-athena/task-trove",
                dataset_id="open-athena/task-trove",
                revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
                revised_at="2026-10-08T09:34:47.000Z",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="llm-judge-freeform",
                environment="Harbor",
                type="Agentic",
                turns="Multi-turn",
                task_count=1052,
                count_basis="Released Harbor tasks: manifest by_source.converted",
                count_precision="exact",
                count_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                notes=(
                    "Multi-turn with four factual or formatting criteria per task. The converter "
                    "preserves the source's explicit positive or negated pass condition."
                ),
                benchmark_basis="Release manifest does not designate benchmarks",
                family_basis="Task Trove release manifest source_verdicts.family",
                family_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
                canonical_source="laion/nemotron-gym-multichallenge-advanced-v4",
                canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
                provenance_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5"
                    "bbccb803e826563f5718dbf/manifest.json"
                ),
                verification="judge",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-multichallenge-advanced-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-multichallenge-advanced-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1068,
                modes=("judge",),
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-multichallenge",
                source=tasktrove_source(CONFIG),
                convert=convert_multichallenge,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                grader=GRADER_PACKAGES,
            ),
        )
    ]
