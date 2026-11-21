# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert grounded research questions into separate actor and private grader contracts."""

import hashlib
import json
from pathlib import Path

from taskcompendium.browsecomp import browsecomp_tools
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import EmptyOutputPolicy, JudgeSpec

from experiments.post_training.task_curation.datasets.openresearcher.assets import DATASET, DATASET_REVISION
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import CONTRACT_SHA256

VERSION = "2026.11.21.1"
SOURCE_ID = "OpenResearcher:OpenResearcher-Dataset"
GRADER_ROOT = Path(__file__).parent.parent / "browsecomp"
RESEARCH_INSTRUCTION = """Answer the question using the offline search and get_document tools.
Find evidence for every necessary clue and reasoning step. Cite supporting document IDs in the explanation.
Return your response in this format:
Explanation: your explanation with supporting document IDs in square brackets
Exact Answer: your succinct final answer
Confidence: your confidence from 0% to 100%

Question:
{question}"""


def convert_question(
    row: dict[str, str], *, state: OpenResearcherState, actor_image: str, grader_image: str
) -> TaskSpec:
    """Lower one retained QA pair; the preparation entrypoint enforces grounding and heldout evidence."""
    question, answer = row["question"], row["answer"]
    if not question.strip() or not answer.strip():
        raise ValueError("Grounded research contracts require a question and private answer")
    # Actor-visible identities depend only on public question text, never on the gold answer.
    question_id = hashlib.sha256(question.encode()).hexdigest()
    references = json.loads(row["source_refs"])
    if not references:
        raise ValueError("Research questions require their released source row references")
    original_row = references[0]
    files = (
        inline_resource("config.json", json.dumps({"question": question, "answer": answer}).encode()),
        inline_resource("research_grade/__init__.py", b""),
        inline_resource("research_grade/grade.py", (GRADER_ROOT / "grade.py").read_bytes()),
        inline_resource("research_grade/scorers/__init__.py", b""),
        inline_resource("research_grade/scorers/prompts.py", (GRADER_ROOT / "scorers/prompts.py").read_bytes()),
        inline_resource(
            "research_grade/scorers/judge_parser.py", (GRADER_ROOT / "scorers/judge_parser.py").read_bytes()
        ),
        inline_resource("research_grade/LICENSE", (GRADER_ROOT / "scorers/LICENSE").read_bytes()),
        inline_resource("source-refs.json", row["source_refs"].encode()),
    )
    return TaskSpec(
        id="openresearcher-" + question_id[:24],
        source=Source(
            dataset=DATASET,
            revision=DATASET_REVISION,
            row=f"{original_row['file']}:{original_row['row']}",
            importer_revision=VERSION,
        ),
        context=ConversationInput(
            events=(TextMessage(role="user", content=RESEARCH_INSTRUCTION.format(question=question)),)
        ),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=actor_image,
            working_directory="/app",
            tool_providers={
                "retrieval": ProviderRequirement(
                    action_interface=OPENRESEARCHER_ACTION_INTERFACE,
                    initial_state=state.model_dump(mode="json"),
                    network_access="service_only",
                )
            },
        ),
        interaction_tools=browsecomp_tools(state),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        resources=ResourceGroups(verifier=files),
        grader=ScriptGrader(
            argv=("python3", "-m", "research_grade.grade"),
            cwd="/tests",
            env={"PYTHONPATH": "/tests"},
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=grader_image,
                capabilities=("network",),
            ),
            answer_path="/app/answer.txt",
            reward=FileReward(files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.VERDICT_JSON),)),
            timeout=60,
            empty_output=EmptyOutputPolicy.GRADE,
            judge=JudgeSpec(references=(answer,), question=question, exact_gate=False, temperature=0.7),
        ),
        tags=("train", "license:mit", "corpus-license:odc-by", "environment:offline-bm25"),
        labels={"grounding_contract_sha256": CONTRACT_SHA256, "index_manifest_sha256": state.index_manifest_sha256},
    )
