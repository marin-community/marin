# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the complete frozen BrowseComp population into offline-search contracts."""

import hashlib
import json
from pathlib import Path

from taskcompendium.browsecomp import (
    BROWSECOMP_ACTION_INTERFACE,
    BrowseCompState,
    browsecomp_tools,
)
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
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import EmptyOutputPolicy, JudgeSpec

from .scorers.decode import DEFAULT_CANARY, decrypt_string
from .scorers.prompts import QUERY_TEMPLATE_NO_GET_DOCUMENT

HERE = Path(__file__).parent
QUERY_REPO = "Tevatron/browsecomp-plus"
QUERY_REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"
CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"
CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
INDEX_REPO = "Tevatron/browsecomp-plus-indexes"
INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
CODE_REVISION = "046949032b0328319cc9a02663a759ec601d9402"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VERSION = "2026.11.15.1"
POPULATION = 830
STATE = BrowseCompState(
    code_revision=CODE_REVISION,
    corpus_revision=CORPUS_REVISION,
    index_revision=INDEX_REVISION,
    tokenizer_revision=TOKENIZER_REVISION,
    top_k=5,
    snippet_max_tokens=512,
    include_get_document=False,
)


def convert_query(row: dict[str, str], actor_image: str) -> TaskSpec:
    question = decrypt_string(row["query"], DEFAULT_CANARY)
    answer = decrypt_string(row["answer"], DEFAULT_CANARY)
    query_id = str(row["query_id"])
    task_id = "browsecomp-plus-" + hashlib.sha256(query_id.encode()).hexdigest()[:24]
    # Only the private verifier receives the reference; relevance labels are not agent inputs.
    files = (
        inline_resource("config.json", json.dumps({"question": question, "answer": answer}).encode()),
        inline_resource("browsecomp_grade/__init__.py", b""),
        inline_resource("browsecomp_grade/grade.py", (HERE / "grade.py").read_bytes()),
        inline_resource("browsecomp_grade/scorers/__init__.py", b""),
        inline_resource("browsecomp_grade/scorers/prompts.py", (HERE / "scorers/prompts.py").read_bytes()),
        inline_resource("browsecomp_grade/scorers/judge_parser.py", (HERE / "scorers/judge_parser.py").read_bytes()),
        inline_resource("browsecomp_grade/LICENSE", (HERE / "scorers/LICENSE").read_bytes()),
    )
    return TaskSpec(
        id=task_id,
        source=Source(dataset=QUERY_REPO, revision=QUERY_REVISION, row=query_id, importer_revision=VERSION),
        context=ConversationInput(
            events=(TextMessage(role="user", content=QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=question)),)
        ),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=actor_image,
            working_directory="/app",
            tool_providers={
                "retrieval": ProviderRequirement(
                    action_interface=BROWSECOMP_ACTION_INTERFACE, initial_state=STATE.model_dump(mode="json")
                )
            },
        ),
        interaction_tools=browsecomp_tools(STATE),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        resources=ResourceGroups(verifier=files),
        grader=ScriptGrader(
            argv=("python3", "-m", "browsecomp_grade.grade"),
            cwd="/tests",
            env={"PYTHONPATH": "/tests"},
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image=actor_image,
                capabilities=("network",),
            ),
            answer_path="/app/answer.txt",
            reward=FileReward(files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.VERDICT_JSON),)),
            timeout=60,
            empty_output=EmptyOutputPolicy.GRADE,
            judge=JudgeSpec(references=(answer,), question=question, exact_gate=False, temperature=0.7),
        ),
        tags=("eval", "benchmark:BrowseComp-Plus", "license:mit"),
        labels={"query_id": query_id},
    )
