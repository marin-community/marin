# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove MultiChallenge: the next assistant turn of a conversation, graded by the source's RewardKit judge.

Each archive's ``tests/judge.toml`` lists yes/no criteria that RewardKit asks a Together-hosted
judge about and aggregates all-pass. ``multichallenge_grade.py`` runs the source's own
``tests/test.sh`` in the RewardKit image and reports its reward. Only the observed runtime files,
test layout, provider settings and judge scope are accepted, so the source files cannot point the
judge at another provider or at other files. The judge needs network access and a provider key,
so the source has no offline controls.
"""

import hashlib
import tomllib
from pathlib import Path

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasktrove import archive_files
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import ScriptSpec

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.judged import REWRITE_REASON, response_instruction
from experiments.post_training.task_curation.datasets.tasktrove.multichallenge_grade import (
    SOURCE_JUDGE,
    SOURCE_TIMEOUT,
    VERDICT_FILENAME,
)
from experiments.post_training.task_curation.images import REWARDKIT_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

GRADE_SCRIPT = "multichallenge_grade.py"
GRADE_PATH = f"__runtime/{GRADE_SCRIPT}"
"""RewardKit scans visible child directories of the tests directory, so added files sit under ``__`` names."""
GRADER_TIMEOUT = SOURCE_TIMEOUT + 30.0
RUNTIME_FILES = {
    "tests/test.sh": "db2681b4a2e86cdfd2699e2b12fd14806cf98040473a9e8956c570c78c99b97b",
    "tests/sitecustomize.py": "87e2c2c69264159f7d755442320a8f048f0e9687a2fba5cc8873433736ca1628",
    "environment/Dockerfile": "7f92fb2192bd4721685459bfc1b18954b68dcdb383f3be24f9bb53d649e00e65",
}
SOURCE_TEST_FILES = frozenset({"test.sh", "sitecustomize.py", "judge.toml", "conversation.txt", "verifier_data.json"})
PROVIDER_SETTINGS = {"REWARDKIT_JUDGE": SOURCE_JUDGE, "TOGETHER_API_KEY": "${TOGETHER_API_KEY}"}
JUDGE_SETTINGS = {
    "judge": SOURCE_JUDGE,
    "files": ["/tests/conversation.txt", "/app/response.txt"],
    "mode": "individual",
    "reasoning_effort": "low",
    "timeout": 300,
}
CRITERION_KEYS = frozenset({"name", "description", "type", "min", "max"})

RUBRIC = """
Read the full persona and conversation; grade only the requested next response, not a historical turn.

Compare every hidden criterion with the public conversation and final user request, including earlier rules.

Preserve negated criterion polarity and the source aggregation rule; all-pass is not a mean score.

RewardKit 0.1.4 all_pass requires every normalized criterion score to be greater than zero. For numeric criteria, even
a small positive score passes; flag this permissive threshold without changing it.

Reject conflicting required formats, absent context, and invented checklist conditions.

Judge availability is a verification limitation; no canonical response should be fabricated.
"""


def convert_multichallenge(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    files = archive_files(row.data).files
    for path, digest in RUNTIME_FILES.items():
        if hashlib.sha256(files.get(path, b"")).hexdigest() != digest:
            return unsupported("unsupported_rewardkit_runtime", f"Unrecognized source {path}")
    if {path.removeprefix("tests/") for path in files if path.startswith("tests/")} != SOURCE_TEST_FILES:
        return unsupported("unsupported_rewardkit_layout", "Source tests differ from observed layout")
    settings = tomllib.loads(files.get("task.toml", b"").decode()).get("verifier", {})
    if settings.get("timeout_sec") != SOURCE_TIMEOUT or settings.get("env") != PROVIDER_SETTINGS:
        return unsupported("unsupported_rewardkit_provider", "Unrecognized source provider settings")
    configuration = tomllib.loads(files["tests/judge.toml"].decode())
    criteria = configuration.get("criterion", [])
    if configuration.get("judge") != JUDGE_SETTINGS or any(set(criterion) != CRITERION_KEYS for criterion in criteria):
        # The grader runs with a provider key, so source files must not redirect the provider or attach
        # other files to judge requests.
        return unsupported("unsupported_rewardkit_judge", "Unrecognized judge provider or file scope")
    if not criteria or any(not criterion["description"].strip() for criterion in criteria):
        return source_defect("empty_criteria", "The source provides no complete checklist criteria")
    instruction = row.data["instruction"]
    if not instruction.strip():
        return source_defect("missing_instruction", "Public instruction is required")
    resources = (
        *(
            inline_resource(path.removeprefix("tests/"), content)
            for path, content in files.items()
            if path.startswith("tests/")
        ),
        *(
            inline_resource(f"__source/{path}", content)
            for path, content in files.items()
            if not path.startswith(("tests/", "solution/"))
        ),
        inline_resource(GRADE_PATH, Path(__file__).with_name(GRADE_SCRIPT).read_bytes()),
    )
    package = verifyit_package(
        ScriptSpec(path=GRADE_PATH, timeout=GRADER_TIMEOUT, verdict_file=VERDICT_FILENAME),
        resources,
        environment=REWARDKIT_IMAGE.requirements(),
    )
    prompt = TextMessage(role="user", content=response_instruction(instruction))
    task = conversation_task(row, events=(prompt,), package=package)
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="tasktrove-multichallenge",
            source=tasktrove_source("laion__nemotron-gym-multichallenge-advanced-v4"),
            convert=convert_multichallenge,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=RUBRIC,
            atlas_id="Task Trove:laion__nemotron-gym-multichallenge-advanced-v4",
        )
    ]
