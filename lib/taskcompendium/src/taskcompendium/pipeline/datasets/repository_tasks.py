# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve repository repair tasks for static curation without pretending they can run."""

import base64
import hashlib
import json
import re
from pathlib import Path

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentFixture,
    EnvironmentRequirements,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.datasets.shell_files import BASH
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
    VerificationReport,
)
from taskcompendium.verifiers.repository_patch import RepositoryPatchVerifier

CHECKOUT = re.compile(r"\bgit checkout\s+([^\s;&]+)")
WORKSPACE = "/testbed"


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    instruction = row.data.get("instruction")
    encoded = row.data.get("files")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(encoded, dict):
        return ImportRejection(reason="missing_repository_input", detail="Public request and source files are required")
    required = ("tests/config.json", "tests/test.sh", "environment/Dockerfile")
    if any(path not in encoded for path in required):
        return ImportRejection(
            reason="missing_repository_contract", detail="config.json, test.sh, and Dockerfile required"
        )
    files = {path: base64.b64decode(content, validate=True) for path, content in encoded.items()}
    config = json.loads(files["tests/config.json"])
    repository = config.get("repo")
    checkout = CHECKOUT.search(instruction)
    if not isinstance(repository, str) or not repository.strip() or checkout is None:
        return ImportRejection(
            reason="missing_repository_ref", detail="Source repository and public checkout ref required"
        )
    if not config.get("FAIL_TO_PASS") and not config.get("PASS_TO_PASS"):
        return ImportRejection(
            reason="missing_repository_tests", detail="No source FAIL_TO_PASS or PASS_TO_PASS test IDs"
        )
    verifier = RepositoryPatchVerifier(
        repository=repository,
        source_ref=checkout[1],
        workspace=WORKSPACE,
        source_config=config,
        source_grader_paths=tuple(sorted("/" + path for path in files if path.startswith("tests/"))),
        source_environment_sha256=hashlib.sha256(files["environment/Dockerfile"]).hexdigest(),
    )
    resources = tuple(
        task_resource(
            "/" + path,
            content,
            (
                ResourceVisibility.VERIFIER
                if path.startswith("tests/")
                else (ResourceVisibility.AGENT if path.startswith("setup_files/") else ResourceVisibility.CONTROL)
            ),
        )
        for path, content in files.items()
        if path.startswith(("tests/", "environment/", "solution/", "setup_files/"))
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem", "git_repository"), action_interfaces=("shell:v1",)
        ),
        fixture=EnvironmentFixture(
            interface="shell:v1",
            revision="repository-unbound-v1",
            initial_state_json=json.dumps(
                {
                    "repository": repository,
                    "source_ref": checkout[1],
                    "workspace": WORKSPACE,
                    "binding_status": "unbound",
                }
            ),
        ),
        interaction_tools=(BASH,),
        resources=resources,
        answer_type=AnswerType.STATE,
        verifier=VerifierSpec(kind=VerifierKind.REPOSITORY_PATCH, parameters_json=verifier.model_dump_json()),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    verifier = RepositoryPatchVerifier.model_validate_json(task.verifier.parameters_json)
    return VerificationReport(
        checks=[
            CheckResult(
                check="isolated_repository_patch_runtime",
                status=CheckStatus.UNSUPPORTED,
                detail=f"Source {verifier.repository}@{verifier.source_ref}, trusted tests, and environment retained; "
                "checkout, dependencies, patch capture, and isolated source grading are not bound",
            )
        ]
    )


def recipe(name: str, snapshot: Path, *, config: str, revision: str, rubric: ReviewRubric) -> DatasetRecipe:
    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", revision, config, "train", str(snapshot)),
        normalize=normalize,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="repository-contract-unbound-runtime", revision="1", parameters={}, run=verification_report
        ),
    )
