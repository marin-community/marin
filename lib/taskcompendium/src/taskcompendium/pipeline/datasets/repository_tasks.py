# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve repository repair tasks for static curation without pretending they can run."""

import base64
import hashlib
import json
import re

from taskcompendium.grader import grader_config
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ProviderRequirement,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.datasets.direct_contracts import source_contract_package
from taskcompendium.pipeline.datasets.shell_files import BASH
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import INTERFACE

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
    package = source_contract_package(
        "source repository patch grader",
        row.source.revision,
        {
            "repository": repository,
            "source_ref": checkout[1],
            "workspace": WORKSPACE,
            "source_config": config,
            "source_grader_paths": sorted("/" + path for path in files if path.startswith("tests/")),
            "source_environment_sha256": hashlib.sha256(files["environment/Dockerfile"]).hexdigest(),
        },
        ("Isolated repository checkout and patch capture",),
    )
    resources = ResourceGroups(
        worker=tuple(
            inline_resource(path, content) for path, content in files.items() if path.startswith("setup_files/")
        ),
        oracle=tuple(
            inline_resource(path, content)
            for path, content in files.items()
            if path.startswith(("environment/", "solution/"))
        ),
        verifier=package.resources
        + tuple(inline_resource(path, content) for path, content in files.items() if path.startswith("tests/")),
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem", "git_repository"),
            tool_providers={
                "shell": ProviderRequirement(
                    action_interface=INTERFACE,
                    initial_state={
                        "repository": repository,
                        "source_ref": checkout[1],
                        "workspace": WORKSPACE,
                        "binding_status": "unbound",
                    },
                )
            },
        ),
        interaction_tools=(BASH,),
        resources=resources,
        answer_type=AnswerType.STATE,
        verifier=package.verifier,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    contract = grader_config(task)["contract"]
    return VerificationReport(
        checks=[
            CheckResult(
                check="isolated_repository_patch_runtime",
                status=CheckStatus.UNSUPPORTED,
                detail=(
                    f"Source {contract['repository']}@{contract['source_ref']}, "
                    "trusted tests and environment retained; "
                    "checkout, dependencies, patch capture, and isolated source grading are not bound"
                ),
            )
        ]
    )


REPOSITORY_COMMON_CRITERIA = (
    "The public repository and checkout identify necessary context; unavailable local checkout is a "
    "runtime limitation rather than proof that the issue is underspecified.",
    "Flag hidden requirements unrelated to the public issue, wrong base references, and inconsistent test IDs.",
    "Repository source, multi-file changes, dependencies, and trusted-test restoration require an "
    "isolated runtime; do not certify a repair using a generic solution.py sandbox.",
    "Source oracle scripts are private review controls; their existence does not prove the issue or grader correct.",
    "Distinguish installation/network failures from task defects and retain concrete unresolved evidence.",
)

RUBRICS: dict[str, ReviewRubric] = {
    "swe_rebench": ReviewRubric(
        id="swe_rebench-answerability",
        version="1",
        criteria=(
            "Compare the issue request and source checkout with the hidden test patch, restored trusted paths, "
            "and test IDs.",
            *REPOSITORY_COMMON_CRITERIA,
        ),
    ),
    "swesmith": ReviewRubric(
        id="swesmith-answerability",
        version="1",
        criteria=(
            "Compare the stated repository bug and behavioral requirements with FAIL_TO_PASS and PASS_TO_PASS tests.",
            *REPOSITORY_COMMON_CRITERIA,
        ),
    ),
}


def pipeline(name: str) -> TaskPipeline:
    """Build repository normalization and the unbound-runtime review policy."""
    return TaskPipeline(
        normalize=normalize,
        rubric=RUBRICS[name],
        check_suite=CheckSuite(
            id="repository-contract-unbound-runtime", revision="1", parameters={}, run=verification_report
        ),
    )
