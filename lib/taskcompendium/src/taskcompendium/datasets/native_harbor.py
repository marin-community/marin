# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain native Harbor contracts whose source runtimes are not bound locally."""

import base64
import hashlib
from functools import partial

from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.datasets.source_definitions import archive_resources
from taskcompendium.grader import grader_config
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.runtime.shell import BASH


def normalize_native_harbor(row: RawRow, *, runtime_requirements: tuple[str, ...]) -> TaskSpec | ImportRejection:
    """Preserve public instruction and private native grader materials without executing them."""
    instruction, encoded = row.data.get("instruction"), row.data.get("files")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(encoded, dict):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_native_harbor_input",
            detail="The native instruction and archived source files are required",
        )
    if row.data["archive_links"]:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="native_harbor_archive_links",
            detail="The source archive requires symbolic or hard links that are not representable as files",
        )
    files = {path: base64.b64decode(content, validate=True) for path, content in encoded.items()}
    if "tests/test.sh" not in files:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_native_harbor_grader",
            detail="No native tests/test.sh was retained in the source archive",
        )
    package = source_contract_package(
        "native Harbor source grader",
        row.source.revision,
        {
            "source_path": str(row.data["path"]),
            "archive_sha256": str(row.data["archive_sha256"]),
            "source_grader_paths": sorted(path for path in files if path.startswith("tests/")),
            "source_file_sha256": {path: hashlib.sha256(content).hexdigest() for path, content in files.items()},
            "binding_status": "unbound",
        },
        runtime_requirements,
    )
    resources = archive_resources(row.data)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(capabilities=("shell", "filesystem", "native_harbor")),
        interaction_tools=(BASH,),
        resources=ResourceGroups(
            worker=resources.worker, verifier=package.resources + resources.verifier, oracle=resources.oracle
        ),
        answer_type=AnswerType.STATE,
        answer_format=PlainText(),
        grader=package.grader,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    config = grader_config(task)
    return VerificationReport(
        [
            CheckResult(
                check="native_harbor_runtime",
                status=CheckStatus.UNSUPPORTED,
                detail="; ".join(config["runtime_requirements"]),
            )
        ]
    )


def policy(source_name: str, family: str, runtime_requirements: tuple[str, ...]) -> TaskPolicy:
    """Build static curation policy with explicit native runtime limitations."""
    rubric = ReviewRubric(
        id=f"{source_name}-native-harbor",
        version="1",
        criteria=(
            f"Inspect the native {family} task's public instruction against its retained source tests and environment.",
            "Flag contradictory instructions, hidden requirements, and incorrect source grader assumptions.",
            "Private solution files are review evidence and must remain unavailable to the solving actor.",
            "Unavailable images, repository dependencies, service credentials, or native plugins are runtime "
            "limitations rather than demonstrated task defects.",
            "Preserve the original tests/test.sh semantics; a generic code or text grader cannot certify this task.",
        ),
    )
    return TaskPolicy(
        normalize=partial(normalize_native_harbor, runtime_requirements=runtime_requirements),
        rubric=rubric,
        check_suite=CheckSuite(
            "native-harbor-unbound", "1", {"requirements": runtime_requirements}, verification_report
        ),
    )
