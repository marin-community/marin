# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import executable R2E-Gym rows without reconstructing their repositories."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tasktrove_verify.spec import Mode

from taskcompendium.models import (
    AnswerRequirements,
    Capability,
    ContainerRuntime,
    Embedded,
    ImageOverlay,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskTroveVerifier,
    WorkspaceState,
    relative_path,
)

DATASET = "R2E-Gym/R2E-Gym-V1"
REVISION = "903d405799ac435061c41e72260c81ca5100f964"
SOURCE_REVISION = "0d94c4eb9431cd195c55a7ea3abd54006c9a1735"
IMPORTER_REVISION = "taskcompendium-r2egym-v4"
# Source images contain a dangling datasets alias; it is not a candidate repair.
SNAPSHOT_EXCLUSIONS = (".venv", "datasets")

# Docker tags in the dataset are commit-looking tags, but remain mutable. These
# manifest digests were resolved from Docker Hub before importing the pinned rows.
_IMAGE_DIGESTS = {
    "2d9617bd0cb1f0ba61771258410ab8fae8e7e24d": "d546d21f56bb6b9c99045bd8fea196c04e43c9fbd6d9bc36fba4a90f88a45276",
    "a95245e37f35446f9870feb68f5ebb97d1c279ce": "b94e6cc6eddc8bee73a035c3a17bd4f93acd4a2ae1f5bad54c9adc8af53267e2",
    "9764be4fa30b1ae85db3b875039f0a2983531a9d": "f915aa089e288e33669fe3b6876c8a0176f38270f047d3b139c7c378976dd4ec",
    "ef8363b650134f6415704b064573707a6d304ca3": "28e1d7e0e313a201d6ac51ec6763c034f0fd0b5faf96f8bc72071ff010d71c90",
    "e48a32f29bd3736e53efb7290cae64d5688357d6": "5756a094938d63382224b624ef20abf6a80affbcdb3b6dadf4b14da9b130f741",
    "74ffccc1d4a1dbb877490c9d6b59e6f501d5e438": "e83eba5546a9d714edd4aa33e672c4556bf224bcd20bba43d580086a340b1f44",
    "269e2a176b9b2b8e00be46f5a59e840625879bf9": "b51ae26833b9714589578f44de5118845d874c7489a247cbb6531a9763e804a3",
    "9b5494e26f407b75e79699c9d40be6df1d80a040": "a5aa510cf45da43f32775bd0cdbf1ac81ef5e621732ba30a12f120624e3358f9",
    "70a4df334835bcb9ea7c6b0eb11b4f5f7e117f36": "20e78ae3243794f55f742896e6e1e56bf3b5ae74018d255647d16fd76db0f5a7",
    "d61803f7181e7ad525d4bb1dd71ca500bb41617e": "073312ac010026ab652f475af9555ce8f30d62d61a02df38e3de053990d45817",
    "df34d9081e1fbe25466980b17a082ce210e821b5": "2ce5c7a809dea42be9511184e83e5a3d0ba936a4feabbecb6bdf171ca4227c11",
    "f5b986b11e518191441420df8f8b84154c92e81c": "82efe4b80310bd1962cfd4857e5d84372e9f6da556e06b40269911a67e3c4d9f",
    "3758d8594d8738e51b3a738e06858ae7fc3cbdf1": "23eb91283d45a802bd61ebb368faffb3ade7b38009c5a830ee09c4a968c09365",
    "4651e73c8ea9a502d4fd6ee25f33ece07b693b37": "924c7b5da201d773bdda8b4809b086b3aa005877bcda118fbb3d8b5d60523e76",
    "7f5c9dace89a846e6fb7aed86e4ff7eb05e01dc0": "518000ecded5d26bd284422574b56ef00f50f2dbeb2024795f6baec5c94d3aef",
    "aae8c47f823197a04010a74570793464eb7352fe": "1380135c2069305aa4c6eafc429d08a19a909ec6dccd3e5e0afabef0a057ffea",
    "b983e25212aa3c006c37089ef970688e3c01c369": "db1324588ce11088e47f8464c6a0c710b1d79d365f3932fb12db1bd2202e356b",
    "7c9de553a279791535aa8ec927bef47801b819f1": "f719c808a015c883cabe3152ea8159346569028105c008273dbf6154c4dd4ec8",
    "ea8c3b0ce9ff87f849b1462aab0b34bd3e35d4ed": "928fec0305886115f1c7ccd22275da5d4aab9dd8df3c03497cb06bd5e6e7cf3e",
    "ad202ae0d526f208bfde4ed9ef47190f078be7de": "1f51967789c2f9172ad05d2c7ca7ab1112266feb804d13f96961bc96a1281eb6",
    # These two SymPy rows are pinned source-family broadening examples.  The
    # manifest digests were resolved from Docker Hub; image layers are not
    # downloaded by the importer.
    "b0d83c555eda061014ae1d04dc0af07a569650e6": "0c6511c00df0564ad7513261f1c643d31130fe7157b34457d9f36fed5d674d56",
    "d635509acdf6aa448200797e2d3393068ed29dc7": "e1dba0c4da4398c49840bea0a5f7398be5787e7150d4ddea600b531ac0d80a0f",
}
_IMAGE_REPOSITORIES = {
    **{
        tag: "namanjain12/orange3_final"
        for tag in _IMAGE_DIGESTS
        if tag
        not in {
            "b0d83c555eda061014ae1d04dc0af07a569650e6",
            "d635509acdf6aa448200797e2d3393068ed29dc7",
        }
    },
    "b0d83c555eda061014ae1d04dc0af07a569650e6": "namanjain12/sympy_final",
    "d635509acdf6aa448200797e2d3393068ed29dc7": "namanjain12/sympy_final",
}
_REQUIRES_XVFB = {"orange3": True, "sympy": False}


def _source(row: Mapping[str, Any], row_id: str) -> Source:
    return Source(DATASET, REVISION, row_id, IMPORTER_REVISION)


def source_image(row: Mapping[str, Any]) -> str | None:
    """Return the inspected immutable source image, or None for an unsupported tag."""
    value = row.get("docker_image")
    if not isinstance(value, str) or ":" not in value:
        return None
    repository, tag = value.rsplit(":", 1)
    if _IMAGE_REPOSITORIES.get(tag) != repository:
        return None
    digest = _IMAGE_DIGESTS.get(tag)
    return f"docker.io/{repository}@sha256:{digest}" if digest else None


def import_row(
    row: Mapping[str, Any],
    source: Source | None = None,
    *,
    verifier_runtime: ContainerRuntime | None = None,
) -> TaskSpecification | Rejected:
    """Convert one complete R2E-Gym row, or return a concrete rejection.

    R2E tests execute the source repository's Python environment.  The task
    image is therefore not implicitly a valid verifier image: callers must
    supply a separately provisioned, immutable image containing the
    TaskCompendium supervisor and the source test dependencies.
    """
    row_id = str(row.get("commit_hash", ""))
    source = source or _source(row, row_id)
    if source.dataset != DATASET or source.revision != REVISION:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, "R2E-Gym source is not the pinned revision")
    required = (
        "repo_name",
        "commit_hash",
        "docker_image",
        "problem_statement",
        "expected_output_json",
        "execution_result_content",
    )
    if any(key not in row for key in required):
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, "R2E-Gym row is truncated")
    expected_repository = f"namanjain12/{row['repo_name']}_final"
    if (
        not isinstance(row["repo_name"], str)
        or not str(row["docker_image"]).startswith(expected_repository + ":")
        or not str(row["docker_image"]).endswith(":" + row_id)
    ):
        return Rejected(
            source, RejectionReason.UNRECOVERABLE_SOURCE, "image repository or commit does not match the row"
        )
    image = source_image(row)
    if image is None:
        return Rejected(
            source, RejectionReason.UNSUPPORTED_ENVIRONMENT, "R2E-Gym image tag has no resolved immutable digest"
        )
    if verifier_runtime is None:
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_ENVIRONMENT,
            "R2E-Gym requires an explicit compatible verifier runtime image",
        )
    if verifier_runtime.image == image:
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_ENVIRONMENT,
            "R2E-Gym verifier runtime must be separate from the task image",
        )
    if (
        not isinstance(verifier_runtime.workspace, ImageOverlay)
        or ".venv" not in verifier_runtime.workspace.preserved_directories
    ):
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_ENVIRONMENT,
            "R2E-Gym verifier runtime must preserve the image's .venv dependency directory",
        )
    if not PurePosixPath(verifier_runtime.supervisor_python).is_absolute():
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_ENVIRONMENT,
            "R2E-Gym requires an absolute supervisor Python path; its image PATH selects the source interpreter",
        )
    instruction = row["problem_statement"]
    if not isinstance(instruction, str) or not instruction.strip():
        return Rejected(source, RejectionReason.UNDERSPECIFIED, "R2E-Gym problem statement is empty")
    try:
        expected = json.loads(row["expected_output_json"])
        execution = json.loads(row["execution_result_content"])
    except (TypeError, json.JSONDecodeError) as error:
        return Rejected(source, RejectionReason.BROKEN_GRADER, f"invalid source JSON: {error}")
    if (
        not isinstance(execution, dict)
        or execution.get("repo_name") != row["repo_name"]
        or execution.get("new_commit_hash") != row["commit_hash"]
    ):
        return Rejected(
            source, RejectionReason.UNRECOVERABLE_SOURCE, "execution result is for a different source commit"
        )
    names = row.get("test_file_names", execution.get("test_file_names"))
    codes = row.get("test_file_codes", execution.get("test_file_codes"))
    if (
        not isinstance(expected, dict)
        or not expected
        or any(not isinstance(key, str) or value not in {"PASSED", "FAILED", "ERROR"} for key, value in expected.items())
        or not isinstance(names, list)
        or not isinstance(codes, list)
        or len(names) != len(codes)
    ):
        return Rejected(source, RejectionReason.BROKEN_GRADER, "R2E-Gym tests and expected status map are malformed")
    try:
        valid_names = all(isinstance(name, str) and name.endswith(".py") and not relative_path(name) for name in names)
    except ValueError:
        valid_names = False
    if not valid_names or len(set(names)) != len(names) or not all(isinstance(code, str) for code in codes):
        return Rejected(source, RejectionReason.BROKEN_GRADER, "R2E-Gym test assets are malformed")
    resources = [Resource("r2e_tests/__init__.py", (ResourceRole.VERIFIER,), Embedded(b""))]
    resources.extend(
        Resource(f"r2e_tests/{name}", (ResourceRole.VERIFIER,), Embedded(code.encode()))
        for name, code in zip(names, codes, strict=True)
    )
    resources.extend(
        (
            Resource(
                "r2e_assets/expected_output.json",
                (ResourceRole.VERIFIER,),
                Embedded(row["expected_output_json"].encode()),
            ),
            Resource(
                "r2e_assets/verify.py", (ResourceRole.VERIFIER,), Embedded(_VERIFIER_SCRIPT.encode()), executable=True
            ),
            Resource(
                "r2e_assets/source_parser.py",
                (ResourceRole.VERIFIER,),
                Embedded(_SOURCE_PARSER.encode()),
            ),
            Resource(
                "oracle/execution_result.json",
                (ResourceRole.ORACLE,),
                Embedded(json.dumps(execution, sort_keys=True).encode()),
            ),
        )
    )
    setup = (
        "rm -rf /r2e_tests /expected_test_output.json /testbed/r2e_tests "
        "/testbed/expected_test_output.json /testbed/.git",
    )
    parameters = {
        "path": "r2e_assets/verify.py",
        "args": ["--require-xvfb"] if _REQUIRES_XVFB[row["repo_name"]] else [],
    }
    verifier = TaskTroveVerifier(Mode.SCRIPT, parameters, runtime=verifier_runtime)
    return TaskSpecification(
        id=f"r2egym/{row_id}",
        steps=(
            StepSpecification(
                instructions=instruction.strip(),
                verifier=verifier,
                answer_requirements=AnswerRequirements("final_state"),
            ),
        ),
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS),
            WorkspaceState(image=image, workdir="/testbed", setup_commands=setup),
        ),
        resources=tuple(resources),
        metadata=TaskMetadata(
            source=source, competencies=("software-engineering", "debugging"), task_shape="environment-modification"
        ),
    )


def import_rows(
    rows: Iterable[Mapping[str, Any]], *, verifier_runtime: ContainerRuntime | None = None
) -> tuple[TaskSpecification | Rejected, ...]:
    """Import rows while retaining one result per source row."""
    return tuple(import_row(row, verifier_runtime=verifier_runtime) for row in rows)


_SOURCE_PARSER = (Path(__file__).parent / "r2e_assets/source_parser.py").read_text()
_VERIFIER_SCRIPT = (Path(__file__).parent / "r2e_assets/verify.py").read_text()
