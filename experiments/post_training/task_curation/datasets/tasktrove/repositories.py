# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository repair tasks with source build recipes and private trusted tests.

SWE-smith's grader is mechanically recoverable without building its environment. SWE-rebench
retains its source contract until patched and non-Python grader migration is complete.
"""

import json
from functools import cache
from pathlib import Path

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.executable import shell_environment, swe_task
from taskcompendium.convert.tasktrove import TEST_SH_REWARD, archive_files
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    DockerBuildContext,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    VerifierArtifact,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import (
    Converter,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)
from taskcompendium.runtime.local import context_paths
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import render_spec

from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.repository_pytest import (
    TRUSTED_PATHS,
    repository_dockerfile,
    repository_test_ids,
    trusted_pytest,
)
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

WORKSPACE = "/testbed"
VERIFYIT_PACKAGE = Path(__file__).resolve().parents[5] / "lib/verifyit"

REPOSITORY_CRITERIA = """
The public repository and checkout identify necessary context; unavailable local checkout is a runtime
limitation rather than proof that the issue is underspecified.

Flag hidden requirements unrelated to the public issue, wrong base references, and inconsistent test IDs.

Repository source, multi-file changes, dependencies, and trusted-test restoration require an isolated runtime;
do not certify a repair using a generic solution.py sandbox.

Source oracle scripts are hidden review controls; their existence does not prove the issue or grader correct.

Distinguish installation/network failures from task defects and retain concrete unresolved evidence.
"""

SWE_REBENCH_RUBRIC = f"""
Compare the issue request and source checkout with the hidden test patch, restored trusted paths, and test IDs.
{REPOSITORY_CRITERIA}"""

SWESMITH_RUBRIC = f"""
Compare the stated repository bug and behavioral requirements with FAIL_TO_PASS and PASS_TO_PASS tests.
{REPOSITORY_CRITERIA}"""


def convert_repository_task(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return swe_task(row, workspace=WORKSPACE)


VERIFYIT_CONTEXT = "taskcompendium-verifyit"
PUBLIC_SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"
REPOSITORY_SETUP = "taskcompendium-repository-setup.sh"
VERIFYIT_INSTALL = f"""
COPY {VERIFYIT_CONTEXT}/ /opt/taskcompendium-verifyit/
RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python 3.12 /opt/taskcompendium-verifyit
"""
VERIFYIT_SCRIPT = b"#!/bin/bash\nset -euo pipefail\nexec verifyit /tests/verifier.toml\n"


@cache
def verifyit_build_files() -> tuple[TaskResource, ...]:
    """Bundle the checked-out verifier package so the recipe names the code used in conversion."""
    package = VERIFYIT_PACKAGE
    paths = [package / "pyproject.toml", package / "README.md", *context_paths(package / "src/verifyit")]
    return tuple(
        inline_resource(f"{VERIFYIT_CONTEXT}/{path.relative_to(package).as_posix()}", path.read_bytes())
        for path in paths
    )


def convert_swesmith_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Recover SWE-smith's pytest grader while leaving its build recipe unresolved."""
    task = convert_repository_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    if row.data.get("archive_links"):
        return unsupported("unsupported_archive_links", "Repository archives with links need explicit build lowering")
    files = archive_files(row.data)
    spec = trusted_pytest(files)
    if isinstance(spec, ImportRejection):
        return spec
    instruction = files.text("instruction.md")
    if not instruction.startswith(PUBLIC_SETUP_PREFIX):
        return unsupported("unsupported_repository_setup", "Expected the source's explicit Environment Setup bash block")
    setup, end, _ = instruction.removeprefix(PUBLIC_SETUP_PREFIX).partition("\n```")
    if not end:
        return unsupported("unsupported_repository_setup", "Environment Setup bash block is not closed")
    original_dockerfile = files.text("environment/Dockerfile")
    # Separate verification needs the packages installed by the public setup, not just its base image.
    # Remove the build-time clone so the public clone instruction and later full artifact copy start empty.
    dockerfile = repository_dockerfile(original_dockerfile, instruction) + VERIFYIT_INSTALL
    if task.resources.worker:
        dockerfile += "COPY taskcompendium-public/ /\n"
    dockerfile += (
        f"COPY {REPOSITORY_SETUP} /opt/{REPOSITORY_SETUP}\n"
        f"RUN bash -e /opt/{REPOSITORY_SETUP} && rm -rf /testbed && mkdir -p /testbed\n"
    )
    # Original environment bytes stay with provenance; the edited recipe is an environment input.
    context_files = tuple(
        resource.model_copy(
            update={
                "path": resource.path.removeprefix("environment/"),
                **(
                    {"source": inline_resource("Dockerfile", dockerfile.encode()).source}
                    if resource.path == "environment/Dockerfile"
                    else {}
                ),
            }
        )
        for resource in task.resources.oracle
        if resource.path.startswith("environment/")
    )
    if any(
        resource.path.startswith((VERIFYIT_CONTEXT, "taskcompendium-public", REPOSITORY_SETUP))
        for resource in context_files
    ):
        return unsupported("build_context_collision", "Source context occupies the bundled verifier path")
    public_context = tuple(
        resource.model_copy(update={"path": "taskcompendium-public/" + resource.path})
        for resource in task.resources.worker
    )
    environment = EnvironmentRequirements(
        docker_build=DockerBuildContext(
            files=(
                *context_files,
                *verifyit_build_files(),
                *public_context,
                inline_resource(REPOSITORY_SETUP, setup.encode()),
            )
        )
    )
    verifier = (
        *(
            resource
            for resource in task.resources.verifier
            if resource.path in (TRUSTED_PATHS, "taskcompendium/archive-provenance.json")
        ),
        inline_resource("verifier.toml", render_spec(spec).encode()),
        inline_resource("test.sh", VERIFYIT_SCRIPT).model_copy(update={"mode": "0755"}),
    )
    original_tests = tuple(
        resource.model_copy(update={"path": "source_archive/tests/" + resource.path})
        for resource in task.resources.verifier
        if resource.path != "taskcompendium/archive-provenance.json"
    )
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        cwd="/",
        environment=environment,
        artifacts=(VerifierArtifact(source=WORKSPACE, target=WORKSPACE, kind=ArtifactKind.DIRECTORY),),
        answer_path=None,
        reward=TEST_SH_REWARD,
    )
    changes = [
        NormalizationChange(
            field="environment/Dockerfile",
            reason="Carry repository test dependencies and the bundled verifier as an unresolved build recipe",
            original=original_dockerfile,
            replacement=dockerfile,
        )
    ]
    pass_to_pass = repository_test_ids(json.loads(files.text("tests/config.json")).get("PASS_TO_PASS"))
    if pass_to_pass != spec.must_not_break:
        changes.append(
            NormalizationChange(
                field="PASS_TO_PASS",
                reason="Match legacy pytest conversion by excluding doctest and truncated node ids it cannot collect",
                original=json.dumps(pass_to_pass),
                replacement=json.dumps(spec.must_not_break),
            )
        )
    return NormalizedTask(
        task.model_copy(
            update={
                "environment_requirements": shell_environment(environment),
                "answer_type": AnswerType.WORKSPACE_STATE,
                "grader": grader,
                "resources": ResourceGroups(
                    worker=task.resources.worker, verifier=verifier, oracle=(*task.resources.oracle, *original_tests)
                ),
                "tags": ("code", "swe", "swe-repo", "trusted-test-paths"),
            }
        ),
        tuple(changes),
    )


def repository_source(
    name: str,
    config: str,
    rubric: str,
    info: SourceInfo,
    *,
    convert: Converter,
    version: str,
    ships: tuple[Path, ...] = (),
) -> RlDataSource:
    return RlDataSource(
        info=info,
        pipeline=RlDataPipeline(
            name=f"tasktrove-{name}",
            source=tasktrove_source(config),
            convert=convert,
            version=version,
            ships=ships,
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=rubric,
        ),
    )


def sources() -> list[RlDataSource]:
    return [
        repository_source(
            "swe_rebench",
            "DCAgent__swe_rebench_v2_patched_oracle-v2",
            SWE_REBENCH_RUBRIC,
            convert=convert_repository_task,
            version="1",
            info=SourceInfo(
                id="Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2",
                title="DCAgent/swe_rebench_v2_patched_oracle-v2",
                origin="Task Trove",
                family="swe-repo",
                tags=(
                    "agentic",
                    "multi-turn",
                    "language:python",
                    "language:go",
                    "language:rust",
                    "language:java",
                    "language:julia",
                    "language:kotlin",
                    "language:swift",
                    "language:dart",
                    "language:c",
                    "language:scala",
                    "language:php",
                    "language:csharp",
                    "language:elixir",
                    "language:lua",
                    "language:cpp",
                    "language:ocaml",
                ),
                count=18319,
                notes=(
                    "Real repos, hidden FAIL_TO_PASS, git gate, trusted-test restore. Bake the verify-"
                    "time installs into the image."
                ),
            ),
        ),
        repository_source(
            "swesmith",
            "laion__swesmith-oracle-filtered-v2",
            SWESMITH_RUBRIC,
            convert=convert_swesmith_task,
            version="2",
            ships=(VERIFYIT_PACKAGE,),
            info=SourceInfo(
                id="Task Trove:laion__swesmith-oracle-filtered-v2",
                title="laion/swesmith-oracle-filtered-v2",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "language:python"),
                count=12927,
                notes=(
                    "Trusted repository pytest tests with unresolved Docker build contexts; "
                    "original config patches remain oracle-only."
                ),
            ),
        ),
    ]
