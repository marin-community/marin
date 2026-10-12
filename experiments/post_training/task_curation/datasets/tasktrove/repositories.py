# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository repair tasks with source build recipes and private trusted tests.

SWE-smith and SWE-rebench keep public setup as unresolved image recipes and grade
private copies of the repository with restored trusted tests.
"""

import json

from taskcompendium.convert.answers import unsupported
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import (
    Converter,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import archive_files
from experiments.post_training.task_curation.datasets.tasktrove.conversion.executable import swe_task
from experiments.post_training.task_curation.datasets.tasktrove.repository_build import (
    WORKSPACE,
    repository_build_task,
)
from experiments.post_training.task_curation.datasets.tasktrove.repository_pytest import (
    TRUSTED_PATHS,
    repository_dockerfile,
    repository_test_ids,
    trusted_pytest,
)
from experiments.post_training.task_curation.datasets.tasktrove.swe_rebench import convert_swe_rebench_task
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

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


def convert_swesmith_task(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Recover SWE-smith's pytest grader while leaving its build recipe unresolved."""
    task = swe_task(row, workspace=WORKSPACE)
    if isinstance(task, ImportRejection):
        return task
    if row.data.get("archive_links"):
        return unsupported("unsupported_archive_links", "Repository archives with links need explicit build lowering")
    files = archive_files(row.data)
    spec = trusted_pytest(files)
    if isinstance(spec, ImportRejection):
        return spec
    changes = []
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
    return repository_build_task(
        task,
        files,
        spec=spec,
        dockerfile=repository_dockerfile(files.text("environment/Dockerfile"), files.text("instruction.md")),
        verifier=tuple(
            resource
            for resource in task.resources.verifier
            if resource.path in (TRUSTED_PATHS, "taskcompendium/archive-provenance.json")
        ),
        tags=("code", "swe", "swe-repo", "trusted-test-paths", "language:python"),
        changes=tuple(changes),
    )


def repository_source(
    name: str,
    config: str,
    rubric: str,
    info: SourceInfo,
    *,
    convert: Converter,
    version: str,
) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        pipeline=process_rows,
        info=info,
        config=CurationRecipe(
            name=f"tasktrove-{name}",
            source=tasktrove_source(config),
            convert=TaskTroveConverter(config, convert),
            version=version,
            intended_use=IntendedUse.TRAIN,
            rubric=rubric,
        ),
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        repository_source(
            "swe_rebench",
            "DCAgent__swe_rebench_v2_patched_oracle-v2",
            SWE_REBENCH_RUBRIC,
            convert=convert_swe_rebench_task,
            version="3",
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
                    "Patched trusted repository tests with deferred Docker builds. Python uses pytest; "
                    "retained non-Python sources keep their parser with exit-code credit disabled."
                ),
            ),
        ),
        repository_source(
            "swesmith",
            "laion__swesmith-oracle-filtered-v2",
            SWESMITH_RUBRIC,
            convert=convert_swesmith_task,
            version="4",
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
