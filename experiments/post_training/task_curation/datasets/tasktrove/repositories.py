# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove repository repair sources, kept for review without a runnable grader.

Each task's source grader applies the agent's patch in that task's own repository image and runs the
FAIL_TO_PASS and PASS_TO_PASS tests. No committed image covers those per-task repositories, so the tasks
record the source grader's files and terms under a ``NoGrader`` and never reach the final export.
"""

from dataclasses import replace

from taskcompendium.convert.executable import swe_task
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="Task Trove",
    url="https://huggingface.co/datasets/open-athena/task-trove",
    dataset_id="open-athena/task-trove",
    revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
    revised_at="2026-10-08T09:34:47.000Z",
    verifier_revision=None,
    family="swe-repo",
    environment="Harbor",
    type="Agentic",
    turns="Multi-turn",
    count_basis="Released Harbor tasks: manifest by_source.converted",
    count_precision="exact",
    count_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    benchmark_basis="Release manifest does not designate benchmarks",
    family_basis="Task Trove release manifest source_verdicts.family",
    family_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
    canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
    provenance_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb8"
        "03e826563f5718dbf/manifest.json"
    ),
    snapshot_safe=True,
    snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
    upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
    recorded_at="2026-10-08",
)

WORKSPACE = "/testbed"

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


def repository_pipeline(name: str, config: str, rubric: str, metadata: DataSourceMetadata) -> RlDataSource:
    return RlDataSource(
        metadata=metadata,
        pipeline=RlDataPipeline(
            name=f"tasktrove-{name}",
            source=tasktrove_source(config),
            convert=convert_repository_task,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=rubric,
        ),
    )


def sources() -> list[RlDataSource]:
    return [
        repository_pipeline(
            "swe_rebench",
            "DCAgent__swe_rebench_v2_patched_oracle-v2",
            SWE_REBENCH_RUBRIC,
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2",
                name="DCAgent__swe_rebench_v2_patched_oracle-v2",
                display_name="DCAgent/swe_rebench_v2_patched_oracle-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                task_count=13559,
                notes=(
                    "Real repos, hidden FAIL_TO_PASS, git gate, trusted-test restore. Bake the "
                    "verify-time installs into the image."
                ),
                canonical_source="DCAgent/swe_rebench_v2_patched_oracle-v2",
                verification="script, pytest",
                upstream_repository="DCAgent/swe_rebench_v2_patched_oracle-v2",
                upstream_url="https://huggingface.co/datasets/DCAgent/swe_rebench_v2_patched_oracle-v2",
                input_count=18319,
                languages=(
                    "python",
                    "go",
                    "rust",
                    "java",
                    "julia",
                    "kotlin",
                    "swift",
                    "dart",
                    "c",
                    "scala",
                    "php",
                    "csharp",
                    "elixir",
                    "lua",
                    "cpp",
                    "ocaml",
                ),
                modes=("script", "pytest"),
            ),
        ),
        repository_pipeline(
            "swesmith",
            "laion__swesmith-oracle-filtered-v2",
            SWESMITH_RUBRIC,
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__swesmith-oracle-filtered-v2",
                name="laion__swesmith-oracle-filtered-v2",
                display_name="laion/swesmith-oracle-filtered-v2",
                dataset_revision="eb0efd4c530101032b870f62aa5590b4888a8b01",
                task_count=12720,
                notes="Real repo tests. Strip the oracle patch from tests/config.json at conversion.",
                canonical_source="laion/swesmith-oracle-filtered-v2",
                verification="pytest",
                upstream_repository="laion/swesmith-oracle-filtered-v2",
                upstream_url="https://huggingface.co/datasets/laion/swesmith-oracle-filtered-v2",
                input_count=12927,
                languages=("python",),
                modes=("pytest",),
            ),
        ),
    ]
