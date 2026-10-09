# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove repository repair sources, kept for review without a runnable grader.

Each task's source grader applies the agent's patch in that task's own repository image and runs the
FAIL_TO_PASS and PASS_TO_PASS tests. No committed image covers those per-task repositories, so the tasks
record the source grader's files and terms under a ``NoGrader`` and never reach the final export.
"""

from taskcompendium.convert.executable import swe_task
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

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


def repository_pipeline(name: str, config: str, rubric: str) -> RlDataPipeline:
    return RlDataPipeline(
        name=f"tasktrove-{name}",
        source=tasktrove_source(config),
        convert=convert_repository_task,
        version="1",
        environment=ShellSim(),
        intended_use=IntendedUse.TRAIN,
        rubric=rubric,
        atlas_id=f"Task Trove:{config}",
    )


def pipelines() -> list[RlDataPipeline]:
    return [
        repository_pipeline("swe_rebench", "DCAgent__swe_rebench_v2_patched_oracle-v2", SWE_REBENCH_RUBRIC),
        repository_pipeline("swesmith", "laion__swesmith-oracle-filtered-v2", SWESMITH_RUBRIC),
    ]
