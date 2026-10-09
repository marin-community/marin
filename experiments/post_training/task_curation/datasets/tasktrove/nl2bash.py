# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove natural-language-to-bash tasks, graded on the agent's captured command output.

The agent runs a command in the nl2bash image and writes its combined output to the capture file. A
checker with the grader packages (``GRADER_PACKAGES``) compares the capture with the oracle command's
recorded output as an order-insensitive multiset of normalized lines. The oracle control runs the
source's ``solution/solve.sh``.
"""

from dataclasses import replace

from taskcompendium.convert.executable import solve_script, tasktrove_archive_task
from taskcompendium.convert.tasktrove_nl2bash import OUTPUT_PATH, convert_nl2bash
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_RELEASE, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.code import ANSWERABILITY_CRITERIA
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import RlDataPipeline, environment_requirements
from experiments.post_training.task_curation.source import RlDataSource

AGENT_IMAGE = Environment(
    image="ghcr.io/marin-community/iris-task@sha256:66cba7cb3eb682f9a53e444876ef2468670336a71e03559de85b5b2b5d4cdde6"
)
"""The nl2bash image the agent's shell runs in."""

CONFIG = "DCAgent2__nl2bash-tasks-cleaned-oracle-v2"

NL2BASH_RUBRIC = f"""
{ANSWERABILITY_CRITERIA}
The public seed/setup files must recreate the command's input files. Flag unavailable tools or inputs.

The hidden comparator is a normalized multiset: ordering is ignored and non-error extra lines are allowed.
Check whether the public request requires distinctions that this comparator cannot grade.

The expected output is an oracle capture, not an instruction to print that output without doing the work.

Every mandatory package or side-effect deliverable needs a public specification or provided helper. An
undefined 'sandboxes task' package is a missing-context defect even if only stdout is graded. Test a literal
minimal answer to the public request; unstated oracle output prefixes are a grading mismatch.
"""


def convert_nl2bash_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return tasktrove_archive_task(
        row,
        convert=convert_nl2bash,
        environment=environment_requirements(AGENT_IMAGE),
        grader_environment=required_grader_environment(context),
        output_paths=(OUTPUT_PATH,),
    )


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=replace(
                TASKTROVE_RELEASE,
                id="Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                name="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                display_name="DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                dataset_revision="8c85912822da0a77978e285af923eecc48ae34a3",
                verifier_revision=None,
                family="shell-cmd",
                task_count=1497,
                notes="Semantic output comparison against an oracle command run in the same sandbox.",
                canonical_source="DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                verification="script",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                upstream_url="https://huggingface.co/datasets/DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1498,
                languages=("bash",),
                modes=("script",),
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-nl2bash",
                source=tasktrove_source(CONFIG),
                convert=convert_nl2bash_task,
                version="1",
                environment=AGENT_IMAGE,
                intended_use=IntendedUse.TRAIN,
                rubric=NL2BASH_RUBRIC,
                controls=Controls(golden=solve_script),
                grader=GRADER_PACKAGES,
            ),
        )
    ]
