# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove natural-language-to-bash tasks, graded on the agent's captured command output.

The agent runs a command in the nl2bash image and writes its combined output to the capture file. A
checker with the grader packages (``GRADER_PACKAGES``) compares the capture with the oracle command's
recorded output as an order-insensitive multiset of normalized lines. The oracle control runs the
source's ``solution/solve.sh``.
"""


from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.code import ANSWERABILITY_CRITERIA
from experiments.post_training.task_curation.datasets.tasktrove.conversion.executable import (
    solve_script,
    tasktrove_archive_task,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.nl2bash import OUTPUT_PATH, convert_nl2bash
from experiments.post_training.task_curation.environment_fidelity import archive_environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

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
        environment=archive_environment(row.data["files"]),
        grader_environment=required_grader_environment(context),
        output_paths=(OUTPUT_PATH,),
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                upstream_dataset="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                upstream_name="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                subset=None,
                build_path="Task Trove",
                origin=SourceOrigin.TASKTROVE,
                family="shell-cmd",
                tags=("agentic", "multi-turn", "language:bash"),
                count=1498,
                notes="Semantic output comparison against an oracle command run in the same sandbox.",
                task_format=TaskFormat.TASKCOMPENDIUM,
            ),
            config=CurationRecipe(
                name="tasktrove-nl2bash",
                source=tasktrove_source(CONFIG),
                convert=TaskTroveConverter(CONFIG, convert_nl2bash_task),
                version="3",
                intended_use=IntendedUse.TRAIN,
                rubric=NL2BASH_RUBRIC,
                controls=Controls(golden=solve_script),
                oracle_solutions=True,
                grader=GRADER_PACKAGES,
            ),
        )
    ]
