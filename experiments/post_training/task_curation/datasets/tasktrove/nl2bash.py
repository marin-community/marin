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
from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.code import ANSWERABILITY_CRITERIA
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import RlDataPipeline, environment_requirements
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = DataSourceMetadata(id="", name="", origin="Task Trove", recorded_at="2026-10-08")

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
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                name="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
                display_name="DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                url="https://huggingface.co/datasets/open-athena/task-trove",
                dataset_id="open-athena/task-trove",
                revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
                revised_at="2026-10-08T09:34:47.000Z",
                dataset_revision="8c85912822da0a77978e285af923eecc48ae34a3",
                verifier_revision=None,
                family="shell-cmd",
                environment="Harbor",
                type="Agentic",
                turns="Multi-turn",
                task_count=1497,
                count_basis="Released Harbor tasks: manifest by_source.converted",
                count_precision="exact",
                count_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                notes="Semantic output comparison against an oracle command run in the same sandbox.",
                benchmark_basis="Release manifest does not designate benchmarks",
                family_basis="Task Trove release manifest source_verdicts.family",
                family_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
                canonical_source="DCAgent2/nl2bash-tasks-cleaned-oracle-v2",
                canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
                provenance_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5"
                    "bbccb803e826563f5718dbf/manifest.json"
                ),
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
