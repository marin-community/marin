# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill2Env's pinned native release, with cached Harbor Hub acquisition."""


from taskcompendium.pipeline.models import Controls, IntendedUse

from experiments.post_training.task_curation.datasets.skill2env.conversion import (
    RELEASE,
    convert_task,
    golden,
    read_task,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HarborHubSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

RUBRIC = """
Compare the original task instructions with the unchanged programmatic verifier and private reference solution.
Flag hidden requirements, reward hacking opportunities, and valid solutions incorrectly rejected by static checks.

The grader receives the captured /workspace in a separate native image. Check whether grading depends on paths
outside that workspace, file modes, symlinks, empty directories, network access, or undeclared services. These
contracts need additional runtime support; do not infer equivalence from a passing reference solution alone.

The scalar reward is the arithmetic mean of native components, following the authors' outcome-reward method.
The optional rubric.md reward is not executed. Distinguish curation opinions from executable rewards.
"""


def sources() -> list[RlDataSource[CurationRecipe]]:
    name, revision = RELEASE.split("@")
    return [
        RlDataSource(
            info=SourceInfo(
                id="Harbor Hub:skill2env",
                title=name,
                origin="Harbor Hub",
                family="terminal-agent",
                tags=("agentic", "multi-turn"),
                count=7496,
                notes=(
                    "Native Harbor training tasks with reference solutions and programmatic component-reward verifiers. "
                    "The paper's private S2EBench is a separate evaluation set. "
                    "Project-owned source is Apache-2.0; third-party skills and assets retain their own terms."
                ),
            ),
            config=CurationRecipe(
                name="skill2env",
                source=HarborHubSource(name, revision, read=read_task),
                version="7",
                convert=convert_task,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=golden, memory_mb=4096),
            ),
            pipeline=process_rows,
        )
    ]
