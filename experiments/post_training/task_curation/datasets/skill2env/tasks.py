# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill2Env's pinned native release, with explicit native-image preparation."""

from dataclasses import dataclass, replace
from typing import Any

from marin.execution.lazy import ArtifactStep
from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.models import Controls, IntendedUse

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.datasets.skill2env.conversion import RELEASE, convert_task, golden
from experiments.post_training.task_curation.pipeline import CurationRecipe, HarborSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

RUBRIC = """
Compare the original task instructions with the unchanged programmatic verifier and private reference solution.
Flag hidden requirements, reward hacking opportunities, and valid solutions incorrectly rejected by static checks.

The grader receives the captured /workspace in a separate native image. Check whether grading depends on paths
outside that workspace, file modes, symlinks, empty directories, network access, or undeclared services. These
contracts need additional runtime support; do not infer equivalence from a passing reference solution alone.

The scalar reward is the arithmetic mean of native components; full pass requires every component to be one
within 1e-9. The optional rubric.md reward is not executed. Distinguish curation opinions from executable rewards.
"""


@dataclass(frozen=True)
class NativeMachines:
    machines: GradingMachines

    def identity(self) -> dict[str, Any]:
        return {"backend": self.machines.identity(), "cpus": 2, "memory_mb": 4096, "storage_mb": 4096}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        factory, spec = self.machines.machine(environment, memory_mb)
        return factory, replace(spec, cpus=2, memory_mb=4096, storage_mb=4096)


def process_skill2env(source: RlDataSource[CurationRecipe], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    settings = options.recipe_settings
    if settings is not None:
        if settings.config is not None and settings.config.machines is not None:
            settings = replace(
                settings, config=replace(settings.config, machines=NativeMachines(settings.config.machines))
            )
        elif settings.machines is not None:
            settings = replace(settings, machines=NativeMachines(settings.machines))
        options = replace(options, recipe_settings=settings)
    return process_rows(source, options)


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
                source=HarborSource(name, revision),
                version="1",
                convert=convert_task,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=golden, memory_mb=4096),
            ),
            pipeline=process_skill2env,
        )
    ]
