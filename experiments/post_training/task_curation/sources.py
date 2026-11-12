# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task catalog: training and evaluation dataset declarations, keyed by name."""

from dataclasses import replace

from experiments.post_training.task_curation.datasets import (
    campaign_releases,
    structure_closure,
    thoughtcrime,
)
from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.daytona_measurements import DAYTONA_MEASUREMENTS
from experiments.post_training.task_curation.datasets.evalchemy import code as eval_code
from experiments.post_training.task_curation.datasets.evalchemy import financebench as eval_financebench
from experiments.post_training.task_curation.datasets.evalchemy import harness as eval_harness
from experiments.post_training.task_curation.datasets.evalchemy import instructions as eval_instructions
from experiments.post_training.task_curation.datasets.evalchemy import jee as eval_jee
from experiments.post_training.task_curation.datasets.evalchemy import long_context as eval_long_context
from experiments.post_training.task_curation.datasets.evalchemy import math as eval_math
from experiments.post_training.task_curation.datasets.evalchemy import mmlu as eval_mmlu
from experiments.post_training.task_curation.datasets.evalchemy import nupa as eval_nupa
from experiments.post_training.task_curation.datasets.evalchemy import reasoning as eval_reasoning
from experiments.post_training.task_curation.datasets.evalchemy import simpleqa as eval_simpleqa
from experiments.post_training.task_curation.datasets.nemotron_ultra import components as nemotron_ultra
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as reasoning_gym
from experiments.post_training.task_curation.datasets.skyrl import code as skyrl_code
from experiments.post_training.task_curation.datasets.skyrl import ifeval as skyrl_ifeval
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.datasets.skyrl import mcq as skyrl_mcq
from experiments.post_training.task_curation.datasets.skyrl import preference as skyrl_preference
from experiments.post_training.task_curation.datasets.tasktrove import calendar as tasktrove_calendar
from experiments.post_training.task_curation.datasets.tasktrove import code as tasktrove_code
from experiments.post_training.task_curation.datasets.tasktrove import instruction_following as tasktrove_instructions
from experiments.post_training.task_curation.datasets.tasktrove import judged as tasktrove_judged
from experiments.post_training.task_curation.datasets.tasktrove import math as tasktrove_math
from experiments.post_training.task_curation.datasets.tasktrove import multichallenge as tasktrove_multichallenge
from experiments.post_training.task_curation.datasets.tasktrove import nl2bash as tasktrove_nl2bash
from experiments.post_training.task_curation.datasets.tasktrove import puzzles as tasktrove_puzzles
from experiments.post_training.task_curation.datasets.tasktrove import python_tests as tasktrove_python_tests
from experiments.post_training.task_curation.datasets.tasktrove import qa as tasktrove_qa
from experiments.post_training.task_curation.datasets.tasktrove import repositories as tasktrove_repositories
from experiments.post_training.task_curation.datasets.tasktrove import structured_outputs as tasktrove_structured_outputs
from experiments.post_training.task_curation.source import GRANDFATHERED_MARINSKYRL_SOURCES, RlDataSource, TaskFormat


def all_sources() -> dict[str, RlDataSource]:
    sources = [
        *eval_code.sources(),
        *eval_harness.sources(),
        *eval_instructions.sources(),
        *eval_jee.sources(),
        *eval_math.sources(),
        *eval_reasoning.sources(),
        *eval_mmlu.sources(),
        *eval_financebench.sources(),
        *eval_simpleqa.sources(),
        *eval_long_context.sources(),
        *eval_nupa.sources(),
        *skyrl_math.sources(),
        *skyrl_code.sources(),
        *skyrl_ifeval.sources(),
        *skyrl_mcq.sources(),
        *skyrl_preference.sources(),
        *tasktrove_code.sources(),
        *tasktrove_python_tests.sources(),
        *tasktrove_nl2bash.sources(),
        *tasktrove_structured_outputs.sources(),
        *tasktrove_repositories.sources(),
        *tasktrove_math.sources(),
        *tasktrove_judged.sources(),
        *tasktrove_qa.sources(),
        *tasktrove_calendar.sources(),
        *tasktrove_instructions.sources(),
        *tasktrove_multichallenge.sources(),
        *tasktrove_puzzles.sources(),
        *arc.sources(),
        *reasoning_gym.sources(),
        *nemotron_ultra.sources(),
        *structure_closure.sources(),
        *thoughtcrime.sources(),
        *campaign_releases.sources(),
    ]
    measured_sources = []
    for source in sources:
        measurement = DAYTONA_MEASUREMENTS.get(source.info.id)
        if measurement is not None:
            dataset = source.dataset
            if dataset is None or (dataset.name, dataset.revision) != (
                measurement.dataset_id,
                measurement.dataset_revision,
            ):
                raise ValueError(f"Daytona census does not match source input pin: {source.info.id}")
            fields = measurement.model_dump(
                exclude={"dataset_id", "dataset_revision", "daytona_unsafe_reasons", "daytona_roles"}
            )
            fields["daytona_unsafe_reasons"] = tuple(measurement.daytona_unsafe_reasons)
            fields["daytona_roles"] = measurement.daytona_roles
            source = replace(source, info=replace(source.info, **fields))
        measured_sources.append(source)
    sources = measured_sources
    identifiers = [source.info.id for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate RL source identifiers")
    return {source.info.id: source for source in sources}


def require_accepted_task_format(source: RlDataSource) -> None:
    """Reject an agentic source whose payload is not an accepted task format.

    New agentic sources must ingest TaskCompendium or Harbor task directories.
    MarinSkyRL's internal blend format is accepted only for the frozen
    grandfather list, so a new internal-format source fails selection.
    """
    if "agentic" not in source.info.tags:
        return
    task_format = source.info.task_format
    if task_format is None:
        raise ValueError(
            f"Agentic source {source.info.id} declares no task format; "
            "declare TaskFormat.TASKCOMPENDIUM or TaskFormat.HARBOR"
        )
    if task_format is TaskFormat.MARINSKYRL and source.info.id not in GRANDFATHERED_MARINSKYRL_SOURCES:
        raise ValueError(
            f"Agentic source {source.info.id} ingests MarinSkyRL's internal format and is not grandfathered"
        )


def runnable_sources() -> dict[str, RlDataSource]:
    """Available sources with dataset-owned execution, keyed by invocation name."""
    sources = [
        source
        for source in all_sources().values()
        if not source.admission_blockers and source.declaration.status == "Available"
    ]
    for source in sources:
        require_accepted_task_format(source)
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate RL pipeline names")
    return {source.name: source for source in sources}
