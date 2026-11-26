# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task catalog: training and evaluation dataset declarations, keyed by name."""

from dataclasses import replace

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.datasets import (
    atlas_publications,
    campaign_releases,
    dolci_precise_if_admitted,
    edge_websearch,
    inverse_ifeval_admitted,
    nemo_mcqa,
    nemo_tool_calls,
    qa_formatting,
    structure_closure,
    tasktrove_structured_native,
    thoughtcrime,
    unconverted,
)
from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.lean import numina as lean_numina
from experiments.post_training.task_curation.datasets.lean import proofs as lean_proofs
from experiments.post_training.task_curation.datasets.nemotron_ultra import components as nemotron_ultra
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as reasoning_gym
from experiments.post_training.task_curation.datasets.skyrl import code as skyrl_code
from experiments.post_training.task_curation.datasets.skyrl import ifeval as skyrl_ifeval
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.datasets.skyrl import mcq as skyrl_mcq
from experiments.post_training.task_curation.datasets.skyrl import preference as skyrl_preference
from experiments.post_training.task_curation.datasets.swe_rebench_v2 import sources as swe_rebench_v2
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

RUNTIME_STACK_SOURCES = frozenset(
    {
        "Nvidia:Nemotron-RL-InverseIFEval-v1",
        "Nvidia:Nemotron-RL-QA-Abstention-v1",
        "Nvidia:Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        "Nvidia:Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
    }
)
"""Admitted or ruled sources whose converters need the atlas runtime stack: the
fresh-machine ScriptGrader contract (an unbounded grader deadline and a declared
grader environment on the grader itself) and the judge tiers this catalog's
taskcompendium generation cannot express. They stay inventory records at their
pins until the runtime alignment lands; training waits."""


def _staged_runtime(source: RlDataSource) -> RlDataSource:
    """Project a converter-carrying declaration to an inventory record at its pin."""
    assert source.info.id in RUNTIME_STACK_SOURCES, source.info.id
    return RlDataSource(info=replace(source.info, dataset=source.dataset), pipeline=None)


def all_sources() -> dict[str, RlDataSource]:
    evaluation_sources = {}
    for benchmark in evaluation_benchmarks().values():
        # Shot/chat variants share invocation IDs. Keep their defaults, plus separately
        # named populations such as MRCR's context bins.
        for source in (
            *benchmark.resolve(),
            *(source for variant in benchmark.variants for source in benchmark.build(variant)),
        ):
            evaluation_sources.setdefault(source.info.id, source)
    sources = [
        *evaluation_sources.values(),
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
        *lean_proofs.sources(),
        *lean_numina.sources(),
        *arc.sources(),
        *reasoning_gym.sources(),
        *nemotron_ultra.sources(),
        *swe_rebench_v2.sources(),
        *dolci_precise_if_admitted.sources(),
        *(_staged_runtime(source) for source in inverse_ifeval_admitted.sources()),
        *nemo_mcqa.sources(),
        *tasktrove_structured_native.sources(),
        *edge_websearch.sources(),
        *(_staged_runtime(source) for source in qa_formatting.sources()),
        *structure_closure.sources(),
        *thoughtcrime.sources(),
        *nemo_tool_calls.sources(),
        *campaign_releases.sources(),
        *unconverted.sources(),
    ]
    sources = atlas_publications.bind_publications(sources)
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
    sources = [source for source in all_sources().values() if source.declaration.status == "Available"]
    for source in sources:
        require_accepted_task_format(source)
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate RL pipeline names")
    return {source.name: source for source in sources}
