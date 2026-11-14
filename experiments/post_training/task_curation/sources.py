# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task catalog: training and evaluation dataset declarations, keyed by name."""


from pathlib import Path

from experiments.post_training.task_curation.census_binding import bind_actor_census
from experiments.post_training.task_curation.datasets import (
    campaign_releases,
    crosscode_python_repair,
    native_releases,
    structure_closure,
    thoughtcrime,
)
from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
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
from experiments.post_training.task_curation.datasets.full_content_results import full_content_result_bindings
from experiments.post_training.task_curation.datasets.grading_measurements import grading_census_receipts
from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.datasets.nemotron_ultra import components as nemotron_ultra
from experiments.post_training.task_curation.datasets.oracle_admitted_releases import oracle_admitted_release_receipts
from experiments.post_training.task_curation.datasets.oracle_releases import oracle_release_receipts
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as reasoning_gym
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.datasets.sample_qc_results import sample_qc_receipts
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
from experiments.post_training.task_curation.grading_binding import bind_grading_census
from experiments.post_training.task_curation.lexical_binding import bind_lexical_measurement
from experiments.post_training.task_curation.lexical_policy import source_lexical_use
from experiments.post_training.task_curation.oracle_admitted_release import (
    bind_oracle_admitted_release,
    bind_selected_source_content,
)
from experiments.post_training.task_curation.oracle_release import bind_oracle_filtered_release
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.remediation_binding import bind_retained_release
from experiments.post_training.task_curation.retirement import retired_admission_source_ids
from experiments.post_training.task_curation.source import RlDataSource, TaskFormat
from experiments.post_training.task_curation.source_policy import GRANDFATHERED_MARINSKYRL_SOURCES
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup, read_dedup_receipts


def sources_before_oracle_filtration() -> dict[str, RlDataSource]:
    """Return configured populations before selecting oracle-filtered private pins."""
    actor_receipts = actor_census_receipts()
    lexical_receipts = lexical_measurements()
    retained_receipts = retained_release_receipts()
    grading_receipts = grading_census_receipts()
    dedup_receipts = read_dedup_receipts(
        Path(__file__).with_name("datasets") / "within_source_dedup.json", load_json=read_registry_json
    )
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
        *native_releases.sources(),
    ]
    sources = [bind_actor_census(source, actor_receipts.get(source.info.id)) for source in sources]
    sources = [
        bind_lexical_measurement(source, lexical_receipts.get(source.info.id), source_lexical_use(source))
        for source in sources
    ]
    sources = [bind_within_source_dedup(source, dedup_receipts[source.info.id]) for source in sources]
    sources = [
        (
            bind_retained_release(source, retained_receipts[source.info.id])
            if source.info.id in retained_receipts
            else source
        )
        for source in sources
    ]
    # Retained selection changes the active pin and actor input for new measurements.
    sources = [
        bind_lexical_measurement(source, lexical_receipts.get(source.info.id), source_lexical_use(source))
        for source in sources
    ]
    sources = [
        bind_grading_census(source, grading_receipts[source.info.id]) if source.info.id in grading_receipts else source
        for source in sources
    ]
    identifiers = [source.info.id for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate RL source identifiers")
    return {source.info.id: source for source in sources if source.info.id not in retired_admission_source_ids()}


def all_sources() -> dict[str, RlDataSource]:
    receipts = oracle_release_receipts()
    admitted = oracle_admitted_release_receipts()
    if receipts.keys() & admitted.keys():
        raise ValueError("A source cannot select both empty and admitted oracle releases")
    originals = sources_before_oracle_filtration()
    sources = dict(originals)
    for source_id, receipt in receipts.items():
        sources[source_id] = bind_oracle_filtered_release(sources[source_id], receipt)
    for source_id, receipt in admitted.items():
        sources[source_id] = bind_oracle_admitted_release(sources[source_id], receipt)
    sources = {source_id: crosscode_python_repair.bind_source(source) for source_id, source in sources.items()}
    samples, full_content = sample_qc_receipts(), full_content_result_bindings()
    return {
        source_id: bind_selected_source_content(
            originals[source_id], source, samples.get(source_id), full_content.get(source_id)
        )
        for source_id, source in sources.items()
    }


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
