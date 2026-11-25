# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task catalog: training and evaluation dataset declarations, keyed by name."""

from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import OversightLevel

from experiments.post_training.task_curation.census_binding import bind_actor_census
from experiments.post_training.task_curation.content_identity import read_content_identity_receipts
from experiments.post_training.task_curation.datasets import (
    backfill_releases,
    campaign_releases,
    crosscode_python_repair,
    facet_terminal,
    native_releases,
    nemo_structured,
    repair_releases,
    structure_closure,
    terminus_pivot,
    thoughtcrime,
    workplace,
)
from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.cascade import tasks as cascade
from experiments.post_training.task_curation.datasets.consumer_decisions import consumer_decisions
from experiments.post_training.task_curation.datasets.consumer_receipts import consumer_receipts
from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
from experiments.post_training.task_curation.datasets.derived_contract_measurements import derived_contract_receipts
from experiments.post_training.task_curation.datasets.endless import tasks as endless
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
from experiments.post_training.task_curation.datasets.facet import tasks as facet
from experiments.post_training.task_curation.datasets.fineenvs import tasks as fineenvs
from experiments.post_training.task_curation.datasets.full_content_results import full_content_result_bindings
from experiments.post_training.task_curation.datasets.grading_measurements import grading_census_receipts
from experiments.post_training.task_curation.datasets.ifbench import tasks as ifbench
from experiments.post_training.task_curation.datasets.lean import numina as lean_numina
from experiments.post_training.task_curation.datasets.lean import proofs as lean_proofs
from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.datasets.litecoder import tasks as litecoder
from experiments.post_training.task_curation.datasets.mcp_atlas_easy import tasks as mcp_atlas_easy
from experiments.post_training.task_curation.datasets.mimo import tasks as mimo
from experiments.post_training.task_curation.datasets.nemotron_ultra import components as nemotron_ultra
from experiments.post_training.task_curation.datasets.openresearcher import recipe as openresearcher
from experiments.post_training.task_curation.datasets.oracle_admitted_releases import oracle_admitted_release_receipts
from experiments.post_training.task_curation.datasets.oracle_releases import oracle_release_receipts
from experiments.post_training.task_curation.datasets.pdbthink import tasks as pdbthink
from experiments.post_training.task_curation.datasets.polaris import tasks as polaris
from experiments.post_training.task_curation.datasets.prime_intellect import swe as prime_swe
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe import convert as prime_multiswe
from experiments.post_training.task_curation.datasets.pymethods2test import tasks as pymethods2test
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as reasoning_gym
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.datasets.sample_qc_results import sample_qc_receipts
from experiments.post_training.task_curation.datasets.skyrl import code as skyrl_code
from experiments.post_training.task_curation.datasets.skyrl import ifeval as skyrl_ifeval
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.datasets.skyrl import mcq as skyrl_mcq
from experiments.post_training.task_curation.datasets.skyrl import preference as skyrl_preference
from experiments.post_training.task_curation.datasets.swe_rebench_v2 import sources as swe_rebench_v2
from experiments.post_training.task_curation.datasets.tasktrove import calendar as tasktrove_calendar
from experiments.post_training.task_curation.datasets.tasktrove import code as tasktrove_code
from experiments.post_training.task_curation.datasets.tasktrove import identity_following as tasktrove_identity_repair
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
from experiments.post_training.task_curation.datasets.terminal_lego import tasks as terminal_lego
from experiments.post_training.task_curation.datasets.terminalworld import qualified as terminalworld
from experiments.post_training.task_curation.derived_contract_tags import bind_derived_contract_tags
from experiments.post_training.task_curation.grading_binding import bind_grading_census
from experiments.post_training.task_curation.lexical_binding import bind_lexical_measurement
from experiments.post_training.task_curation.lexical_policy import source_lexical_use
from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    bind_license_receipt,
    read_license_receipts,
)
from experiments.post_training.task_curation.oracle_admitted_release import (
    bind_oracle_admitted_release,
    bind_selected_source_content,
)
from experiments.post_training.task_curation.oracle_release import bind_oracle_filtered_release
from experiments.post_training.task_curation.oversight_binding import bind_derived_oversight
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.remediation_binding import (
    RetainedReleaseReceipt,
    bind_archived_retained_release,
    bind_retained_release,
    retained_lineage,
)
from experiments.post_training.task_curation.retirement import retired_admission_source_ids
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup, read_dedup_receipts

CASCADE_RETAINED_SUBSETS = ("multi-domain-RL", "MOPD")

# Awaiting the intake measurements listed in docs/references/task-curation.md#pending-source-intake.
OUT_OF_REGION_PENDING_IDS = frozenset(
    {
        # Sealed populations and censuses exist only in the rno2a region; the release
        # census refuses out-of-region populations (ruling 2026-11-24: don't block
        # R40). They stage here for R41, re-measured in-region.
        "TaskTrove:laion__stackexchange-codereview-sandboxes-verified-v2",
        "TaskTrove:laion__stackexchange-overflow-sandboxes-verified-v2",
        "TaskTrove:laion__stackexchange-superuser-sandboxes-verified-v2",
        "TaskTrove:laion__stackexchange-unix-sandboxes-verified-v2",
        "TaskTrove:laion__wizardlm-orca-v4",
    }
)

# Sources origin/main declares that the Atlas r39 cut does not carry under any ID
# (user request 2026-11-25, the main-source-sync landing). They enter through the
# standing pending mechanism only — staged for R41 intake at their main pins — never
# the served declarations. The KTO components are additionally license-banned: their
# pinned upstream trl-lib/kto-mix-14k declares no license (doubt-means-ban ruling,
# license registry 2026.11.25.a-main-sync, #11497), so they hold out of the served
# set and never enter intake; the ban lifts only off the registry with new primary
# evidence of a permissive grant.
MAIN_SYNC_STAGED_IDS = frozenset(
    {
        "Anthropic:hh_rlhf/harmless-base",
        "Anthropic:hh_rlhf/helpful-base",
        "Anthropic:hh_rlhf/helpful-online",
        "Anthropic:hh_rlhf/helpful-rejection-sampled",
        "TaskTrove:laion__glaive-code-assistant-sandboxes-verified-v2",
        "TaskTrove:laion__nemotron-gym-structured-outputs-v4",
        "TaskTrove:laion__stackexchange-tezos-sandboxes-verified-v2",
        "Nemotron:nemotron_ultra_mopd/hs3_en",
        "Nemotron:nemotron_ultra_mopd/hs3_multi",
        "Nemotron:nemotron_ultra_mopd/hs3_multiturn",
        "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        "Nemotron:nemotron_ultra_mopd/safety_en",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        "project-numina:NuminaMath-LEAN",
    }
)

MAIN_SYNC_BANNED_IDS = frozenset(
    {
        "TRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized",
        "TRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
        "TRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
    }
)

MAIN_SYNC_PENDING_IDS = MAIN_SYNC_STAGED_IDS | MAIN_SYNC_BANNED_IDS

# The 2026-11-24 freeze ruling: R40's source set was frozen at the cut — sources
# admitted or imported after it route to R41 through the standing mechanism.
# The FACET-Terminal and Endless imports unfroze for R41 (their blockers were fixed
# after the 5:10 PM EST ruling), but the R40 cut is frozen at the 334-source exact
# cover per that ruling — they hold here until the cut completes, then re-enter the
# served set for R41.
R41_PENDING_SOURCE_IDS = (
    OUT_OF_REGION_PENDING_IDS
    | MAIN_SYNC_PENDING_IDS
    | {
        "FACET-Terminal:FACET-Terminal/FACET-Terminal-Tasks-6k",
        "FWeindel:validated-tasks/endless",
        # The pinned population record measures one task more than the declared census
        # binding (13,969 vs 13,968) — the delta needs its owner's reconciliation, so it
        # holds here rather than refusing the release census.
        "TaskTrove:tasktrove-competitive-coding-rebuilt-v1",
        # The seven campaign-release rebuilds whose declared census bindings disagree
        # with every reachable population record by a census-generation delta
        # (ruling 2026-11-25: timebox expired, freeze to pending for R41).
        "TaskTrove:tasktrove-code-contests-built-image",
        "TaskTrove:tasktrove-nl2bash-rebuilt-v1",
        "TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1",
        "TaskTrove:tasktrove-pymethods2test-rebuilt-v1",
        "TaskTrove:tasktrove-taco-built-image",
        "TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1",
        "TaskTrove:tasktrove-unitsyn-python-rebuilt-v1",
        # The five Nemotron lean/rdkit renames: their fresh SAMPLE panels are drawn at
        # the current pins but await the judge review, so their declarations still
        # carry the archived-lineage marker (ruling 2026-11-25: timebox expired).
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
        # The Workplace admission landed after the 2026-11-24 freeze ruling: R40's source
        # set is frozen at the cut, so the newly admitted population routes to R41.
        "Nvidia:Nemotron-RL-agent-workplace_assistant",
    }
)

PENDING_MEASUREMENT_SOURCES = (
    *(source for source in cascade.sources() if source.info.subset not in CASCADE_RETAINED_SUBSETS),
    *lean_proofs.sources(),
    *lean_numina.sources(),
    *swe_rebench_v2.sources(),
    # The repaired identity-following variant is measured and admission-gated at its
    # sealed intake pin; it stages here until its release adopts it (R41), so its
    # registry receipts bind without entering the served declarations.
    *tasktrove_identity_repair.repaired_sources(),
    *(source for source in tasktrove_judged.sources() if source.info.id in OUT_OF_REGION_PENDING_IDS),
    *facet_terminal.sources(),
    *endless.cohort_source(),
    *pymethods2test.sources(),
    *workplace.sources(),
    *(
        source
        for source in campaign_releases.sources()
        if source.info.id == "TaskTrove:tasktrove-competitive-coding-rebuilt-v1"
    ),
    *(
        source
        for source in campaign_releases.sources()
        if source.info.id
        in {
            "TaskTrove:tasktrove-code-contests-built-image",
            "TaskTrove:tasktrove-nl2bash-rebuilt-v1",
            "TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1",
            "TaskTrove:tasktrove-pymethods2test-rebuilt-v1",
            "TaskTrove:tasktrove-taco-built-image",
            "TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1",
            "TaskTrove:tasktrove-unitsyn-python-rebuilt-v1",
        }
    ),
    *(
        source
        for source in nemotron_ultra.sources()
        if source.info.id
        in {
            "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
            "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
            "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
            "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
            "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
        }
    ),
    # The main-sync staged sources (2026-11-25): declared on origin/main, absent
    # from the Atlas r39 cut, staged for R41 intake at their main pins. The
    # license-banned KTO components never enter this listing.
    *(source for source in skyrl_preference.sources() if source.info.id in MAIN_SYNC_STAGED_IDS),
    *(source for source in nemotron_ultra.sources() if source.info.id in MAIN_SYNC_STAGED_IDS),
    *(source for source in tasktrove_structured_outputs.candidate_sources() if source.info.id in MAIN_SYNC_STAGED_IDS),
    *(source for source in tasktrove_judged.candidate_sources() if source.info.id in MAIN_SYNC_STAGED_IDS),
)


def configured_sources() -> list[RlDataSource]:
    """Return source populations registered for Atlas evidence binding."""
    sources = [
        *litecoder.sources(),
        *terminalworld.sources(),
        *mcp_atlas_easy.sources(),
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
        *swe_rebench_v2.admitted_sources(),
        *tasktrove_math.sources(),
        *(source for source in tasktrove_judged.sources() if source.info.id not in OUT_OF_REGION_PENDING_IDS),
        *tasktrove_qa.sources(),
        *tasktrove_calendar.sources(),
        *tasktrove_instructions.sources(),
        *tasktrove_multichallenge.sources(),
        *tasktrove_puzzles.sources(),
        *arc.sources(),
        *reasoning_gym.sources(),
        *nemotron_ultra.sources(),
        *nemo_structured.sources(),
        *workplace.sources(),
        *(source for source in cascade.sources() if source.info.subset in CASCADE_RETAINED_SUBSETS),
        *ifbench.sources(),
        *openresearcher.sources(),
        *polaris.sources(),
        *pdbthink.sources(),
        *facet.sources(),
        *facet_terminal.sources(),
        *endless.cohort_source(),
        *fineenvs.sources(fineenvs.PUBLISHED_IMAGES),
        *mimo.sources(),
        *terminal_lego.sources(),
        *structure_closure.sources(),
        *thoughtcrime.sources(),
        *campaign_releases.sources(),
        *native_releases.sources(),
        *repair_releases.sources(),
        *backfill_releases.sources(),
        *terminus_pivot.sources(),
        *prime_multiswe.sources(),
        *prime_swe.sources(),
    ]
    return [source for source in sources if source.info.id not in R41_PENDING_SOURCE_IDS]


def retained_binding(source: RlDataSource, receipt: RetainedReleaseReceipt) -> RlDataSource:
    """Bind the retained release, or keep the configured pin and record the drift.

    A receipt whose conversion pin no longer matches the configured declaration used to
    crash the whole population build; the audit reports the drift as a typed
    served-catalog gap instead (ruling 2026-11-19), so one drifted source cannot block
    every lane's landing. The configured declaration stays current and carries the note.
    """
    try:
        return bind_retained_release(source, receipt)
    except ValueError as error:
        RETAINED_BINDING_DRIFTS[source.info.id] = receipt.publication.plan.source_revision
        note = f"Retained release drift: {error}."
        return replace(source, info=replace(source.info, notes=f"{source.info.notes} {note}".strip()))


RETAINED_BINDING_DRIFTS: dict[str, str] = {}
"""The retained release receipts whose conversion pin no longer matches the configured
declaration, recorded by the population build for the declaration audit (ruling
2026-11-19): the audit routes each drifted source to a waivable served-catalog gap
instead of failing the run."""


def sources_before_oracle_filtration() -> dict[str, RlDataSource]:
    """Return configured populations before selecting oracle-filtered private pins."""
    actor_receipts = actor_census_receipts()
    lexical_receipts = lexical_measurements()
    retained_receipts = retained_release_receipts()
    grading_receipts = grading_census_receipts()
    dedup_receipts = read_dedup_receipts(
        Path(__file__).with_name("datasets") / "within_source_dedup.json", load_json=read_registry_json
    )
    archived = {
        source_id: receipt.archived_original
        for source_id, receipt in oracle_admitted_release_receipts().items()
        if receipt.archived_original is not None
    }
    sources = [source for source in configured_sources() if source.info.id not in archived]
    sources = [bind_actor_census(source, actor_receipts.get(source.info.id)) for source in sources]
    sources = [
        bind_lexical_measurement(source, lexical_receipts.get(source.info.id), source_lexical_use(source))
        for source in sources
    ]
    sources = [bind_within_source_dedup(source, dedup_receipts[source.info.id]) for source in sources]
    sources = [
        retained_binding(source, retained_receipts[source.info.id]) if source.info.id in retained_receipts else source
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
    sources.extend(bind_archived_retained_release(binding) for binding in archived.values())
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
        if source_id not in sources:
            continue  # the R40 freeze routes sources admitted after the cut to R41
        sources[source_id] = bind_oracle_filtered_release(sources[source_id], receipt)
    for source_id, receipt in admitted.items():
        if source_id not in sources:
            continue  # the R40 freeze routes sources admitted after the cut to R41
        sources[source_id] = bind_oracle_admitted_release(sources[source_id], receipt)
    sources = {source_id: crosscode_python_repair.bind_source(source) for source_id, source in sources.items()}
    samples, full_content = sample_qc_receipts(), full_content_result_bindings()
    content_identities = read_content_identity_receipts()
    retained_records = dict(retained_release_receipts())
    for source_id, receipt in admitted.items():
        if receipt.archived_original is not None:
            retained_records[source_id] = receipt.archived_original.retained
    lineages = {
        source_id: retained_lineage(receipt) for source_id, receipt in retained_records.items() if source_id in sources
    }
    selected = {
        source_id: bind_selected_source_content(
            originals[source_id],
            source,
            samples.get(source_id),
            full_content.get(source_id),
            retained_lineage=lineages.get(source_id),
            content_identity=content_identities.get(source_id),
        )
        for source_id, source in sources.items()
    }
    license_receipts = read_license_receipts(LICENSE_RECEIPT_INDEX)
    selected = {
        source_id: bind_license_receipt(source, license_receipts.get(source_id))
        for source_id, source in selected.items()
    }
    receipts = derived_contract_receipts()
    selected = {
        source_id: bind_derived_oversight(
            source,
            samples.get(source.info.id),
            full_content.get(source.info.id),
            lineages.get(source.info.id),
        )
        for source_id, source in selected.items()
    }
    selected = {
        source_id: bind_derived_contract_tags(source, receipts.get(source.info.id))
        for source_id, source in selected.items()
    }
    consumers = consumer_receipts()
    decisions = consumer_decisions()
    registered = {
        source_id: replace(
            source,
            info=replace(
                source.info,
                consumer_receipt=consumers.get(source_id),
                consumer_requirements=tuple(sorted(consumers[source_id].requirements)) if source_id in consumers else (),
                consumer_decision=decisions.get(source_id),
            ),
        )
        for source_id, source in selected.items()
    }
    undeclared = sorted(source.info.id for source in registered.values() if source.info.task_format is None)
    if undeclared:
        raise ValueError(
            "Sources declare no task format; every source registration states exactly one TaskFormat value: "
            + ", ".join(undeclared)
        )
    return registered


def require_accepted_task_format(source: RlDataSource) -> None:
    """Reject a source whose registration states no task format.

    Every registration states exactly one shape; the conversion census verifies
    the declared shape against every row's detected shape.
    """
    if source.info.task_format is None:
        raise ValueError(
            f"Source {source.info.id} declares no task format; "
            "every source registration states exactly one TaskFormat value"
        )


def runnable_sources() -> dict[str, RlDataSource]:
    """Available sources with dataset-owned execution, keyed by invocation name.

    Sources whose evidence-derived oversight is QUICK stay out of the default
    selection: QUICK QC admits them to the Atlas but not to default training.
    Explicitly requested selections reach them through the driver's registered
    catalog, like Deprecated sources.
    """
    sources = [
        source
        for source in all_sources().values()
        if not source.admission_blockers
        and source.declaration.status == "Available"
        and (source.info.oversight is None or source.info.oversight.level is not OversightLevel.QUICK)
    ]
    for source in sources:
        require_accepted_task_format(source)
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate RL pipeline names")
    return {source.name: source for source in sources}
