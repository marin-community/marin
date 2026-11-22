# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The requirement-backfill's sealed cohorts over their published private releases.

Each entry is the requirement-backfilled population of a tracked TaskTrove source: the
same retained tasks whose declared requirements the static rules completed under rule
revision 2 (``harbor-v2`` policy). The carry-forward binding receipt proves every
backfilled task differs from its parent only in ``environment_requirements.capabilities``
and the evidence label, so the parent's oracle and QC verdicts carry forward; admission
runs through the standard Atlas gates on the new pin. The declaration binds the licensed
TaskTrove upstream as its dataset pin, and ``bind_oracle_admitted_release`` selects the
published private pin as the runnable consumer.
"""

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.repair_releases import (
    TASKTROVE_UPSTREAM_REPO,
    TASKTROVE_UPSTREAM_REVISION,
    TASKTROVE_UPSTREAM_URL,
)
from experiments.post_training.task_curation.prebuilt import (
    PrebuiltCohort,
    import_prebuilt,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

BACKFILL_VARIANT = "backfilled-20261122"
BACKFILL_POLICY_REVISION = "harbor-v2-r3"


def _backfill_cohort(
    *,
    source_id: str,
    upstream_dataset: str,
    cohort_name: str,
    expected_rows: int,
    release_repo: str,
    release_revision: str,
    count_basis: str,
    verification: str,
    notes: str,
) -> RlDataSource[PrebuiltCohort]:
    """One requirement-backfilled cohort over its published private release."""
    return RlDataSource(
        pipeline=import_prebuilt,
        config=PrebuiltCohort(
            intended_use=IntendedUse.TRAIN,
            name=cohort_name,
            version=BACKFILL_POLICY_REVISION,
            repo=TASKTROVE_UPSTREAM_REPO,
            revision=TASKTROVE_UPSTREAM_REVISION,
            expected_rows=expected_rows,
        ),
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=source_id,
            upstream_dataset=upstream_dataset,
            upstream_name=upstream_dataset,
            subset=None,
            variant=BACKFILL_VARIANT,
            build_path="Requirement Backfill",
            origin=SourceOrigin.TASKTROVE,
            upstream_url=TASKTROVE_UPSTREAM_URL,
            family="unit-test-gen",
            tags=("agentic", "language:python", "backfill-release", "converted"),
            count=expected_rows,
            task_format=TaskFormat.TASKSPEC,
            notes=notes,
        ),
    )


def sources() -> list[RlDataSource]:
    return [
        _backfill_cohort(
            source_id="TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@backfilled-20261122",
            upstream_dataset="DCAgent__exp_rpt_pymethods2test-large-v2",
            cohort_name="atlas-backfill-pymethods-20261122",
            expected_rows=4949,
            release_repo="open-athena/atlas-backfill-pymethods-20261122-v1",
            release_revision="565b26154a107c0ba651a33177785d3087125e20",
            count_basis=(
                "The repaired population's actor census over all 4,951 TaskSpecs; the backfill rewrote "
                "declared requirements only and the setup-command repair dropped the redundant Dockerfile "
                "replay, verified per task by the repair binding receipt"
            ),
            verification=(
                "Carry-forward binding receipt: every backfilled task differs from its parent only in "
                "environment_requirements.capabilities (extended) and the requirements.backfill.capabilities "
                "label; the setup-command repair additionally dropped the Dockerfile-replay pip install the "
                "pinned image already provides (fail-closed verified per task against the task's own "
                "Dockerfile RUN step). The local route is confirmed by executed controls on the repaired "
                "population (3-task smoke: golden reward=1.0, route_tier=local); the full 9,900-control "
                "campaign is running. The parent's original-oracle filtration verdicts carry forward "
                "unchanged"
            ),
            notes=(
                "Requirement-backfilled sub-population of TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2: "
                "python declared from the shipped sources on 4,951 tasks, network left not-stated after the "
                "build-time Dockerfile evidence was excluded (rule revision 2, policy harbor-v2-r2); the "
                "Dockerfile-replay setup command dropped so the local tier serves without runtime network; "
                "11 tasks carry a per-capability ambiguous signal for the LLM residual (0.22 percent). The "
                "parent source and its published revision are unchanged; this variant re-enters the Atlas "
                "queue for readmission through the standard gates."
            ),
        ),
        _backfill_cohort(
            source_id="TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2@backfilled-20261122",
            upstream_dataset="DCAgent__exp_rpt_unitsyn-python-large-v2",
            cohort_name="atlas-backfill-unitsyn-20261122",
            expected_rows=4947,
            release_repo="open-athena/atlas-backfill-unitsyn-20261122-v1",
            release_revision="db02de39b8174bd6bb7b45194f05233c55e102a6",
            count_basis=(
                "The repaired population's actor census over all 4,949 TaskSpecs; the backfill rewrote "
                "declared requirements only and the setup-command repair dropped the redundant Dockerfile "
                "replay, verified per task by the repair binding receipt"
            ),
            verification=(
                "Carry-forward binding receipt: every backfilled task differs from its parent only in "
                "environment_requirements.capabilities (extended) and the requirements.backfill.capabilities "
                "label; the setup-command repair additionally dropped the Dockerfile-replay pip install the "
                "pinned image already provides (fail-closed verified per task against the task's own "
                "Dockerfile RUN step). The local route is confirmed by executed controls on the repaired "
                "population (3-task smoke: golden reward=1.0, route_tier=local); the full 9,900-control "
                "campaign is running. The parent's original-oracle filtration verdicts carry forward "
                "unchanged"
            ),
            notes=(
                "Requirement-backfilled sub-population of TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2: "
                "python declared from the shipped sources on 4,949 tasks, network left not-stated after the "
                "build-time Dockerfile evidence was excluded (rule revision 2, policy harbor-v2-r2); the "
                "Dockerfile-replay setup command dropped so the local tier serves without runtime network; "
                "11 tasks "
                "carry a per-capability ambiguous signal for the LLM residual (0.22 percent). The parent "
                "source and its published revision are unchanged; this variant re-enters the Atlas queue "
                "for readmission through the standard gates."
            ),
        ),
    ]
