# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repair lane's sealed cohorts over their published private oracle-filtered releases.

Each entry is the repaired sub-population of a tracked TaskTrove source, rebuilt through the
tracked converter at the pinned revision and admitted by the repair lane's oracle and
physical nop-control campaign. The declaration binds the licensed TaskTrove upstream as its
dataset pin (the filtration evidence's original release); ``bind_oracle_admitted_release``
selects the published private pin as the runnable consumer.
"""

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import (
    PrebuiltCohort,
    import_prebuilt,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

TASKTROVE_UPSTREAM_REPO = "open-thoughts/TaskTrove"
TASKTROVE_UPSTREAM_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
TASKTROVE_UPSTREAM_URL = (
    "https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a"
)
CONVERTER_REVISION = "900cb18689180a5b41747594a8de534518855ea7"


def _repair_cohort(
    *,
    source_id: str,
    upstream_dataset: str,
    variant: str,
    cohort_name: str,
    recipe_version: str,
    expected_rows: int,
    release_repo: str,
    release_revision: str,
    family: str,
    tags: tuple[str, ...],
    upstream_url: str,
    count_basis: str,
    verification: str,
    notes: str,
    upstream_repo: str = TASKTROVE_UPSTREAM_REPO,
    upstream_revision: str = TASKTROVE_UPSTREAM_REVISION,
    origin: SourceOrigin = SourceOrigin.TASKTROVE,
    task_format: TaskFormat = TaskFormat.TASKSPEC,
) -> RlDataSource[PrebuiltCohort]:
    """One sealed repair cohort pending its oracle-admitted release binding."""
    return RlDataSource(
        pipeline=import_prebuilt,
        config=PrebuiltCohort(
            intended_use=IntendedUse.TRAIN,
            name=cohort_name,
            version=recipe_version,
            repo=upstream_repo,
            revision=upstream_revision,
            expected_rows=expected_rows,
        ),
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=source_id,
            upstream_dataset=upstream_dataset,
            upstream_name=upstream_dataset,
            subset=None,
            variant=variant,
            build_path="Repair Lane",
            origin=origin,
            upstream_url=upstream_url,
            family=family,
            tags=tags,
            count=expected_rows,
            task_format=task_format,
            notes=notes,
        ),
    )


def sources() -> list[RlDataSource]:
    return [
        _repair_cohort(
            source_id="TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2@repaired-20261118",
            upstream_dataset="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
            variant="repaired-20261118",
            cohort_name="atlas-repair-nl2bash-20261118",
            recipe_version="3",
            expected_rows=372,
            release_repo="open-athena/atlas-repair-nl2bash-20261118-oracle-filter-v1",
            release_revision="a46af53c31658e57d60c4bde7b18f00e70945c66",
            family="shell-cmd",
            tags=("agentic", "language:bash", "repair-release", "converted"),
            upstream_url=TASKTROVE_UPSTREAM_URL,
            count_basis=(
                "Production machine_spec census over all 372 published TaskSpec+runtime envelopes: "
                "372 machine slots, zero execution failures"
            ),
            verification=(
                "The repair lane's comparative review accepted 372 of the 1,025 mutual-best repaired "
                "nl2bash pairs; every accepted pair's oracle passed the production Harbor trial path and "
                "carries a conclusive nonpassing physical nop control"
            ),
            notes=(
                "Repaired sub-population of TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2: the "
                "372 comparative-review winners rebuilt through the tracked tasktrove-nl2bash recipe v3 "
                "at converter 900cb18689, admitted by the full-population oracle campaign with physical "
                "nop controls. The tracked source itself is unchanged."
            ),
        ),
        _repair_cohort(
            source_id="TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@repaired-20261118",
            upstream_dataset="DCAgent__exp_rpt_pymethods2test-large-v2",
            variant="repaired-20261118",
            cohort_name="atlas-repair-pymethods-20261118",
            recipe_version="5",
            expected_rows=3754,
            release_repo="open-athena/atlas-repair-pymethods-20261118-oracle-filter-v1",
            release_revision="a5cfc803a80fe5eb7e0b3ee4d4d9d70a1d22c0df",
            family="unit-test-gen",
            tags=("agentic", "language:python", "repair-release", "converted"),
            upstream_url=TASKTROVE_UPSTREAM_URL,
            count_basis=(
                "Production machine_spec census over all 3,754 published TaskSpec+runtime envelopes: "
                "3,754 machine slots, zero execution failures"
            ),
            verification=(
                "The repair lane's comparative review accepted 3,754 repaired pymethods2test pairs with "
                "zero conflicting matches; every accepted pair's oracle passed the production Harbor "
                "trial path and carries a conclusive nonpassing physical nop control"
            ),
            notes=(
                "Repaired sub-population of TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2: the "
                "3,754 comparative-review winners rebuilt through the tracked tasktrove-pymethods_large "
                "recipe v5 at converter 900cb18689 (wrapper solve.sh normalization applied), admitted by "
                "the full-population oracle campaign with physical nop controls. The tracked source "
                "itself is unchanged."
            ),
        ),
    ]
