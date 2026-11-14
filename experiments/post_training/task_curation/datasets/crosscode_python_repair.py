# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the private CrossCode Python repair without inheriting qualification."""

import json
from pathlib import Path

from finestore.rl.curation import DatasetRelease, OracleValidation, WithinSourceDedupEvidence
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat
from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_REASON
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup

SOURCE_ID = "TaskTrove:laion__exp_rpt_crosscodeeval-python-v2"


def bind_source(source: RlDataSource) -> RlDataSource:
    if source.info.id != SOURCE_ID or source.info.oracle_filtration is not None:
        return source
    repair = json.loads(Path(__file__).with_suffix(".json").read_text())
    if source.current_release != DatasetRelease.model_validate(repair["original_release"]):
        raise ValueError("CrossCode Python repair belongs to another original source pin")
    actor = ActorCensusReceipt.model_validate(repair["actor_receipt"])
    if actor.source_id != SOURCE_ID or actor.input.task_specs_sha256 != repair["task_specs_sha256"]:
        raise ValueError("CrossCode Python repair census belongs to another source population")
    config = PrebuiltCohort(
        name=source.name + "-encoding-repair",
        version=repair["version"],
        repo=repair["repo"],
        revision=repair["revision"],
        expected_rows=repair["task_count"],
        intended_use=IntendedUse.EVAL,
    )
    info = SourceInfo(
        id=SOURCE_ID,
        origin=source.info.origin,
        upstream_dataset=source.info.upstream_dataset,
        upstream_name=source.info.upstream_name,
        upstream_url=source.info.upstream_url,
        family=source.info.family,
        tags=tuple(dict.fromkeys((*source.info.tags, "eval", "excluded", "repair-candidate"))),
        build_path="CrossCode Python private encoding repair",
        current_release=DatasetRelease(repo=config.repo, revision=config.revision, url=config.dataset.url),
        dataset=config.dataset,
        count=config.expected_rows,
        task_format=TaskFormat.TASKCOMPENDIUM,
        oracle_validation=OracleValidation.SKIPPED,
        notes=(
            "Private repair candidate; full-population oracle filtration and negative controls remain pending. "
            "Against the preceding fragment-adapter candidate, only the grader image encoding and digest change. "
            "Its image configuration, uncompressed layers, fragment oracle, public task contract, four scoring tiers "
            "and source deadlines are preserved. "
            f"Repair provenance: {repair['repair_provenance']}. {CROSSCODE_EVAL_REASON}"
        ),
    )
    candidate = RlDataSource(config=config, pipeline=import_taskcompendium_prebuilt, info=info)
    candidate = bind_within_source_dedup(
        candidate, WithinSourceDedupEvidence.model_validate(repair["within_source_dedup"])
    )
    candidate = bind_actor_census(candidate, actor)
    if not candidate.has_actor_census:
        raise ValueError("CrossCode Python actor census does not cover the configured repair pin")
    return candidate
