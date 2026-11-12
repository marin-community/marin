# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executed QUICK releases recovered from the retired source ledger."""

import json
from pathlib import Path

from finestore.rl.curation import (
    ConvertedExamples,
    GradingEnvironmentEvidence,
    IndependentQC,
    OracleValidation,
)

from experiments.post_training.task_curation.native_release import import_native_quick
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat

LEDGER = Path(__file__).parents[1] / "retired-inventory-only.json"


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    result = []
    for entry in json.loads(LEDGER.read_text())["entries"]:
        if entry["disposition"] != "quick_converted":
            continue
        release = entry["disposition_release"]
        evidence = f"https://huggingface.co/datasets/{release['repo']}/blob/{release['revision']}/manifest.json"
        config = PrebuiltCohort(
            name=entry["name"],
            version="2026.11.12.1",
            repo=release["repo"],
            revision=release["revision"],
            expected_rows=release["converted_rows"],
        )
        tags = tuple(tag for tag in entry["tags"] if tag != "excluded")
        result.append(
            RlDataSource(
                config=config,
                pipeline=import_native_quick,
                info=SourceInfo(
                    id=entry["id"],
                    title=entry["display_name"],
                    origin=entry["origin"],
                    family=entry["family"],
                    tags=tags,
                    task_format=TaskFormat.HARBOR,
                    count=release["converted_rows"],
                    notes=(
                        entry["notes"].split(" Admission blocked:")[0]
                        + " Native Harbor package bytes are preserved in a private TaskCompendium QUICK release. "
                        "QUICK is unreviewed and supplies no oracle or admission claim. "
                        "Native image tags and build downloads may remain mutable. Source defects remain recorded."
                    ),
                    independent_qc=(
                        IndependentQC.model_validate(entry["independent_qc"]) if entry["independent_qc"] else None
                    ),
                    oracle_validation=OracleValidation.SKIPPED,
                    grading_environment=GradingEnvironmentEvidence(
                        **release["grading_environment"], evidence_url=evidence
                    ),
                    converted_examples=ConvertedExamples(
                        count=release["converted_rows"],
                        input_rows=release["input_rows"],
                        rejections=release["rejections"],
                        admission_rejections={},
                        source_dataset_id=entry["dataset_id"],
                        source_revision=entry["dataset_revision"],
                        converter_revision=release["converter_identity"],
                        converter_version="2026.11.12.1",
                        evidence_url=evidence,
                    ),
                ),
            )
        )
    return result
