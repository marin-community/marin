# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published Snowball structural-closure cohorts at their first retained pins.

The 2026.11.13.1 retained releases are the declared conversion inputs: the
original Snowball-Smartie conversion produced them, and the 2026.11.17.1 census
repair re-measured both populations with every TaskSpec byte identical. The
2026.11.18.1-r1 re-publications select these pins through retained-release
receipts; ``import_prebuilt`` admits the pinned published tasks without
repeating conversion. The original converter lives in
``taskcompendium.convert.structure_closure`` and the historical review rubric in
the retained releases' publication evidence.
"""

from finestore.rl.curation import PolicyRewardMeasurement, SourcePopulation
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

CONVERTER_VERSION = "2026.11.13.1"

# Measured rewards realized by the Mochi .123 starting policy on these populations,
# recorded as measured-limitation receipts: the source stays Available and Use=train,
# and the Atlas renders the measurements beside oracle_validation.
SNOWBALL_POLICY_REWARD_MEASUREMENTS = {
    "train": [
        PolicyRewardMeasurement(
            policy_identity="mochi-long-loop/adopted-sft@2026.11.19.110:b8dde345",
            population=SourcePopulation.TRAIN,
            reward=0.2301,
            sample_count=15,
            evidence_url=(
                "s3://marin-us-east-08a/marin/users/benfeuer/smartie-think-control/receipts/"
                "policy-reward/mochi123-snowball-train-reward.json"
            ),
            sha256="24013236e7e8b461422ecae7289ec39b5cac5dd524049ca5f4756b3fb0d73211",
            measured_at="2026-11-23T16:11:34+00:00",
        ),
    ],
    "validation": [
        PolicyRewardMeasurement(
            policy_identity="mochi-long-loop/adopted-sft@2026.11.19.110:b8dde345",
            population=SourcePopulation.HOLDOUT,
            reward=0.2841,
            sample_count=88,
            evidence_url=(
                "s3://marin-us-east-08a/marin/users/benfeuer/smartie-think-control/receipts/"
                "policy-reward/mochi123-snowball-holdout-think-control.jsonl"
            ),
            sha256="7122ae7782dd35d01253878b372ce9ecd31c0db1ce5ea5d790fa47c98a835ded",
            measured_at="2026-11-23T21:37:00+00:00",
        ),
    ],
}

RELEASES = {
    "train": (
        "open-athena/snowballv01-structure-closure-d9e5f81a-retained-20261113-v1",
        "0062f4f286d4e4f1568436f5e9c7749fcdd8fa69",
        57_312,
        IntendedUse.TRAIN,
    ),
    "validation": (
        "open-athena/snowballv01-structure-closure-validation-abaae0c5-retained-20261113-v1",
        "787917a504e14d59e2c4532ae7da083f6868c529",
        238,
        IntendedUse.EVAL,
    ),
}


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    return [
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                intended_use=intended_use,
                name=f"snowball-v01-structure-closure-{split}",
                version=CONVERTER_VERSION,
                repo=release,
                revision=revision,
                expected_rows=rows,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="OpenAthena:structure-closure" + ("/validation" if intended_use == IntendedUse.EVAL else ""),
                upstream_dataset="structure-closure",
                upstream_name="structure-closure",
                subset="validation" if intended_use == IntendedUse.EVAL else None,
                build_path="SnowballV01",
                origin=SourceOrigin.OPENATHENA,
                family="structural",
                tags=("rlvr", "single-turn", "structural-closure", "assistant-prefill"),
                task_format=TaskFormat.TASKSPEC,
                count=rows,
                policy_reward_measurements=SNOWBALL_POLICY_REWARD_MEASUREMENTS[split],
                notes=(
                    "Text assistant prefills; validation is a reused development panel. Declared cohort is the "
                    f"{split} retained release published from the Snowball-Smartie-RL-2026.10.15.32-Data "
                    "conversion; the 2026.11.17.1 census repair re-measured the same pin with every TaskSpec "
                    "byte identical to the original conversion (content-identity receipts), and the "
                    "2026.11.18.1-r1 retained re-publication selects this pin."
                ),
            ),
        )
        for split, (release, revision, rows, intended_use) in RELEASES.items()
    ]
