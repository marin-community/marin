# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import TypeVar

from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from rigging.filesystem import check_path_in_region, marin_temp_bucket, region_from_metadata
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    assert value is not None
    return value


def exect_dedup_steps_10BT() -> list[StepSpec]:
    raw_data_step = StepSpec(
        name="raw_fineweb_edu",
        override_output_path="gs://marin-eu-west4/raw/fineweb-edu-87f0914",
    )

    # assert we are not reading cross-region
    check_path_in_region(raw_data_step.name, raw_data_step.output_path, assert_not_none(region_from_metadata()))

    dedup_step = StepSpec(
        name="exact_dedup",
        output_path_prefix=marin_temp_bucket(ttl_days=1, prefix="rav"),
        deps=[raw_data_step],
        fn=lambda op: dedup_exact_paragraph(
            input_paths=os.path.join(raw_data_step.output_path, "sample/10BT"),
            output_path=op,
            max_parallelism=1024,
        ),
    )

    return [raw_data_step, dedup_step]


def exact_dedup_steps() -> list[StepSpec]:
    raw_data_step = StepSpec(
        name="raw_nemotron",
        override_output_path="gs://marin-eu-west4/raw/nemotro-cc-eeb783/",
    )

    # assert we are not reading cross-region
    check_path_in_region(raw_data_step.name, raw_data_step.output_path, assert_not_none(region_from_metadata()))

    dedup_step = StepSpec(
        name="exact_dedup_high_medium_1",
        output_path_prefix=marin_temp_bucket(ttl_days=2, prefix="rav"),
        deps=[raw_data_step],
        fn=lambda op: dedup_exact_paragraph(
            input_paths=[
                os.path.join(raw_data_step.output_path, "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high"),
                os.path.join(raw_data_step.output_path, "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium-high"),
            ],
            output_path=op,
            max_parallelism=2048,
        ),
    )

    return [raw_data_step, dedup_step]


def fuzzy_dedup_steps() -> list[StepSpec]:
    raw_data_step = StepSpec(
        name="raw_nemotron",
        override_output_path="gs://marin-eu-west4/raw/nemotro-cc-eeb783/",
    )

    # assert we are not reading cross-region
    check_path_in_region(raw_data_step.name, raw_data_step.output_path, assert_not_none(region_from_metadata()))

    # Normalize each quality bucket separately so we get one MinHashAttrData per
    # source dataset, and dedup them globally in a single fuzzy_dups step.
    quality_paths = {
        "high": "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high",
        "medium-high": "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium-high",
    }

    normalize_steps = {
        q: normalize_step(
            name=f"normalized_nemotron/{q}",
            download=raw_data_step,
            relative_input_path=path,
        )
        for q, path in quality_paths.items()
    }

    minhash_steps = {
        q: StepSpec(
            name=f"minhash_nemotron/{q}",
            deps=[norm],
            fn=lambda op, norm=norm: compute_minhash_attrs(
                source=Artifact.load(norm, NormalizedData),
                output_path=op,
                worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
            ),
        )
        for q, norm in normalize_steps.items()
    }

    dedup_step = StepSpec(
        name="fuzzy_dedup_full",
        output_path_prefix=marin_temp_bucket(ttl_days=2, prefix="rav"),
        deps=list(minhash_steps.values()),
        fn=lambda op: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(s, MinHashAttrData) for s in minhash_steps.values()],
            output_path=op,
            max_parallelism=2048,
            worker_resources=ResourceConfig(cpu=1, ram="32g", disk="5g"),
        ),
    )

    return [raw_data_step, *normalize_steps.values(), *minhash_steps.values(), dedup_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(exact_dedup_steps())
