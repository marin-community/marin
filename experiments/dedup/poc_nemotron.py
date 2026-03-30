# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from rigging.log_setup import configure_logging
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
import os
from typing import TypeVar
from marin.execution.step_runner import StepRunner
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from fray.v2 import ResourceConfig
from rigging.filesystem import marin_temp_bucket, region_from_metadata, check_path_in_region
from marin.execution.step_spec import StepSpec

import logging

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

    dedup_step = StepSpec(
        name="fuzzy_dedup_full",
        output_path_prefix=marin_temp_bucket(ttl_days=2, prefix="rav"),
        deps=[raw_data_step],
        fn=lambda op: dedup_fuzzy_document(
            input_paths=[
                os.path.join(raw_data_step.output_path, "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high"),
                os.path.join(raw_data_step.output_path, "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium-high"),
            ],
            output_path=op,
            max_parallelism=2048,
            worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
        ),
    )

    return [raw_data_step, dedup_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(exact_dedup_steps())
