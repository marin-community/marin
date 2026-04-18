# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.log_setup import configure_logging
from rigging.filesystem import marin_prefix
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def build_steps() -> list[StepSpec]:
    raw = StepSpec(
        name="raw/fineweb-edu-sample-10bt",
        # TODO: allow to override via relative override path in StepSpec
        override_output_path=f"{marin_prefix()}/raw/fineweb-edu-87f0914",
    )
    dedup = StepSpec(
        name="dedup_sample/10BT",
        deps=[raw],
        fn=lambda op: dedup_fuzzy_document(
            input_paths=raw.output_path + "/sample/10BT",
            output_path=op,
            max_parallelism=1024,
        ),
    )
    return [raw, dedup]


if __name__ == "__main__":
    configure_logging()
    StepRunner().run(build_steps())
