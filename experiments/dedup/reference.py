# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.logging import configure_logging
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def build_steps() -> list[StepSpec]:
    raw = StepSpec(
        name="raw/fineweb-edu-sample-10bt",
        override_output_path="raw/fineweb-edu-87f0914/sample/10BT",
    )
    dedup = StepSpec(
        name="dedup_sample/10BT",
        deps=[raw],
        fn=lambda op: dedup_fuzzy_document(
            input_paths=raw.output_path,
            output_path=op,
            max_parallelism=1024,
        ),
    )
    return [dedup]


if __name__ == "__main__":
    configure_logging()
    StepRunner().run(build_steps())
