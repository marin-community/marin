# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging


def build_steps() -> list[StepSpec]:
    raw = StepSpec(
        name="raw/fineweb-edu-sample-10bt",
        # TODO: allow to override via relative override path in StepSpec
        override_output_path=f"{marin_prefix()}/raw/fineweb-edu-87f0914",
    )
    normalized = normalize_step(
        name="normalized/fineweb-edu-sample-10bt",
        download=raw,
        relative_input_path="sample/10BT",
    )
    minhash = StepSpec(
        name="minhash/fineweb-edu-sample-10bt",
        deps=[normalized],
        fn=lambda op: compute_minhash_attrs(
            source=Artifact.load(normalized, NormalizedData),
            output_path=op,
        ),
    )
    dedup = StepSpec(
        name="dedup_sample/10BT",
        deps=[minhash],
        fn=lambda op: compute_fuzzy_dups_attrs(
            inputs=[Artifact.load(minhash, MinHashAttrData)],
            output_path=op,
            max_parallelism=1024,
        ),
    )
    return [raw, normalized, minhash, dedup]


if __name__ == "__main__":
    configure_logging()
    StepRunner().run(build_steps())
