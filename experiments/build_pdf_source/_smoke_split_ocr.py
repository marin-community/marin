# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- smoke the two-phase OCR step end to end on 4 GPUs before the full-sample run.

DELETE once the split ships. Nothing in the pipeline imports this.

This drives the *production* :func:`~experiments.build_pdf_source.extract_ocr.ocr_pdf_text` --
not a bench copy of it -- over a 1/96 slice of the source (about 18 shards, ~750 OCR-route
documents) at 4 instances, on the merged zephyr stack (#7145 shared pools + the polars
addressing fix). What it proves that unit tests cannot:

* phase 1 writes raw shards through ``Dataset.write_parquet`` and the fleet is released
  before the shuffle starts;
* phase 2 runs the group_by on the same warm worker pool, and its polars scatter reads
  survive CoreWeave's gateways (the failure that killed ceiling bench attempt 2);
* the split writer's outputs land under ``outputs/main`` with the production schema.

Submit::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name ocr-split-smoke \\
        -- python -m experiments.build_pdf_source._smoke_split_ocr
"""

import logging
from functools import partial

from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.extract_ocr import ocr_pdf_text
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# 1/96 of the 1,773 source shards is ~18 shards: ~750 OCR-route documents, ~15k pages, a few
# minutes of phase 1 at 4 instances and a phase 2 small enough to finish in one more.
PARTITION = (0, 96)
INSTANCES = 4

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    classify = classify_step(fetch, model_step())
    step = StepSpec(
        name="data/datakit/validate/ocr_split_smoke",
        deps=[fetch, classify],
        hash_attrs={"partition": list(PARTITION), "instances": INSTANCES, "attempt": 1},
        fn=remote(
            partial(
                ocr_pdf_text,
                source_output_path=fetch.output_path,
                classification_output_path=classify.output_path,
                instances=INSTANCES,
                partition=PARTITION,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    StepRunner().run([step])


if __name__ == "__main__":
    main()
