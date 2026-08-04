# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry job for the docling backfill of the router's OCR route.

The production fleet run converted the 214,444 text-extractable documents; this converts the
101,332 the router sent to OCR, so the union of the two outputs is a docling conversion of the
full 10% sample. Same fleet, same senders, same record shape -- only the routing key set and the
output prefix differ.

A separate module from :mod:`~experiments.build_pdf_source.extract_fleet` for the same reason as
:mod:`~experiments.build_pdf_source.extract_fleet_run`: the step's functions must pickle by
reference to an importable module, not to ``__main__``.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-extract-backfill \\
        -- python -m experiments.build_pdf_source.extract_fleet_backfill_run
"""

import logging

from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.extract_fleet import (
    _MAX_WORKERS,
    _POOL_INSTANCES,
    _PROCESSES_PER_INSTANCE,
    _SENDER_TASKS,
    fleet_backfill_step,
)
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(logging.INFO)
    logger.info(
        "Backfill fleet request: %d pods x %d converters, fed by %d sender tasks on %d workers",
        _POOL_INSTANCES,
        _PROCESSES_PER_INSTANCE,
        _SENDER_TASKS,
        _MAX_WORKERS,
    )
    plan = plan_step()
    fetch = fetch_step(plan)
    classify = classify_step(fetch, model_step())
    layout_model = layout_model_step(fetch)
    StepRunner().run([layout_model, fleet_backfill_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
