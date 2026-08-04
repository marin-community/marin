# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry job for the production fleet extraction over the full 10% sample.

A separate module from :mod:`~experiments.build_pdf_source.extract_fleet` on purpose: the step
function and everything it references must live in an importable module, because running a module
as ``__main__`` makes cloudpickle serialize its functions by value -- and the ``@cache``-wrapped
client/pool helpers in ``extract_fleet`` can only pickle as references, which the driver's
``__main__`` cannot resolve. Same split as ``extract_ocr_all`` over ``extract_ocr``.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-extract-fleet \\
        -- python -m experiments.build_pdf_source.extract_fleet_run
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
    fleet_extract_step,
)
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(logging.INFO)
    logger.info(
        "Fleet request: %d pods x %d converters, fed by %d sender tasks on %d workers",
        _POOL_INSTANCES,
        _PROCESSES_PER_INSTANCE,
        _SENDER_TASKS,
        _MAX_WORKERS,
    )
    plan = plan_step()
    fetch = fetch_step(plan)
    classify = classify_step(fetch, model_step())
    layout_model = layout_model_step(fetch)
    StepRunner().run([layout_model, fleet_extract_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
