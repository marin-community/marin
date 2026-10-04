# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Datakit reference DAG as a ferry.

Each datakit ferry supplies its normalized sources and a :class:`PipelineScale`;
the DAG itself comes from :func:`reference_datakit_steps`, so the canaries test
the production pipeline. Every step output goes to a one-day temporary prefix
below ``MARIN_PREFIX``. The quality model and the decontamination eval corpus
resolve against ``MARIN_PREFIX``, so they must be staged in the ferry's region.
"""

import logging
import os

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.timing import log_time

from experiments.datakit.reference_pipeline import (
    QUALITY_MODEL_VERSION,
    PipelineScale,
    quality_model_path,
    reference_datakit_steps,
    shared_zephyr_context,
)
from infra.ci.run_status import run_status

logger = logging.getLogger(__name__)

FERRY_OUTPUT_TTL_DAYS = 1
FERRY_MAX_CONCURRENT_STEPS = 8


def ferry_output_prefix(ferry_name: str) -> str:
    """Return the temporary output root for this run of ``ferry_name``."""
    return marin_temp_bucket(ttl_days=FERRY_OUTPUT_TTL_DAYS, prefix=f"{ferry_name}/{os.environ['SMOKE_RUN_ID']}")


def run_reference_ferry(
    *,
    ferry_name: str,
    sources: dict[str, StepSpec],
    scale: PipelineScale,
    output_prefix: str,
) -> None:
    """Run the reference DAG over ``sources`` with every output under ``output_prefix``."""
    logger.info("Output prefix: %s", output_prefix)
    zephyr_context = shared_zephyr_context(scale, name=ferry_name)
    steps = reference_datakit_steps(
        sources,
        quality_model=quality_model_path(),
        quality_model_version=QUALITY_MODEL_VERSION,
        scale=scale,
        zephyr_context=zephyr_context,
        output_prefix=output_prefix,
    )
    with run_status(os.environ.get("FERRY_STATUS_PATH"), marin_prefix=output_prefix):
        with log_time(f"{ferry_name} total wall time"), zephyr_context:
            StepRunner().run(steps.all_steps, max_concurrent=FERRY_MAX_CONCURRENT_STEPS)
