# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample whichever sources have a completed normalize, end-to-end test for the sampler.

Subsets :func:`marin.datakit.sources.all_sources` to only those whose
terminal normalize step is already cached on GCS, then builds a
:func:`sample_normalized_shards_step` per available source and runs them via
``StepRunner``. The proportional-fractions math is intentionally bypassed:
a fixed ``SAMPLE_FRACTION`` keeps output small and predictable so this
script can run against whatever subset of normalized data exists right now.

Outputs land under ``gs://marin-us-central1/datakit-testbed/<RUN_ID>/sample/<src.name>``
so the test artifacts are easy to identify and drop later.
"""

import logging
import os

from experiments.datakit_testbed.sampler import sample_normalized_shards_step
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner, check_cache
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
RUN_ID = "sample-test"
SAMPLE_FRACTION = 0.05


def main() -> None:
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX
    base = f"datakit-testbed/{RUN_ID}"

    steps = []
    skipped: list[str] = []
    for src in all_sources().values():
        if not check_cache(src.normalized.output_path):
            skipped.append(src.name)
            continue
        steps.append(
            sample_normalized_shards_step(
                name=f"datakit-testbed/sample/{src.name}",
                normalized=src.normalized,
                sample_fraction=SAMPLE_FRACTION,
                override_output_path=f"{base}/sample/{src.name}",
            )
        )

    logger.info(
        "Sampling %d / %d sources at fraction=%.2f under %s/%s/ (skipped %d not-yet-normalized)",
        len(steps),
        len(all_sources()),
        SAMPLE_FRACTION,
        STAGING_PREFIX,
        base,
        len(skipped),
    )
    StepRunner().run(steps)
    logger.info("All %d sample steps reached a terminal state", len(steps))


if __name__ == "__main__":
    configure_logging()
    main()
