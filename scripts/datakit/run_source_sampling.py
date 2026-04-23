# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample whichever sources have a completed normalize, end-to-end test for the sampler.

Subsets :func:`marin.datakit.sources.all_sources` to only those whose
terminal normalize step is already cached on GCS, then builds a
:func:`sample_normalized_shards_step` per available source and runs them via
``StepRunner``. The proportional-fractions math is intentionally bypassed:
a fixed ``SAMPLE_FRACTION`` keeps output small and predictable so this
script can run against whatever subset of normalized data exists right now.

Outputs land in the region-local marin temp bucket
(``gs://marin-tmp-us-central1/ttl=Nd/datakit-testbed/<RUN_ID>/sample/<src.name>``)
so they're auto-deleted after the configured TTL — no manual cleanup.
"""

import logging
import os

from rigging.filesystem import marin_temp_bucket
from rigging.log_setup import configure_logging

from experiments.datakit_testbed.sampler import sample_normalized_shards_step
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner, check_cache

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
RUN_ID = "sample-test"
SAMPLE_FRACTION = 0.05
TTL_DAYS = 1


def main() -> None:
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX
    base = marin_temp_bucket(ttl_days=TTL_DAYS, prefix=f"datakit-testbed/{RUN_ID}")

    # For each source whose normalize is already cached, feed the sample step
    # AND its full ``normalize_steps`` chain (with transitive deps) into the
    # iterable. StepRunner's dep-satisfaction loop only marks a step as
    # ``completed`` when it sees the step go through its own cache check, so
    # the normalize must be in the iterable even though it'll no-op — without
    # it, the sample step's dep check fails and the run dies with "Iterable
    # exhausted with unsatisfied dependencies".
    seen: set[str] = set()
    steps = []
    skipped: list[str] = []

    def visit(step):
        if step.output_path in seen:
            return
        for dep in step.deps:
            visit(dep)
        seen.add(step.output_path)
        steps.append(step)

    for src in all_sources().values():
        if not check_cache(src.normalized.output_path):
            skipped.append(src.name)
            continue
        for step in src.normalize_steps:
            visit(step)
        steps.append(
            sample_normalized_shards_step(
                name=f"datakit-testbed/sample/{src.name}",
                normalized=src.normalized,
                sample_fraction=SAMPLE_FRACTION,
                override_output_path=f"{base}/sample/{src.name}",
            )
        )

    logger.info(
        "Sampling %d / %d sources at fraction=%.2f under %s/ (TTL=%dd, skipped %d not-yet-normalized)",
        len(steps),
        len(all_sources()),
        SAMPLE_FRACTION,
        base,
        TTL_DAYS,
        len(skipped),
    )
    StepRunner().run(steps)
    logger.info("All %d sample steps reached a terminal state", len(steps))


if __name__ == "__main__":
    configure_logging()
    main()
