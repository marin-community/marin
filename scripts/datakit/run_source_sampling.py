# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample whichever sources have a completed normalize, end-to-end test for the sampler.

Subsets :func:`marin.datakit.sources.all_sources` to only those whose
terminal normalize step is already cached on GCS, then builds a
:func:`sample_normalized_shards_step` per available source and runs them
via ``StepRunner``. Per-source sample fractions are computed from
``TARGET_TOTAL_TOKENS_B`` via :func:`proportional_sample_fractions` — the
math that the full ferry uses — so the subset actually reflects the
requested target size rather than a blanket fraction.

Outputs land in the region-local marin temp bucket
(``gs://marin-tmp-us-central1/ttl=Nd/data/datakit/mini/<RUN_ID>/sample/<src.name>``)
so they're auto-deleted after the configured TTL — no manual cleanup.
"""

import logging
import os

from rigging.filesystem import marin_temp_bucket
from rigging.log_setup import configure_logging

from experiments.datakit_testbed.sampler import (
    proportional_sample_fractions,
    sample_normalized_shards_step,
)
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner, check_cache

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
RUN_ID = "sample-test"
# Target total token count (billions) across all cached sources. Drives
# per-source sample_fraction via proportional_sample_fractions. Small
# default keeps this a cheap smoke test; bump for larger rehearsals.
TARGET_TOTAL_TOKENS_B = 10.0
TTL_DAYS = 1


def main() -> None:
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX
    base = marin_temp_bucket(ttl_days=TTL_DAYS, prefix=f"data/datakit/mini/{RUN_ID}")

    available = [s for s in all_sources().values() if check_cache(s.normalized.output_path)]
    skipped = [s.name for s in all_sources().values() if not check_cache(s.normalized.output_path)]

    fractions = proportional_sample_fractions(available, target_total_tokens_b=TARGET_TOTAL_TOKENS_B)

    # For each source, feed its full normalize chain (+ transitive deps) into
    # the iterable alongside the sample step. StepRunner's dep-satisfaction
    # loop only marks a step as ``completed`` when it sees the step go
    # through its own cache check, so the normalize must be in the iterable
    # even though it'll no-op — otherwise the sample step's dep check fails
    # with "Iterable exhausted with unsatisfied dependencies".
    seen: set[str] = set()
    steps = []

    def visit(step):
        if step.output_path in seen:
            return
        for dep in step.deps:
            visit(dep)
        seen.add(step.output_path)
        steps.append(step)

    for src in available:
        for step in src.normalize_steps:
            visit(step)
        steps.append(
            sample_normalized_shards_step(
                name=f"data/datakit/mini/{src.name}",
                normalized=src.normalized,
                sample_fraction=fractions[src.name],
                override_output_path=f"{base}/sample/{src.name}",
            )
        )

    logger.info(
        "Sampling %d / %d sources targeting %.1fB tokens under %s/ (TTL=%dd, skipped %d not-yet-normalized)",
        len(available),
        len(all_sources()),
        TARGET_TOTAL_TOKENS_B,
        base,
        TTL_DAYS,
        len(skipped),
    )
    StepRunner().run(steps)
    logger.info("All %d sample steps reached a terminal state", len(steps))


if __name__ == "__main__":
    configure_logging()
    main()
