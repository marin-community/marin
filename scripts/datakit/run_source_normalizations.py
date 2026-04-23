# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run every Datakit source's full ``normalize_steps`` chain via ``StepRunner``.

For each :class:`marin.datakit.sources.DatakitSource`, every step in the
chain (download, any transforms, normalize) is fed to ``StepRunner``. Steps
that have already succeeded are skipped automatically by the runner's
on-disk status check, so this is safe to re-run: it advances whatever
hasn't completed yet and no-ops the rest.

Shared family downloads (e.g. Nemotron v2 subsets) land in the iterable as
the same Python object; dedup by ``output_path`` so the runner doesn't see
them twice. Staging region is pinned via ``MARIN_PREFIX`` so every
``step.output_path`` resolves under ``gs://marin-us-central1``.
"""

import logging
import os

from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"


def main() -> None:
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    seen: set[str] = set()
    steps: list = []
    for src in all_sources().values():
        for step in src.normalize_steps:
            if step.output_path in seen:
                continue
            seen.add(step.output_path)
            steps.append(step)

    logger.info("Running %d unique StepSpecs across %d sources", len(steps), len(all_sources()))
    StepRunner().run(steps)
    logger.info("All %d steps reached a terminal state", len(steps))


if __name__ == "__main__":
    configure_logging()
    main()
