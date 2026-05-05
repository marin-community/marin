# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run every Datakit source's full ``normalize_steps`` chain via ``StepRunner``.

For each :class:`marin.datakit.sources.DatakitSource`, hand ``StepRunner`` the
terminal normalize step; the runner walks back through every transitive dep
(transforms, downloads) in post-order and dedupes by ``output_path`` — so
shared family downloads (e.g. Nemotron v2 subsets) are materialized once.
Already-succeeded steps short-circuit via the on-disk cache check, so this
is safe to re-run: it advances whatever hasn't completed yet and no-ops the
rest. Staging region is pinned via ``MARIN_PREFIX`` so every
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

    sources = list(all_sources().values())
    terminals = [src.normalized for src in sources]
    logger.info("Running normalize chains for %d sources", len(sources))
    StepRunner().run(terminals)
    logger.info("All %d sources reached a terminal state", len(sources))


if __name__ == "__main__":
    configure_logging()
    main()
