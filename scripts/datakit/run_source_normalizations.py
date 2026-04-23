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

    # Walk each source's chain AND its transitive deps — e.g. coderforge's
    # ``processed`` step depends on a hidden ``raw_download`` that isn't in
    # ``normalize_steps``. Without it in the iterable, StepRunner's dep-
    # satisfaction loop leaves ``processed`` in ``waiting`` forever and the
    # run dies with "Iterable exhausted with unsatisfied dependencies".
    seen: set[str] = set()
    steps: list = []

    def visit(step):
        if step.output_path in seen:
            return
        for dep in step.deps:
            visit(dep)
        seen.add(step.output_path)
        steps.append(step)

    for src in all_sources().values():
        for step in src.normalize_steps:
            visit(step)

    logger.info("Running %d unique StepSpecs across %d sources", len(steps), len(all_sources()))
    StepRunner().run(steps)
    logger.info("All %d steps reached a terminal state", len(steps))


if __name__ == "__main__":
    configure_logging()
    main()
