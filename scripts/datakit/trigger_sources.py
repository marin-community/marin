# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trigger every Datakit source's chain via ``StepRunner``, optionally downloads-only.

For each :class:`marin.datakit.sources.DatakitSource`, hand ``StepRunner`` the
terminal normalize step (or, with ``--downloads-only``, just the chain's
download — ``normalize_steps[0]``); the runner walks back through every
transitive dep in post-order and dedupes by ``output_path`` — so shared
family downloads (e.g. Nemotron v2 subsets) are materialized once.
Already-succeeded steps short-circuit via the on-disk cache check, so this
is safe to re-run: it advances whatever hasn't completed yet and no-ops
the rest. The staging region is pinned by the caller (e.g. via
``iris job run --region us-east5 ...``); the iris worker exports a
region-appropriate ``MARIN_PREFIX`` automatically, so this script does
not set or validate it.
"""

import argparse
import logging

from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads-only",
        action="store_true",
        help="Run only the download step of each source's chain (normalize_steps[0]).",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Source name to trigger (repeatable). Default: all registered sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registered_sources = all_sources()
    if args.source:
        missing = [name for name in args.source if name not in registered_sources]
        if missing:
            raise SystemExit(f"unknown source(s): {', '.join(missing)}")
        sources = [registered_sources[name] for name in args.source]
    else:
        sources = list(registered_sources.values())
    terminals = [src.normalize_steps[0] if args.downloads_only else src.normalized for src in sources]
    stage = "downloads" if args.downloads_only else "normalize chains"
    logger.info("Running %s for %d sources", stage, len(sources))
    StepRunner().run(terminals)
    logger.info("All %d sources reached a terminal state", len(sources))


if __name__ == "__main__":
    configure_logging()
    main()
