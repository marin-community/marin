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
the rest. ``--list-pending`` prints each source whose terminal step is not
cached, then exits. The caller pins the staging region. For example, use
``iris job run --region us-east5 ...``. The Iris worker exports a
region-appropriate ``MARIN_PREFIX`` automatically.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STATUS_CHECK_WORKERS = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads-only",
        action="store_true",
        help="Run only the download step of each source's chain (normalize_steps[0]).",
    )
    parser.add_argument(
        "--sources",
        help="Comma-separated source names to run (default: all). Use to re-drive specific sources.",
    )
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="List sources whose selected terminal step is not cached, without starting pipeline steps.",
    )
    return parser.parse_args(argv)


def _print_pending(source_terminals: list[tuple[DatakitSource, StepSpec]]) -> None:
    with ThreadPoolExecutor(max_workers=STATUS_CHECK_WORKERS) as pool:
        cached = list(pool.map(lambda item: step_is_built(item[1]), source_terminals))

    pending = [item for item, is_cached in zip(source_terminals, cached, strict=True) if not is_cached]
    for source, terminal in pending:
        print(f"{source.name}\t{terminal.output_path}")
    print(f"{len(pending)}/{len(source_terminals)} source(s) would run.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry = all_sources()
    if args.sources:
        names = [name.strip() for name in args.sources.split(",") if name.strip()]
        unknown = [name for name in names if name not in registry]
        if unknown:
            raise SystemExit(f"Unknown sources {unknown}; available: {sorted(registry)}")
        sources = [registry[name] for name in names]
    else:
        sources = list(registry.values())
    source_terminals = [
        (source, source.normalize_steps[0] if args.downloads_only else source.normalized) for source in sources
    ]
    if args.list_pending:
        _print_pending(source_terminals)
        return

    stage = "downloads" if args.downloads_only else "normalize chains"
    logger.info("Running %s for %d sources", stage, len(sources))
    StepRunner().run([terminal for _, terminal in source_terminals])
    logger.info("All %d sources reached a terminal state", len(sources))


if __name__ == "__main__":
    configure_logging()
    main()
