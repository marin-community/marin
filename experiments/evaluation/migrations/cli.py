# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Administrative sweeps over the eval archive fleet.

``backfill-samples`` brings archives up to the current contract from each run's kept
``samples_*.jsonl``; ``rebuild-samples`` re-derives them from the sources preserved inside the
archive, for a run whose results tree is gone. Both are operator tools, not part of launching an
evaluation.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click
from finestore.eval import ARCHIVE_SAMPLES_TABLE, SCHEMA_VERSION
from finestore.reader import CompositeReader
from marin.evaluation.lm_eval_samples import (
    export_lm_eval_samples,
    preserved_sample_sources,
    rebuild_lm_eval_samples,
)
from marin.evaluation.records import DEFAULT_SCAN_PREFIXES, list_records
from rigging.filesystem.s3_compat import configure_coreweave_s3

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Fleet-wide maintenance of eval sample archives."""


# One worker holds a run's whole sample file in memory while normalizing it, and the largest are a
# few hundred megabytes, so this trades throughput against a bounded footprint on a CPU node.
_DEFAULT_BACKFILL_WORKERS = 8


@dataclass(frozen=True)
class SweepOutcome:
    """What a sweep did to one archive: ``category`` for the tally, ``detail`` for the reported line."""

    category: str
    detail: str


@cli.command("backfill-samples")
@click.argument("results_paths", nargs=-1)
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    help=f"Object-store prefix(es) to scan for records; repeatable. Defaults to {DEFAULT_SCAN_PREFIXES}.",
)
@click.option(
    "--workers",
    default=_DEFAULT_BACKFILL_WORKERS,
    show_default=True,
    help="Archives to export concurrently. Each holds one run's sample file in memory.",
)
def backfill_samples(results_paths: tuple[str, ...], prefixes: tuple[str, ...], workers: int) -> None:
    """Bring the named RESULTS_PATHS, or every recorded run, up to the current contract.

    Reads each run's kept ``samples_*.jsonl`` and rewrites its ``samples`` table. An archive a
    completed export already brought to the current version is skipped, so an interrupted sweep
    resumes where it stopped; use ``rebuild-samples`` to force one that is already current.
    """
    configure_coreweave_s3()
    _sweep_archives(selected_archives(_resolve_prefixes(prefixes, results_paths), results_paths), workers, _backfill_one)


def _resolve_prefixes(prefixes: tuple[str, ...], results_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Fall back to the whole fleet only when the caller named neither a prefix nor a path."""
    if prefixes or results_paths:
        return prefixes
    return tuple(DEFAULT_SCAN_PREFIXES)


def _backfill_one(results_path: str) -> SweepOutcome:
    """Export one archive unless a completed export already brought it to the current contract."""
    reader = CompositeReader(results_path)
    # The version alone is stamped as soon as the new table is created, so an export that died
    # partway would read as current and never be retried. The seal is what says it finished.
    if reader.schema_version(ARCHIVE_SAMPLES_TABLE) == SCHEMA_VERSION and reader.is_sealed():
        return SweepOutcome("already_current", "current")
    written = export_lm_eval_samples(results_path).samples
    return SweepOutcome("exported", f"{written} sample(s)")


def selected_archives(prefixes: tuple[str, ...], results_paths: tuple[str, ...]) -> dict[str, list[str]]:
    """Group run ids by archive path, over explicit paths and every record under each prefix.

    Grouping is what keeps a sweep correct: several runs can be recorded against one results tree,
    and letting two workers write the same archive concurrently makes one of them compact shards the
    other is still reading.
    """
    runs_by_path: dict[str, list[str]] = defaultdict(list)
    for path in results_paths:
        runs_by_path[path.rstrip("/")] = []
    for prefix in prefixes:
        for record in list_records(prefix):
            runs_by_path[record.results_path.rstrip("/")].append(record.run_id)
    return runs_by_path


def _sweep_archives(runs_by_path: dict[str, list[str]], workers: int, work: Callable[[str], SweepOutcome], /) -> None:
    """Apply ``work`` once per archive, reporting each outcome and a final tally."""
    tally: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, path): path for path in runs_by_path}
        for future in as_completed(futures):
            path = futures[future]
            # A path named directly has no record to take a run id from; the tree names it well enough.
            runs = " ".join(sorted(runs_by_path[path])) or path.rstrip("/").rsplit("/", 2)[-2]
            try:
                outcome = future.result()
            except Exception as exc:
                tally["failed"] += 1
                # One unreadable archive must not abandon the rest of a fleet-wide sweep; the tally
                # and the non-zero exit below are what surface it.
                logger.exception("sweep failed for %s", path)
                click.echo(f"{runs}  FAILED {type(exc).__name__}: {exc}  {path}")
                continue
            tally[outcome.category] += 1
            click.echo(f"{runs}  {outcome.detail}  {path}")
    click.echo(json.dumps(tally, sort_keys=True))
    if tally["failed"]:
        raise SystemExit(1)


@cli.command("rebuild-samples")
@click.argument("results_paths", nargs=-1)
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    help=f"Object-store prefix(es) to scan for records; repeatable. Defaults to {DEFAULT_SCAN_PREFIXES}.",
)
@click.option(
    "--workers",
    default=_DEFAULT_BACKFILL_WORKERS,
    show_default=True,
    help="Archives to rebuild concurrently. Each holds one run's sample file in memory.",
)
def rebuild_samples(results_paths: tuple[str, ...], prefixes: tuple[str, ...], workers: int) -> None:
    """Rebuild sample archives from the sources preserved inside them, ignoring the results tree.

    Use this after a contract change when a run's evaluator-native
    files are no longer beside it, or to force a re-export of an archive ``backfill-samples`` would
    consider current. Runs whose archive predates source preservation are reported and skipped.
    """
    configure_coreweave_s3()
    _sweep_archives(selected_archives(_resolve_prefixes(prefixes, results_paths), results_paths), workers, _rebuild_one)


def _rebuild_one(results_path: str) -> SweepOutcome:
    """Rebuild one archive from its preserved sources, or report that it has none."""
    if not preserved_sample_sources(results_path):
        return SweepOutcome("no_sources", "no preserved sources")
    written = rebuild_lm_eval_samples(results_path)
    return SweepOutcome("rebuilt", f"{written} sample(s)")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
