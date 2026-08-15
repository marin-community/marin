# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Administrative sweeps over the eval archive fleet.

``upgrade-format`` moves sealed FineStore v1 archives onto transactional manifests;
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
from dataclasses import asdict, dataclass
from enum import StrEnum

import click
from finestore.migrations import migrate
from finestore.reader import ReadView
from marin.evaluation.archive import ARCHIVE_SAMPLES_TABLE, SCHEMA_VERSION
from marin.evaluation.lm_eval_samples import (
    export_lm_eval_samples,
    preserved_sample_sources,
    rebuild_lm_eval_samples,
)
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_SCAN_PREFIXES,
    LEGACY_CW_RECORDS_PREFIX,
    list_records,
)
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.migrations.format_smoke import (
    SmokeUpgradeResult,
    fleet_destination,
    select_v1_archive,
    select_v1_fleet,
    smoke_destination,
    smoke_upgrade,
    smoke_upgrade_fleet,
)

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Fleet-wide maintenance of eval sample archives."""


# One worker holds a run's whole sample file in memory while normalizing it, and the largest are a
# few hundred megabytes, so this trades throughput against a bounded footprint on a CPU node.
_DEFAULT_SWEEP_WORKERS = 8
_DEFAULT_SMOKE_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_FLEET_SMOKE_WORKERS = 4
_CW_RECORDS_PREFIXES = (CW_RECORDS_PREFIX, LEGACY_CW_RECORDS_PREFIX)


class FleetSmokeMode(StrEnum):
    """Write behavior for a fleet migration rehearsal."""

    INVENTORY = "inventory"
    KEEP_TEMP = "keep-temp"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class SweepOutcome:
    """What a sweep did to one archive: ``category`` for the tally, ``detail`` for the reported line."""

    category: str
    detail: str


@cli.command("upgrade-format")
@click.argument("results_paths", nargs=-1)
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    help=f"Object-store prefix(es) to scan for records; repeatable. Defaults to {DEFAULT_SCAN_PREFIXES}.",
)
@click.option(
    "--workers",
    default=_DEFAULT_SWEEP_WORKERS,
    show_default=True,
    help="Sealed archives to migrate concurrently.",
)
def upgrade_format(results_paths: tuple[str, ...], prefixes: tuple[str, ...], workers: int) -> None:
    """Upgrade named RESULTS_PATHS, or every recorded run, from FineStore v1 to v2."""
    configure_coreweave_s3()
    _sweep_archives(selected_archives(_resolve_prefixes(prefixes, results_paths), results_paths), workers, _upgrade_one)


def _upgrade_one(results_path: str) -> SweepOutcome:
    result = migrate(results_path)
    if not result.applied:
        return SweepOutcome("already_current", "current")
    token = result.token
    assert token is not None
    return SweepOutcome("upgraded", f"commit {token.sequence}:{token.commit_id}")


@cli.command("smoke-upgrade")
@click.argument("results_path", required=False)
@click.option("--records-prefix", help="Choose one sealed v1 archive recorded under this prefix.")
@click.option("--destination", help="Temporary destination. Defaults to a same-region one-day prefix.")
@click.option(
    "--max-bytes",
    default=_DEFAULT_SMOKE_MAX_BYTES,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum FineStore-owned bytes when selecting from --records-prefix.",
)
@click.option("--ttl-days", default=1, show_default=True, type=click.IntRange(min=1))
def smoke_upgrade_command(
    results_path: str | None,
    records_prefix: str | None,
    destination: str | None,
    max_bytes: int,
    ttl_days: int,
) -> None:
    """Clone one v1 archive to temporary storage, migrate it, and validate exact contents."""
    if (results_path is None) == (records_prefix is None):
        raise click.UsageError("pass exactly one RESULTS_PATH or --records-prefix")
    if any(path and path.startswith("s3://") for path in (results_path, records_prefix, destination)):
        configure_coreweave_s3()
    if results_path is not None:
        source = results_path
    else:
        assert records_prefix is not None
        source = select_v1_archive(records_prefix, max_bytes=max_bytes)
    resolved_destination = destination or smoke_destination(source, ttl_days=ttl_days)
    click.echo(json.dumps({"status": "copying", "source": source, "destination": resolved_destination}, sort_keys=True))
    result = smoke_upgrade(source, resolved_destination)
    click.echo(json.dumps({"status": "validated", **asdict(result)}, sort_keys=True))


@cli.command("smoke-upgrade-fleet")
@click.option(
    "--records-prefix",
    "records_prefixes",
    multiple=True,
    help=f"Record roots to scan; repeatable. Defaults to the CoreWeave roots {_CW_RECORDS_PREFIXES}.",
)
@click.option("--destination-root", help="Generated fleet root. Defaults to same-region one-day storage.")
@click.option(
    "--workers",
    default=_DEFAULT_FLEET_SMOKE_WORKERS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Archives to clone and validate concurrently.",
)
@click.option("--ttl-days", default=1, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--mode",
    type=click.Choice(tuple(mode.value for mode in FleetSmokeMode)),
    default=FleetSmokeMode.KEEP_TEMP.value,
    show_default=True,
    help="Inventory only, retain the validated temporary fleet, or delete it after complete success.",
)
def smoke_upgrade_fleet_command(
    records_prefixes: tuple[str, ...],
    destination_root: str | None,
    workers: int,
    ttl_days: int,
    mode: str,
) -> None:
    """Clone and validate every sealed CoreWeave v1 archive as one fleet rehearsal."""
    resolved_prefixes = records_prefixes or _CW_RECORDS_PREFIXES
    if any(path.startswith("s3://") for path in (*resolved_prefixes, destination_root or "")):
        configure_coreweave_s3()
    selection = select_v1_fleet(resolved_prefixes)
    click.echo(json.dumps({"status": "selected", **asdict(selection)}, sort_keys=True))
    blockers = selection.record_failures + selection.unsealed_v1 + selection.unsupported
    if blockers:
        raise click.ClickException(f"fleet selection has {len(blockers)} blocking record or archive issue(s)")
    fleet_mode = FleetSmokeMode(mode)
    if fleet_mode is FleetSmokeMode.INVENTORY:
        return

    root = destination_root or fleet_destination(resolved_prefixes[0], ttl_days=ttl_days)
    completed = 0

    def report(result: SmokeUpgradeResult) -> None:
        nonlocal completed
        completed += 1
        click.echo(
            json.dumps(
                {
                    "status": "validated_archive",
                    "completed": completed,
                    "total": len(selection.sources),
                    **asdict(result),
                },
                sort_keys=True,
            )
        )

    click.echo(json.dumps({"status": "starting", "destination_root": root}, sort_keys=True))
    result = smoke_upgrade_fleet(
        selection.sources,
        root,
        workers=workers,
        cleanup=fleet_mode is FleetSmokeMode.CLEANUP,
        on_result=report,
    )
    click.echo(
        json.dumps(
            {
                "status": "validated_fleet",
                "destination_root": result.destination_root,
                "archives": len(result.results),
                "source_objects": result.source_objects,
                "source_bytes": result.source_bytes,
                "rows": result.rows,
                "cleaned_up": result.cleaned_up,
            },
            sort_keys=True,
        )
    )


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
    default=_DEFAULT_SWEEP_WORKERS,
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
    reader = ReadView(results_path)
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
    default=_DEFAULT_SWEEP_WORKERS,
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
