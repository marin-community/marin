# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that a run's finestore archive holds every sample its evaluator scored.

A run's ``results_*.json`` states, per leaf task, how many documents were evaluated and which
extraction filters scored them. The archive should therefore hold ``documents x filters`` sample
rows for that task. Comparing the two catches the failure this tool exists for: rows that were
written but silently collapsed because two of them shared a merge key.

Aggregate group rows (``mmlu`` over its 57 subtasks) carry no samples of their own. Evalchemy marks
them with a ``sample_count`` field that leaf tasks lack, which is how they are excluded here —
counting them would inflate the expectation by a factor of the group nesting depth.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import click
from finestore.eval import RESULTS_PREFIX, is_scratch_artifact
from marin.evaluation.records import list_records
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.migrate_archive import archive_sample_count

logger = logging.getLogger(__name__)

# The documents a task evaluated, and the keys that describe a task rather than score it.
_DOCUMENT_COUNT = "sample_len"
_STRUCTURAL_KEYS = frozenset({"name", "alias", _DOCUMENT_COUNT})

# Present only on an aggregate group row, so its absence identifies a leaf task.
_GROUP_MARKER = "sample_count"

_MAX_WORKERS = 16


@dataclasses.dataclass(frozen=True)
class TaskExpectation:
    """What one leaf task should contribute to the archive."""

    documents: int
    filters: tuple[str, ...]

    @property
    def expected_rows(self) -> int:
        return self.documents * len(self.filters)


@dataclasses.dataclass(frozen=True)
class RunVerification:
    """One run's expected and actual sample-row counts."""

    results_path: str
    tasks: int
    documents: int
    expected_rows: int
    actual_rows: int
    multi_filter_tasks: tuple[str, ...]
    error: str | None = None

    @property
    def missing_rows(self) -> int:
        return self.expected_rows - self.actual_rows

    @property
    def ok(self) -> bool:
        return self.error is None and self.missing_rows == 0


def leaf_task_expectations(results_json: dict) -> dict[str, TaskExpectation]:
    """Per leaf task, the documents evaluated and the extraction filters that scored them."""
    expectations = {}
    for task, values in (results_json.get("results") or {}).items():
        if not isinstance(values, dict) or _GROUP_MARKER in values:
            continue
        filters = set()
        for key in values:
            if key in _STRUCTURAL_KEYS or "," not in key:
                continue
            metric, _, extraction_filter = key.partition(",")
            if metric.endswith("_stderr"):
                continue
            filters.add(extraction_filter)
        documents = values.get(_DOCUMENT_COUNT)
        if filters and isinstance(documents, (int, float)):
            expectations[task] = TaskExpectation(documents=int(documents), filters=tuple(sorted(filters)))
    return expectations


def verify_run(results_path: str) -> RunVerification:
    """Compare one run's archive sample count against what its results files claim was scored."""
    expectations: dict[str, TaskExpectation] = {}
    try:
        root = StoragePath(results_path)
        for directory, _, names in root.walk():
            for name in names:
                if not (name.startswith(RESULTS_PREFIX) and name.endswith(".json")):
                    continue
                # A retried evaluation leaves a second results tree the export does not index, so
                # counting it here would hold the archive to a total it was never meant to reach.
                if is_scratch_artifact((directory / name).relative_to(root)):
                    continue
                expectations.update(leaf_task_expectations(json.loads((directory / name).read_text())))
        actual_rows = archive_sample_count(results_path)
    except Exception as exc:
        return RunVerification(
            results_path=results_path,
            tasks=0,
            documents=0,
            expected_rows=0,
            actual_rows=0,
            multi_filter_tasks=(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return RunVerification(
        results_path=results_path,
        tasks=len(expectations),
        documents=sum(item.documents for item in expectations.values()),
        expected_rows=sum(item.expected_rows for item in expectations.values()),
        actual_rows=actual_rows,
        multi_filter_tasks=tuple(sorted(name for name, item in expectations.items() if len(item.filters) > 1)),
    )


def _selected_results_paths(results_paths: tuple[str, ...], records_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(results_paths)
    for records_prefix in records_prefixes:
        selected.update(record.results_path for record in list_records(records_prefix))
    return tuple(sorted(selected))


@click.command()
@click.argument("results_paths", nargs=-1)
@click.option(
    "--records-prefix",
    "records_prefixes",
    multiple=True,
    help="Verify every recorded run under this prefix. Repeatable.",
)
@click.option("--only-failures", is_flag=True, help="Print a line per run that is short or errored, not every run.")
def main(results_paths: tuple[str, ...], records_prefixes: tuple[str, ...], only_failures: bool) -> None:
    """Verify one or more RESULTS_PATHS, or every run under --records-prefix.

    Exits non-zero when any run holds fewer samples than its results files account for.
    """
    logging.basicConfig(level=logging.INFO)
    if not results_paths and not records_prefixes:
        raise click.UsageError("pass at least one RESULTS_PATH or --records-prefix")
    if any(path.startswith("s3://") for path in (*results_paths, *records_prefixes)):
        configure_coreweave_s3()
    selected = _selected_results_paths(results_paths, records_prefixes)

    verified = short = errored = missing_rows = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for result in pool.map(verify_run, selected):
            if result.error is not None:
                errored += 1
            elif result.missing_rows > 0:
                short += 1
                missing_rows += result.missing_rows
            else:
                verified += 1
            if not only_failures or not result.ok:
                click.echo(
                    json.dumps({**dataclasses.asdict(result), "missing_rows": result.missing_rows}, sort_keys=True)
                )
    click.echo(
        json.dumps(
            {
                "status": "complete",
                "selected_runs": len(selected),
                "verified_runs": verified,
                "short_runs": short,
                "errored_runs": errored,
                "missing_rows": missing_rows,
            },
            sort_keys=True,
        )
    )
    if short or errored:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
