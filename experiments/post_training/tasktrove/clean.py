# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble TaskTrove Clean from the graded rows.

    tasks/part-<n>.parquet   every surviving task with its selection columns
    ledger.parquet           one row per task that did not survive: its status and the reason
    manifest.json            revision, tool ref, counts per status, source, converter, mode, and check
    report.md                per-source and per-converter summary, regenerated every run

The survivors are copied by a Zephyr stage that reads only converted rows; the ledger and counts
come from one pass over the graded rows' small columns, never their binaries.

``export_task`` writes one row back out as a Harbor task directory for hand inspection.
"""

import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.convert import CONVERTED_SCHEMA
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.raw_tasks import APPROX_SHARD_BYTES, WORKER_RESOURCES
from experiments.post_training.tasktrove.sources import TASKTROVE_HF_ID, TASKTROVE_REVISION
from experiments.post_training.tasktrove.taskbinary import read_task_binary
from experiments.post_training.tasktrove.verify import GRADED_GLOB, VERIFIED_STATUS

logger = logging.getLogger(__name__)

TASK_COLUMNS = (
    "path",
    "source",
    "family",
    "template_id",
    "converter",
    "mode",
    "dockerfile_id",
    "language",
    "tags",
    "has_solution",
    "task_binary",
    "solution_binary",
)
TASKS_SCHEMA = pa.schema([CONVERTED_SCHEMA.field(name) for name in TASK_COLUMNS])
LEDGER_COLUMNS = ("source", "path", "status", "error")
SUMMARY_COLUMNS = ("source", "path", "status", "error", "converter", "mode", "dockerfile_id")
_READERS = 32


def _survivors(graded_path: str, output_path: str) -> None:
    """Zephyr stage: copy the converted rows' task columns into ``tasks/``."""
    files = Dataset.from_files(str(StoragePath(graded_path) / GRADED_GLOB))
    ds = files.load_parquet(columns=[*TASK_COLUMNS, "status"], approx_shard_bytes=APPROX_SHARD_BYTES)
    ds = ds.filter(lambda row: row["status"] == ConvertStatus.CONVERTED)
    ds = ds.map(lambda row: {name: row[name] for name in TASK_COLUMNS})
    ds = ds.write_parquet(str(StoragePath(output_path) / "tasks/part-{shard:05d}.parquet"), schema=TASKS_SCHEMA)
    ZephyrContext(name="tasktrove-clean", resources=WORKER_RESOURCES).execute(ds)


def read_columns(glob: StoragePath, columns: tuple[str, ...]) -> pa.Table:
    """The named columns of every parquet the glob matches, read concurrently."""

    def read(path: StoragePath) -> pa.Table:
        with path.open("rb") as handle:
            return pq.read_table(handle, columns=list(columns))

    files = sorted(glob.glob(), key=str)
    with ThreadPoolExecutor(_READERS) as pool:
        tables = list(pool.map(read, files))
    return pa.concat_tables(tables)


def build_manifest(graded: pa.Table, tool_ref: str) -> dict:
    by_status: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_converter: dict[str, Counter] = defaultdict(Counter)
    by_mode: Counter = Counter()
    by_check: Counter = Counter()
    dockerfiles: dict[str, set[str]] = defaultdict(set)
    columns = {
        name: graded.column(name).to_pylist() for name in ("source", "status", "converter", "mode", "dockerfile_id")
    }
    for source, status, converter, mode, dockerfile_id in zip(*columns.values(), strict=True):
        by_status[status] += 1
        by_source[source][status] += 1
        if converter:
            by_converter[converter][status] += 1
        if status.startswith(VERIFIED_STATUS):
            by_check[status.removeprefix(VERIFIED_STATUS)] += 1
        if status == ConvertStatus.CONVERTED:
            by_mode[mode] += 1
            dockerfiles[converter].add(dockerfile_id)
    return {
        "tasktrove": {"hf_id": TASKTROVE_HF_ID, "revision": TASKTROVE_REVISION},
        "verify_tool_ref": tool_ref,
        "input_tasks": graded.num_rows,
        "clean_tasks": by_status[ConvertStatus.CONVERTED],
        "by_status": dict(by_status.most_common()),
        "by_check": dict(by_check.most_common()),
        "by_mode": dict(by_mode.most_common()),
        "by_source": {s: dict(c.most_common()) for s, c in sorted(by_source.items())},
        "by_converter": {s: dict(c.most_common()) for s, c in sorted(by_converter.items())},
        "dockerfiles_by_converter": {c: len(ids) for c, ids in sorted(dockerfiles.items())},
    }


def build_clean(graded_path: str, output_path: str, tool_ref: str) -> None:
    out = StoragePath(output_path)
    for stale in ("tasks", "ledger.parquet"):
        if (out / stale).exists():
            (out / stale).rmtree()
    _survivors(graded_path, output_path)
    graded = read_columns(StoragePath(graded_path) / GRADED_GLOB, SUMMARY_COLUMNS)
    ledger = graded.filter(pa.compute.not_equal(graded.column("status"), ConvertStatus.CONVERTED.value))
    with (out / "ledger.parquet").open("wb") as handle:
        pq.write_table(ledger.select(list(LEDGER_COLUMNS)), handle)
    manifest = build_manifest(graded, tool_ref)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "report.md").write_text(render_report(manifest))
    logger.info("clean: %d of %d tasks; %s", manifest["clean_tasks"], manifest["input_tasks"], manifest["by_status"])


def render_report(manifest: dict) -> str:
    converted = ConvertStatus.CONVERTED.value
    lines = [
        "# TaskTrove Clean",
        "",
        f"{manifest['clean_tasks']} of {manifest['input_tasks']} tasks from {manifest['tasktrove']['hf_id']}"
        f" @ {manifest['tasktrove']['revision']}, graded by tasktrove-verify @ {manifest['verify_tool_ref']}.",
        "",
        "## By status",
        "",
        "| status | tasks |",
        "|---|---:|",
        *(f"| {s} | {n} |" for s, n in manifest["by_status"].items()),
        "",
        "## By source",
        "",
        "| source | tasks | clean | main reason for the rest |",
        "|---|---:|---:|---|",
    ]
    for source, statuses in manifest["by_source"].items():
        rest = [(s, n) for s, n in statuses.items() if s != converted]
        reason = f"{rest[0][0]} ({rest[0][1]})" if rest else ""
        lines.append(f"| {source} | {sum(statuses.values())} | {statuses.get(converted, 0)} | {reason} |")
    lines += [
        "",
        "## By converter",
        "",
        "| converter | clean | rejected | dockerfiles |",
        "|---|---:|---:|---:|",
    ]
    for converter, statuses in manifest["by_converter"].items():
        rejected = sum(n for s, n in statuses.items() if s != converted)
        lines.append(
            f"| {converter} | {statuses.get(converted, 0)} | {rejected} |"
            f" {manifest['dockerfiles_by_converter'].get(converter, 0)} |"
        )
    lines += [
        "",
        "## By mode",
        "",
        "| mode | tasks |",
        "|---|---:|",
        *(f"| {m} | {n} |" for m, n in manifest["by_mode"].items()),
        "",
        "## Verifier rejections by check",
        "",
        "| check | tasks |",
        "|---|---:|",
        *(f"| {c} | {n} |" for c, n in manifest["by_check"].items()),
    ]
    return "\n".join(lines) + "\n"


def export_task(tasks_dir: str, path: str, dest: Path) -> Path:
    """Write the task ``path`` from a clean ``tasks/`` directory as a Harbor task directory under ``dest``."""
    rows: list[dict] = []
    for shard in (StoragePath(tasks_dir) / "*.parquet").glob():
        with shard.open("rb") as handle:
            rows = pq.read_table(handle, filters=[("path", "=", path)]).to_pylist()
        if rows:
            break
    if not rows:
        raise KeyError(f"{path} is not in {tasks_dir}")
    row = rows[0]
    task_dir = dest / Path(path).name.removesuffix(".tar.gz")
    for blob in (row["task_binary"], row["solution_binary"]):
        if blob is not None:
            read_task_binary(blob).write_to(task_dir)
    return task_dir


@click.command(help="Export one clean task as a Harbor task directory.")
@click.argument("tasks_dir")
@click.argument("path")
@click.option("--dest", type=click.Path(path_type=Path), default=Path("."))
def main(tasks_dir: str, path: str, dest: Path) -> None:
    print(export_task(tasks_dir, path, dest))


if __name__ == "__main__":
    main()
