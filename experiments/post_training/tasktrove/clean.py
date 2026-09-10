# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble TaskTrove Clean from the deduped rows and the verified ledger.

    tasks/<source>/part-<n>.parquet  every surviving task with its selection columns
    ledger/convert.parquet           one row per task that did not survive conversion or dedup
    ledger/verified.parquet          one row per task the verified step rejected
    manifest.json                    revision, tool ref, counts per step, status, source, converter, mode
    report.md                        per-converter summary, regenerated every run

Deduped shards are streamed one at a time and each writes its survivors as one parquet per
source, so the step holds one shard in memory however large the source.

``export_task`` writes one row back out as a Harbor task directory for hand inspection.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.dedup import DEDUPED_GLOB
from experiments.post_training.tasktrove.sources import TASKTROVE_HF_ID, TASKTROVE_REVISION
from experiments.post_training.tasktrove.taskbinary import read_task_binary
from experiments.post_training.tasktrove.verify import read_ledger

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


@dataclass(frozen=True)
class ConvertLedgerRow:
    """Why one task is absent from ``tasks/``: its convert, dedup, or ``verified:<check>`` status."""

    source: str
    path: str
    status: str
    detail: str


@dataclass
class CleanCounts:
    input_tasks: int = 0
    clean_tasks: int = 0
    by_status: Counter = None  # type: ignore[assignment]
    by_source: dict[str, Counter] = None  # type: ignore[assignment]
    by_converter: dict[str, Counter] = None  # type: ignore[assignment]
    by_mode: Counter = None  # type: ignore[assignment]
    dockerfiles_by_converter: dict[str, set[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.by_status = Counter()
        self.by_source = defaultdict(Counter)
        self.by_converter = defaultdict(Counter)
        self.by_mode = Counter()
        self.dockerfiles_by_converter = defaultdict(set)


def build_clean(deduped_path: str, verified_path: str, output_path: str, tool_ref: str) -> None:
    out = StoragePath(output_path)
    verified = [asdict(row) for row in read_ledger(verified_path)]
    rejected = {(r["source"], r["path"]): r["check"] for r in verified}

    counts = CleanCounts()
    convert_ledger: list[dict] = []
    for shard in sorted((StoragePath(deduped_path) / DEDUPED_GLOB).glob(), key=str):
        by_source_rows: dict[str, list[dict]] = defaultdict(list)
        for row in load_parquet(str(shard)):
            counts.input_tasks += 1
            status = row["status"]
            if status == ConvertStatus.CONVERTED and (row["source"], row["path"]) in rejected:
                status = f"verified:{rejected[(row['source'], row['path'])]}"
            counts.by_status[status] += 1
            counts.by_source[row["source"]][status] += 1
            if row["converter"]:
                counts.by_converter[row["converter"]][status] += 1
            if status != ConvertStatus.CONVERTED:
                convert_ledger.append(asdict(ConvertLedgerRow(row["source"], row["path"], status, row["error"])))
                continue
            counts.clean_tasks += 1
            counts.by_mode[row["mode"]] += 1
            counts.dockerfiles_by_converter[row["converter"]].add(row["dockerfile_id"])
            by_source_rows[row["source"]].append({k: row[k] for k in TASK_COLUMNS})
        for source, rows in sorted(by_source_rows.items()):
            _write_parquet(out / "tasks" / source / shard.name, rows)

    _write_parquet(out / "ledger" / "convert.parquet", convert_ledger)
    _write_parquet(out / "ledger" / "verified.parquet", verified)
    manifest = {
        "tasktrove": {"hf_id": TASKTROVE_HF_ID, "revision": TASKTROVE_REVISION},
        "verify_tool_ref": tool_ref,
        "input_tasks": counts.input_tasks,
        "clean_tasks": counts.clean_tasks,
        "by_status": dict(counts.by_status.most_common()),
        "by_mode": dict(counts.by_mode.most_common()),
        "by_source": {s: dict(c) for s, c in sorted(counts.by_source.items())},
        "by_converter": {s: dict(c) for s, c in sorted(counts.by_converter.items())},
        "dockerfiles_by_converter": {c: len(ids) for c, ids in sorted(counts.dockerfiles_by_converter.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "report.md").write_text(render_report(manifest))
    logger.info("clean: %d of %d tasks; %s", counts.clean_tasks, counts.input_tasks, manifest["by_status"])


def _write_parquet(target: StoragePath, rows: list[dict]) -> None:
    target.parent.mkdirs()
    with target.open("wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle)


def render_report(manifest: dict) -> str:
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
        "## By converter",
        "",
        "| converter | clean | rejected | dockerfiles |",
        "|---|---:|---:|---:|",
    ]
    for converter, statuses in manifest["by_converter"].items():
        clean = statuses.get(ConvertStatus.CONVERTED.value, 0)
        rejected = sum(n for s, n in statuses.items() if s != ConvertStatus.CONVERTED.value)
        lines.append(
            f"| {converter} | {clean} | {rejected} | {manifest['dockerfiles_by_converter'].get(converter, 0)} |"
        )
    lines += [
        "",
        "## By mode",
        "",
        "| mode | tasks |",
        "|---|---:|",
        *(f"| {m} | {n} |" for m, n in manifest["by_mode"].items()),
    ]
    return "\n".join(lines) + "\n"


def export_task(tasks_dir: str, path: str, dest: Path) -> Path:
    """Write the task ``path`` from a clean ``tasks/<source>`` directory as a Harbor task directory under ``dest``."""
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
