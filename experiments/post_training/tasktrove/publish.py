# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble TaskTrove Clean from the graded rows.

    tasks/part-00000.parquet every surviving task with its selection columns
    ledger.parquet           one row per task that did not survive: its status and the reason
    manifest.json            revision, tool ref, counts per status, source, converter, mode, tag, check
                             and Dockerfile, plus every source's verdict
    report.md                the manifest as tables, regenerated every run

The survivors are copied by a Zephyr stage that reads only converted rows; the ledger and counts
come from one pass over the graded rows' small columns. The only binaries read are one task per
distinct Dockerfile, for its base image.

    python -m experiments.post_training.tasktrove.publish summary <graded_path> <output_path> <tool_ref>
    python -m experiments.post_training.tasktrove.publish export <tasks_dir> <path> [--dest DIR]

``summary`` rewrites the ledger, manifest and report of an existing output without touching
``tasks/``; ``export`` writes one row back out as a Harbor task directory for hand inspection.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.convert import CONVERTED_SCHEMA
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.dataset import APPROX_SHARD_BYTES, WORKER_RESOURCES
from experiments.post_training.tasktrove.dataset import TASKTROVE_HF_ID, TASKTROVE_REVISION, load_source_verdicts
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, read_task_binary
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
SUMMARY_COLUMNS = ("source", "path", "status", "error", "converter", "mode", "dockerfile_id", "tags")
_READERS = 32
FINAL_SHARDS = 1
_FROM_LINE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE | re.IGNORECASE)


def _survivors(graded_path: str, output_path: str) -> None:
    """Zephyr stage: copy the converted rows' task columns into ``tasks/``."""
    files = Dataset.from_files(str(StoragePath(graded_path) / GRADED_GLOB))
    ds = files.load_parquet(columns=[*TASK_COLUMNS, "status"], approx_shard_bytes=APPROX_SHARD_BYTES)
    ds = ds.filter(lambda row: row["status"] == ConvertStatus.CONVERTED)
    ds = ds.map(lambda row: {name: row[name] for name in TASK_COLUMNS}).reshard(FINAL_SHARDS)
    ds = ds.write_parquet(str(StoragePath(output_path) / "tasks/part-{shard:05d}.parquet"), schema=TASKS_SCHEMA)
    ZephyrContext(name="tasktrove-clean", resources=WORKER_RESOURCES).execute(ds)


def read_columns(glob: StoragePath, columns: tuple[str, ...]) -> pa.Table:
    """The named columns of every parquet the glob matches, read concurrently, plus each row's ``file``
    and ``row`` position so a caller can go back for one row's binaries."""

    def read(path: StoragePath) -> pa.Table:
        with path.open("rb") as handle:
            table = pq.read_table(handle, columns=list(columns))
        return table.append_column("file", pa.array([str(path)] * table.num_rows, pa.string())).append_column(
            "row", pa.array(range(table.num_rows), pa.int64())
        )

    files = sorted(glob.glob(), key=str)
    with ThreadPoolExecutor(_READERS) as pool:
        tables = list(pool.map(read, files))
    return pa.concat_tables(tables)


def dockerfile_texts(graded: pa.Table) -> dict[str, str]:
    """One Dockerfile per distinct ``dockerfile_id`` among the converted rows, read from the first task
    that carries it."""
    first: dict[str, tuple[str, int]] = {}
    for status, dockerfile_id, file, row in zip(
        *(graded.column(name).to_pylist() for name in ("status", "dockerfile_id", "file", "row")), strict=True
    ):
        if status == ConvertStatus.CONVERTED and dockerfile_id not in first:
            first[dockerfile_id] = (file, row)
    by_file: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for dockerfile_id, (file, row) in first.items():
        by_file[file].append((dockerfile_id, row))

    def read(file: str) -> list[tuple[str, str]]:
        with StoragePath(file).open("rb") as handle:
            binaries = pq.read_table(handle, columns=["task_binary"]).column("task_binary")
        return [
            (dockerfile_id, read_task_binary(binaries[row].as_py()).text(DOCKERFILE))
            for dockerfile_id, row in by_file[file]
        ]

    with ThreadPoolExecutor(_READERS) as pool:
        return dict(pair for pairs in pool.map(read, by_file) for pair in pairs)


def build_manifest(graded: pa.Table, tool_ref: str, dockerfiles: dict[str, str]) -> dict:
    """Counts per status, source, converter, mode, check and Dockerfile, plus every source's verdict."""
    verdicts = load_source_verdicts()
    by_status: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_converter: dict[str, Counter] = defaultdict(Counter)
    by_mode: Counter = Counter()
    by_check: Counter = Counter()
    by_tag: Counter = Counter()
    by_dockerfile: dict[str, dict] = {
        dockerfile_id: {
            "base_image": " / ".join(_FROM_LINE.findall(text)),
            "tasks": 0,
            "converters": Counter(),
            "sources": Counter(),
        }
        for dockerfile_id, text in dockerfiles.items()
    }
    columns = {
        name: graded.column(name).to_pylist()
        for name in ("source", "status", "converter", "mode", "dockerfile_id", "tags")
    }
    for source, status, converter, mode, dockerfile_id, tags in zip(*columns.values(), strict=True):
        by_status[status] += 1
        by_source[source][status] += 1
        if converter:
            by_converter[converter][status] += 1
        if status.startswith(VERIFIED_STATUS):
            by_check[status.removeprefix(VERIFIED_STATUS)] += 1
        if status == ConvertStatus.CONVERTED:
            by_mode[mode] += 1
            by_tag.update(tags)
            entry = by_dockerfile[dockerfile_id]
            entry["tasks"] += 1
            entry["converters"][converter] += 1
            entry["sources"][source] += 1
    return {
        "tasktrove": {"hf_id": TASKTROVE_HF_ID, "revision": TASKTROVE_REVISION},
        "verify_tool_ref": tool_ref,
        "input_tasks": graded.num_rows,
        "clean_tasks": by_status[ConvertStatus.CONVERTED],
        "by_status": dict(by_status.most_common()),
        "by_check": dict(by_check.most_common()),
        "by_mode": dict(by_mode.most_common()),
        "by_tag": dict(by_tag.most_common()),
        "by_source": {s: dict(c.most_common()) for s, c in sorted(by_source.items())},
        "source_verdicts": {
            s: {"verdict": v.verdict.value, "family": v.family, "reason": v.reason} for s, v in sorted(verdicts.items())
        },
        "by_converter": {s: dict(c.most_common()) for s, c in sorted(by_converter.items())},
        "dockerfiles": {
            dockerfile_id: {
                **entry,
                "converters": dict(entry["converters"].most_common()),
                "sources": dict(entry["sources"].most_common()),
            }
            for dockerfile_id, entry in sorted(by_dockerfile.items(), key=lambda item: -item[1]["tasks"])
        },
    }


def build_clean(graded_path: str, output_path: str, tool_ref: str) -> None:
    out = StoragePath(output_path)
    for stale in ("tasks", "ledger.parquet"):
        if (out / stale).exists():
            (out / stale).rmtree()
    _survivors(graded_path, output_path)
    write_summary(graded_path, output_path, tool_ref)


def write_summary(graded_path: str, output_path: str, tool_ref: str) -> None:
    """Write ``ledger.parquet``, ``manifest.json`` and ``report.md`` from the graded rows."""
    out = StoragePath(output_path)
    graded = read_columns(StoragePath(graded_path) / GRADED_GLOB, SUMMARY_COLUMNS)
    ledger = graded.filter(pa.compute.not_equal(graded.column("status"), ConvertStatus.CONVERTED.value))
    with (out / "ledger.parquet").open("wb") as handle:
        pq.write_table(ledger.select(list(LEDGER_COLUMNS)), handle)
    manifest = build_manifest(graded, tool_ref, dockerfile_texts(graded))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "report.md").write_text(render_report(manifest))
    logger.info("clean: %d of %d tasks; %s", manifest["clean_tasks"], manifest["input_tasks"], manifest["by_status"])


def render_report(manifest: dict) -> str:
    converted = ConvertStatus.CONVERTED.value
    verdicts = manifest["source_verdicts"]
    kept = [s for s in manifest["by_source"] if verdicts[s]["verdict"] == "keep"]
    dropped = [s for s in manifest["by_source"] if verdicts[s]["verdict"] == "drop"]
    lines = [
        "# TaskTrove Clean",
        "",
        f"{manifest['clean_tasks']} of {manifest['input_tasks']} tasks from {manifest['tasktrove']['hf_id']}"
        f" @ {manifest['tasktrove']['revision']}, graded by tasktrove-verify @ {manifest['verify_tool_ref']}:"
        f" {len(kept)} of {len(kept) + len(dropped)} sources, {len(manifest['by_converter'])} converters,"
        f" {len(manifest['by_mode'])} modes, {len(manifest['dockerfiles'])} distinct Dockerfiles.",
        "",
        "## By status",
        "",
        "| status | tasks |",
        "|---|---:|",
        *(f"| {s} | {n} |" for s, n in manifest["by_status"].items()),
        "",
        "## Kept sources",
        "",
        "| source | family | tasks | clean | main reason for the rest |",
        "|---|---|---:|---:|---|",
    ]
    for source in kept:
        statuses = manifest["by_source"][source]
        rest = [(s, n) for s, n in statuses.items() if s != converted]
        reason = f"{rest[0][0]} ({rest[0][1]})" if rest else ""
        lines.append(
            f"| {source} | {verdicts[source]['family']} | {sum(statuses.values())} |"
            f" {statuses.get(converted, 0)} | {reason} |"
        )
    lines += [
        "",
        "## Dropped sources",
        "",
        "| source | family | tasks | reason |",
        "|---|---|---:|---|",
        *(
            f"| {s} | {verdicts[s]['family']} | {sum(manifest['by_source'][s].values())} | {verdicts[s]['reason']} |"
            for s in dropped
        ),
        "",
        "## By converter",
        "",
        "| converter | clean | rejected | dockerfiles |",
        "|---|---:|---:|---:|",
    ]
    dockerfiles_by_converter: Counter = Counter()
    for entry in manifest["dockerfiles"].values():
        dockerfiles_by_converter.update(entry["converters"].keys())
    for converter, statuses in manifest["by_converter"].items():
        rejected = sum(n for s, n in statuses.items() if s != converted)
        lines.append(
            f"| {converter} | {statuses.get(converted, 0)} | {rejected} | {dockerfiles_by_converter[converter]} |"
        )
    lines += [
        "",
        "## Dockerfiles",
        "",
        "| id | base image | tasks | converters | sources |",
        "|---|---|---:|---|---:|",
        *(
            f"| {dockerfile_id} | {entry['base_image']} | {entry['tasks']} |"
            f" {', '.join(f'{c} ({n})' for c, n in entry['converters'].items())} | {len(entry['sources'])} |"
            for dockerfile_id, entry in manifest["dockerfiles"].items()
        ),
        "",
        "## By mode",
        "",
        "| mode | tasks |",
        "|---|---:|",
        *(f"| {m} | {n} |" for m, n in manifest["by_mode"].items()),
        "",
        "## By tag",
        "",
        "| tag | tasks |",
        "|---|---:|",
        *(f"| {t} | {n} |" for t, n in manifest["by_tag"].items()),
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


@click.group(help=__doc__)
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configure_coreweave_s3()


@main.command(help="Export one clean task as a Harbor task directory.")
@click.argument("tasks_dir")
@click.argument("path")
@click.option("--dest", type=click.Path(path_type=Path), default=Path("."))
def export(tasks_dir: str, path: str, dest: Path) -> None:
    print(export_task(tasks_dir, path, dest))


@main.command(help="Rewrite the ledger, manifest and report of an existing clean output.")
@click.argument("graded_path")
@click.argument("output_path")
@click.argument("tool_ref")
def summary(graded_path: str, output_path: str, tool_ref: str) -> None:
    write_summary(graded_path, output_path, tool_ref)


if __name__ == "__main__":
    main()
