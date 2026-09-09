# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint every TaskTrove task and build the template index.

``fingerprint_tasks`` walks every ``*/tasks.parquet`` and writes one row per task with its
template id, so per-template work replaces per-task work. ``build_template_index`` groups those
rows, extracts one exemplar task per template into ``templates/<template_id>/`` and writes
``templates.json`` plus a Markdown summary. That directory is the handoff for agents writing
converters: each template gets one converter, not each task.
"""

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TEST_SH,
    read_task_binary,
    template_fingerprint,
)

logger = logging.getLogger(__name__)

TASKS_GLOB = "*/tasks.parquet"
EXEMPLAR_MIN_TASKS = 20
"""Templates below this many tasks are listed in the index but get no extracted exemplar. The
long tail is per-repository SWE setup scripts, handled per repository rather than per template."""


@dataclass(frozen=True)
class TaskRow:
    """One TaskTrove row and where it sits in its source parquet."""

    source: str
    path: str
    row_group: int
    row_in_group: int
    task_binary: bytes


@dataclass(frozen=True)
class TaskFingerprintRecord:
    source: str
    path: str
    row_group: int
    row_in_group: int
    template_id: str
    dockerfile_id: str
    test_sh_id: str
    base_image: str
    code_files: list[str]
    data_files: list[str]
    instruction_chars: int
    has_solution: bool
    task_bytes: int


def source_name(parquet_path: str) -> str:
    return StoragePath(parquet_path).parent.name


def iter_task_rows(parquet_path: str) -> Iterator[TaskRow]:
    """Yield every task in one source parquet, one row group in memory at a time."""
    source = source_name(parquet_path)
    with StoragePath(parquet_path).open("rb") as handle:
        pf = pq.ParquetFile(handle)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["path", "task_binary"])
            paths = table.column("path").to_pylist()
            blobs = table.column("task_binary").to_pylist()
            for i, (path, blob) in enumerate(zip(paths, blobs, strict=True)):
                yield TaskRow(source, path, rg, i, blob)


def fingerprint_parquet(parquet_path: str) -> Iterator[dict]:
    for row in iter_task_rows(parquet_path):
        task = read_task_binary(row.task_binary)
        fp = template_fingerprint(task)
        dockerfile = task.get_text(DOCKERFILE) or ""
        base_image = next((line.split()[1] for line in dockerfile.splitlines() if line.startswith("FROM ")), "")
        yield asdict(
            TaskFingerprintRecord(
                source=row.source,
                path=row.path,
                row_group=row.row_group,
                row_in_group=row.row_in_group,
                template_id=fp.template_id,
                dockerfile_id=fp.dockerfile_id,
                test_sh_id=fp.test_sh_id,
                base_image=base_image,
                code_files=list(fp.code_files),
                data_files=list(fp.data_files),
                instruction_chars=len(task.files.get(INSTRUCTION, b"")),
                has_solution=task.has_solution,
                task_bytes=len(row.task_binary),
            )
        )


def fingerprint_tasks(input_path: str, output_path: str) -> None:
    """Zephyr stage: one parquet shard of fingerprint records per source parquet."""
    pattern = str(StoragePath(input_path) / TASKS_GLOB)
    ds = Dataset.from_files(pattern).flat_map(fingerprint_parquet)
    ds = ds.write_parquet(str(StoragePath(output_path) / "fingerprints/part-{shard:05d}.parquet"))
    ZephyrContext(name="tasktrove-fingerprint").execute(ds)


@dataclass
class TemplateSummary:
    template_id: str
    tasks: int
    sources: dict[str, int]
    dockerfile_ids: dict[str, int]
    base_images: dict[str, int]
    code_files: list[str]
    data_file_shapes: list[str]
    tasks_with_solution: int
    exemplar_source: str
    exemplar_path: str
    exemplar_row_group: int
    exemplar_row_in_group: int


def build_template_index(input_path: str, fingerprints_path: str, output_path: str) -> None:
    """Group fingerprints by template, extract one exemplar per template, write the index."""
    shards = (StoragePath(fingerprints_path) / "fingerprints/*.parquet").glob()
    rows: list[dict] = []
    for shard in sorted(shards, key=str):
        with shard.open("rb") as handle:
            rows.extend(pq.read_table(handle).to_pylist())
    logger.info("loaded %d fingerprint rows from %d shards", len(rows), len(shards))

    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(r)

    summaries: list[TemplateSummary] = []
    exemplars: list[tuple[str, dict]] = []
    for template_id, members in sorted(by_template.items(), key=lambda kv: -len(kv[1])):
        exemplar = members[0]
        if len(members) >= EXEMPLAR_MIN_TASKS:
            exemplars.append((template_id, exemplar))
        summaries.append(
            TemplateSummary(
                template_id=template_id,
                tasks=len(members),
                sources=dict(Counter(m["source"] for m in members).most_common()),
                dockerfile_ids=dict(Counter(m["dockerfile_id"] for m in members).most_common()),
                base_images=dict(Counter(m["base_image"] for m in members).most_common()),
                code_files=exemplar["code_files"],
                data_file_shapes=sorted({p for m in members[:50] for p in m["data_files"]})[:40],
                tasks_with_solution=sum(1 for m in members if m["has_solution"]),
                exemplar_source=exemplar["source"],
                exemplar_path=exemplar["path"],
                exemplar_row_group=exemplar["row_group"],
                exemplar_row_in_group=exemplar["row_in_group"],
            )
        )
    out = StoragePath(output_path)
    _extract_exemplars(input_path, exemplars, out / "templates")
    (out / "templates.json").write_text(json.dumps([asdict(s) for s in summaries], indent=1))
    (out / "templates.md").write_text(_render_summary(summaries, len(rows)))
    logger.info("wrote %d templates covering %d tasks to %s", len(summaries), len(rows), output_path)


def _extract_exemplars(input_path: str, exemplars: list[tuple[str, dict]], dest_root: StoragePath) -> None:
    """Extract exemplar tasks, reading each parquet row group once for every exemplar it holds."""
    by_group: dict[tuple[str, int], list[tuple[str, dict]]] = defaultdict(list)
    for template_id, record in exemplars:
        by_group[(record["source"], record["row_group"])].append((template_id, record))
    for (source, row_group), wanted in sorted(by_group.items()):
        parquet = StoragePath(input_path) / source / "tasks.parquet"
        with parquet.open("rb") as handle:
            column = pq.ParquetFile(handle).read_row_group(row_group, columns=["task_binary"]).column("task_binary")
            for template_id, record in wanted:
                task = read_task_binary(column[record["row_in_group"]].as_py())
                for path, data in task.files.items():
                    target = dest_root / template_id / "exemplar" / path
                    target.parent.mkdirs()
                    target.write_bytes(data)


def _render_summary(summaries: list[TemplateSummary], total: int) -> str:
    lines = [
        "# TaskTrove templates",
        "",
        f"{len(summaries)} templates over {total} tasks. One converter per template id; see",
        f"`converters/registry.py`. Templates with at least {EXEMPLAR_MIN_TASKS} tasks have one",
        "extracted exemplar under `templates/<template_id>/exemplar/`; the rest are listed only.",
        "",
        "| template | tasks | sources | dockerfiles | with solution | test.sh / code files |",
        "|---|---:|---|---:|---:|---|",
    ]
    for s in summaries:
        srcs = ", ".join(f"{k} ({v})" for k, v in list(s.sources.items())[:3])
        if len(s.sources) > 3:
            srcs += f", +{len(s.sources) - 3} more"
        code = ", ".join(p for p in s.code_files if p != TEST_SH)[:80]
        lines.append(
            f"| `{s.template_id}` | {s.tasks} | {srcs} | {len(s.dockerfile_ids)} | {s.tasks_with_solution} | {code} |"
        )
    return "\n".join(lines) + "\n"
