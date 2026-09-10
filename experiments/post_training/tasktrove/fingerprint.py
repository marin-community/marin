# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint every TaskTrove task and build the template index.

``summarize_templates`` fingerprints every row of every ``*/tasks.parquet`` and groups the
fingerprints by template id, writing one summary per template, so per-template work replaces
per-task work. ``build_template_index`` extracts one exemplar task per template into
``templates/<template_id>/`` and writes ``templates.json`` plus a Markdown summary. That
directory is the handoff for agents writing converters: each template gets one converter, not
each task.
"""

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.readers import load_jsonl

from experiments.post_training.tasktrove.converters.converted_task import ConverterKey
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.raw_tasks import WORKER_RESOURCES, raw_tasks
from experiments.post_training.tasktrove.sources import SourceVerdict, load_source_verdicts
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TEST_SH,
    read_task_binary,
    template_fingerprint,
)

logger = logging.getLogger(__name__)

COVERAGE_JSON = "coverage.json"
SUMMARIES_GLOB = "template_summaries/*.jsonl.gz"
SHAPE_SAMPLE = 50
"""Data-file shapes are collected from this many members of each template."""
EXEMPLAR_MIN_TASKS = 20
"""Templates below this many tasks are listed in the index but get no extracted exemplar. The
long tail is per-repository SWE setup scripts, handled per repository rather than per template."""


@dataclass(frozen=True)
class TaskFingerprintRecord:
    source: str
    path: str
    template_id: str
    dockerfile_id: str
    test_sh_id: str
    base_image: str
    code_files: list[str]
    data_files: list[str]
    instruction_chars: int
    has_solution: bool
    task_bytes: int


def fingerprint_row(row: dict) -> dict:
    task = read_task_binary(row["task_binary"])
    fp = template_fingerprint(task)
    dockerfile = task.get_text(DOCKERFILE) or ""
    base_image = next((line.split()[1] for line in dockerfile.splitlines() if line.startswith("FROM ")), "")
    return asdict(
        TaskFingerprintRecord(
            source=row["source"],
            path=row["path"],
            template_id=fp.template_id,
            dockerfile_id=fp.dockerfile_id,
            test_sh_id=fp.test_sh_id,
            base_image=base_image,
            code_files=list(fp.code_files),
            data_files=list(fp.data_files),
            instruction_chars=len(task.files.get(INSTRUCTION, b"")),
            has_solution=task.has_solution,
            task_bytes=len(row["task_binary"]),
        )
    )


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


@dataclass
class KeyCoverage:
    """One converter key over the kept sources: what it covers and whether a converter exists."""

    family: str
    code_files: list[str]
    tasks: int
    templates: int
    sources: dict[str, int]
    exemplar_template: str
    converter: str | None


def key_coverage(summaries: list[TemplateSummary]) -> list[KeyCoverage]:
    verdicts = load_source_verdicts()
    index = converter_index()
    grouped: dict[ConverterKey, dict] = {}
    for s in summaries:
        for source, count in s.sources.items():
            info = verdicts.get(source)
            if info is None or info.verdict != SourceVerdict.KEEP:
                continue
            key = ConverterKey(info.family, frozenset(s.code_files))
            entry = grouped.setdefault(key, {"tasks": 0, "templates": set(), "sources": Counter(), "exemplar": s})
            entry["tasks"] += count
            entry["templates"].add(s.template_id)
            entry["sources"][source] += count
            if s.tasks > entry["exemplar"].tasks:
                entry["exemplar"] = s
    coverage = [
        KeyCoverage(
            family=key.family,
            code_files=sorted(key.code_files),
            tasks=entry["tasks"],
            templates=len(entry["templates"]),
            sources=dict(entry["sources"].most_common()),
            exemplar_template=entry["exemplar"].template_id,
            converter=index[key].name if key in index else None,
        )
        for key, entry in grouped.items()
    ]
    return sorted(coverage, key=lambda c: -c.tasks)


def uncovered_keys(coverage: list[dict]) -> list[str]:
    """Keys with at least one exemplar-sized template and no converter, largest first."""
    return [
        f"{c['family']} {c['code_files']} ({c['tasks']} tasks)"
        for c in coverage
        if c["converter"] is None and c["tasks"] >= EXEMPLAR_MIN_TASKS
    ]


def summarize_template(template_id: str, members: Iterator[dict]) -> dict:
    """Reduce one template's fingerprints to a ``TemplateSummary``; the exemplar is the lowest (source, path)."""
    tasks = 0
    sources: Counter = Counter()
    dockerfile_ids: Counter = Counter()
    base_images: Counter = Counter()
    with_solution = 0
    shapes: set[str] = set()
    exemplar: dict | None = None
    for m in members:
        tasks += 1
        sources[m["source"]] += 1
        dockerfile_ids[m["dockerfile_id"]] += 1
        base_images[m["base_image"]] += 1
        with_solution += m["has_solution"]
        if tasks <= SHAPE_SAMPLE:
            shapes.update(m["data_files"])
        if exemplar is None or (m["source"], m["path"]) < (exemplar["source"], exemplar["path"]):
            exemplar = m
    assert exemplar is not None
    return asdict(
        TemplateSummary(
            template_id=template_id,
            tasks=tasks,
            sources=dict(sources.most_common()),
            dockerfile_ids=dict(dockerfile_ids.most_common()),
            base_images=dict(base_images.most_common()),
            code_files=exemplar["code_files"],
            data_file_shapes=sorted(shapes)[:40],
            tasks_with_solution=with_solution,
            exemplar_source=exemplar["source"],
            exemplar_path=exemplar["path"],
        )
    )


def summarize_templates(input_path: str, output_path: str) -> None:
    """Zephyr stage: fingerprint every task, group by template id, write one summary per template."""
    ds = raw_tasks(input_path).map(fingerprint_row)
    ds = ds.group_by(key=lambda fp: fp["template_id"], reducer=summarize_template)
    ds = ds.write_jsonl(str(StoragePath(output_path) / "template_summaries/part-{shard:05d}.jsonl.gz"))
    ZephyrContext(name="tasktrove-templates", resources=WORKER_RESOURCES).execute(ds)


def read_summaries(summaries_path: str) -> list[TemplateSummary]:
    files = sorted((StoragePath(summaries_path) / SUMMARIES_GLOB).glob(), key=str)
    summaries = [TemplateSummary(**row) for f in files for row in load_jsonl(str(f))]
    return sorted(summaries, key=lambda s: (-s.tasks, s.template_id))


def build_template_index(input_path: str, summaries_path: str, output_path: str) -> None:
    """Extract one exemplar per template with enough tasks and write the index and coverage."""
    summaries = read_summaries(summaries_path)
    total = sum(s.tasks for s in summaries)
    out = StoragePath(output_path)
    _extract_exemplars(input_path, [s for s in summaries if s.tasks >= EXEMPLAR_MIN_TASKS], out / "templates")
    (out / "templates.json").write_text(json.dumps([asdict(s) for s in summaries], indent=1))
    coverage = key_coverage(summaries)
    (out / COVERAGE_JSON).write_text(json.dumps([asdict(c) for c in coverage], indent=1))
    (out / "templates.md").write_text(_render_summary(summaries, coverage, total))
    logger.info("wrote %d templates covering %d tasks to %s", len(summaries), total, output_path)


def _extract_exemplars(input_path: str, exemplars: list[TemplateSummary], dest_root: StoragePath) -> None:
    """Extract exemplar tasks, scanning each source parquet's paths once and reading only the row groups that hit."""
    by_source: dict[str, dict[str, str]] = defaultdict(dict)
    for s in exemplars:
        by_source[s.exemplar_source][s.exemplar_path] = s.template_id
    for source, wanted in sorted(by_source.items()):
        parquet = StoragePath(input_path) / source / "tasks.parquet"
        with parquet.open("rb") as handle:
            pf = pq.ParquetFile(handle)
            for rg in range(pf.num_row_groups):
                paths = pf.read_row_group(rg, columns=["path"]).column("path").to_pylist()
                hits = [(i, wanted[p]) for i, p in enumerate(paths) if p in wanted]
                if not hits:
                    continue
                column = pf.read_row_group(rg, columns=["task_binary"]).column("task_binary")
                for i, template_id in hits:
                    for path, data in read_task_binary(column[i].as_py()).files.items():
                        target = dest_root / template_id / "exemplar" / path
                        target.parent.mkdirs()
                        target.write_bytes(data)


def _render_summary(summaries: list[TemplateSummary], coverage: list[KeyCoverage], total: int) -> str:
    lines = [
        "# TaskTrove templates",
        "",
        f"{len(summaries)} templates over {total} tasks. Templates with at least {EXEMPLAR_MIN_TASKS} tasks",
        "have one extracted exemplar under `templates/<template_id>/exemplar/`; the rest are listed only.",
        "",
        "## Converter keys over kept sources",
        "",
        "One converter per key (source family plus the template's code files); see `converters/registry.py`.",
        "",
        "| family | code files | tasks | templates | converter | exemplar |",
        "|---|---|---:|---:|---|---|",
    ]
    for c in coverage:
        code = ", ".join(p for p in c.code_files if p != TEST_SH) or "test.sh only"
        lines.append(
            f"| {c.family} | {code} | {c.tasks} | {c.templates} | {c.converter or 'none'} | `{c.exemplar_template}` |"
        )
    lines += [
        "",
        "## Templates",
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
