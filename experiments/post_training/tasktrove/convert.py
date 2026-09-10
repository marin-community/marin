# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert TaskTrove tasks to the Clean layout, one converter per key.

Every task is fingerprinted again (cheap, and keeps this stage independent of the fingerprint
shards), routed by source verdict and converter key, and either rewritten as a new task binary or
recorded with the reason it was not. Output rows keep TaskTrove's ``path`` and ``task_binary``
and add the selection columns; the oracle solution goes in ``solution_binary`` rather than in the
binary the agent sees.
"""

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from tasktrove_verify.spec import mode_of, render_spec
from zephyr.context import ZephyrContext

from experiments.post_training.tasktrove.contract import (
    MODE_EXTRAS,
    VERIFIER_TOML,
    VERIFY_TEST_SH,
    dockerfile_id,
    edit_dockerfile,
    render_task_toml,
)
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.fingerprint import COVERAGE_JSON, uncovered_keys
from experiments.post_training.tasktrove.raw_tasks import WORKER_RESOURCES, raw_tasks
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict, load_source_verdicts
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    SOLUTION_DIR,
    TASK_TOML,
    TEST_SH,
    TaskFiles,
    read_task_binary,
    template_fingerprint,
    write_task_binary,
)

logger = logging.getLogger(__name__)

CONVERTED_GLOB = "converted/*.parquet"
_WHITESPACE = re.compile(r"\s+")


def instruction_key(instruction: str) -> str:
    """Two tasks are duplicates when their instructions match after lowercasing and whitespace collapsing."""
    return hashlib.sha256(_WHITESPACE.sub(" ", instruction).strip().lower().encode()).hexdigest()


@dataclass(frozen=True)
class ConvertedRecord:
    source: str
    path: str
    family: str
    template_id: str
    converter: str
    mode: str
    dockerfile_id: str
    language: str
    tags: list[str]
    has_solution: bool
    status: str
    error: str
    instruction_key: str
    """Hash of the normalized instruction; the deduped step reads this column instead of the binary."""
    task_binary: bytes | None
    solution_binary: bytes | None


CONVERTED_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("path", pa.string()),
        ("family", pa.string()),
        ("template_id", pa.string()),
        ("converter", pa.string()),
        ("mode", pa.string()),
        ("dockerfile_id", pa.string()),
        ("language", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("has_solution", pa.bool_()),
        ("status", pa.string()),
        ("error", pa.string()),
        ("instruction_key", pa.string()),
        ("task_binary", pa.binary()),
        ("solution_binary", pa.binary()),
    ]
)


def build_task_files(converted: ConvertedTask, tool_ref: str, metadata: dict) -> TaskFiles:
    """The binary the agent and Harbor see: instruction, task.toml, edited Dockerfile, shim, spec, data."""
    mode = mode_of(converted.spec)
    files: dict[str, bytes] = {
        INSTRUCTION: converted.instruction.encode(),
        TASK_TOML: render_task_toml(converted.agent_timeout, converted.verifier_timeout, metadata).encode(),
        DOCKERFILE: edit_dockerfile(converted.dockerfile, tool_ref, MODE_EXTRAS.get(mode, ())).encode(),
        TEST_SH: VERIFY_TEST_SH.encode(),
        VERIFIER_TOML: render_spec(converted.spec).encode(),
    }
    for path in converted.data_files:
        if path.startswith(SOLUTION_DIR):
            raise ValueError(f"data file {path} is under {SOLUTION_DIR}; put it in solution_files")
    files.update(converted.data_files)
    return TaskFiles(files)


def convert_rows(rows: Iterator[dict], tool_ref: str) -> Iterator[dict]:
    verdicts = load_source_verdicts()
    index = converter_index()
    for row in rows:
        yield asdict(convert_one(verdicts[row["source"]], row["path"], row["task_binary"], index, tool_ref))


def _unconverted(info: SourceInfo, path: str, template_id: str, status: ConvertStatus, error: str) -> ConvertedRecord:
    return ConvertedRecord(
        source=info.source,
        path=path,
        family=info.family,
        template_id=template_id,
        converter="",
        mode="",
        dockerfile_id="",
        language="",
        tags=[],
        has_solution=False,
        status=status.value,
        error=error,
        instruction_key="",
        task_binary=None,
        solution_binary=None,
    )


def convert_one(
    info: SourceInfo, path: str, blob: bytes, index: dict[ConverterKey, Converter], tool_ref: str
) -> ConvertedRecord:
    task = read_task_binary(blob)
    fingerprint = template_fingerprint(task)
    template_id = fingerprint.template_id
    if info.verdict == SourceVerdict.DROP:
        return _unconverted(info, path, template_id, ConvertStatus.DROPPED_SOURCE, "")
    converter = index.get(ConverterKey(info.family, frozenset(fingerprint.code_files)))
    if converter is None:
        return _unconverted(info, path, template_id, ConvertStatus.NO_CONVERTER, "")
    try:
        result = converter.convert(task)
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        return _unconverted(info, path, template_id, ConvertStatus.CONVERTER_ERROR, f"{type(error).__name__}: {error}")
    if isinstance(result, Rejected):
        return _unconverted(info, path, template_id, result.status, result.detail)
    metadata = {
        **result.metadata,
        "tasktrove_source": info.source,
        "tasktrove_path": path,
        "family": info.family,
        "template_id": template_id,
        "converter": converter.name,
        "mode": mode_of(result.spec).value,
        "language": result.language,
        "tags": list(result.tags),
    }
    new_task = build_task_files(result, tool_ref, metadata)
    solution = write_task_binary(TaskFiles(dict(result.solution_files))) if result.solution_files else None
    return ConvertedRecord(
        source=info.source,
        path=path,
        family=info.family,
        template_id=template_id,
        converter=converter.name,
        mode=metadata["mode"],
        dockerfile_id=dockerfile_id(new_task.text(DOCKERFILE)),
        language=result.language,
        tags=list(result.tags),
        has_solution=solution is not None,
        status=ConvertStatus.CONVERTED.value,
        error="",
        instruction_key=instruction_key(result.instruction),
        task_binary=write_task_binary(new_task),
        solution_binary=solution,
    )


def convert_tasks(input_path: str, templates_path: str, output_path: str, tool_ref: str) -> None:
    """Zephyr stage: one converted parquet per shard, every row tagged with its status.

    Refuses to run while ``coverage.json`` lists a kept key with an exemplar and no converter, so
    a converter that was never written cannot silently become a ``no_converter`` column.
    """
    missing = uncovered_keys(json.loads((StoragePath(templates_path) / COVERAGE_JSON).read_text()))
    if missing:
        raise ValueError(f"{len(missing)} kept converter keys have no converter: {missing[:5]}")
    ds = raw_tasks(input_path).map_shard(lambda rows, _: convert_rows(rows, tool_ref))
    ds = ds.write_parquet(str(StoragePath(output_path) / "converted/part-{shard:05d}.parquet"), schema=CONVERTED_SCHEMA)
    ZephyrContext(name="tasktrove-convert", resources=WORKER_RESOURCES).execute(ds)
