# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert TaskTrove tasks to the Marin verifier format, one template converter at a time.

Every task is fingerprinted again (cheap, and keeps this stage independent of the fingerprint
shards), routed by source verdict and template id, and either rewritten as a new task binary or
recorded as unconverted with the reason. Output parquets keep TaskTrove's ``path`` and
``task_binary`` columns and add ``source``, ``template_id``, ``status`` and ``error``.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum

from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask
from experiments.post_training.tasktrove.converters.registry import CONVERTERS
from experiments.post_training.tasktrove.fingerprint import TASKS_GLOB, iter_task_rows, source_name
from experiments.post_training.tasktrove.sources import SourceVerdict, load_source_verdicts
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TASK_TOML,
    TEST_SH,
    TaskFiles,
    read_task_binary,
    template_fingerprint,
    write_task_binary,
)
from experiments.post_training.tasktrove.verifier_spec import VERIFY_TEST_SH, render_task_toml, tier_dockerfile

logger = logging.getLogger(__name__)


class ConvertStatus(StrEnum):
    CONVERTED = "converted"
    NO_CONVERTER = "no_converter"
    CONVERTER_ERROR = "converter_error"
    DROPPED_SOURCE = "dropped_source"
    REWRITE_SOURCE = "rewrite_source"


@dataclass(frozen=True)
class ConvertedRecord:
    source: str
    path: str
    template_id: str
    status: str
    error: str
    task_binary: bytes | None


def build_task_files(converted: ConvertedTask, source: str, path: str, template_id: str) -> TaskFiles:
    """Assemble the new task binary: instruction, task.toml with ``[verifier]``, tier Dockerfile,
    the three-line test.sh, data files, and the oracle solution if any."""
    metadata = {**converted.metadata, "tasktrove_source": source, "tasktrove_path": path, "template_id": template_id}
    files: dict[str, bytes] = {
        INSTRUCTION: converted.instruction.encode(),
        TASK_TOML: (
            render_task_toml(converted.agent_timeout, converted.verifier_timeout, converted.verifier, metadata).encode()
        ),
        DOCKERFILE: tier_dockerfile(converted.tier, converted.repo_setup).encode(),
        TEST_SH: VERIFY_TEST_SH.encode(),
    }
    files.update(converted.data_files)
    files.update(converted.solution_files)
    return TaskFiles(files)


def convert_parquet(parquet_path: str) -> Iterator[dict]:
    verdict = load_source_verdicts()[source_name(parquet_path)].verdict
    skip_status = {
        SourceVerdict.DROP: ConvertStatus.DROPPED_SOURCE,
        SourceVerdict.REWRITE: ConvertStatus.REWRITE_SOURCE,
    }.get(verdict)
    for row in iter_task_rows(parquet_path):
        yield asdict(_convert_one(row.source, row.path, row.task_binary, skip_status))


def _convert_one(source: str, path: str, blob: bytes, skip_status: ConvertStatus | None) -> ConvertedRecord:
    task = read_task_binary(blob)
    template_id = template_fingerprint(task).template_id
    if skip_status is not None:
        return ConvertedRecord(source, path, template_id, skip_status, "", None)
    converter = CONVERTERS.get(template_id)
    if converter is None:
        return ConvertedRecord(source, path, template_id, ConvertStatus.NO_CONVERTER, "", None)
    try:
        converted = converter(task)
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        return ConvertedRecord(
            source, path, template_id, ConvertStatus.CONVERTER_ERROR, f"{type(error).__name__}: {error}", None
        )
    new_task = build_task_files(converted, source, path, template_id)
    return ConvertedRecord(source, path, template_id, ConvertStatus.CONVERTED, "", write_task_binary(new_task))


def convert_tasks(input_path: str, output_path: str) -> None:
    """Zephyr stage: one output parquet per source parquet, every row tagged with its status."""
    ds = Dataset.from_files(str(StoragePath(input_path) / TASKS_GLOB)).flat_map(convert_parquet)
    ds = ds.write_parquet(str(StoragePath(output_path) / "converted/part-{shard:05d}.parquet"))
    ZephyrContext(name="tasktrove-convert").execute(ds)
