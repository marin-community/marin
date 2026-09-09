# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static gates over converted tasks, plus the conversion ledger.

Container gates (oracle pass, no-op fail, old-versus-new grader agreement) run in the verifier
image and are a later stage. This stage checks what can be checked without a container: every
converted task parses back into a typed :class:`VerifierSpec`, the files the spec references
exist, and the per-source and per-template counts by status are written as ``ledger.json`` so
unconverted templates are visible at a glance.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tasktrove.convert import ConvertStatus
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TASK_TOML, TEST_SH, read_task_binary
from experiments.post_training.tasktrove.verifier_spec import VERIFY_TEST_SH, parse_verifier_spec

logger = logging.getLogger(__name__)

REQUIRED_FILES = (INSTRUCTION, TASK_TOML, DOCKERFILE, TEST_SH)


@dataclass(frozen=True)
class InvalidTask:
    source: str
    path: str
    problem: str


@dataclass(frozen=True)
class ValidationLedger:
    tasks: int
    by_status: dict[str, int]
    by_source: dict[str, dict[str, int]]
    unconverted_templates: dict[str, int]
    invalid: list[InvalidTask]


def check_converted_task(blob: bytes) -> str | None:
    """Return a problem description, or ``None`` when the converted task is well formed."""
    task = read_task_binary(blob)
    missing = [p for p in REQUIRED_FILES if p not in task.files]
    if missing:
        return f"missing {missing}"
    if task.text(TEST_SH) != VERIFY_TEST_SH:
        return "test.sh is not the verify shim"
    spec = parse_verifier_spec(task.text(TASK_TOML))
    if spec.stdio is not None and not any(p.startswith(spec.stdio.cases) for p in task.files):
        return "stdio verifier references no case files"
    if spec.script is not None and spec.script.path not in task.files:
        return f"script verifier path {spec.script.path} not in task"
    if spec.answer is not None and not spec.answer.expected:
        return "answer verifier has empty expected value"
    return None


def validate_converted(converted_path: str, output_path: str) -> None:
    by_status: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    unconverted: Counter = Counter()
    invalid: list[InvalidTask] = []
    total = 0
    for shard in sorted((StoragePath(converted_path) / "converted/*.parquet").glob(), key=str):
        with shard.open("rb") as handle:
            pf = pq.ParquetFile(handle)
            for rg in range(pf.num_row_groups):
                for row in pf.read_row_group(rg).to_pylist():
                    total += 1
                    status = ConvertStatus(row["status"])
                    by_status[status] += 1
                    by_source[row["source"]][status] += 1
                    if status == ConvertStatus.NO_CONVERTER:
                        unconverted[row["template_id"]] += 1
                    if status == ConvertStatus.CONVERTED:
                        problem = check_converted_task(row["task_binary"])
                        if problem is not None:
                            invalid.append(InvalidTask(row["source"], row["path"], problem))
    ledger = ValidationLedger(
        tasks=total,
        by_status=dict(by_status),
        by_source={s: dict(c) for s, c in sorted(by_source.items())},
        unconverted_templates=dict(unconverted.most_common()),
        invalid=invalid[:1000],
    )
    (StoragePath(output_path) / "ledger.json").write_text(json.dumps(asdict(ledger), indent=1))
    logger.info("validated %d tasks: %s; %d invalid", total, dict(by_status), len(invalid))
    if invalid:
        raise ValueError(f"{len(invalid)} converted tasks failed static validation; see {output_path}/ledger.json")
