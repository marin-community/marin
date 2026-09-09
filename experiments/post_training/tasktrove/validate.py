# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static gates over converted tasks, plus the conversion ledger.

Container gates (oracle pass, no-op fail, old-versus-new grader agreement) run in the verifier
image and are a later stage. This stage checks what can be checked without a container: every
converted task parses back, its ``[verifier]`` table names a kind with a matching subtable, the
files the spec references exist, and the per-source and per-template counts by status are
written as ``ledger.json`` so unconverted templates are visible at a glance.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import fsspec
import pyarrow.parquet as pq

from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TASK_TOML, TEST_SH, read_task_binary
from experiments.post_training.tasktrove.verifier_spec import VERIFY_TEST_SH, VerifierKind, parse_verifier_table

logger = logging.getLogger(__name__)

REQUIRED_FILES = (INSTRUCTION, TASK_TOML, DOCKERFILE, TEST_SH)


@dataclass(frozen=True)
class ValidationLedger:
    tasks: int
    by_status: dict[str, int]
    by_source: dict[str, dict[str, int]]
    unconverted_templates: dict[str, int]
    invalid: list[dict]


def check_converted_task(blob: bytes) -> str | None:
    """Return a problem description, or ``None`` when the converted task is well formed."""
    task = read_task_binary(blob)
    missing = [p for p in REQUIRED_FILES if p not in task.files]
    if missing:
        return f"missing {missing}"
    if task.text(TEST_SH) != VERIFY_TEST_SH:
        return "test.sh is not the verify shim"
    table = parse_verifier_table(task.text(TASK_TOML))
    kind = VerifierKind(table["kind"])
    body = table[kind.value]
    if kind == VerifierKind.STDIO and not any(p.startswith(body.get("cases", "tests/cases")) for p in task.files):
        return "stdio verifier references no case files"
    if kind == VerifierKind.SCRIPT and body["path"] not in task.files:
        return f"script verifier path {body['path']} not in task"
    if kind == VerifierKind.ANSWER and not body.get("expected"):
        return "answer verifier has empty expected value"
    return None


def validate_converted(converted_path: str, output_path: str) -> None:
    fs, _ = fsspec.core.url_to_fs(converted_path)
    by_status: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    unconverted: Counter = Counter()
    invalid: list[dict] = []
    total = 0
    for f in sorted(fs.glob(f"{converted_path}/converted/*.parquet")):
        with fs.open(f, "rb") as handle:
            pf = pq.ParquetFile(handle)
            for rg in range(pf.num_row_groups):
                for row in pf.read_row_group(rg).to_pylist():
                    total += 1
                    by_status[row["status"]] += 1
                    by_source[row["source"]][row["status"]] += 1
                    if row["status"] == "no_converter":
                        unconverted[row["template_id"]] += 1
                    if row["status"] == "converted":
                        problem = check_converted_task(row["task_binary"])
                        if problem is not None:
                            invalid.append({"source": row["source"], "path": row["path"], "problem": problem})
    ledger = ValidationLedger(
        tasks=total,
        by_status=dict(by_status),
        by_source={s: dict(c) for s, c in sorted(by_source.items())},
        unconverted_templates=dict(unconverted.most_common()),
        invalid=invalid[:1000],
    )
    out_fs, _ = fsspec.core.url_to_fs(output_path)
    with out_fs.open(f"{output_path}/ledger.json", "w") as handle:
        json.dump(asdict(ledger), handle, indent=1)
    logger.info("validated %d tasks: %s; %d invalid", total, dict(by_status), len(invalid))
    if invalid:
        raise ValueError(f"{len(invalid)} converted tasks failed static validation; see {output_path}/ledger.json")
