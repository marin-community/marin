# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parquet contract for generated curriculum tasks and their oracle results."""

from typing import Any

import pyarrow as pa

from experiments.post_training.curriculum_sft.ablation.matrix import AblationCell
from experiments.post_training.curriculum_sft.ablation.verifier import verify_task_payload

GENERATION_FILENAME = "generation.json"
GENERATED_TASKS_FILENAME = "tasks/part-00000-of-00001.parquet"
RAW_RESPONSES_FILENAME = "raw-responses.jsonl"
GENERATED_TASK_SCHEMA = pa.schema(
    [
        pa.field("cell", pa.string(), nullable=False),
        pa.field("curriculum", pa.string(), nullable=False),
        pa.field("generation_spec", pa.string(), nullable=False),
        pa.field("replicate", pa.int64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("task_id", pa.string(), nullable=True),
        pa.field("issuer", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("revenue", pa.int64(), nullable=False),
        pa.field("operating_cost", pa.int64(), nullable=False),
        pa.field("gross_profit", pa.int64(), nullable=False),
        pa.field("margin_bps", pa.int64(), nullable=False),
        pa.field("evidence", pa.list_(pa.string()), nullable=False),
        pa.field("format_valid", pa.bool_(), nullable=False),
        pa.field("arithmetic_valid", pa.bool_(), nullable=False),
        pa.field("evidence_valid", pa.bool_(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
    ]
)


def generated_task_record(
    cell: AblationCell,
    *,
    replicate: int,
    seed: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one generated task and its oracle result into the Parquet contract."""

    verification = verify_task_payload(payload)
    return {
        "cell": cell.name,
        "curriculum": cell.curriculum.value,
        "generation_spec": cell.generation_spec.value,
        "replicate": replicate,
        "seed": seed,
        "task_id": payload.get("task_id"),
        "issuer": payload["issuer"],
        "question": payload["question"],
        "revenue": payload["facts"]["revenue"],
        "operating_cost": payload["facts"]["operating_cost"],
        "gross_profit": payload["answer"]["gross_profit"],
        "margin_bps": payload["answer"]["margin_bps"],
        "evidence": payload["evidence"],
        "format_valid": verification.format_valid,
        "arithmetic_valid": verification.arithmetic_valid,
        "evidence_valid": verification.evidence_valid,
        "accepted": verification.accepted,
    }


def task_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the verifier's task payload from one Parquet record."""

    payload = {
        "issuer": record["issuer"],
        "question": record["question"],
        "facts": {
            "revenue": record["revenue"],
            "operating_cost": record["operating_cost"],
        },
        "answer": {
            "gross_profit": record["gross_profit"],
            "margin_bps": record["margin_bps"],
        },
        "evidence": record["evidence"],
    }
    if record["task_id"] is not None:
        payload["task_id"] = record["task_id"]
    return payload
