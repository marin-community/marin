# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate and mechanically filter the judge-free ablation task matrix.

The prompts deliberately expose only a broad financial-reporting area and a
short curriculum outcome.  They do not contain FinanceBench examples or
questions.  GLM is used only for task generation; acceptance is determined by
the local arithmetic/evidence oracle.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    SftDose,
    build_generation_prompt,
    unique_accepted_payloads,
)
from experiments.post_training.curriculum_sft.ablation.tasks import (
    EVIDENCE_IDS,
    FACT_FIELDS,
    RESULT_FIELDS,
    TASK_PAYLOAD_FIELDS,
)
from experiments.post_training.curriculum_sft.ablation.verifier import verify_task_payload
from experiments.post_training.curriculum_sft.glm_responses import tool_arguments
from experiments.post_training.tasktrove.mcqa_routing import (
    GLM_BULK_TOKEN_ENV,
    read_batch_output,
    resolve_base_url,
    submit_batch,
    wait_for_batch,
)

logger = logging.getLogger(__name__)

RELAY_JOB = "/muchanem/glm53-relay-08a"
CURRICULUM_PACKET = "\n".join(
    (
        "Curriculum catalog: 2026.09.20-cross-domain-v3",
        "Subject curriculum: D27 production-candidate-v1",
        "Capability ID: d27.reporting.analysis",
        "Name: Calculate Reported Financial Measures",
        "Outcome: Select, calculate, and reconcile comparable financial measures "
        "from statements and disclosures under stated definitions.",
        "Includes:",
        "- profitability, liquidity, leverage, and efficiency ratios",
        "- common-size analysis",
        "- trend and growth measures",
        "- earnings-quality and cash-versus-profit measures",
        "Excludes:",
        "- standalone disclosure lookup",
        "- forward-looking enterprise valuation",
        "- evidence-attributed driver bridges",
    )
)
TASK_FAMILY = "fictional company questions"


def _task_schema(expected_tasks: int) -> dict[str, Any]:
    task = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(TASK_PAYLOAD_FIELDS),
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "issuer": {"type": "string", "minLength": 1},
            "facts": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(FACT_FIELDS),
                "properties": {"revenue": {"type": "integer"}, "operating_cost": {"type": "integer"}},
            },
            "question": {"type": "string", "minLength": 1},
            "answer": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(RESULT_FIELDS),
                "properties": {"gross_profit": {"type": "integer"}, "margin_bps": {"type": "integer"}},
            },
            "evidence": {
                "type": "array",
                "description": f"Use exactly {json.dumps(EVIDENCE_IDS)} in this order.",
                "items": {"type": "string"},
                "minItems": len(EVIDENCE_IDS),
                "maxItems": len(EVIDENCE_IDS),
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tasks"],
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": expected_tasks,
                "maxItems": expected_tasks,
                "items": task,
            }
        },
    }


def generation_body(cell: AblationCell, expected_tasks: int, seed: int) -> dict[str, Any]:
    prompt = build_generation_prompt(
        cell,
        subject_area="financial reporting",
        task_family=TASK_FAMILY,
        curriculum_packet=CURRICULUM_PACKET,
        task_count=expected_tasks,
    )
    return {
        "model": "glm-5.3",
        "messages": [
            {"role": "system", "content": "Generate fictional tasks and call submit_tasks exactly once."},
            {"role": "user", "content": prompt},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_tasks",
                    "description": "Submit the generated fictional tasks.",
                    "strict": True,
                    "parameters": _task_schema(expected_tasks),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "submit_tasks"}},
        "parallel_tool_calls": False,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0.8,
        "seed": seed,
        "max_tokens": 12000,
    }


def run_matrix(
    output: str,
    *,
    tasks_per_replicate: int = 16,
    replicate_count: int = 4,
    seed: int = 17,
    relay_job: str = RELAY_JOB,
) -> None:
    """Run paired generation replicates and write raw quality plus all payloads."""

    cells = tuple(
        AblationCell(curriculum, generation_spec, SftDose.LOW)
        for curriculum in CurriculumCondition
        for generation_spec in GenerationSpec
    )
    base_url = resolve_base_url(relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    lines = []
    cell_by_request: dict[str, tuple[AblationCell, int]] = {}
    for replicate in range(replicate_count):
        replicate_seed = seed + replicate
        for cell in cells:
            custom_id = f"ablation-{cell.name}-replicate-{replicate}"
            cell_by_request[custom_id] = (cell, replicate_seed)
            lines.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": generation_body(cell, tasks_per_replicate, replicate_seed),
                }
            )
    _, batch_id = submit_batch(base_url, token, lines, "curriculum-sft-ablation-paired-replicates.jsonl")
    batch = wait_for_batch(base_url, token, batch_id, 5.0)
    raw_output, raw_errors = read_batch_output(base_url, token, batch)
    if raw_errors:
        raise RuntimeError("GLM returned batch errors for paired ablation generation")

    results: dict[str, dict[str, Any]] = {}
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        custom_id = response["custom_id"]
        result = response.get("response") or {}
        if response.get("error") or result.get("status_code") != 200:
            raise RuntimeError(f"GLM request {custom_id} failed")
        payload = tool_arguments(result["body"], "submit_tasks")["tasks"]
        if not isinstance(payload, list):
            raise ValueError(f"GLM request {custom_id} returned tasks as {type(payload).__name__}, expected array")
        if len(payload) != tasks_per_replicate:
            raise ValueError(f"GLM request {custom_id} returned {len(payload)} tasks, expected {tasks_per_replicate}")
        results[custom_id] = payload
    missing = set(cell_by_request) - set(results)
    if missing:
        raise RuntimeError(f"GLM batch omitted requests: {sorted(missing)}")

    rows = []
    for cell in cells:
        tasks: list[dict[str, Any]] = []
        replicates = []
        for replicate in range(replicate_count):
            custom_id = f"ablation-{cell.name}-replicate-{replicate}"
            payload = results[custom_id]
            checks = [verify_task_payload(task) for task in payload]
            tasks.extend(payload)
            replicates.append(
                {
                    "replicate": replicate,
                    "seed": cell_by_request[custom_id][1],
                    "requested": len(payload),
                    "accepted": sum(check.accepted for check in checks),
                    "format_rate": sum(check.format_valid for check in checks) / len(checks),
                    "arithmetic_rate": sum(check.arithmetic_valid for check in checks) / len(checks),
                    "evidence_rate": sum(check.evidence_valid for check in checks) / len(checks),
                }
            )
        checks = [verify_task_payload(task) for task in tasks]
        unique_accepted = unique_accepted_payloads(tasks)
        curriculum_packet = None
        if cell.curriculum is CurriculumCondition.CURRICULUM_CONDITIONED:
            curriculum_packet = CURRICULUM_PACKET
        rows.append(
            {
                "cell": cell.name,
                "curriculum": cell.curriculum,
                "generation_spec": cell.generation_spec,
                "curriculum_packet": curriculum_packet,
                "replicates": replicates,
                "requested": len(tasks),
                "accepted": sum(check.accepted for check in checks),
                "unique_accepted": len(unique_accepted),
                "format_rate": sum(check.format_valid for check in checks) / len(checks),
                "arithmetic_rate": sum(check.arithmetic_valid for check in checks) / len(checks),
                "evidence_rate": sum(check.evidence_valid for check in checks) / len(checks),
                "tasks": tasks,
            }
        )

    output_path = StoragePath(output)
    output_path.mkdirs()
    ledger = {
        "batch_id": batch_id,
        "base_seed": seed,
        "replicate_count": replicate_count,
        "tasks_per_replicate": tasks_per_replicate,
        "cells": rows,
    }
    (output_path / "generation.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks-per-replicate", type=int, default=16)
    parser.add_argument("--replicate-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run_matrix(
        args.output,
        tasks_per_replicate=args.tasks_per_replicate,
        replicate_count=args.replicate_count,
        seed=args.seed,
    )
