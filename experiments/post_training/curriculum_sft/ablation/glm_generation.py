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

from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from pydantic import Field
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_parquet_file

from experiments.post_training.curriculum_sft.ablation.generated_tasks import (
    GENERATED_TASK_SCHEMA,
    GENERATED_TASKS_FILENAME,
    GENERATION_FILENAME,
    RAW_RESPONSES_FILENAME,
    generated_task_record,
)
from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    build_generation_prompt,
    unique_accepted_payloads,
)
from experiments.post_training.curriculum_sft.ablation.verifier import verify_task_payload
from experiments.post_training.glm import (
    DEFAULT_GLM_RELAY_JOB,
    GLM_BULK_TOKEN_ENV,
    GLM_MODEL,
    resolve_glm_base_url,
)
from experiments.post_training.task_curriculum.models import StrictModel

logger = logging.getLogger(__name__)

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


class FinanceFacts(StrictModel):
    revenue: int
    operating_cost: int


class FinanceAnswer(StrictModel):
    gross_profit: int
    margin_bps: int


class GeneratedFinanceTask(StrictModel):
    task_id: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    facts: FinanceFacts
    question: str = Field(min_length=1)
    answer: FinanceAnswer
    evidence: list[str]


class GeneratedFinanceTasks(StrictModel):
    tasks: list[GeneratedFinanceTask]


GENERATED_TASKS_TOOL = StructuredTool(
    name="submit_tasks",
    description="Submit the generated fictional tasks.",
    output_type=GeneratedFinanceTasks,
)


def generation_body(cell: AblationCell, expected_tasks: int, seed: int) -> dict[str, Any]:
    prompt = build_generation_prompt(
        cell,
        subject_area="financial reporting",
        task_family=TASK_FAMILY,
        curriculum_packet=CURRICULUM_PACKET,
        task_count=expected_tasks,
    )
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Generate fictional tasks and call submit_tasks exactly once."},
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0.8,
        "seed": seed,
        "max_tokens": 12000,
    }
    body.update(GENERATED_TASKS_TOOL.request_fields())
    return body


def _request_id(cell: AblationCell, replicate: int) -> str:
    return f"ablation-{cell.name}-replicate-{replicate}"


def run_matrix(
    output: str,
    *,
    tasks_per_replicate: int = 16,
    replicate_count: int = 4,
    seed: int = 17,
    relay_job: str = DEFAULT_GLM_RELAY_JOB,
) -> None:
    """Run paired generation replicates and write task Parquet plus an audit manifest."""

    cells = tuple(
        AblationCell(curriculum, generation_spec)
        for curriculum in CurriculumCondition
        for generation_spec in GenerationSpec
    )
    base_url = resolve_glm_base_url(relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    batch_client = OpenAIBatchClient(base_url, token)
    lines = []
    cell_by_request: dict[str, tuple[AblationCell, int]] = {}
    for replicate in range(replicate_count):
        replicate_seed = seed + replicate
        for cell in cells:
            custom_id = _request_id(cell, replicate)
            cell_by_request[custom_id] = (cell, replicate_seed)
            lines.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": CHAT_COMPLETIONS_ENDPOINT,
                    "body": generation_body(cell, tasks_per_replicate, replicate_seed),
                }
            )
    submission = batch_client.submit(lines, "curriculum-sft-ablation-paired-replicates.jsonl")
    batch_id = submission.batch_id
    batch = batch_client.wait(batch_id, 5.0)
    batch_output = batch_client.output(batch)
    raw_output, raw_errors = batch_output.output, batch_output.errors
    if raw_errors:
        raise RuntimeError("GLM returned batch errors for paired ablation generation")

    results: dict[str, list[dict[str, Any]]] = {}
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        custom_id = response["custom_id"]
        if custom_id not in cell_by_request:
            raise ValueError(f"GLM batch returned unknown request ID: {custom_id}")
        if custom_id in results:
            raise ValueError(f"GLM batch returned duplicate request ID: {custom_id}")
        result = response.get("response") or {}
        if response.get("error") or result.get("status_code") != 200:
            raise RuntimeError(f"GLM request {custom_id} failed")
        payload = [task.model_dump(mode="json") for task in GENERATED_TASKS_TOOL.parse(result["body"]).tasks]
        if len(payload) != tasks_per_replicate:
            raise ValueError(f"GLM request {custom_id} returned {len(payload)} tasks, expected {tasks_per_replicate}")
        results[custom_id] = payload
    missing = set(cell_by_request) - set(results)
    if missing:
        raise RuntimeError(f"GLM batch omitted requests: {sorted(missing)}")

    rows = []
    task_records = []
    for cell in cells:
        tasks: list[dict[str, Any]] = []
        replicates = []
        for replicate in range(replicate_count):
            custom_id = _request_id(cell, replicate)
            payload = results[custom_id]
            checks = [verify_task_payload(task) for task in payload]
            tasks.extend(payload)
            task_records.extend(
                generated_task_record(
                    cell,
                    replicate=replicate,
                    seed=cell_by_request[custom_id][1],
                    payload=task,
                )
                for task in payload
            )
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
            }
        )

    output_path = StoragePath(output)
    output_path.mkdirs()
    task_data_path = output_path / GENERATED_TASKS_FILENAME
    task_data_path.parent.mkdirs()
    write_parquet_file(task_records, str(task_data_path), schema=GENERATED_TASK_SCHEMA)
    (output_path / RAW_RESPONSES_FILENAME).write_text(raw_output.rstrip() + "\n")
    ledger = {
        "batch_id": batch_id,
        "base_seed": seed,
        "replicate_count": replicate_count,
        "tasks_per_replicate": tasks_per_replicate,
        "task_data": GENERATED_TASKS_FILENAME,
        "raw_responses": RAW_RESPONSES_FILENAME,
        "cells": rows,
    }
    (output_path / GENERATION_FILENAME).write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")


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
