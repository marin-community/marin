# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched curriculum/specification/dose matrix for a judge-free canary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from experiments.post_training.curriculum_sft.ablation.tasks import (
    SyntheticFinanceTask,
    held_out_task,
    task_from_payload,
)
from experiments.post_training.curriculum_sft.ablation.verifier import verify_task_payload

SFT_SYSTEM_PROMPT = (
    "Solve the fictional financial calculation. Return exactly one JSON object with keys result and evidence. "
    "result must contain integer gross_profit and margin_bps; evidence must contain the cited evidence IDs."
)


class CurriculumCondition(StrEnum):
    TASK_ONLY = "task-only"
    CURRICULUM_CONDITIONED = "curriculum-conditioned"


class GenerationSpec(StrEnum):
    WEAK = "weak"
    STRICT = "strict-executable"


class SftDose(StrEnum):
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class AblationCell:
    """One arm; all arms share the same held-out task IDs and token budget."""

    curriculum: CurriculumCondition
    generation_spec: GenerationSpec
    dose: SftDose
    accepted_examples: int = 16

    @property
    def name(self) -> str:
        return f"{self.curriculum}__{self.generation_spec}__dose-{self.dose}"

    @property
    def train_epochs(self) -> int:
        return {SftDose.LOW: 1, SftDose.HIGH: 4}[self.dose]


def ablation_matrix(*, accepted_examples: int = 16) -> tuple[AblationCell, ...]:
    """Return the full 2x2x2 matrix in a stable order."""

    return tuple(
        AblationCell(curriculum, generation_spec, dose, accepted_examples=accepted_examples)
        for curriculum in CurriculumCondition
        for generation_spec in GenerationSpec
        for dose in SftDose
    )


def build_generation_prompt(
    cell: AblationCell,
    *,
    subject_area: str,
    task_family: str,
    curriculum_packet: str,
    task_count: int,
) -> str:
    """Build a GLM prompt whose factors are explicit and auditable."""

    curriculum = ""
    if cell.curriculum is CurriculumCondition.CURRICULUM_CONDITIONED:
        curriculum = (
            "Use this exact curriculum section as design guidance, without copying benchmark questions:\n"
            f"{curriculum_packet}\n"
        )
    if cell.generation_spec is GenerationSpec.STRICT:
        specification = (
            "Every task must ask for gross profit (revenue minus operating cost) and gross margin in basis points "
            "(gross profit divided by revenue times 10000). Use positive integer revenue and operating_cost with "
            "operating_cost below revenue. Choose values whose basis-point answer is integral, make the exact answer "
            "consistent with the facts, include both figures in the question, and use exactly the evidence IDs "
            "disclosure.revenue and disclosure.operating_cost."
        )
    else:
        specification = (
            "Make each question self-contained, accurate, and answerable from its supplied figures. Give the correct "
            "answer and relevant evidence IDs. Vary the fictional issuers, values, and wording."
        )
    return (
        f"Subject area: {subject_area}\nTask family: {task_family}\n"
        f"Generate {task_count} distinct fictional tasks through the provided tool. Do not use or imitate any "
        "benchmark examples.\n"
        f"{curriculum}{specification}"
    )


def _assistant_target(task: SyntheticFinanceTask) -> str:
    return json.dumps(
        {
            "result": task.expected_result,
            "evidence": list(task.evidence_ids),
        },
        separators=(",", ":"),
    )


def _sft_row(cell: AblationCell, payload: Mapping[str, Any]) -> dict[str, Any]:
    task = task_from_payload(dict(payload))
    verification = verify_task_payload(payload)
    if not verification.accepted:
        raise ValueError(f"unverified task reached SFT row construction: {task.task_id}")
    return {
        "id": task.task_id,
        "messages": [
            {
                "role": "system",
                "content": SFT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": payload["question"]},
            {"role": "assistant", "content": _assistant_target(task)},
        ],
        "metadata": {
            "source_task_id": task.task_id,
            "curriculum_condition": cell.curriculum,
            "generation_spec": cell.generation_spec,
            "accepted_example_budget": cell.accepted_examples,
            "format_valid": verification.format_valid,
            "arithmetic_valid": verification.arithmetic_valid,
            "evidence_valid": verification.evidence_valid,
            "deterministic_task_verification": "accepted",
        },
    }


def generated_payloads_to_rows(
    cell: AblationCell,
    payloads: Sequence[object],
) -> list[dict[str, Any]]:
    """Convert accepted GLM payloads into canonical SFT messages.

    Every generation arm uses one structured payload contract so output format
    cannot masquerade as a prompt-specification effect. The local oracle filters
    every arm before training. Curriculum and specification affect only what GLM
    generates; all accepted rows receive the same SFT wrapper.
    """

    accepted: list[Mapping[str, Any]] = []
    task_ids: set[str] = set()
    questions: set[str] = set()
    fact_tuples: set[tuple[int, int]] = set()
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        verification = verify_task_payload(payload)
        if not verification.accepted:
            continue
        facts = payload["facts"]
        task_id = payload["task_id"]
        question = " ".join(payload["question"].lower().split())
        fact_tuple = (facts["revenue"], facts["operating_cost"])
        if task_id in task_ids or question in questions or fact_tuple in fact_tuples:
            continue
        task_ids.add(task_id)
        questions.add(question)
        fact_tuples.add(fact_tuple)
        accepted.append(payload)
        if len(accepted) == cell.accepted_examples:
            break
    if len(accepted) != cell.accepted_examples:
        raise ValueError(
            f"found {len(accepted)} unique oracle-accepted {cell.generation_spec} payloads, "
            f"needed {cell.accepted_examples}"
        )
    return [_sft_row(cell, payload) for payload in accepted]


def build_sft_rows(
    cell: AblationCell,
    *,
    generated_payloads: Sequence[object],
) -> list[dict[str, Any]]:
    """Build matched canonical rows from accepted generated payloads."""

    return generated_payloads_to_rows(cell, generated_payloads)


def build_held_out_tasks(*, task_count: int, seed: int = 17) -> list[SyntheticFinanceTask]:
    """Build the same held-out oracle set for every ablation arm."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return [held_out_task(index, seed=seed) for index in range(task_count)]


def row_task_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the strict answer contract from an SFT row for verification."""

    messages = row["messages"]
    question = next(message["content"] for message in messages if message["role"] == "user")
    assistant = json.loads(next(message["content"] for message in messages if message["role"] == "assistant"))
    return {"question": question, "answer": assistant["result"], "evidence": assistant["evidence"]}
