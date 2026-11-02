# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's judge-backed abstention and MultiChallenge rewards."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, final_verdict, last_boxed_answer

_MULTICHALLENGE_SYSTEM = (
    "You are a precise evaluator. Assess responses objectively based on the given criteria. "
    "Analyze the response carefully against the evaluation question."
)
_MULTICHALLENGE_PROMPT = """You are evaluating whether a model's response meets a specific criterion.

CONVERSATION CONTEXT:
{context}

MODEL'S FINAL RESPONSE:
{response}

EVALUATION QUESTION:
{question}

EXPECTED ANSWER: {pass_criteria}

Does the model's response satisfy the criterion described in the evaluation question?
Think step by step, then respond with exactly [[YES]] or [[NO]] on the last line."""


def _normalize_answer(value: str) -> str:
    value = re.sub(r"\b(a|an|the)\b", " ", value.lower().strip())
    return " ".join(re.sub(r"[^\w\s]", "", value).split())


def grade_abstention(response: str, record: dict[str, Any], judge: OpenAIJudge) -> tuple[float, dict[str, Any]]:
    response = final_answer_text(response)
    extracted = last_boxed_answer(response) or response
    if _normalize_answer(extracted) == _normalize_answer("[IDK]"):
        return 0.5, {"verdict": "abstain", "extracted_answer": extracted}

    template = files(__package__).joinpath("abstention_prompt.txt").read_text()
    prompt = template.format(
        question=record.get("question", ""),
        target=record.get("answer", ""),
        predicted_answer=extracted,
    )
    judge_output = judge.generate([{"role": "user", "content": prompt}])
    grade = final_verdict(judge_output, {"A", "B", "C"})
    verdict = {"A": "correct", "B": "incorrect", "C": "abstain"}[grade]
    reward = {"A": 1.0, "B": 0.0, "C": 0.5}[grade]
    return reward, {
        "verdict": verdict,
        "extracted_answer": extracted,
        "judge_output": judge_output,
        "omniscience_index": 1.0 if grade == "A" else -1.0 if grade == "B" else 0.0,
    }


def grade_multichallenge(
    response: str,
    record: dict[str, Any],
    judge: OpenAIJudge,
) -> tuple[float, dict[str, Any]]:
    response = final_answer_text(response)
    rubric = record.get("rubric") or record.get("metadata", {}).get("rubric") or []
    context = record.get("context", "")

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        pass_criteria = str(item.get("pass_criteria", "YES"))
        prompt = _MULTICHALLENGE_PROMPT.format(
            context=context,
            response=response,
            question=item.get("question", ""),
            pass_criteria=pass_criteria,
        )
        output = judge.generate(
            [
                {"role": "system", "content": _MULTICHALLENGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        verdict = final_verdict(output, {"YES", "NO", "[[YES]]", "[[NO]]"}).removeprefix("[[").removesuffix("]]")
        expected = pass_criteria.upper()
        score = float(verdict == expected) if expected in {"YES", "NO"} else float(verdict == "YES")
        return {"question": item.get("question", ""), "verdict": verdict, "score": score, "judge_output": output}

    with ThreadPoolExecutor(max_workers=max(1, len(rubric))) as executor:
        evaluations = list(executor.map(evaluate, rubric))
    reward = sum(item["score"] for item in evaluations) / len(evaluations) if evaluations else 0.0
    return reward, {
        "rubric_evaluations": evaluations,
        "num_passed": sum(item["score"] >= 0.99 for item in evaluations),
        "num_total": len(evaluations),
    }
