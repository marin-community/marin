# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PHYSICS-Verified prompts and conversion.

The system prompt and judge prompt are copied from ``eval/prompts.py`` of ``yale-nlp/physics-verified``
@ 724170025e35, Copyright (c) 2025 Yale-NLP, MIT License. The judge prompt's ``{response}`` field is
renamed ``{candidate}`` and its numeric tolerance is fixed at the harness's 1%.
"""

from collections.abc import Mapping
from typing import Any

from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

PHYSICS_VERIFIED_SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answers}\nAnswers: {your final answers, one per line, in the order the question asks for them; begin each line with the subquestion label, such as (a), when the question labels its parts}\nConfidence: {your confidence score between 0% and 100% for your answers}"  # noqa: E501 - exact source prompt
PHYSICS_VERIFIED_JUDGE_TEMPLATE = 'Judge whether the following [response] to [question] correctly gives each result in the list of [reference_answers] below. The reference answers are precise and authoritative.\n\n[question]: {question}\n\n[response]: {candidate}\n\n[reference_answers]:\n{references}\n\nFor EACH reference answer, in order, decide independently:\n\nextracted_final_answer: the response\'s final answer for the same requested result, copied from its Answers section or, if absent there, from its final statement of that result. When the reference label is a part marker such as (a) or (b)(ii), take only what the response gives for that same part; an answer the response places under a different part does not count. When the label is an ordinal (1, 2, ...), the question has no part markers, so identify the answer by the quantity it gives. If the response\'s answer line for that part states only a choice, a comparison, or a qualitative statement, but its explanation for that same part states the requested quantity, extract that quantity. Use \'None\' if the response gives no final answer for it.\n\nreasoning: explain why the extracted answer matches the reference answer or not, focusing only on meaningful differences. Do not comment on background, do not solve the problem, and do not argue for any answer other than the reference answer.\n\ncorrect: \'yes\' only if the extracted answer is equivalent to the reference answer under these rules, otherwise \'no\':\n1. Symbolic answers: algebraically equivalent expressions are correct, including rearrangements and notation defined in the question. A missing or different factor, sign, exponent, or variable dependence is incorrect. Quantum state vectors or wavefunctions that differ only by an overall phase factor are equivalent.\n2. Numeric answers: correct if within 1% relative difference of the reference value, or equal to it when rounded to the reference\'s significant figures. For order-of-magnitude estimates, the same order of magnitude is correct. Units must be equivalent (unit conversions are fine); a missing unit is incorrect unless the question fixes the unit.\n3. If the reference states a formula followed by its value, compare the final result; the response must not contradict either.\n4. Verbal answers: correct if they state the same conclusion, choice, or identification. Extra correct detail is fine; any contradiction is incorrect. When the reference gives a formula, value, or explicit result, a qualitative description of it is incorrect unless the response also states that formula or value for the same part.\n5. Incorrect if the response gives several different candidate answers for one result, hedges between alternatives, answers a different quantity, gives the result only under a different part, or gives no final answer. Intermediate steps shown in the reference are not required.\n6. An answer that equals the reference once a convention or conversion the response itself states is applied is correct. The response must state the convention, and the two must then agree exactly; a convention it does not state, or one that still leaves a difference in factor, sign, exponent, or variable dependence, does not make a wrong answer correct.\n\nAlso extract confidence: the confidence score between 0 and 100 stated in the response; use 100 if none is given.\n\nReturn only json in exactly this shape, with one entry per reference answer in the same order:\n{{"answers": [{{"index": 0, "label": "(a)", "extracted_final_answer": "...", "reasoning": "...", "correct": "yes"}}], "confidence": 85}}'  # noqa: E501 - exact source prompt
# The harness's judge budget; reasoning judges spend much of it before the JSON verdict.
JUDGE_MAX_COMPLETION_TOKENS = 32768
# GLM-5.3-Flash often reasons past the default 120 s on long responses. Each token budget gets its own
# transport deadline, so both budgets together fit inside the evaluator's 600 s verifier timeout.
JUDGE_REQUEST_TIMEOUT = 290.0
JUDGE_TRANSPORT_BUDGET = 290.0


def is_text_only(row: dict[str, Any], _context: ConversionContext) -> bool:
    return not row["figures"]


def physics_verified_judge(row: Mapping[str, Any]) -> JudgeSpec:
    return JudgeSpec(
        rubric="answers",
        question=row["questions"],
        references=tuple(
            f"label {label}: {answer}" for label, answer in zip(row["answer_labels"], row["final_answers"], strict=True)
        ),
        prompt_template=PHYSICS_VERIFIED_JUDGE_TEMPLATE,
        exact_gate=False,
        max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
        # A reply that exhausts the first budget is retried once with twice the room.
        incomplete_retry_tokens=2 * JUDGE_MAX_COMPLETION_TOKENS,
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=3,
        transport_budget=JUDGE_TRANSPORT_BUDGET,
        # Use the judge deployment selected by the evaluation launcher.
        model="",
    )


def convert_physics_verified(row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(
            TextMessage(role="system", content=PHYSICS_VERIFIED_SYSTEM_PROMPT),
            TextMessage(role="user", content=row.data["questions"]),
        ),
        package=verifyit_package(physics_verified_judge(row.data)),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:physics-verified", "benchmark:PHYSICS-Verified")})


def physics_verified_item(task: TaskSpec) -> EvaluationItem:
    """Judge only the final answer, as the harness does with text after any closing think tag."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"domain": source_evidence(task)["domain"]}}
    )
