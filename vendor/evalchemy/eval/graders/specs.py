"""Reusable item graders selected by sources through GraderSpec."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from eval.contracts.eval_spec import EvalSpec, Grade
from eval.execution.grading import Grader
from eval.runtime.resources import ExecutionMode
from eval.graders.program_suites import program_suite
from eval.graders.instructions import instruction_following
from eval.graders.conversations import livebench, mtbench, turn_ratings
from eval.graders.structured import code_line, gsm8k_flexible, json_answer
from eval.graders.judged import mixeval, mixeval_fallback, mixeval_score, wildbench, wildbench_score
from eval.graders.swebench import swe_patch
from eval.graders.alpaca import alpaca, alpaca_preference
from eval.graders.math_specs import answer_number, boxed_math, gsm8k, jee, letter_choice, math_equivalence, matharena, mmlu_choice, nupa
from eval.graders.programs import crux, live_code, program
from eval.graders.judged import equivalence_judge, equivalence_label, judge_failed, simpleqa, simpleqa_label
from eval.graders.structured import nonce_similarity
from eval.graders.graphwalks import graphwalks


def exact_match(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = str(responses[-1]).strip()
    expected = str(spec.target).strip()
    return Grade({"accuracy": float(answer == expected)}, filtered_resps=[answer])


GRADERS = {
    "graphwalks": Grader(graphwalks),
    "exact_match": Grader(exact_match), "boxed_math": Grader(boxed_math), "gsm8k": Grader(gsm8k),
    "letter_choice": Grader(letter_choice), "mmlu_choice": Grader(mmlu_choice),
    "live_code": Grader(live_code, ExecutionMode.PROCESS_ISOLATED),
    "matharena": Grader(matharena),
    "program": Grader(program, ExecutionMode.PROCESS_ISOLATED),
    "answer_number": Grader(answer_number), "jee": Grader(jee), "nupa": Grader(nupa),
    # The shared worker runs canonical fork-based sandboxes off the driver loop.
    "program_suite": Grader(program_suite, ExecutionMode.SANDBOXED),
    "simpleqa": Grader(simpleqa), "simpleqa_label": Grader(simpleqa_label),
    "judge_failed": Grader(judge_failed),
    "instructions": Grader(instruction_following),
    "nonce_similarity": Grader(nonce_similarity), "crux": Grader(crux, ExecutionMode.PROCESS_ISOLATED),
    "math_equivalence": Grader(math_equivalence), "equivalence_judge": Grader(equivalence_judge),
    "equivalence_label": Grader(equivalence_label),
    "mtbench": Grader(mtbench), "turn_ratings": Grader(turn_ratings),
    "livebench": Grader(livebench, ExecutionMode.PROCESS_ISOLATED),
    "gsm8k_flexible": Grader(gsm8k_flexible), "json_answer": Grader(json_answer), "code_line": Grader(code_line),
    "mixeval": Grader(mixeval), "mixeval_score": Grader(mixeval_score), "mixeval_fallback": Grader(mixeval_fallback),
    "wildbench": Grader(wildbench), "wildbench_score": Grader(wildbench_score),
    "swe_patch": Grader(swe_patch, ExecutionMode.PROCESS_ISOLATED),
    "alpaca": Grader(alpaca), "alpaca_preference": Grader(alpaca_preference),
}
